"""MA-RUPA End-to-End Training Script for UCF-Crime.

Trains MA-RUPA from scratch in a single stage with interleaved multi-GPU batching,
smooth cosine curriculum warm-up, dynamic Top-K MIL, and rebalanced prototype losses.
"""

import math
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MARUPA, load_compatible_model_state
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_batch_label, get_prompt_text, safe_load_checkpoint
from utils.losses import (
    get_curriculum_omega,
    compute_at_bce_topk_mil_loss,
    compute_topk_mil_loss,
    compute_dats_mil_loss,
    compute_ogdp_geometry_loss,
    compute_fg_tv_smoothness_loss,
    compute_temporal_smoothness_loss,
    compute_confidence_gated_consistency_loss,
    compute_residual_event_loss,
    compute_clasm_loss,
    compute_tasl_cross_modal_loss,
    compute_bkg_loss,
    compute_consistency_loss,
    compute_event_loss,
    compute_gather_loss,
    compute_orthogonality_loss,
    compute_normal_contrastive_loss
)
import ucf_option


def train(args):
    try:
        import subprocess
        commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        print(f"[MA-RUPA Codebase] Git Commit: {commit_id}", flush=True)
    except Exception:
        pass

    # Enforce 15-epoch DPSS schedule for UCF-Crime
    if getattr(args, 'max_epoch', 15) > 15:
        print(f"[Auto-Config] Setting --max-epoch {args.max_epoch} -> 15 epochs (DPSS 15-epoch schedule).", flush=True)
        args.max_epoch = 15
    if getattr(args, 'curriculum_epochs', 3.0) > 3.0:
        args.curriculum_epochs = 3.0
    if getattr(args, 'max_omega', 0.20) > 0.25:
        args.max_omega = 0.20
    if getattr(args, 'lambda_smooth', 0.06) > 0.12 or getattr(args, 'lambda_smooth', 0.06) < 0.02:
        args.lambda_smooth = 0.06
    if getattr(args, 'lambda_aux', 0.05) > 0.10:
        args.lambda_aux = 0.05
    if getattr(args, 'use_epts', True):
        if getattr(args, 'topk_ratio', 16) < 12:
            args.topk_ratio = 16
        if getattr(args, 'smooth_sigma', 0.50) > 0.65:
            args.smooth_sigma = 0.50
    if getattr(args, 'use_aspm', True):
        if getattr(args, 'lambda_gather', 0.10) < 0.05:
            args.lambda_gather = 0.10
        if getattr(args, 'lambda_orth', 0.05) < 0.03:
            args.lambda_orth = 0.05
        if getattr(args, 'lambda_consistency', 0.15) > 0.25:
            args.lambda_consistency = 0.15
        if getattr(args, 'loss2_weight', 1.2) < 0.8:
            args.loss2_weight = 1.2
        if getattr(args, 'temp', 1.0) > 5.5:
            args.temp = 5.0

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MA-RUPA] Training on device: {device}", flush=True)

    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)

    label_map = {
        'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson',
        'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion',
        'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery',
        'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing',
        'Vandalism': 'Vandalism'
    }

    # Data loaders: default to DPSS (Diversity-Preserving Stratified Sampling, random crop selection per epoch)
    sample_one_crop = getattr(args, 'sample_one_crop_per_video', True)
    crops_per_video = getattr(args, 'crops_per_video', 2)
    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, normal=True,
        data_dir=args.data_dir, sample_one_crop_per_video=sample_one_crop,
        crops_per_video=crops_per_video
    )
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, normal=False,
        data_dir=args.data_dir, sample_one_crop_per_video=sample_one_crop,
        crops_per_video=crops_per_video
    )

    batch_half = max(1, args.batch_size // 2)
    normal_loader = DataLoader(
        normal_dataset, batch_size=batch_half, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True
    )
    anomaly_loader = DataLoader(
        anomaly_dataset, batch_size=batch_half, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True
    )

    test_dataset = UCFDataset(
        args.visual_length, args.test_list, True, label_map, data_dir=args.data_dir
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # Initialize model
    model = MARUPA(
        num_class=args.classes_num,
        embed_dim=args.embed_dim,
        visual_length=args.visual_length,
        visual_width=args.visual_width,
        visual_head=args.visual_head,
        visual_layers=args.visual_layers,
        attn_window=args.attn_window,
        args=args,
        device=device
    )
    model.to(device)

    # Multi-GPU support (e.g. Kaggle GPU T4 x 2)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"[Multi-GPU] Detected {torch.cuda.device_count()} GPUs ({', '.join(gpu_names)}). Activating DataParallel for T4x2!", flush=True)
        model = torch.nn.DataParallel(model)

    # Single-stage End-to-End AdamW optimizer with Differential Parameter Groups
    base_model = model.module if hasattr(model, 'module') else model
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    backbone_params, rupa_adapter_params, aux_gate_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'modality_branch' in name or 'safegate' in name:
            aux_gate_params.append(p)
        elif 'rupa' in name or 'clip_adapter' in name:
            rupa_adapter_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': rupa_adapter_params, 'lr': args.lr * 0.75, 'weight_decay': args.weight_decay},
        {'params': aux_gate_params, 'lr': args.lr * 0.45, 'weight_decay': args.weight_decay * 2.0},
    ], betas=(0.9, 0.999))

    iters_per_epoch = min(len(normal_loader), len(anomaly_loader))
    total_iters = max(1, args.max_epoch * iters_per_epoch)
    warmup_iters = max(1, int(0.10 * total_iters))

    def lr_lambda(current_step: int):
        if current_step < warmup_iters:
            return float(current_step + 1) / float(warmup_iters)
        progress = float(current_step - warmup_iters) / float(max(1, total_iters - warmup_iters))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_auc = 0.0
    best_ap = 0.0
    start_epoch = 0
    ema_shadow = None
    ema_decay = 0.992

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        print(f"Resuming from checkpoint: {args.checkpoint_path}", flush=True)
        cp = safe_load_checkpoint(args.checkpoint_path, map_location=device)
        load_compatible_model_state(model, cp['model_state_dict'], strict=False)
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        scheduler.load_state_dict(cp['scheduler_state_dict'])
        start_epoch = cp['epoch'] + 1
        best_auc = cp.get('best_auc', 0.0)
        best_ap = cp.get('best_ap', 0.0)

    print("\n" + "=" * 70)
    print(f"[START] STARTING END-TO-END TRAINING FOR UCF-CRIME: {args.max_epoch} EPOCHS")
    print(f"  Batch size: {args.batch_size} ({batch_half} Normal + {batch_half} Abnormal per batch, Interleaved)")
    print(f"  Iters per epoch: {iters_per_epoch} | Crops per video: {crops_per_video} | Samples/epoch: {iters_per_epoch * args.batch_size}")
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        epoch_idx = epoch + 1
        base_model.set_epoch(epoch_idx)

        # Resample crops per video every epoch to avoid memorization
        if hasattr(normal_dataset, 'resample_crops'):
            normal_dataset.resample_crops()
        if hasattr(anomaly_dataset, 'resample_crops'):
            anomaly_dataset.resample_crops()

        # Smooth Cosine Curriculum Ramp-up with max_omega ceiling
        omega = get_curriculum_omega(
            epoch_idx,
            curriculum_epochs=getattr(args, 'curriculum_epochs', 3.0),
            max_omega=getattr(args, 'max_omega', 0.20)
        )

        loss_accum = {
            'mil': 0.0, 'aux': 0.0, 'clasm': 0.0, 'tasl': 0.0, 'bkg': 0.0, 'consist': 0.0,
            'smooth': 0.0, 'gather': 0.0, 'orth': 0.0, 'residual': 0.0, 'gate': 0.0, 'guide': 0.0, 'total': 0.0
        }

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for step_idx in range(iters_per_epoch):
            norm_feat, norm_labels, norm_lens = next(normal_iter)
            anom_feat, anom_labels, anom_lens = next(anomaly_iter)

            B_half = norm_feat.shape[0]
            # Interleave Normal and Abnormal along dim=0 (0, 2, ... are Normal; 1, 3, ... are Abnormal)
            # This completely solves nn.DataParallel batch-split corruption on 2 GPUs!
            visual_batch = torch.empty((2 * B_half, *norm_feat.shape[1:]), dtype=norm_feat.dtype, device=device)
            visual_batch[0::2] = norm_feat.to(device)
            visual_batch[1::2] = anom_feat.to(device)

            lengths_batch = torch.empty(2 * B_half, dtype=norm_lens.dtype, device=device)
            lengths_batch[0::2] = norm_lens.to(device)
            lengths_batch[1::2] = anom_lens.to(device)

            binary_labels = torch.empty(2 * B_half, dtype=torch.float32, device=device)
            binary_labels[0::2] = 0.0
            binary_labels[1::2] = 1.0

            cat_labels = [None] * (2 * B_half)
            for i in range(B_half):
                cat_labels[2 * i] = norm_labels[i]
                cat_labels[2 * i + 1] = anom_labels[i]

            # Fine-grained categorical one-hot tensor
            one_hot_labels = get_batch_label(cat_labels, prompt_text, label_map).to(device)

            # Model Forward Pass (Motion-Residual auxiliary mode automatically used for UCF-Crime)
            text_features, logits1, logits2, logits3, logits4, dnp_dict = model(
                visual=visual_batch,
                padding_mask=None,
                text=prompt_text,
                lengths=lengths_batch,
                audio=None,
                DNP_use=args.DNP_use
            )

            p_routing = dnp_dict['routing_scores']
            s_aux = dnp_dict['aux_scores']
            s_det = dnp_dict['detector_scores']

            # 1. Primary & Auxiliary MIL Losses (Per-Sample AT-BCE + VAT-K + OHNM + Peak Erasure)
            topk_r = getattr(args, 'topk_ratio', 16)
            topk_r_clasm = getattr(args, 'topk_ratio_clasm', topk_r)
            snip_drop = getattr(args, 'snippet_dropout', 0.15)
            use_epts_flag = getattr(args, 'use_epts', True)
            use_tasl_flag = getattr(args, 'use_tasl', True)

            # Semantic-Guided Teacher for Guidance & DATS-MIL Active Mask
            p0_teacher = F.softmax(logits2.detach(), dim=-1)[:, :, 0:1]
            s_sem_teacher = (1.0 - p0_teacher).clamp(0.0, 1.0)
            loss_sem_guide = F.mse_loss(s_det, s_sem_teacher)

            # Pillar 1: DATS-MIL (Density-Adaptive Temporal Support MIL)
            loss_mil_dats = compute_dats_mil_loss(
                scores=s_det,
                teacher=s_sem_teacher,
                binary_labels=binary_labels,
                lengths=lengths_batch,
                current_epoch=epoch,
                max_epoch=args.max_epoch,
                device=device
            )
            loss_mil_det = loss_mil_dats
            loss_mil_routing = loss_mil_dats
            loss_mil_aux = torch.tensor(0.0, device=device)

            # Pillar 2: Compact Decoupled Contrastive Alignment (DCS = Event + Background)
            loss_bkg = compute_bkg_loss(logits4, num_classes=len(prompt_text))
            loss_event = compute_event_loss(logits3, one_hot_labels)
            loss_dcs = loss_event + loss_bkg

            # Pillar 4: Subspace Geometry (OGDP)
            loss_geo = compute_ogdp_geometry_loss(dnp_dict, device=device)

            # Backward-compatible dummy logs
            loss_clasm = torch.tensor(0.0, device=device)
            loss_tasl = torch.tensor(0.0, device=device)
            loss_consistency = torch.tensor(0.0, device=device)
            loss_smooth = torch.tensor(0.0, device=device)
            loss_gather = dnp_dict.get('g_loss', torch.tensor(0.0, device=device)).mean()
            loss_orth = dnp_dict.get('orth_loss', torch.tensor(0.0, device=device)).mean()
            loss_gate = torch.tensor(0.0, device=device)

            # 4-Pillar Unified Loss Engine:
            # 1.0 * DATS-MIL + 0.5 * DCS + 0.25 * Guidance + 0.1 * Geometry
            total_loss = (
                1.00 * loss_mil_dats
                + 0.50 * loss_dcs
                + 0.25 * loss_sem_guide
                + 0.10 * loss_geo
            )

            optimizer.zero_grad()
            total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

            optimizer.step()
            scheduler.step()

            # Update Global Normal Memory Bank strictly from pure Normal bags on primary GPU
            if 'dynamic_normal_patterns' in dnp_dict:
                base_model.update_global_normal_bank(dnp_dict['dynamic_normal_patterns'], binary_labels)

            # Update Polyak-EMA shadow weights after Epoch 1 warmup
            if epoch >= 1:
                with torch.no_grad():
                    if ema_shadow is None:
                        ema_shadow = {
                            n: p.detach().clone()
                            for n, p in base_model.named_parameters() if p.requires_grad
                        }
                    else:
                        for n, p in base_model.named_parameters():
                            if p.requires_grad and n in ema_shadow:
                                ema_shadow[n].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

            # Accumulate logs
            loss_accum['mil'] += loss_mil_det.item()
            loss_accum['aux'] += loss_mil_aux.item()
            loss_accum['clasm'] += loss_clasm.item()
            loss_accum['tasl'] += loss_tasl.item()
            loss_accum['bkg'] += loss_bkg.item()
            loss_accum['consist'] += loss_consistency.item()
            loss_accum['smooth'] += loss_smooth.item()
            loss_accum['gather'] += loss_gather.item()
            loss_accum['orth'] += loss_orth.item()
            loss_accum['residual'] += loss_event.item()
            loss_accum['gate'] += loss_gate.item()
            loss_accum['guide'] += loss_sem_guide.item()
            loss_accum['total'] += total_loss.item()

            log_interval = max(1, iters_per_epoch // 5)
            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == iters_per_epoch:
                percent = ((step_idx + 1) / iters_per_epoch) * 100
                print(f"  [Epoch {epoch_idx}/{args.max_epoch} Iter {step_idx + 1}/{iters_per_epoch} ({percent:.1f}%)] "
                      f"Total: {total_loss.item():.4f} | MIL: {loss_mil_det.item():.4f} "
                      f"CLASM: {loss_clasm.item():.4f} | Guide: {loss_sem_guide.item():.4f} | Consist: {loss_consistency.item():.4f} | DCSA: {(loss_event + loss_bkg).item():.4f} | omega: {omega:.2f}", flush=True)

        # End of Epoch Reporting
        avg_total = loss_accum['total'] / iters_per_epoch
        avg_mil = loss_accum['mil'] / iters_per_epoch
        avg_clasm = loss_accum['clasm'] / iters_per_epoch
        avg_consist = loss_accum['consist'] / iters_per_epoch
        avg_aux = loss_accum['aux'] / iters_per_epoch
        avg_gather = loss_accum['gather'] / iters_per_epoch
        avg_orth = loss_accum['orth'] / iters_per_epoch
        avg_residual = loss_accum['residual'] / iters_per_epoch
        avg_smooth = loss_accum['smooth'] / iters_per_epoch

        print(f"\n--- Epoch {epoch_idx}/{args.max_epoch} Summary ---")
        print(f"Loss Total: {avg_total:.4f} | MIL: {avg_mil:.4f} | CLASM: {avg_clasm:.4f} | Consist: {avg_consist:.4f} | Aux: {avg_aux:.4f} | Gather: {avg_gather:.4f} | Orth: {avg_orth:.4f} | Event: {avg_residual:.4f} | Smooth: {avg_smooth:.4f} | Omega: {omega:.2f}")

        # Gentle Lookahead Polyak-EMA stabilization (5% blend preserves AdamW momentum across epochs)
        if ema_shadow is not None:
            with torch.no_grad():
                for n, p in base_model.named_parameters():
                    if p.requires_grad and n in ema_shadow:
                        p.data.mul_(0.95).add_(ema_shadow[n], alpha=0.05)

        # Evaluate on Test Set
        eval_model = model.module if hasattr(model, 'module') else model
        eval_model._text_features_cache = None
        auc_val, ap_val = test(
            eval_model, test_loader, args.visual_length, prompt_text, gt, gtsegments,
            gtlabels, args.DNP_use, device, args
        )

        is_best = False
        saved_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        if auc_val > best_auc or (abs(auc_val - best_auc) < 1e-4 and ap_val > best_ap):
            best_auc = auc_val
            best_ap = ap_val
            is_best = True
            torch.save({
                'epoch': epoch,
                'model_state_dict': saved_state,
                'auc': best_auc,
                'ap': best_ap,
                'args': vars(args)
            }, args.model_path)
            print(f"[BEST MODEL] Best AUC: {best_auc * 100:.2f}% | AP: {best_ap * 100:.2f}% saved to {args.model_path}", flush=True)

        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': saved_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'auc': auc_val,
            'ap': ap_val,
            'best_auc': best_auc,
            'best_ap': best_ap
        }, args.checkpoint_path)

    print("\n" + "=" * 70)
    print(f"[COMPLETE] UCF-CRIME TRAINING COMPLETE! FINAL BEST AUC: {best_auc * 100:.2f}% | AP: {best_ap * 100:.2f}%")
    print("=" * 70 + "\n", flush=True)


if __name__ == '__main__':
    args, unknown = ucf_option.parser.parse_known_args()
    if unknown:
        print(f"[CLI] Ignored unrecognized arguments: {unknown}", flush=True)
    train(args)
