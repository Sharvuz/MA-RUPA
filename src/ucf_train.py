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
    compute_fg_tv_smoothness_loss,
    compute_temporal_smoothness_loss,
    compute_confidence_gated_consistency_loss,
    compute_residual_event_loss,
    compute_clasm_loss,
    compute_bkg_loss,
    compute_consistency_loss,
    compute_event_loss,
    compute_gather_loss,
    compute_orthogonality_loss,
    compute_normal_contrastive_loss
)
import ucf_option


def train(args):
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

    # Data loaders: default to full 10 crops per video (16,000 steps/epoch, 250 iters/epoch)
    sample_one_crop = getattr(args, 'sample_one_crop_per_video', getattr(args, 'enable_crop_reduction', False))
    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, normal=True,
        data_dir=args.data_dir, sample_one_crop_per_video=sample_one_crop
    )
    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, label_map, normal=False,
        data_dir=args.data_dir, sample_one_crop_per_video=sample_one_crop
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

    # Single-stage End-to-End AdamW optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    iters_per_epoch = min(len(normal_loader), len(anomaly_loader))
    total_iters = max(1, args.max_epoch * iters_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=args.lr * 0.01
    )

    best_auc = 0.0
    best_ap = 0.0
    start_epoch = 0

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
    print(f"  Iters per epoch: {iters_per_epoch} | Sample 1 crop per video: {sample_one_crop}")
    print("=" * 70, flush=True)

    for epoch in range(start_epoch, args.max_epoch):
        model.train()
        epoch_idx = epoch + 1

        # Resample crops per video every epoch to avoid memorization
        if hasattr(normal_dataset, 'resample_crops'):
            normal_dataset.resample_crops()
        if hasattr(anomaly_dataset, 'resample_crops'):
            anomaly_dataset.resample_crops()

        # Smooth Cosine Curriculum Ramp-up
        omega = get_curriculum_omega(epoch_idx, getattr(args, 'curriculum_epochs', 5.0))

        loss_accum = {
            'mil': 0.0, 'aux': 0.0, 'clasm': 0.0, 'bkg': 0.0, 'consist': 0.0,
            'smooth': 0.0, 'gather': 0.0, 'orth': 0.0, 'residual': 0.0, 'gate': 0.0, 'total': 0.0
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

            # 1. Primary & Auxiliary MIL Losses (AT-BCE + VAT-K + OHNM + Snippet Dropout)
            topk_r = getattr(args, 'topk_ratio', 10)
            snip_drop = getattr(args, 'snippet_dropout', 0.15)

            loss_mil_routing = compute_at_bce_topk_mil_loss(
                p_routing, binary_labels, lengths_batch, device=device,
                topk_ratio=topk_r, vat_k=True, snippet_dropout=snip_drop,
                lambda_ent=0.02, lambda_ohnm=0.2, target_loss=0.15, is_training=True
            )
            loss_mil_det = compute_at_bce_topk_mil_loss(
                s_det, binary_labels, lengths_batch, device=device,
                topk_ratio=topk_r, vat_k=True, snippet_dropout=snip_drop,
                lambda_ent=0.02, lambda_ohnm=0.2, target_loss=0.15, is_training=True
            )
            loss_mil_aux = compute_at_bce_topk_mil_loss(
                s_aux, binary_labels, lengths_batch, device=device,
                topk_ratio=topk_r, vat_k=True, snippet_dropout=snip_drop,
                lambda_ent=0.02, lambda_ohnm=0.2, target_loss=0.15, is_training=True
            )

            # 2. Multi-class Semantic Alignment & Consistency Losses
            loss_clasm_bb = compute_clasm_loss(logits2, one_hot_labels, lengths_batch, k_ratio=topk_r)
            if 'semantic_logits' in dnp_dict:
                loss_clasm_rupa = compute_clasm_loss(dnp_dict['semantic_logits'], one_hot_labels, lengths_batch, k_ratio=topk_r)
            else:
                loss_clasm_rupa = torch.tensor(0.0, device=device)

            loss_bkg = compute_bkg_loss(logits4, num_classes=len(prompt_text))

            # Confidence-Gated Asymmetric Consistency
            if 'reconstruction_scores' in dnp_dict:
                loss_consistency = compute_confidence_gated_consistency_loss(
                    s_det=s_det, s_rupa=dnp_dict['reconstruction_scores'], lengths=lengths_batch, device=device
                )
            elif 'original_features' in dnp_dict and 'reconstructed_features' in dnp_dict:
                loss_consistency = compute_consistency_loss(
                    p_routing, dnp_dict['original_features'], dnp_dict['reconstructed_features'], lengths_batch, s_det=s_det
                )
            else:
                loss_consistency = torch.tensor(0.0, device=device)

            loss_event = compute_event_loss(logits3, one_hot_labels)
            loss_dnp_normal = dnp_dict.get('dnp_normal_loss', torch.tensor(0.0, device=device)).mean()

            # 3. Feature-Guided Edge-Preserving TV Smoothness & Compactness Loss (DODO Training Stage)
            loss_smooth = compute_fg_tv_smoothness_loss(
                scores=p_routing,
                visual_features=visual_batch,
                lengths=lengths_batch,
                gamma_edge=getattr(args, 'gamma_edge', 5.0),
                lambda_compact=0.1,
                binary_labels=binary_labels,
                device=device
            )

            # 4. RUPA Prototype Losses (modulated by smooth cosine curriculum omega(e))
            loss_gather = compute_gather_loss(dnp_dict)
            loss_orth = compute_orthogonality_loss(dnp_dict)

            if 'residual_semantic_logits' in dnp_dict:
                loss_residual = compute_residual_event_loss(
                    dnp_dict['residual_semantic_logits'], one_hot_labels, lengths_batch, k_ratio=topk_r
                )
            else:
                loss_residual = torch.tensor(0.0, device=device)

            gate_val = dnp_dict.get('gate', torch.tensor(0.0, device=device))
            loss_gate = (gate_val ** 2).mean()

            # Text Orthogonality Regularization
            loss_text_orth = compute_normal_contrastive_loss(
                text_features, num_classes=len(prompt_text)
            )

            # Total Loss Computation with Pareto Optimal Weighting
            loss_clasm = loss_clasm_bb + 0.8 * loss_clasm_rupa
            total_loss = (
                loss_mil_routing
                + 0.8 * loss_mil_det
                + getattr(args, 'lambda_aux', 0.5) * loss_mil_aux
                + getattr(args, 'loss2_weight', 1.0) * loss_clasm_bb
                + 0.8 * loss_clasm_rupa
                + getattr(args, 'lambda_bkg', 0.5) * loss_bkg
                + getattr(args, 'lambda_consistency', 0.25) * loss_consistency
                + getattr(args, 'lambda_smooth', 0.05) * loss_smooth
                + 0.1 * loss_text_orth
                + omega * (
                    getattr(args, 'lambda_gather', 0.05) * loss_gather
                    + getattr(args, 'lambda_orth', 0.02) * loss_orth
                    + getattr(args, 'lambda_residual', 0.05) * loss_residual
                    + 0.5 * loss_event
                    + 0.2 * loss_dnp_normal
                    + getattr(args, 'lambda_gate', 0.005) * loss_gate
                )
            )

            optimizer.zero_grad()
            total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

            optimizer.step()
            scheduler.step()

            # Accumulate logs
            loss_accum['mil'] += loss_mil_routing.item()
            loss_accum['aux'] += loss_mil_aux.item()
            loss_accum['clasm'] += loss_clasm.item()
            loss_accum['bkg'] += loss_bkg.item()
            loss_accum['consist'] += loss_consistency.item()
            loss_accum['smooth'] += loss_smooth.item()
            loss_accum['gather'] += loss_gather.item()
            loss_accum['orth'] += loss_orth.item()
            loss_accum['residual'] += loss_residual.item()
            loss_accum['gate'] += loss_gate.item()
            loss_accum['total'] += total_loss.item()

            if (step_idx + 1) % 20 == 0 or (step_idx + 1) == iters_per_epoch:
                current_step = (step_idx + 1) * args.batch_size
                percent = ((step_idx + 1) / iters_per_epoch) * 100
                print(f"  [Epoch {epoch_idx}/{args.max_epoch} Step {current_step}/{iters_per_epoch * args.batch_size} ({percent:.1f}%)] "
                      f"Total: {total_loss.item():.4f} | MIL: {loss_mil_routing.item():.4f} "
                      f"CLASM: {loss_clasm.item():.4f} | Consist: {loss_consistency.item():.4f} | Aux: {loss_mil_aux.item():.4f} | omega: {omega:.2f}", flush=True)

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
        print(f"Loss Total: {avg_total:.4f} | MIL: {avg_mil:.4f} | CLASM: {avg_clasm:.4f} | Consist: {avg_consist:.4f} | Aux: {avg_aux:.4f} | Gather: {avg_gather:.4f} | Orth: {avg_orth:.4f} | Res: {avg_residual:.4f} | Smooth: {avg_smooth:.4f} | Omega: {omega:.2f}")

        # Evaluate on Test Set
        eval_model = model.module if hasattr(model, 'module') else model
        auc_val, ap_val = test(
            eval_model, test_loader, args.visual_length, prompt_text, gt, gtsegments,
            gtlabels, args.DNP_use, device, args
        )

        is_best = False
        saved_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        if auc_val > best_auc:
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
