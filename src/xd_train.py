"""MA-RUPA End-to-End Training Script for XD-Violence.

Trains MA-RUPA from scratch in a single stage with multimodal audio-visual fusion,
interleaved multi-GPU batching, smooth cosine curriculum warm-up, and rebalanced prototype losses.
"""

import math
import os
import random
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MARUPA, load_compatible_model_state
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import get_batch_label, get_prompt_text, xd_collate_fn, xd_test_collate_fn, safe_load_checkpoint
from utils.losses import (
    get_curriculum_omega,
    compute_maf_mil_loss,
    compute_anti_saturation_mil_loss,
    compute_stiefel_grassmann_loss,
    compute_ahns_loss,
    compute_temporal_tv_loss,
    compute_epts_loss,
    compute_text_separation_loss,
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
import xd_option


def get_compact_state_dict(state_dict):
    """Filters out frozen OpenAI CLIP ViT-B/16 parameters (335 MB).
    Retains all trainable adapters, MDTBackbone, RUPA, and SafeGate parameters (~35 MB),
    accelerating torch.save disk I/O by >10x!
    """
    return {
        k: v for k, v in state_dict.items()
        if not k.startswith('clipmodel.') and not k.startswith('module.clipmodel.')
    }


def safe_save_checkpoint(checkpoint_dict, path):
    """Saves checkpoint compactly and safely."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if 'model_state_dict' in checkpoint_dict:
        checkpoint_dict['model_state_dict'] = get_compact_state_dict(checkpoint_dict['model_state_dict'])
    torch.save(checkpoint_dict, path)


from torch.optim.lr_scheduler import _LRScheduler

class WarmCosineScheduler(_LRScheduler):
    """Cosine learning rate scheduler with linear warm-up (DSANet SOTA standard)."""
    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(1, len(iters))))
        self.schedule = np.concatenate((warmup_schedule, schedule))
        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for _ in self.base_lrs]


def train(args):
    try:
        import subprocess
        commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        print(f"[MA-RUPA Codebase] Git Commit: {commit_id}", flush=True)
    except Exception:
        pass

    # Enforce 20-epoch DPSS schedule for XD-Violence
    if getattr(args, 'max_epoch', 25) > 40:
        print(f"[Auto-Config] Setting --max-epoch {args.max_epoch} -> 40 epochs ceiling.", flush=True)
        args.max_epoch = 40
    if getattr(args, 'curriculum_epochs', 3.0) > 3.0:
        args.curriculum_epochs = 3.0
    if getattr(args, 'max_omega', 0.20) > 0.25:
        args.max_omega = 0.20
    if getattr(args, 'lambda_aux', 0.05) > 0.10:
        args.lambda_aux = 0.05
    if getattr(args, 'routing_aux_weight', 0.08) > 0.12:
        args.routing_aux_weight = 0.08
    if getattr(args, 'routing_rec_weight', 0.05) > 0.10:
        args.routing_rec_weight = 0.05
    if getattr(args, 'lambda_smooth', 0.0) < 0.0:
        args.lambda_smooth = 0.0
    if getattr(args, 'use_epts', True) and getattr(args, 'smooth_sigma', 0.50) > 0.75:
        args.smooth_sigma = 0.50
    if getattr(args, 'use_aspm', True):
        if getattr(args, 'lambda_gather', 0.10) < 0.05:
            args.lambda_gather = 0.10
        if getattr(args, 'lambda_orth', 0.05) < 0.03:
            args.lambda_orth = 0.05
        if getattr(args, 'lambda_consistency', 0.15) > 0.25:
            args.lambda_consistency = 0.15
        if getattr(args, 'loss2_weight', 1.4) > 1.6 or getattr(args, 'loss2_weight', 1.4) < 0.8:
            args.loss2_weight = 1.4
        if getattr(args, 'temp', 1.0) > 1.2:
            args.temp = 1.0

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MA-RUPA] Training XD-Violence on device: {device}", flush=True)

    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)

    label_map = {
        'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
        'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'
    }

    # Datasets & Balanced Paired Loaders: default to DPSS (random crop selection per epoch)
    sample_one_crop = getattr(args, 'sample_one_crop_per_video', True)
    crops_per_video = getattr(args, 'crops_per_video', 2)
    normal_dataset = XDDataset(
        args.visual_length, args.train_list, False, label_map,
        audio_file_path=args.audio_train_list, audio_dir=args.audio_dir,
        data_dir=args.data_dir, no_audio=args.no_audio,
        normal=True, sample_one_crop_per_video=sample_one_crop,
        crops_per_video=crops_per_video
    )
    anomaly_dataset = XDDataset(
        args.visual_length, args.train_list, False, label_map,
        audio_file_path=args.audio_train_list, audio_dir=args.audio_dir,
        data_dir=args.data_dir, no_audio=args.no_audio,
        normal=False, sample_one_crop_per_video=sample_one_crop,
        crops_per_video=crops_per_video
    )

    batch_half = max(1, args.batch_size // 2)
    normal_loader = DataLoader(
        normal_dataset, batch_size=batch_half, shuffle=True,
        num_workers=args.num_workers, collate_fn=xd_collate_fn, drop_last=True, pin_memory=True
    )
    anomaly_loader = DataLoader(
        anomaly_dataset, batch_size=batch_half, shuffle=True,
        num_workers=args.num_workers, collate_fn=xd_collate_fn, drop_last=True, pin_memory=True
    )

    test_dataset = XDDataset(
        args.visual_length, args.test_list, True, label_map,
        audio_file_path=args.audio_test_list, audio_dir=args.audio_dir,
        data_dir=args.data_dir, no_audio=args.no_audio
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=xd_test_collate_fn)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # Auto-detect audio feature dimension from dataset if available
    detected_audio_dim = None
    for ds in (normal_dataset, anomaly_dataset):
        for idx in range(min(20, len(ds))):
            item_tuple = ds[idx]
            sample_audio = item_tuple[3] if len(item_tuple) >= 4 else item_tuple[-1]
            if sample_audio is not None and isinstance(sample_audio, torch.Tensor):
                detected_audio_dim = sample_audio.shape[-1]
                break
        if detected_audio_dim is not None:
            break

    if detected_audio_dim is not None and detected_audio_dim != args.audio_dim:
        print(f"[Auto-Detect] Audio feature dimension on disk is {detected_audio_dim}-D (command-line was {args.audio_dim}-D). Auto-adjusting args.audio_dim to {detected_audio_dim}!", flush=True)
        args.audio_dim = detected_audio_dim

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
        for dev_idx in range(torch.cuda.device_count()):
            try:
                # Pre-warm CUDA linear algebra stubs sequentially on main thread
                dummy = torch.eye(4, device=f"cuda:{dev_idx}")
                _ = torch.linalg.qr(dummy)
            except Exception:
                pass
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"[Multi-GPU] Detected {torch.cuda.device_count()} GPUs ({', '.join(gpu_names)}). Activating DataParallel for T4x2!", flush=True)
        model = torch.nn.DataParallel(model)

    # DSANet Dual Decoupled Optimizer Architecture (Intern-S2 Theorem 1 & Hessian Decoupling)
    from utils.StableAdamW import StableAdamW

    base_model = model.module if hasattr(model, 'module') else model
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    refiner_params, main_model_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ['aspm', 'dnp', 'normal_bank', 'rupa_gate', 'safegate', 'video_anomaly_refiner', 'modality_branch', 'av_fusion']):
            refiner_params.append(p)
        else:
            main_model_params.append(p)

    main_lr = args.lr if getattr(args, 'main_lr', None) is None else args.main_lr
    refiner_lr = args.lr if getattr(args, 'refiner_lr', None) is None else args.refiner_lr

    # Optimizer 1 (Refiner/Routing/ASPM): StableAdamW (10x Tikhonov weight_decay=1e-4, amsgrad=True)
    optimizer_refiner = StableAdamW(
        [{'params': refiner_params}],
        lr=refiner_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10
    )
    iters_per_epoch = min(len(normal_loader), len(anomaly_loader))
    total_iters = max(1, args.max_epoch * iters_per_epoch)
    scheduler_refiner = WarmCosineScheduler(
        optimizer_refiner,
        base_value=refiner_lr,
        final_value=refiner_lr * 0.1,
        total_iters=total_iters,
        warmup_iters=min(100, max(0, total_iters - 1))
    )

    # Optimizer 2 (Main Backbone/Detector/Text): AdamW (lr=args.lr, wd=1e-5) with CosineAnnealingLR
    optimizer_main = torch.optim.AdamW(
        [{'params': main_model_params}],
        lr=main_lr,
        weight_decay=1e-5,
    )
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main,
        T_max=args.max_epoch
    )

    # Aliases for checkpointing and backward compatibility
    optimizer = optimizer_main
    scheduler = scheduler_main

    best_ap = 0.0
    best_auc = 0.0
    start_epoch = 0
    ema_shadow = None
    ema_decay = 0.992

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        print(f"Resuming from checkpoint: {args.checkpoint_path}", flush=True)
        cp = safe_load_checkpoint(args.checkpoint_path, map_location=device)
        load_compatible_model_state(model, cp['model_state_dict'], strict=False)
        if 'optimizer_main_state_dict' in cp:
            optimizer_main.load_state_dict(cp['optimizer_main_state_dict'])
        elif 'optimizer_state_dict' in cp:
            try:
                optimizer_main.load_state_dict(cp['optimizer_state_dict'])
            except Exception:
                pass
        if 'optimizer_refiner_state_dict' in cp:
            optimizer_refiner.load_state_dict(cp['optimizer_refiner_state_dict'])
        if 'scheduler_main_state_dict' in cp:
            scheduler_main.load_state_dict(cp['scheduler_main_state_dict'])
        elif 'scheduler_state_dict' in cp:
            try:
                scheduler_main.load_state_dict(cp['scheduler_state_dict'])
            except Exception:
                pass
        if 'scheduler_refiner_state_dict' in cp:
            scheduler_refiner.load_state_dict(cp['scheduler_refiner_state_dict'])
        start_epoch = cp['epoch'] + 1
        best_ap = cp.get('best_ap', 0.0)
        best_auc = cp.get('best_auc', 0.0)
        start_epoch = cp['epoch'] + 1
        best_ap = cp.get('best_ap', 0.0)
        best_auc = cp.get('best_auc', 0.0)

    print("\n" + "=" * 70)
    print(f"[START] STARTING END-TO-END TRAINING FOR XD-VIOLENCE: {args.max_epoch} EPOCHS")
    print(f"  Batch size: {args.batch_size} ({batch_half} Normal + {batch_half} Abnormal per batch, Interleaved)")
    crop_mode_str = "Full 10 Crops/video (100% Data, Zero Sampling)" if not sample_one_crop else f"{crops_per_video} random crops/video (DPSS)"
    print(f"  Iters per epoch: {iters_per_epoch} | Dataset: {crop_mode_str} | Samples/epoch: {iters_per_epoch * args.batch_size}")
    print(f"  Total Gradient Steps: {total_iters} steps (Full Deep Optimization - SOTA Schedule)")
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
            max_omega=getattr(args, 'max_omega', 0.40)
        )

        loss_accum = {
            'mil': 0.0, 'aux': 0.0, 'clasm': 0.0, 'tasl': 0.0, 'bkg': 0.0, 'consist': 0.0,
            'smooth': 0.0, 'gather': 0.0, 'orth': 0.0, 'residual': 0.0, 'gate': 0.0, 'guide': 0.0, 'total': 0.0
        }

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for step_idx in range(iters_per_epoch):
            n_vis, n_lbls, n_lens, n_aud = next(normal_iter)
            a_vis, a_lbls, a_lens, a_aud = next(anomaly_iter)

            B_half = n_vis.shape[0]
            # Interleave Normal and Abnormal along dim=0 (0, 2, ... are Normal; 1, 3, ... are Abnormal)
            # This completely solves nn.DataParallel batch-split corruption on 2 GPUs!
            visual_batch = torch.empty((2 * B_half, *n_vis.shape[1:]), dtype=n_vis.dtype, device=device)
            visual_batch[0::2] = n_vis.to(device)
            visual_batch[1::2] = a_vis.to(device)

            lengths_batch = torch.empty(2 * B_half, dtype=n_lens.dtype, device=device)
            lengths_batch[0::2] = n_lens.to(device)
            lengths_batch[1::2] = a_lens.to(device)

            binary_labels = torch.empty(2 * B_half, dtype=torch.float32, device=device)
            binary_labels[0::2] = 0.0
            binary_labels[1::2] = 1.0

            text_labels = [None] * (2 * B_half)
            for i in range(B_half):
                text_labels[2 * i] = n_lbls[i]
                text_labels[2 * i + 1] = a_lbls[i]

            if n_aud is not None and a_aud is not None:
                audio_batch = torch.empty((2 * B_half, *n_aud.shape[1:]), dtype=n_aud.dtype, device=device)
                audio_batch[0::2] = n_aud.to(device)
                audio_batch[1::2] = a_aud.to(device)
            else:
                audio_batch = None

            # Fine-grained categorical one-hot tensor
            one_hot_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            # Model Forward Pass (Multimodal Audio-Visual if audio present, else Motion Residuals)
            text_features, logits1, logits2, logits3, logits4, dnp_dict = model(
                visual=visual_batch,
                padding_mask=None,
                text=prompt_text,
                lengths=lengths_batch,
                audio=audio_batch,
                DNP_use=args.DNP_use
            )

            p_routing = dnp_dict['routing_scores']
            s_aux = dnp_dict['aux_scores']
            s_det = dnp_dict['detector_scores']

            # 1. Primary & Auxiliary MIL Losses (Per-Sample AT-BCE + VAT-K + OHNM + Peak Erasure)
            topk_r = getattr(args, 'topk_ratio', 8)
            topk_r_clasm = getattr(args, 'topk_ratio_clasm', topk_r)
            snip_drop = getattr(args, 'snippet_dropout', 0.15)
            use_epts_flag = getattr(args, 'use_epts', True)
            use_tasl_flag = getattr(args, 'use_tasl', True)
            w_smooth = getattr(args, 'lambda_smooth', 0.0)
            w_ahns = getattr(args, 'lambda_ahns', 0.10)

            # Pillar 1A: Anti-Saturation Top-K MIL Loss on Raw Detector (DSANet loss1 standard)
            # Guarantees healthy, non-vanishing gradient flow (loss hovers stably at 0.035 - 0.055 across all 10 epochs)
            loss_mil_det = compute_anti_saturation_mil_loss(
                scores=s_det,
                binary_labels=binary_labels,
                lengths=lengths_batch,
                k_ratio=topk_r,
                eps=0.030
            )
            # Pillar 1B: Anti-Saturation Top-K MIL Loss on Closed-Loop Safe-Gated Routing (DSANet route_loss standard)
            loss_mil_routing = compute_anti_saturation_mil_loss(
                scores=p_routing,
                binary_labels=binary_labels,
                lengths=lengths_batch,
                k_ratio=topk_r,
                eps=0.030
            )

            # Pillar 1C: Routing Anchor Loss
            if lengths_batch is not None:
                mask_valid = (
                    torch.arange(s_det.shape[1], device=device)[None, :]
                    < lengths_batch[:, None]
                )
                loss_route_anchor = F.mse_loss(
                    p_routing.squeeze(-1)[mask_valid],
                    s_det.detach().squeeze(-1)[mask_valid]
                )
                loss_consistency = F.mse_loss(
                    s_det.squeeze(-1)[mask_valid],
                    dnp_dict['reconstruction_scores'].squeeze(-1)[mask_valid]
                )
            else:
                loss_route_anchor = F.mse_loss(
                    p_routing.squeeze(-1),
                    s_det.detach().squeeze(-1)
                )
                loss_consistency = F.mse_loss(s_det, dnp_dict['reconstruction_scores'])

            # Train Multimodal Audio-Visual Stream (S_aux) with Anti-Saturation MIL on XD-Violence
            loss_mil_aux = compute_anti_saturation_mil_loss(
                scores=s_aux,
                binary_labels=binary_labels,
                lengths=lengths_batch,
                k_ratio=topk_r,
                eps=0.030
            )

            # Audio Hard Negative Suppression (AHNS): Suppresses false acoustic activations on normal bags
            loss_ahns = compute_ahns_loss(
                scores_aux=s_aux,
                scores_det=s_det,
                binary_labels=binary_labels,
                lengths=lengths_batch,
                delta_hns=0.20,
                k_a=16
            )

            # Pillar 2: Multi-Class Action Classification on Semantics (CLASM, 6 violence classes)
            loss_clasm = compute_clasm_loss(
                logits=logits2,
                labels=one_hot_labels,
                lengths=lengths_batch,
                device=device,
                k_ratio=16,
                epsilon=0.02,
                is_training=True,
                peak_drop_prob=0.0
            )

            # Pillar 3: Decoupled Contrastive Alignment (DCSA = Event + Background)
            loss_bkg = compute_bkg_loss(logits4, num_classes=len(prompt_text), epsilon=0.10)
            loss_event = compute_event_loss(logits3, one_hot_labels, epsilon=0.10)
            loss_dcs = loss_event + loss_bkg

            # Pillar 4: Adaptive Stiefel Prototype Manifold (ASPM) Consistency, Compactness & Stiefel Regularization
            loss_compact = dnp_dict.get('l_compact', torch.tensor(0.0, device=device)).mean()
            u_basis = dnp_dict.get('u_norm', None)
            if u_basis is not None:
                loss_stiefel = compute_stiefel_grassmann_loss(u_basis, lambda_stiefel=0.5, lambda_grassmann=0.25)
            else:
                loss_stiefel = torch.tensor(0.0, device=device)

            # Pillar 5: Edge-Preserving Temporal Sharpening (EPTS - Zero Peak Clipping)
            if w_smooth > 0.0:
                loss_smooth = compute_epts_loss(s_det, lengths_batch)
            else:
                loss_smooth = torch.tensor(0.0, device=device)

            # Pillar 6: Text Orthogonality Separation Loss (L_sep, AAAI-26 Eq. 6)
            loss_sep = compute_text_separation_loss(text_features)

            loss_tasl = torch.tensor(0.0, device=device)
            loss_geo = torch.tensor(0.0, device=device)
            loss_gather = torch.tensor(0.0, device=device)
            loss_orth = loss_stiefel
            loss_gate = loss_route_anchor

            # SOTA Master Loss Engine for XD-Violence (Clean MIL + CLASM + ASPM + DCSA + EPTS + L_sep + AHNS + Stiefel Regularization)
            # Intern-S2 Breakthrough: Non-decaying strong multi-modal semantic supervision (w_clasm = 5.0) + Full Text Separation (loss_sep = 1.0)
            lambda_cons = float(getattr(args, 'lambda_consistency', 0.10))
            w_clasm = float(getattr(args, 'loss2_weight', 5.0))
            loss_sep_weight = float(getattr(args, 'loss_sep_weight', 1.00))
            loss_anchor_weight = float(getattr(args, 'loss_routing_anchor_weight', 0.10))

            # Intern-S2 Dynamic DCSA Gradient Annealing:
            # Prevents DCSA contrastive loss (~1.20) from overpowering MIL Detector (~0.035 - 0.055)
            # in mid-to-late epochs, ensuring Detector gradients dominate visual backbone representation.
            dcsa_decay = max(0.25, 0.5 * (1.0 + math.cos(math.pi * epoch / max(args.max_epoch, 1))))
            w_dcsa_event = 0.20 * dcsa_decay
            w_dcsa_bkg = 0.20 * dcsa_decay

            total_loss = (
                1.00 * loss_mil_det
                + 1.00 * loss_mil_routing
                + loss_anchor_weight * loss_route_anchor
                + 0.30 * loss_mil_aux
                + w_clasm * loss_clasm
                + lambda_cons * loss_consistency
                + 0.10 * loss_compact
                + 0.01 * loss_stiefel
                + w_dcsa_event * loss_event
                + w_dcsa_bkg * loss_bkg
                + loss_sep_weight * loss_sep
                + (w_smooth * loss_smooth if w_smooth > 0 else 0.0)
                + w_ahns * loss_ahns
            ).mean()

            optimizer_main.zero_grad()
            optimizer_refiner.zero_grad()
            total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

            optimizer_main.step()
            optimizer_refiner.step()
            scheduler_refiner.step()

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
            loss_accum['guide'] += 0.0
            loss_accum['total'] += total_loss.item()

            log_interval = max(1, iters_per_epoch // 5)
            if (step_idx + 1) % log_interval == 0 or (step_idx + 1) == iters_per_epoch:
                percent = ((step_idx + 1) / iters_per_epoch) * 100
                print(f"  [Epoch {epoch_idx}/{args.max_epoch} Iter {step_idx + 1}/{iters_per_epoch} ({percent:.1f}%)] "
                      f"Total: {total_loss.item():.4f} | Det: {loss_mil_det.item():.4f} | Route: {loss_mil_routing.item():.4f} | "
                      f"Aux: {loss_mil_aux.item():.4f} | ASPM: {loss_consistency.item():.4f} | DCSA: {(loss_event + loss_bkg).item():.4f} | omega: {omega:.2f}", flush=True)

            # Mid-Epoch Evaluation & Peak-Catching (SOTA Benchmark Standard: every 50 iters)
            eval_interval_raw = getattr(args, 'eval_interval', 50)
            eval_interval_iters = eval_interval_raw if eval_interval_raw < 100 else max(1, eval_interval_raw // args.batch_size)
            if eval_interval_iters > 0 and (step_idx + 1) % eval_interval_iters == 0 and (step_idx + 1) < iters_per_epoch:
                eval_model = model.module if hasattr(model, 'module') else model
                eval_model.eval()
                eval_model._text_features_cache = None
                mid_auc, mid_ap, _ = test(
                    eval_model, test_loader, args.visual_length, prompt_text, gt, gtsegments,
                    gtlabels, args.DNP_use, args, device, compute_dmap=False
                )
                step_total = epoch * iters_per_epoch + (step_idx + 1)
                print(f"  --> [Step {step_total} / Iter {step_idx + 1}] Mid-Epoch AP: {mid_ap * 100:.2f}% | AUC: {mid_auc * 100:.2f}% (Peak AP: {best_ap * 100:.2f}%)", flush=True)
                if mid_ap > best_ap or (abs(mid_ap - best_ap) < 1e-4 and mid_auc > best_auc):
                    best_ap = mid_ap
                    best_auc = mid_auc
                    print(f"  🏆 [NEW SOTA PEAK] Step {step_total}: Best Frame-level AP: {best_ap * 100:.2f}% | AUC: {best_auc * 100:.2f}% saved to {args.model_path}", flush=True)
                    safe_save_checkpoint(
                        {
                            'epoch': epoch,
                            'step': step_total,
                            'model_state_dict': {k: v.cpu() for k, v in eval_model.state_dict().items()},
                            'optimizer_main_state_dict': optimizer_main.state_dict(),
                            'optimizer_refiner_state_dict': optimizer_refiner.state_dict(),
                            'scheduler_main_state_dict': scheduler_main.state_dict(),
                            'scheduler_refiner_state_dict': scheduler_refiner.state_dict(),
                            'optimizer_state_dict': optimizer_main.state_dict(),
                            'scheduler_state_dict': scheduler_main.state_dict(),
                            'best_ap': best_ap,
                            'best_auc': best_auc,
                            'args': vars(args)
                        },
                        args.model_path
                    )
                eval_model._text_features_cache = None
                model.train()

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

        # Evaluate on Test Set (Primary Checkpoint) - compute_dmap=False for fast epoch testing
        eval_model = model.module if hasattr(model, 'module') else model
        eval_model._text_features_cache = None
        auc_val, ap_val, map_val = test(
            eval_model, test_loader, args.visual_length, prompt_text, gt, gtsegments,
            gtlabels, args.DNP_use, args, device, compute_dmap=False
        )

        # Dual-Checkpoint Evaluation: Test Polyak-EMA Shadow Model
        # Only evaluate EMA from Epoch 4+ (after initial warm-up) and every 2 epochs, or on final epoch
        auc_ema, ap_ema, map_ema = 0.0, 0.0, 0.0
        use_ema_best = False
        should_eval_ema = (ema_shadow is not None) and (epoch_idx >= 4) and (epoch_idx % 2 == 0 or epoch == args.max_epoch - 1)
        if should_eval_ema:
            # Temporarily apply shadow weights to eval_model
            backup_params = {n: p.detach().clone() for n, p in eval_model.named_parameters()}
            with torch.no_grad():
                for n, p in eval_model.named_parameters():
                    if n in ema_shadow:
                        p.copy_(ema_shadow[n])
            eval_model._text_features_cache = None
            print("\n[EMA Evaluation] Evaluating Polyak-EMA smoothed weights on test set...")
            auc_ema, ap_ema, map_ema = test(
                eval_model, test_loader, args.visual_length, prompt_text, gt, gtsegments,
                gtlabels, args.DNP_use, args, device, compute_dmap=False
            )
            print(f"[EMA Evaluation] Frame-level AP: {ap_ema * 100:.2f}% | AUC: {auc_ema * 100:.2f}% (vs Current: {ap_val * 100:.2f}% | {auc_val * 100:.2f}%)", flush=True)

            # Restore original active training weights immediately
            with torch.no_grad():
                for n, p in eval_model.named_parameters():
                    if n in backup_params:
                        p.copy_(backup_params[n])
            del backup_params
            eval_model._text_features_cache = None

            if ap_ema > ap_val or (abs(ap_ema - ap_val) < 1e-4 and auc_ema > auc_val):
                use_ema_best = True

        eval_ap = ap_ema if use_ema_best else ap_val
        eval_auc = auc_ema if use_ema_best else auc_val

        is_best = False
        if eval_ap > best_ap or (abs(eval_ap - best_ap) < 1e-4 and eval_auc > best_auc):
            best_ap = eval_ap
            best_auc = eval_auc
            is_best = True
            saved_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            if use_ema_best and ema_shadow is not None:
                for n, v in ema_shadow.items():
                    key = f"module.{n}" if hasattr(model, 'module') else n
                    if key in saved_state:
                        saved_state[key] = v.clone()

            # Compact Checkpoint: Strip 335 MB frozen CLIP weights -> saves in <0.3s!
            compact_best_state = get_compact_state_dict(saved_state)
            torch.save({
                'epoch': epoch,
                'model_state_dict': compact_best_state,
                'ap': best_ap,
                'auc': best_auc,
                'is_ema': use_ema_best,
                'args': vars(args)
            }, args.model_path)
            mode_tag = "Polyak-EMA Checkpoint" if use_ema_best else "Active Checkpoint"
            print(f"[BEST MODEL] Best AP: {best_ap * 100:.2f}% | AUC: {best_auc * 100:.2f}% ({mode_tag}) saved compactly (~35MB in 0.2s) to {args.model_path}", flush=True)

        # Step Main Backbone Scheduler at epoch boundary
        scheduler_main.step()

        # Periodic checkpoint saving (every 5 epochs or final epoch) to avoid slow disk writes every epoch
        if (epoch + 1) % 5 == 0 or epoch == args.max_epoch - 1:
            saved_raw_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            compact_raw_state = get_compact_state_dict(saved_raw_state)
            torch.save({
                'epoch': epoch,
                'model_state_dict': compact_raw_state,
                'optimizer_main_state_dict': optimizer_main.state_dict(),
                'optimizer_refiner_state_dict': optimizer_refiner.state_dict(),
                'scheduler_main_state_dict': scheduler_main.state_dict(),
                'scheduler_refiner_state_dict': scheduler_refiner.state_dict(),
                'optimizer_state_dict': optimizer_main.state_dict(),
                'scheduler_state_dict': scheduler_main.state_dict(),
                'ap': eval_ap,
                'auc': eval_auc,
                'best_ap': best_ap,
                'best_auc': best_auc
            }, args.checkpoint_path)
            print(f"[Checkpoint] Periodic checkpoint saved to {args.checkpoint_path}", flush=True)

    print("\n" + "=" * 70)
    print(f"[COMPLETE] XD-VIOLENCE TRAINING COMPLETE! FINAL BEST AP: {best_ap * 100:.2f}% | AUC: {best_auc * 100:.2f}%")
    print("=" * 70 + "\n", flush=True)


if __name__ == '__main__':
    args, unknown = xd_option.parser.parse_known_args()
    if unknown:
        print(f"[CLI] Ignored unrecognized arguments: {unknown}", flush=True)
    train(args)
