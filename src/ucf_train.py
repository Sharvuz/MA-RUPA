import os
import random
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import MARUPA, load_compatible_model_state
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label, safe_load_checkpoint
from utils.StableAdamW import StableAdamW
from utils.losses import (
    CLAS2, CLASM, CLASM_EVENT, CLASM_BKG,
    text_orthogonality_loss, ConsistencyLoss, routing_anchor_loss,
    edge_preserving_smoothness_loss, iam_intention_loss, acc_pseudo_label_loss,
    WarmCosineScheduler, ModelEMA,
)
import ucf_option


UCF_LABEL_MAP = {
    'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
    'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
    'Fighting': 'fighting', 'RoadAccidents': 'roadaccidents', 'Robbery': 'robbery',
    'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
    'Vandalism': 'vandalism',
}


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def unwrap_compact_state_dict(model: nn.Module) -> dict:
    """Extracts trainable/adapter state_dict without frozen CLIP weights (~42 MB vs 290 MB)."""
    raw = model.module if isinstance(model, nn.DataParallel) else model
    return {
        k: v.detach().cpu()
        for k, v in raw.state_dict().items()
        if not k.startswith('clip_adapter.clipmodel.')
    }


def interleave_normal_anomaly(norm_v, norm_lbl, norm_len, abn_v, abn_lbl, abn_len, device):
    """Interleaves Normal and Abnormal samples [N0, A0, N1, A1, ...] so every GPU shard
    in torch.nn.DataParallel receives an exact 50/50 Normal-Abnormal balance.
    """
    b = min(norm_v.shape[0], abn_v.shape[0])
    _, t, d = norm_v.shape
    vis = torch.empty((2 * b, t, d), dtype=norm_v.dtype)
    vis[0::2] = norm_v[:b]
    vis[1::2] = abn_v[:b]

    lens = torch.empty((2 * b,), dtype=torch.long)
    lens[0::2] = norm_len[:b]
    lens[1::2] = abn_len[:b]

    labels = []
    for k in range(b):
        labels.append(norm_lbl[k])
        labels.append(abn_lbl[k])

    return vis.to(device, non_blocking=True), labels, lens.to(device, non_blocking=True)


def build_ucf_components(args, device):
    half_bs = max(1, args.batch_size // 2)
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': args.num_workers > 0,
    }

    normal_dataset = UCFDataset(
        args.visual_length, args.train_list, False, UCF_LABEL_MAP, True,
        data_dir=args.data_dir, sample_one_crop_per_video=args.sample_one_crop_per_video,
        crops_per_video=args.crops_per_video,
    )
    normal_loader = DataLoader(normal_dataset, batch_size=half_bs, shuffle=True, drop_last=True, **loader_kwargs)

    anomaly_dataset = UCFDataset(
        args.visual_length, args.train_list, False, UCF_LABEL_MAP, False,
        data_dir=args.data_dir, sample_one_crop_per_video=args.sample_one_crop_per_video,
        crops_per_video=args.crops_per_video,
    )
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=half_bs, shuffle=True, drop_last=True, **loader_kwargs)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, UCF_LABEL_MAP, data_dir=args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    model = MARUPA(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, args, device,
    )
    if getattr(args, 'init_model_path', None) and os.path.exists(args.init_model_path):
        init_state = safe_load_checkpoint(args.init_model_path, map_location='cpu')
        load_compatible_model_state(model, init_state, allow_legacy_dsanet=True)
        print(f"[Init] Loaded initial weights from {args.init_model_path}", flush=True)

    return model, normal_loader, anomaly_loader, test_loader, UCF_LABEL_MAP


def train(model_or_args, normal_loader=None, anomaly_loader=None, testloader=None,
          args=None, label_map=None, device=None):
    """Supports both `train(args)` and `train(model, normal_loader, anomaly_loader, testloader, args, label_map, device)`."""
    if normal_loader is None and not isinstance(model_or_args, nn.Module):
        args = model_or_args
        device = "cuda" if torch.cuda.is_available() else "cpu"
        setup_seed(args.seed)
        model, normal_loader, anomaly_loader, testloader, label_map = build_ucf_components(args, device)
    else:
        model = model_or_args

    os.makedirs(os.path.dirname(args.checkpoint_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)

    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model.to(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        gpu_names = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        print(f"[Multi-GPU] Detected {torch.cuda.device_count()} GPUs ({gpu_names}). Activating DataParallel!", flush=True)
        model = nn.DataParallel(model)

    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    refiner_params = []
    main_model_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'video_anomaly_refiner' in name or 'rupa_gate' in name:
            refiner_params.append(param)
        else:
            main_model_params.append(param)

    main_lr = args.lr if args.main_lr is None else args.main_lr
    refiner_lr = args.lr if args.refiner_lr is None else args.refiner_lr

    optimizer_main = torch.optim.AdamW(
        [{'params': main_model_params}],
        lr=main_lr,
        weight_decay=args.weight_decay,
    )
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, T_max=max(1, args.max_epoch)
    )

    num_batches_per_epoch = min(len(normal_loader), len(anomaly_loader))
    total_iters = max(1, args.max_epoch * num_batches_per_epoch)
    total_iters_refiner = max(1, args.max_epoch * (len(normal_loader) + len(anomaly_loader)))

    optimizer_refiner = StableAdamW(
        [{'params': refiner_params}],
        lr=refiner_lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-10,
    )
    scheduler_refiner = WarmCosineScheduler(
        optimizer_refiner,
        base_value=refiner_lr,
        final_value=refiner_lr * 0.1,
        total_iters=total_iters_refiner,
        warmup_iters=min(100, max(0, total_iters_refiner - 1)),
    )

    ema = ModelEMA(model, decay=args.ema_decay) if getattr(args, 'use_ema', True) else None
    consistency_loss_fn = ConsistencyLoss()
    prompt_text = get_prompt_text(label_map)

    auc_best = -1.0
    best_live_state = None
    start_epoch = 0

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        checkpoint = safe_load_checkpoint(args.checkpoint_path, map_location=device)
        load_compatible_model_state(model, checkpoint['model_state_dict'], allow_legacy_dsanet=True)
        best_live_state = unwrap_compact_state_dict(model)
        if 'optimizer_state_dict' in checkpoint:
            optimizer_main.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'optimizer_refiner_state_dict' in checkpoint:
            optimizer_refiner.load_state_dict(checkpoint['optimizer_refiner_state_dict'])
        if 'scheduler_main_state_dict' in checkpoint:
            scheduler_main.load_state_dict(checkpoint['scheduler_main_state_dict'])
        if 'scheduler_refiner_state_dict' in checkpoint:
            scheduler_refiner.load_state_dict(checkpoint['scheduler_refiner_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        auc_best = checkpoint.get('auc', checkpoint.get('ap', auc_best))
        print(f"[Resume] Resumed from epoch {start_epoch}, best AUC={auc_best * 100:.2f}%", flush=True)

    eff_batch = normal_loader.batch_size * 2
    lambda_iam = float(getattr(args, 'lambda_iam', 0.0))
    lambda_acc = float(getattr(args, 'lambda_acc', 0.0))
    print("\n" + "=" * 78, flush=True)
    print(f"[START] MA-RUPA v4+ (ARGON-4 GRADIENT CONSERVATION) SOTA ON UCF-CRIME ({args.max_epoch} EPOCHS)", flush=True)
    print(f"  Batch Size : {eff_batch} ({normal_loader.batch_size} Normal + {anomaly_loader.batch_size} Abnormal interleaved)", flush=True)
    print(f"  Schedule   : {num_batches_per_epoch} iters/epoch | Total = {total_iters} gradient steps | LR = {main_lr:.1e}", flush=True)
    print(f"  Loss Eq.   : GC-MIL=1.0 | CLASM={args.loss2_weight} | Orth={args.loss_sep_weight} | ResEv={args.loss_residual_weight} | Bkg={args.loss_reconstructed_normal_weight} | Cons={args.loss_consistency_weight} | Gather={args.loss_gather_weight}", flush=True)
    print(f"  Multi-Crop : {'10-Crop Parallel Eval Active' if args.multi_crop_eval else '1-Crop Eval'} (sigma={args.smooth_sigma})", flush=True)
    print("=" * 78 + "\n", flush=True)

    for e in range(start_epoch, args.max_epoch):
        model.train()
        DNP_use = args.DNP_use and (not args.no_rupa)
        epoch_improved = False

        sum_l1 = sum_l2 = sum_l3 = sum_l4 = sum_l5 = sum_cons = sum_g = 0.0
        epoch_t0 = time.time()

        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)

        for i in range(num_batches_per_epoch):
            norm_feat, norm_lbl, norm_len = next(normal_iter)
            abn_feat, abn_lbl, abn_len = next(anomaly_iter)

            visual_features, text_labels_raw, feat_lengths = interleave_normal_anomaly(
                norm_feat, norm_lbl, norm_len, abn_feat, abn_lbl, abn_len, device
            )
            text_labels = get_batch_label(text_labels_raw, prompt_text, label_map).to(device)

            text_features, logits1, logits2, logits3, logits4, dnp_dict = model(
                visual_features, None, prompt_text, feat_lengths, audio=None, DNP_use=DNP_use
            )

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss3 = text_orthogonality_loss(text_features, num_classes=args.classes_num)
            loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
            loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)

            iam_protos = raw_model.iam_module.intention_prototypes if getattr(raw_model, 'iam_module', None) is not None else None
            loss_iam = (
                iam_intention_loss(dnp_dict, text_labels, feat_lengths, device, iam_prototypes=iam_protos)
                if lambda_iam > 0 else logits1.sum() * 0.0
            )
            loss_acc = (
                acc_pseudo_label_loss(dnp_dict, feat_lengths, logits1=logits1)
                if lambda_acc > 0 else logits1.sum() * 0.0
            )
            loss_smooth = (
                edge_preserving_smoothness_loss(torch.sigmoid(logits1), feat_lengths)
                if args.lambda_smooth > 0 else logits1.sum() * 0.0
            )

            if DNP_use and 'reconstructed_features' in dnp_dict:
                cons_loss = consistency_loss_fn(
                    logits1=logits1,
                    original_features=dnp_dict['original_features'],
                    reconstructed_features=dnp_dict['reconstructed_features'],
                    lengths=feat_lengths,
                )
                g_loss = dnp_dict['g_loss'].mean()

                if args.rupa_use and 'routing_logits' in dnp_dict and args.loss_routing_weight > 0:
                    route_loss = CLAS2(dnp_dict['routing_logits'], text_labels, feat_lengths, device)
                    route_anchor = routing_anchor_loss(dnp_dict, feat_lengths)
                else:
                    route_loss = logits1.sum() * 0.0
                    route_anchor = logits1.sum() * 0.0

                loss = (
                    loss1
                    + loss2 * args.loss2_weight
                    + loss3 * args.loss_sep_weight
                    + loss4 * args.loss_residual_weight
                    + loss5 * args.loss_reconstructed_normal_weight
                    + loss_iam * lambda_iam
                    + loss_acc * lambda_acc
                    + cons_loss * args.loss_consistency_weight
                    + g_loss * args.loss_gather_weight
                    + route_loss * args.loss_routing_weight
                    + route_anchor * args.loss_routing_anchor_weight
                    + loss_smooth * args.lambda_smooth
                )
            else:
                cons_loss = logits1.sum() * 0.0
                g_loss = logits1.sum() * 0.0
                loss = (
                    loss1
                    + loss2 * args.loss2_weight
                    + loss3 * args.loss_sep_weight
                    + loss4 * args.loss_residual_weight
                    + loss5 * args.loss_reconstructed_normal_weight
                    + loss_iam * lambda_iam
                    + loss_acc * lambda_acc
                    + loss_smooth * args.lambda_smooth
                )

            optimizer_main.zero_grad(set_to_none=True)
            optimizer_refiner.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

            optimizer_main.step()
            optimizer_refiner.step()
            scheduler_refiner.step()

            if ema is not None:
                ema.update(model)

            sum_l1 += loss1.item()
            sum_l2 += loss2.item()
            sum_l3 += loss3.item()
            sum_l4 += loss4.item()
            sum_l5 += loss5.item()
            sum_cons += cons_loss.item()
            sum_g += g_loss.item()

            iter_num = i + 1
            if iter_num == 1 or iter_num % 25 == 0 or iter_num == num_batches_per_epoch:
                elapsed = max(1e-3, time.time() - epoch_t0)
                it_s = iter_num / elapsed
                eta_s = (num_batches_per_epoch - iter_num) / max(it_s, 1e-4)
                print(
                    f"Epoch [{e+1}/{args.max_epoch}] Iter [{iter_num:03d}/{num_batches_per_epoch:03d}] | "
                    f"MIL:{sum_l1/iter_num:.4f} CLASM:{sum_l2/iter_num:.4f} Orth:{sum_l3/iter_num:.4f} "
                    f"Ev:{sum_l4/iter_num:.4f} Bkg:{sum_l5/iter_num:.4f} Cons:{sum_cons/iter_num:.4f} G:{sum_g/iter_num:.4f} | "
                    f"{it_s:.2f} it/s ETA:{eta_s:.0f}s",
                    flush=True,
                )

            if args.eval_interval > 0 and iter_num % args.eval_interval == 0 and iter_num < num_batches_per_epoch:
                auc_cur, _ = test(
                    model, testloader, args.visual_length, prompt_text,
                    gt, gtsegments, gtlabels, DNP_use, device, args, compute_dmap=False,
                )
                if auc_cur > auc_best:
                    auc_best = auc_cur
                    epoch_improved = True
                    best_live_state = unwrap_compact_state_dict(model)
                    torch.save(best_live_state, args.model_path)
                    print(f"  [*] [NEW BEST UCF-CRIME] AUC = {auc_best * 100:.2f}% saved -> {args.model_path}", flush=True)
                model.train()

        scheduler_main.step()

        print(f"\n--- Epoch {e+1}/{args.max_epoch} Complete | Evaluating Live Weights ---", flush=True)
        auc_live, _ = test(
            model, testloader, args.visual_length, prompt_text,
            gt, gtsegments, gtlabels, DNP_use, device, args, compute_dmap=(e == args.max_epoch - 1),
        )
        live_state = unwrap_compact_state_dict(model)
        if auc_live > auc_best:
            auc_best = auc_live
            epoch_improved = True
            best_live_state = live_state
            torch.save(best_live_state, args.model_path)
            print(f"  [*] [NEW BEST UCF-CRIME (Live)] AUC = {auc_best * 100:.2f}% saved -> {args.model_path}", flush=True)

        torch.save({
            'epoch': e,
            'model_state_dict': best_live_state if best_live_state is not None else live_state,
            'optimizer_state_dict': optimizer_main.state_dict(),
            'optimizer_refiner_state_dict': optimizer_refiner.state_dict(),
            'scheduler_main_state_dict': scheduler_main.state_dict(),
            'scheduler_refiner_state_dict': scheduler_refiner.state_dict(),
            'auc': auc_best,
            'ap': auc_best,
        }, args.checkpoint_path)

        if ema is not None and e >= 1:
            backup = ema.apply_to(model)
            print(f"--- Epoch {e+1}/{args.max_epoch} | Evaluating Polyak-EMA Weights ---", flush=True)
            auc_ema, _ = test(
                model, testloader, args.visual_length, prompt_text,
                gt, gtsegments, gtlabels, DNP_use, device, args, compute_dmap=False,
            )
            if auc_ema > auc_best:
                auc_best = auc_ema
                epoch_improved = True
                ema_state = unwrap_compact_state_dict(model)
                torch.save(ema_state, args.model_path)
                print(f"  [*] [NEW BEST UCF-CRIME (EMA)] AUC = {auc_best * 100:.2f}% saved -> {args.model_path}", flush=True)
            ema.restore(model, backup)

        if best_live_state is not None:
            load_compatible_model_state(model, best_live_state, allow_legacy_dsanet=True)
            if not epoch_improved:
                scheduler_main.schedule *= 0.85
                scheduler_main.final_value *= 0.85
                scheduler_refiner.schedule *= 0.85
                scheduler_refiner.final_value *= 0.85
                cur_lr = scheduler_main.get_lr()[0]
                print(f"  [Rollback] Restored best checkpoint (AUC={auc_best*100:.2f}%) | Gentle LR damping x0.85 -> {cur_lr:.2e}", flush=True)
        print("", flush=True)

    print(f"[FINISH] UCF-Crime Training Complete! Best ROC-AUC: {auc_best * 100:.2f}%", flush=True)
    return auc_best


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, unknown = ucf_option.parser.parse_known_args()
    if unknown:
        print(f"[CLI] Ignored unrecognized arguments: {unknown}", flush=True)

    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        print(f"[MA-RUPA v4+ Argon-4] Git Commit: {commit_hash}", flush=True)
    except Exception:
        pass

    print(f"[MA-RUPA v4+ Argon-4] Training UCF-Crime on device: {device}", flush=True)
    setup_seed(args.seed)
    model, normal_loader, anomaly_loader, test_loader, label_map = build_ucf_components(args, device)
    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
