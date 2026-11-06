import math
import os
import random
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import MARUPA, load_compatible_model_state
from xd_test import test
from utils.dataset import XDDataset
from utils.tools import (
    get_prompt_text, get_batch_label, safe_load_checkpoint,
    xd_collate_fn, xd_test_collate_fn,
)
from utils.StableAdamW import StableAdamW
from utils.losses import (
    CLAS2, CLASM, CLASM_EVENT, CLASM_BKG,
    text_orthogonality_loss, ConsistencyLoss, routing_anchor_loss,
    edge_preserving_smoothness_loss, audio_guided_mil_loss,
    iam_intention_loss, acc_pseudo_label_loss,
    WarmCosineScheduler, ModelEMA,
)
import xd_option


XD_LABEL_MAP = {
    'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
    'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion',
}


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def unwrap_compact_state_dict(model: nn.Module) -> dict:
    """Extracts trainable/adapter state_dict without frozen CLIP weights (~45 MB vs 295 MB)."""
    raw = model.module if isinstance(model, nn.DataParallel) else model
    return {
        k: v.detach().cpu()
        for k, v in raw.state_dict().items()
        if not k.startswith('clip_adapter.clipmodel.')
    }


def interleave_multimodal_batch(n_vis, n_lbls, n_lens, n_aud,
                                a_vis, a_lbls, a_lens, a_aud, device):
    """Interleaves Normal and Abnormal visual+audio samples [N0, A0, N1, A1, ...]
    when optional `--balanced-sampling True` mode is enabled.
    """
    b = min(n_vis.shape[0], a_vis.shape[0])
    _, t, d = n_vis.shape

    vis = torch.empty((2 * b, t, d), dtype=n_vis.dtype)
    vis[0::2] = n_vis[:b]
    vis[1::2] = a_vis[:b]

    lens = torch.empty((2 * b,), dtype=torch.long)
    lens[0::2] = n_lens[:b]
    lens[1::2] = a_lens[:b]

    labels = []
    for k in range(b):
        labels.append(n_lbls[k])
        labels.append(a_lbls[k])

    aud_dev = None
    if n_aud is not None or a_aud is not None:
        ref = n_aud if n_aud is not None else a_aud
        da = ref.shape[-1]
        ta = ref.shape[1]
        aud = torch.zeros((2 * b, ta, da), dtype=ref.dtype)
        if n_aud is not None:
            aud[0::2] = n_aud[:b]
        if a_aud is not None:
            aud[1::2] = a_aud[:b]
        aud_dev = aud.to(device, non_blocking=True)

    return vis.to(device, non_blocking=True), labels, lens.to(device, non_blocking=True), aud_dev


def build_xd_components(args, device):
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': args.num_workers > 0,
    }

    use_balanced = bool(getattr(args, 'balanced_sampling', False))
    if use_balanced:
        half_bs = max(1, args.batch_size // 2)
        normal_dataset = XDDataset(
            args.visual_length, args.train_list, False, XD_LABEL_MAP,
            audio_file_path=args.audio_train_list, audio_dir=args.audio_dir,
            data_dir=args.data_dir, no_audio=args.no_audio,
            normal=True, sample_one_crop_per_video=args.sample_one_crop_per_video,
            crops_per_video=args.crops_per_video,
        )
        anomaly_dataset = XDDataset(
            args.visual_length, args.train_list, False, XD_LABEL_MAP,
            audio_file_path=args.audio_train_list, audio_dir=args.audio_dir,
            data_dir=args.data_dir, no_audio=args.no_audio,
            normal=False, sample_one_crop_per_video=args.sample_one_crop_per_video,
            crops_per_video=args.crops_per_video,
        )
        primary_train_loader = DataLoader(
            normal_dataset, batch_size=half_bs, shuffle=True, drop_last=True,
            collate_fn=xd_collate_fn, **loader_kwargs,
        )
        secondary_train_loader = DataLoader(
            anomaly_dataset, batch_size=half_bs, shuffle=True, drop_last=True,
            collate_fn=xd_collate_fn, **loader_kwargs,
        )
        probe_datasets = (normal_dataset, anomaly_dataset)
    else:
        train_dataset = XDDataset(
            args.visual_length, args.train_list, False, XD_LABEL_MAP,
            audio_file_path=args.audio_train_list, audio_dir=args.audio_dir,
            data_dir=args.data_dir, no_audio=args.no_audio,
            normal=None, sample_one_crop_per_video=args.sample_one_crop_per_video,
            crops_per_video=args.crops_per_video,
        )
        primary_train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
            collate_fn=xd_collate_fn, **loader_kwargs,
        )
        secondary_train_loader = None
        probe_datasets = (train_dataset,)

    test_dataset = XDDataset(
        args.visual_length, args.test_list, True, XD_LABEL_MAP,
        audio_file_path=args.audio_test_list, audio_dir=args.audio_dir,
        data_dir=args.data_dir, no_audio=args.no_audio,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=xd_test_collate_fn, **loader_kwargs,
    )

    if not args.no_audio:
        detected_audio_dim = None
        for ds in probe_datasets:
            for idx in range(min(20, len(ds))):
                item_tuple = ds[idx]
                sample_audio = item_tuple[3] if len(item_tuple) >= 4 else None
                if sample_audio is not None and isinstance(sample_audio, torch.Tensor):
                    detected_audio_dim = int(sample_audio.shape[-1])
                    break
            if detected_audio_dim is not None:
                break
        if detected_audio_dim is not None and detected_audio_dim != args.audio_dim:
            print(f"[Auto-Detect] Audio feature dim is {detected_audio_dim}-D; updating args.audio_dim.", flush=True)
            args.audio_dim = detected_audio_dim

    model = MARUPA(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, args, device,
    )
    if getattr(args, 'init_model_path', None) and os.path.exists(args.init_model_path):
        init_state = safe_load_checkpoint(args.init_model_path, map_location='cpu')
        load_compatible_model_state(model, init_state, allow_legacy_dsanet=True)
        print(f"[Init] Loaded initial weights from {args.init_model_path}", flush=True)

    return model, primary_train_loader, secondary_train_loader, test_loader, XD_LABEL_MAP


def train(model_or_args, normal_loader=None, anomaly_loader=None, testloader=None,
          args=None, label_map=None, device=None):
    """Supports both `train(args)` and `train(model, train_loader, anomaly_loader, testloader, args, label_map, device)`."""
    if normal_loader is None and not isinstance(model_or_args, nn.Module):
        args = model_or_args
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
            print(f"[MA-RUPA v10 SOTA] Git Commit: {commit_hash}", flush=True)
        except Exception:
            pass
        print(f"[MA-RUPA v10 SOTA] Training XD-Violence on device: {device}", flush=True)
        setup_seed(args.seed)
        model, normal_loader, anomaly_loader, testloader, label_map = build_xd_components(args, device)
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
    audio_adapter_params = []
    main_decay_params = []
    main_nodecay_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'video_anomaly_refiner' in name or (args.rupa_use and 'rupa_gate' in name):
            refiner_params.append(param)
        elif 'audio_adapter' in name:
            if not args.no_audio:
                audio_adapter_params.append(param)
        elif 'rupa_gate' not in name:
            if param.ndim >= 2:
                main_decay_params.append(param)
            else:
                main_nodecay_params.append(param)

    main_lr = args.lr if args.main_lr is None else args.main_lr
    audio_lr = getattr(args, 'audio_lr', None) or max(main_lr, 3e-5)
    refiner_lr = args.lr if args.refiner_lr is None else args.refiner_lr

    dual_loader_mode = anomaly_loader is not None
    if dual_loader_mode:
        num_batches_per_epoch = min(len(normal_loader), len(anomaly_loader))
        eff_batch = normal_loader.batch_size * 2
        sampling_desc = f"{eff_batch} ({normal_loader.batch_size} Normal + {anomaly_loader.batch_size} Abnormal interleaved)"
    else:
        num_batches_per_epoch = len(normal_loader)
        eff_batch = normal_loader.batch_size
        ds = normal_loader.dataset
        if getattr(ds, 'sample_one_crop_per_video', False) and getattr(ds, 'video_groups', None):
            sampling_desc = (
                f"{eff_batch} (Stratified {ds.crops_per_video}-of-10 Random Spatial Crops -- "
                f"{len(ds)} crops/epoch across {len(ds.video_groups)} unique clips)"
            )
        else:
            sampling_desc = f"{eff_batch} (Single Natural Shuffled Loader -- 100% of {len(ds)} crops)"

    total_iters = max(1, args.max_epoch * num_batches_per_epoch)
    warmup_iters = min(80, max(10, total_iters // 20)) if total_iters > 10 else 0

    main_param_groups = [{'params': main_decay_params, 'lr': main_lr, 'weight_decay': args.weight_decay}]
    if main_nodecay_params:
        main_param_groups.append({'params': main_nodecay_params, 'lr': main_lr, 'weight_decay': 0.0})
    if audio_adapter_params and not args.no_audio:
        main_param_groups.append({'params': audio_adapter_params, 'lr': audio_lr, 'weight_decay': 1e-4})

    optimizer_main = torch.optim.AdamW(main_param_groups)
    scheduler_main = WarmCosineScheduler(
        optimizer_main,
        base_value=main_lr,
        final_value=main_lr * 0.05,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
    )

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
        final_value=refiner_lr * 0.05,
        total_iters=total_iters,
        warmup_iters=warmup_iters,
    )

    ema = ModelEMA(model, decay=args.ema_decay) if getattr(args, 'use_ema', True) else None
    consistency_loss_fn = ConsistencyLoss()
    prompt_text = get_prompt_text(label_map)

    ap_best = -1.0
    auc_best = -1.0
    start_epoch = 0

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        checkpoint = safe_load_checkpoint(args.checkpoint_path, map_location=device)
        saved_ep = checkpoint.get('epoch', -1) + 1
        is_compatible_ckpt = (checkpoint.get('arch_version') == 'marupa_v10_sota') or (args.max_epoch <= 3)
        if 0 < saved_ep < args.max_epoch and is_compatible_ckpt:
            load_compatible_model_state(model, checkpoint['model_state_dict'], allow_legacy_dsanet=True)
            try:
                if 'optimizer_state_dict' in checkpoint:
                    optimizer_main.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'optimizer_refiner_state_dict' in checkpoint:
                    optimizer_refiner.load_state_dict(checkpoint['optimizer_refiner_state_dict'])
                if 'scheduler_main_state_dict' in checkpoint:
                    scheduler_main.load_state_dict(checkpoint['scheduler_main_state_dict'])
                if 'scheduler_refiner_state_dict' in checkpoint:
                    scheduler_refiner.load_state_dict(checkpoint['scheduler_refiner_state_dict'])
                if ema is not None and 'ema_state_dict' in checkpoint:
                    ema.load_state_dict(checkpoint['ema_state_dict'])
            except Exception:
                pass
            start_epoch = saved_ep
            ap_best = checkpoint.get('best_ap', checkpoint.get('ap', ap_best))
            auc_best = checkpoint.get('best_auc', checkpoint.get('auc', auc_best))
            print(f"[Resume] Resumed from epoch {start_epoch + 1}/{args.max_epoch}, best AP={ap_best * 100:.2f}% | AUC={auc_best * 100:.2f}%", flush=True)
        elif 0 < saved_ep < args.max_epoch:
            print(f"[Checkpoint] Ignoring legacy pre-v10 checkpoint (epoch {saved_ep}) -> Starting fresh v10 Confidence-Margin SOTA run!", flush=True)

    lambda_iam = float(getattr(args, 'lambda_iam', 0.0))
    lambda_acc = float(getattr(args, 'lambda_acc', 0.0))
    mode_tag = "CONFIDENCE-MARGIN LGT-SGNM VISUAL SOTA" if args.no_audio else "MULTIMODAL SOTA"
    print("\n" + "=" * 78, flush=True)
    print(f"[START] MA-RUPA v10 ({mode_tag}) ON XD-VIOLENCE ({args.max_epoch} EPOCHS)", flush=True)
    print(f"  Batch Size : {sampling_desc}", flush=True)
    print(f"  Schedule   : {num_batches_per_epoch} iters/epoch | Total = {total_iters} steps (Warmup={warmup_iters}) | LR = {main_lr:.1e}->{main_lr*0.05:.1e} (WD={args.weight_decay:.0e})", flush=True)
    print(f"  Loss Eq.   : MIL=1.0 | CLASM={args.loss2_weight} | Orth={args.loss_sep_weight} | Event={args.loss_residual_weight} | Bkg={args.loss_reconstructed_normal_weight} | Cons={args.loss_consistency_weight} | Gather={args.loss_gather_weight}", flush=True)
    print(f"  Regularize : Per-Bag Confidence Margin | MLPDrop={args.mlp_dropout:.2f} | WarmupEMA={args.ema_decay} | Zero Rollback | {'1-Crop + 5-Crop Eval' if args.multi_crop_eval else '1-Crop Eval'}", flush=True)
    print("=" * 78 + "\n", flush=True)

    for e in range(start_epoch, args.max_epoch):
        model.train()
        DNP_use = args.DNP_use and (not args.no_rupa)

        sum_l1 = sum_l2 = sum_l3 = sum_l4 = sum_l5 = sum_cons = sum_g = sum_aux = 0.0
        epoch_t0 = time.time()

        if dual_loader_mode:
            normal_iter = iter(normal_loader)
            anomaly_iter = iter(anomaly_loader)
        else:
            single_iter = iter(normal_loader)

        for i in range(num_batches_per_epoch):
            if dual_loader_mode:
                n_vis, n_lbls, n_lens, n_aud = next(normal_iter)
                a_vis, a_lbls, a_lens, a_aud = next(anomaly_iter)
                visual_features, text_labels_raw, feat_lengths, audio_features = interleave_multimodal_batch(
                    n_vis, n_lbls, n_lens, n_aud, a_vis, a_lbls, a_lens, a_aud, device
                )
            else:
                vis_b, text_labels_raw, lens_b, aud_b = next(single_iter)
                visual_features = vis_b.to(device, non_blocking=True)
                feat_lengths = lens_b.to(device, non_blocking=True)
                audio_features = aud_b.to(device, non_blocking=True) if aud_b is not None else None

            text_labels = get_batch_label(text_labels_raw, prompt_text, label_map).to(device)

            forward_module = (
                raw_model
                if (isinstance(model, nn.DataParallel) and visual_features.shape[0] < torch.cuda.device_count())
                else model
            )
            text_features, logits1, logits2, logits3, logits4, dnp_dict = forward_module(
                visual_features, None, prompt_text, feat_lengths, audio=audio_features, DNP_use=DNP_use
            )

            loss1 = CLAS2(logits1, text_labels, feat_lengths, device)
            loss2 = CLASM(logits2, text_labels, feat_lengths, device) if args.loss2_weight > 0 else logits1.sum() * 0.0
            loss3 = text_orthogonality_loss(text_features, num_classes=args.classes_num) if args.loss_sep_weight > 0 else logits1.sum() * 0.0
            loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device) if args.loss_residual_weight > 0 else logits1.sum() * 0.0
            loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device) if args.loss_reconstructed_normal_weight > 0 else logits1.sum() * 0.0

            iam_protos = raw_model.iam_module.intention_prototypes if getattr(raw_model, 'iam_module', None) is not None else None
            loss_iam = (
                iam_intention_loss(dnp_dict, text_labels, feat_lengths, device, iam_prototypes=iam_protos)
                if lambda_iam > 0 else logits1.sum() * 0.0
            )
            loss_acc = (
                acc_pseudo_label_loss(dnp_dict, feat_lengths, logits1=logits1)
                if lambda_acc > 0 else logits1.sum() * 0.0
            )

            aux_logits = dnp_dict.get('aux_logits')
            aux_sem_logits = dnp_dict.get('aux_sem_logits')
            if aux_logits is not None and args.lambda_aux > 0:
                loss_aux = audio_guided_mil_loss(
                    aux_logits, text_labels, feat_lengths, aux_sem_logits=aux_sem_logits
                )
            else:
                loss_aux = logits1.sum() * 0.0

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
                    + loss_aux * args.lambda_aux
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
                    + loss_aux * args.lambda_aux
                    + loss_smooth * args.lambda_smooth
                )

            optimizer_main.zero_grad(set_to_none=True)
            optimizer_refiner.zero_grad(set_to_none=True)
            loss.backward()

            for p in raw_model.parameters():
                if p.grad is not None:
                    torch.nan_to_num_(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)

            optimizer_main.step()
            optimizer_refiner.step()
            scheduler_main.step()
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
            sum_aux += loss_aux.item()

            iter_num = i + 1
            if iter_num == 1 or iter_num % 45 == 0 or iter_num == num_batches_per_epoch:
                elapsed = max(1e-3, time.time() - epoch_t0)
                it_s = iter_num / elapsed
                eta_s = (num_batches_per_epoch - iter_num) / max(it_s, 1e-4)
                cur_lr = optimizer_main.param_groups[0]['lr']
                aux_str = f" AuxAV:{sum_aux/iter_num:.4f}" if (not args.no_audio and args.lambda_aux > 0) else ""
                print(
                    f"Epoch [{e+1}/{args.max_epoch}] Iter [{iter_num:03d}/{num_batches_per_epoch:03d}] | "
                    f"MIL:{sum_l1/iter_num:.4f} CLASM:{sum_l2/iter_num:.4f} Orth:{sum_l3/iter_num:.4f} "
                    f"Ev:{sum_l4/iter_num:.4f} Bkg:{sum_l5/iter_num:.4f} "
                    f"Cons:{sum_cons/iter_num:.4f} G:{sum_g/iter_num:.4f} LR:{cur_lr:.2e}{aux_str} | "
                    f"{it_s:.2f} it/s ETA:{eta_s:.0f}s",
                    flush=True,
                )

            if args.eval_interval > 0 and iter_num % args.eval_interval == 0 and iter_num < num_batches_per_epoch:
                auc_cur, ap_cur, _ = test(
                    model, testloader, args.visual_length, prompt_text,
                    gt, gtsegments, gtlabels, DNP_use, args, device, compute_dmap=False,
                )
                if ap_cur > ap_best or (abs(ap_cur - ap_best) < 1e-5 and auc_cur > auc_best):
                    ap_best = ap_cur
                    auc_best = auc_cur
                    torch.save(
                        {'epoch': e, 'model_state_dict': unwrap_compact_state_dict(model), 'ap': ap_best, 'auc': auc_best},
                        args.model_path,
                    )
                    print(
                        f"  [*] [NEW BEST XD-VIOLENCE] AP = {ap_best * 100:.2f}% | AUC = {auc_best * 100:.2f}% saved -> {args.model_path}",
                        flush=True,
                    )
                model.train()

        print(f"\n--- Epoch {e+1}/{args.max_epoch} Complete | Evaluating Live Weights ---", flush=True)
        auc_live, ap_live, _ = test(
            model, testloader, args.visual_length, prompt_text,
            gt, gtsegments, gtlabels, DNP_use, args, device, compute_dmap=(e == args.max_epoch - 1),
        )
        live_state = unwrap_compact_state_dict(model)
        if ap_live > ap_best or (abs(ap_live - ap_best) < 1e-5 and auc_live > auc_best):
            ap_best = ap_live
            auc_best = auc_live
            torch.save({'epoch': e, 'model_state_dict': live_state, 'ap': ap_best, 'auc': auc_best}, args.model_path)
            print(
                f"  [*] [NEW BEST XD-VIOLENCE (Live)] AP = {ap_best * 100:.2f}% | AUC = {auc_best * 100:.2f}% saved -> {args.model_path}",
                flush=True,
            )

        if ema is not None:
            backup = ema.apply_to(model)
            print(f"--- Epoch {e+1}/{args.max_epoch} | Evaluating Polyak-EMA Weights ---", flush=True)
            auc_ema, ap_ema, _ = test(
                model, testloader, args.visual_length, prompt_text,
                gt, gtsegments, gtlabels, DNP_use, args, device, compute_dmap=(e == args.max_epoch - 1),
            )
            if ap_ema > ap_best or (abs(ap_ema - ap_best) < 1e-5 and auc_ema > auc_best):
                ap_best = ap_ema
                auc_best = auc_ema
                ema_state = unwrap_compact_state_dict(model)
                torch.save({'epoch': e, 'model_state_dict': ema_state, 'ap': ap_best, 'auc': auc_best}, args.model_path)
                print(
                    f"  [*] [NEW BEST XD-VIOLENCE (EMA)] AP = {ap_best * 100:.2f}% | AUC = {auc_best * 100:.2f}% saved -> {args.model_path}",
                    flush=True,
                )
            ema.restore(model, backup)

        ckpt_payload = {
            'epoch': e,
            'arch_version': 'marupa_v10_sota',
            'model_state_dict': live_state,
            'optimizer_state_dict': optimizer_main.state_dict(),
            'optimizer_refiner_state_dict': optimizer_refiner.state_dict(),
            'scheduler_main_state_dict': scheduler_main.state_dict(),
            'scheduler_refiner_state_dict': scheduler_refiner.state_dict(),
            'ap': ap_live,
            'auc': auc_live,
            'best_ap': ap_best,
            'best_auc': auc_best,
        }
        if ema is not None:
            ckpt_payload['ema_state_dict'] = ema.state_dict()
        torch.save(ckpt_payload, args.checkpoint_path)
        print("", flush=True)

    print(f"[FINISH] XD-Violence Training Complete! Best AP: {ap_best * 100:.2f}% | Best AUC: {auc_best * 100:.2f}%", flush=True)
    return ap_best


if __name__ == '__main__':
    args, unknown = xd_option.parser.parse_known_args()
    if unknown:
        print(f"[CLI] Ignored unrecognized arguments: {unknown}", flush=True)
    train(args)
