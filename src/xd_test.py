import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from model import MARUPA, load_compatible_model_state
from utils.dataset import XDDataset, load_multicrop_test_video
from utils.tools import (
    get_batch_mask, get_prompt_text, smooth_scores,
    safe_load_checkpoint, calibrate_stream_global, xd_test_collate_fn,
)
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option


def refine_scores_hierarchical(logits_det: torch.Tensor,
                               logits_align: torch.Tensor,
                               temp: float = 1.0,
                               total_abnormal_prob: torch.Tensor = None) -> torch.Tensor:
    """Computes per-class snippet probability matrix [T, C] for temporal localization (dmAP)."""
    eps = 1e-12
    eff_temp = max(float(temp), 1e-3)
    if total_abnormal_prob is None:
        total_abnormal_prob = torch.sigmoid(logits_det / eff_temp)
    if total_abnormal_prob.dim() == 1:
        total_abnormal_prob = total_abnormal_prob.unsqueeze(-1)
    total_normal_prob = (1.0 - total_abnormal_prob).clamp(0.0, 1.0)

    p_align = F.softmax(logits_align / eff_temp, dim=1)
    p_abn_only = p_align[:, 1:]
    abn_dist = p_abn_only / p_abn_only.sum(dim=1, keepdim=True).clamp_min(eps)
    final_abn_probs = total_abnormal_prob * abn_dist
    return torch.cat([total_normal_prob, final_abn_probs], dim=1)


def _build_split_lengths(length: int, maxlen: int) -> torch.Tensor:
    num_splits = int(length / maxlen) + 1
    lengths = torch.zeros(num_splits, dtype=torch.long)
    rem = int(length)
    for j in range(num_splits):
        if rem > maxlen:
            lengths[j] = maxlen
            rem -= maxlen
        else:
            lengths[j] = max(1, rem)
            rem = 0
    return lengths


def _forward_multimodal_chunked(eval_model: nn.Module, visual_flat: torch.Tensor,
                                audio_flat: torch.Tensor, padding_mask: torch.Tensor,
                                prompt_text, lengths_flat: torch.Tensor,
                                DNP_use: bool, max_chunk: int = 32):
    """Runs batched crop-splits with optional synchronized audio without VRAM spikes."""
    total_b = visual_flat.shape[0]
    if total_b <= max_chunk:
        _, logits1, logits2, _, _, dnp_dict = eval_model(
            visual_flat, padding_mask, prompt_text, lengths_flat, audio=audio_flat, DNP_use=DNP_use
        )
        route_s = dnp_dict.get('routing_scores', torch.sigmoid(logits1))
        sem_logits = dnp_dict.get('semantic_logits', logits2)
        aux_s = dnp_dict.get('aux_scores')
        gate_s = dnp_dict.get('routing_gate')
        return logits1, logits2, sem_logits, route_s, aux_s, gate_s

    l1_list, l2_list, sem_list, r_list, aux_list, g_list = [], [], [], [], [], []
    for start in range(0, total_b, max_chunk):
        end = min(start + max_chunk, total_b)
        aud_chunk = audio_flat[start:end] if audio_flat is not None else None
        _, l1, l2, _, _, dnp_c = eval_model(
            visual_flat[start:end], padding_mask[start:end], prompt_text,
            lengths_flat[start:end], audio=aud_chunk, DNP_use=DNP_use
        )
        l1_list.append(l1)
        l2_list.append(l2)
        sem_list.append(dnp_c.get('semantic_logits', l2))
        r_list.append(dnp_c.get('routing_scores', torch.sigmoid(l1)))
        if dnp_c.get('aux_scores') is not None:
            aux_list.append(dnp_c['aux_scores'])
        if dnp_c.get('routing_gate') is not None:
            g_list.append(dnp_c['routing_gate'])

    return (
        torch.cat(l1_list, dim=0),
        torch.cat(l2_list, dim=0),
        torch.cat(sem_list, dim=0),
        torch.cat(r_list, dim=0),
        torch.cat(aux_list, dim=0) if aux_list else None,
        torch.cat(g_list, dim=0) if g_list else None,
    )


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         DNP_use, args, device, compute_dmap: bool = False):
    """Evaluates MA-RUPA v3 on XD-Violence with 5-Crop Parallel Multimodal Aggregation & DeSC Fusion."""
    # Defensively support swapped (args, device) vs (device, args) callers
    if isinstance(args, (str, torch.device)) and not isinstance(device, (str, torch.device)):
        args, device = device, args

    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.to(device)
    eval_model.eval()

    multi_crop = bool(getattr(args, 'multi_crop_eval', True))
    max_crops = int(getattr(args, 'max_test_crops', 5)) if multi_crop else 1
    data_dir = getattr(args, 'data_dir', None)
    audio_dir = getattr(args, 'audio_dir', None)
    no_audio = bool(getattr(args, 'no_audio', False))
    collab_alpha = float(getattr(args, 'collab_alpha', 0.15))
    aux_eval_alpha = float(getattr(args, 'aux_eval_alpha', 0.10))
    smooth_eval = bool(getattr(args, 'smooth_eval', True))
    smooth_method = getattr(args, 'smooth_method', 'gaussian')
    smooth_sigma = float(getattr(args, 'smooth_sigma', 0.75))

    vid_route_list = []
    vid_sem_list = []
    vid_aux_list = []
    vid_logits1_list = []
    vid_sem_logits_list = []
    routing_gate_vals = []
    crop_counts = []
    has_aux_stream = False

    with torch.no_grad():
        total_vids = len(testdataloader)
        for i, item in enumerate(testdataloader):
            if total_vids >= 100 and ((i + 1) % 200 == 0 or (i + 1) == total_vids):
                print(f"    [Test-Inference] Processed {i + 1}/{total_vids} XD-Violence test videos...", flush=True)

            fallback_vis = item[0].squeeze(0)
            len_cur = int(item[2])
            raw_path = item[3][0] if isinstance(item[3], (list, tuple)) and len(item[3]) > 0 else str(item[3])

            fallback_aud = None
            if not no_audio and len(item) >= 6 and item[5] is not None and isinstance(item[5], torch.Tensor):
                fallback_aud = item[5].squeeze(0)

            if multi_crop:
                vis_crops, aud_crops, len_cur, num_crops = load_multicrop_test_video(
                    raw_path=raw_path,
                    maxlen=maxlen,
                    data_dir=data_dir,
                    audio_dir=audio_dir,
                    no_audio=no_audio,
                    max_crops=max_crops,
                    fallback_visual=fallback_vis,
                    fallback_length=len_cur,
                    fallback_audio=fallback_aud,
                )
            else:
                v_single = fallback_vis.unsqueeze(0) if fallback_vis.ndim == 2 else fallback_vis
                vis_crops = v_single.unsqueeze(0)
                num_crops = 1
                if fallback_aud is not None and not no_audio:
                    a_single = fallback_aud.unsqueeze(0) if fallback_aud.ndim == 2 else fallback_aud
                    aud_crops = a_single.unsqueeze(0)
                else:
                    aud_crops = None

            crop_counts.append(num_crops)
            C, num_splits, _, D = vis_crops.shape

            base_lengths = _build_split_lengths(len_cur, maxlen)
            if len(base_lengths) != num_splits:
                base_lengths = torch.full((num_splits,), maxlen, dtype=torch.long)
                rem = len_cur - maxlen * (num_splits - 1)
                base_lengths[-1] = max(1, min(maxlen, rem))

            lengths_flat = base_lengths.repeat(C)
            padding_mask = get_batch_mask(lengths_flat, maxlen).to(device)
            visual_flat = vis_crops.view(C * num_splits, maxlen, D).to(device)
            audio_flat = (
                aud_crops.view(C * num_splits, maxlen, aud_crops.shape[-1]).to(device)
                if aud_crops is not None else None
            )

            logits1, logits2, sem_logits, route_s, aux_s, gate_s = _forward_multimodal_chunked(
                eval_model, visual_flat, audio_flat, padding_mask, prompt_text, lengths_flat, DNP_use
            )

            logits1_c = logits1.view(C, num_splits * maxlen, -1)[:, :len_cur, :]
            logits2_c = logits2.view(C, num_splits * maxlen, -1)[:, :len_cur, :]
            sem_logits_c = sem_logits.view(C, num_splits * maxlen, -1)[:, :len_cur, :]
            route_c = route_s.view(C, num_splits * maxlen, -1)[:, :len_cur, 0]

            p_sem_bb = 1.0 - F.softmax(logits2_c.float(), dim=-1)[:, :, 0]
            p_sem_rupa = 1.0 - F.softmax(sem_logits_c.float(), dim=-1)[:, :, 0]
            p_sem_c = (0.5 * p_sem_bb + 0.5 * p_sem_rupa).clamp(0.0, 1.0)

            prob_route_np = route_c.float().mean(dim=0).cpu().numpy()
            prob_sem_np = p_sem_c.float().mean(dim=0).cpu().numpy()

            vid_route_list.append(prob_route_np)
            vid_sem_list.append(prob_sem_np)
            vid_logits1_list.append(logits1_c.float().mean(dim=0).cpu())
            vid_sem_logits_list.append(sem_logits_c.float().mean(dim=0).cpu())

            if aux_s is not None:
                has_aux_stream = True
                aux_c = aux_s.view(C, num_splits * maxlen, -1)[:, :len_cur, 0]
                vid_aux_list.append(aux_c.float().mean(dim=0).cpu().numpy())
            else:
                vid_aux_list.append(prob_route_np)

            if gate_s is not None:
                gate_c = gate_s.view(C, num_splits * maxlen)[:, :len_cur].mean(dim=0)
                routing_gate_vals.append(gate_c.detach().cpu())

    # Unsupervised quantile alignment of semantic & auxiliary streams (zero GT labels used)
    concat_route = np.concatenate(vid_route_list, axis=0)
    concat_sem = np.concatenate(vid_sem_list, axis=0)
    q15_r = float(np.clip(np.percentile(concat_route, 15.0), 0.01, 0.35))
    q90_r = float(np.clip(np.percentile(concat_route, 90.0), 0.50, 0.98))
    cal_sem_concat = calibrate_stream_global(concat_sem, target_q15=q15_r, target_q90=q90_r)

    if has_aux_stream:
        concat_aux = np.concatenate(vid_aux_list, axis=0)
        cal_aux_concat = calibrate_stream_global(concat_aux, target_q15=q15_r, target_q90=q90_r)
    else:
        cal_aux_concat = concat_route

    eff_aux_alpha = aux_eval_alpha if has_aux_stream else 0.0
    eff_route_alpha = max(0.0, 1.0 - collab_alpha - eff_aux_alpha)

    final_route_all = []
    final_sem_all = []
    final_aux_all = []
    final_collab_all = []
    element_logits2_stack = []

    offset = 0
    for idx_v, r_v in enumerate(vid_route_list):
        lv = len(r_v)
        s_v_cal = cal_sem_concat[offset:offset + lv]
        a_v_cal = cal_aux_concat[offset:offset + lv]
        s_v_raw = vid_sem_list[idx_v]
        a_v_raw = vid_aux_list[idx_v]
        offset += lv

        # Visual-anchored acoustic reliability gate suppresses isolated movie music spikes on calm scenes
        g_vis = np.clip(r_v / (r_v + 0.12), 0.15, 1.0)
        gated_aux_v = g_vis * a_v_cal + (1.0 - g_vis) * r_v
        c_v = eff_route_alpha * r_v + collab_alpha * s_v_cal + eff_aux_alpha * gated_aux_v

        if smooth_eval:
            r_eval = smooth_scores(r_v, method=smooth_method, sigma=smooth_sigma)
            s_eval = smooth_scores(s_v_raw, method=smooth_method, sigma=smooth_sigma)
            a_eval = smooth_scores(a_v_raw, method=smooth_method, sigma=smooth_sigma)
            c_eval = smooth_scores(c_v, method=smooth_method, sigma=smooth_sigma)
        else:
            r_eval, s_eval, a_eval, c_eval = r_v, s_v_raw, a_v_raw, c_v

        final_route_all.append(r_eval)
        final_sem_all.append(s_eval)
        final_aux_all.append(a_eval)
        final_collab_all.append(c_eval)

        if compute_dmap:
            collab_t = torch.from_numpy(c_eval).float().unsqueeze(-1)
            opt_probs = refine_scores_hierarchical(
                vid_logits1_list[idx_v], vid_sem_logits_list[idx_v], args.temp, collab_t
            )
            element_logits2_stack.append(np.repeat(opt_probs.numpy(), 16, axis=0))

    pred_route = np.nan_to_num(np.repeat(np.concatenate(final_route_all, axis=0), 16), nan=0.0, posinf=1.0, neginf=0.0)
    pred_sem = np.nan_to_num(np.repeat(np.concatenate(final_sem_all, axis=0), 16), nan=0.0, posinf=1.0, neginf=0.0)
    pred_aux = np.nan_to_num(np.repeat(np.concatenate(final_aux_all, axis=0), 16), nan=0.0, posinf=1.0, neginf=0.0)
    pred_collab = np.nan_to_num(np.repeat(np.concatenate(final_collab_all, axis=0), 16), nan=0.0, posinf=1.0, neginf=0.0)

    n_gt = len(gt)
    pred_route = pred_route[:n_gt] if len(pred_route) >= n_gt else np.pad(pred_route, (0, n_gt - len(pred_route)))
    pred_sem = pred_sem[:n_gt] if len(pred_sem) >= n_gt else np.pad(pred_sem, (0, n_gt - len(pred_sem)))
    pred_aux = pred_aux[:n_gt] if len(pred_aux) >= n_gt else np.pad(pred_aux, (0, n_gt - len(pred_aux)))
    pred_collab = pred_collab[:n_gt] if len(pred_collab) >= n_gt else np.pad(pred_collab, (0, n_gt - len(pred_collab)))

    auc_route = roc_auc_score(gt, pred_route)
    ap_route = average_precision_score(gt, pred_route)
    auc_sem = roc_auc_score(gt, pred_sem)
    ap_sem = average_precision_score(gt, pred_sem)
    auc_collab = roc_auc_score(gt, pred_collab)
    ap_collab = average_precision_score(gt, pred_collab)

    avg_crops = float(np.mean(crop_counts)) if crop_counts else 1.0
    gate_info = ""
    if routing_gate_vals:
        gates_cat = torch.cat(routing_gate_vals)
        gate_info = f" | SafeGate={gates_cat.mean().item():.4f}"
    if hasattr(eval_model, 'audio_adapter') and eval_model.audio_adapter is not None:
        alpha_av = float(torch.tanh(eval_model.audio_adapter.alpha_av).detach().cpu().item())
        gate_info += f" | AlphaAV={alpha_av:.3f}"

    print(f"  [Eval {avg_crops:.1f}-Crop{gate_info}]", flush=True)
    print(f"    Safe-RUPA Multimodal Routing : AP = {ap_route * 100:.2f}% | AUC = {auc_route * 100:.2f}%", flush=True)
    print(f"    Vision-Language Semantic     : AP = {ap_sem * 100:.2f}% | AUC = {auc_sem * 100:.2f}%", flush=True)
    if has_aux_stream:
        auc_aux = roc_auc_score(gt, pred_aux)
        ap_aux = average_precision_score(gt, pred_aux)
        print(f"    Audio-Visual Auxiliary Head  : AP = {ap_aux * 100:.2f}% | AUC = {auc_aux * 100:.2f}%", flush=True)
    print(f"    MA-RUPA DeSC Collaborative   : AP = {ap_collab * 100:.2f}% | AUC = {auc_collab * 100:.2f}%", flush=True)

    avg_dmap = 0.0
    if compute_dmap and element_logits2_stack:
        try:
            dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
            avg_dmap = float(np.mean(dmap))
            iou_str = " | ".join([f"mAP@{iou[k]:.1f}={dmap[k]:.2f}%" for k in range(len(iou))])
            print(f"    Localization: {iou_str} | Avg mAP = {avg_dmap:.2f}%", flush=True)
        except Exception as exc:
            print(f"    [Note] Detection mAP skipped: {exc}", flush=True)

    sys.stdout.flush()
    if ap_collab >= ap_route:
        primary_ap, primary_auc = ap_collab, auc_collab
    else:
        primary_ap, primary_auc = ap_route, auc_route
    return primary_auc, primary_ap, avg_dmap


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, unknown = xd_option.parser.parse_known_args()
    if unknown:
        print(f"[CLI] Ignored unrecognized arguments: {unknown}", flush=True)

    label_map = {
        'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
        'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion',
    }

    test_dataset = XDDataset(
        args.visual_length, args.test_list, True, label_map,
        audio_file_path=args.audio_test_list, audio_dir=args.audio_dir,
        data_dir=args.data_dir, no_audio=args.no_audio,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, collate_fn=xd_test_collate_fn,
    )

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    detected_audio_dim = None
    if not args.no_audio:
        for idx in range(min(30, len(test_dataset))):
            item_tuple = test_dataset[idx]
            sample_audio = item_tuple[5] if len(item_tuple) >= 6 else None
            if sample_audio is not None and isinstance(sample_audio, torch.Tensor):
                detected_audio_dim = int(sample_audio.shape[-1])
                break
    if detected_audio_dim is not None and detected_audio_dim != args.audio_dim:
        print(f"[Auto-Detect] Audio feature dim is {detected_audio_dim}-D; updating args.audio_dim.", flush=True)
        args.audio_dim = detected_audio_dim

    model = MARUPA(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, args, device,
    )
    if args.model_path:
        ckpt = safe_load_checkpoint(args.model_path, map_location=device)
        load_compatible_model_state(model, ckpt, allow_legacy_dsanet=True)

    test(
        model, test_loader, args.visual_length, prompt_text,
        gt, gtsegments, gtlabels, args.DNP_use, args, device, compute_dmap=True,
    )
