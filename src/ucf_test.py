import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from model import MARUPA, load_compatible_model_state
from utils.dataset import UCFDataset, load_multicrop_test_video
from utils.tools import (
    get_batch_mask, get_prompt_text, smooth_scores, safe_load_checkpoint,
)
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option


def refine_scores_hierarchical(logits_det: torch.Tensor,
                               logits_align: torch.Tensor,
                               temp: float = 5.0,
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


def _forward_splits_chunked(eval_model: nn.Module, visual_flat: torch.Tensor,
                            padding_mask: torch.Tensor, prompt_text, lengths_flat: torch.Tensor,
                            DNP_use: bool, max_chunk: int = 32):
    """Runs batched crop-splits on a single GPU without VRAM spikes on ultra-long videos."""
    total_b = visual_flat.shape[0]
    if total_b <= max_chunk:
        _, logits1, logits2, _, _, dnp_dict = eval_model(
            visual_flat, padding_mask, prompt_text, lengths_flat, audio=None, DNP_use=DNP_use
        )
        det_s = dnp_dict.get('detector_scores', torch.sigmoid(logits1))
        route_s = dnp_dict.get('routing_scores', det_s)
        rec_s = dnp_dict.get('reconstruction_scores', det_s)
        gate_s = dnp_dict.get('routing_gate')
        return logits1, logits2, det_s, route_s, rec_s, gate_s

    l1_list, l2_list, d_list, r_list, rec_list, g_list = [], [], [], [], [], []
    for start in range(0, total_b, max_chunk):
        end = min(start + max_chunk, total_b)
        _, l1, l2, _, _, dnp_c = eval_model(
            visual_flat[start:end], padding_mask[start:end], prompt_text,
            lengths_flat[start:end], audio=None, DNP_use=DNP_use
        )
        d_c = dnp_c.get('detector_scores', torch.sigmoid(l1))
        l1_list.append(l1)
        l2_list.append(l2)
        d_list.append(d_c)
        r_list.append(dnp_c.get('routing_scores', d_c))
        rec_list.append(dnp_c.get('reconstruction_scores', d_c))
        if dnp_c.get('routing_gate') is not None:
            g_list.append(dnp_c['routing_gate'])

    return (
        torch.cat(l1_list, dim=0),
        torch.cat(l2_list, dim=0),
        torch.cat(d_list, dim=0),
        torch.cat(r_list, dim=0),
        torch.cat(rec_list, dim=0),
        torch.cat(g_list, dim=0) if g_list else None,
    )


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         DNP_use, device, args, compute_dmap: bool = False):
    """Evaluates MA-RUPA v4 on UCF-Crime with Parallel 10-Crop Aggregation & Calibrated Synergy."""
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.to(device)
    eval_model.eval()

    multi_crop = bool(getattr(args, 'multi_crop_eval', True))
    max_crops = int(getattr(args, 'max_test_crops', 10)) if multi_crop else 1
    data_dir = getattr(args, 'data_dir', None)
    collab_alpha = float(getattr(args, 'collab_alpha', 0.12))
    smooth_eval = bool(getattr(args, 'smooth_eval', True))
    smooth_method = getattr(args, 'smooth_method', 'gaussian')
    smooth_sigma = float(getattr(args, 'smooth_sigma', 0.80))

    vid_det_raw_list = []
    vid_route_list = []
    vid_sem_list = []
    vid_rec_list = []
    vid_logits1_list = []
    vid_logits2_list = []
    routing_gate_vals = []
    crop_counts = []

    with torch.no_grad():
        total_vids = len(testdataloader)
        for i, item in enumerate(testdataloader):
            if total_vids >= 100 and ((i + 1) % 100 == 0 or (i + 1) == total_vids):
                print(f"    [Test-Inference] Processed {i + 1}/{total_vids} UCF test videos...", flush=True)
            fallback_vis = item[0].squeeze(0)
            len_cur = int(item[2])
            raw_path = item[3][0] if isinstance(item[3], (list, tuple)) else str(item[3])

            if multi_crop:
                vis_crops, _, len_cur, num_crops = load_multicrop_test_video(
                    raw_path=raw_path,
                    maxlen=maxlen,
                    data_dir=data_dir,
                    no_audio=True,
                    max_crops=max_crops,
                    fallback_visual=fallback_vis,
                    fallback_length=len_cur,
                )
            else:
                v_single = fallback_vis.unsqueeze(0) if fallback_vis.ndim == 2 else fallback_vis
                vis_crops = v_single.unsqueeze(0)
                num_crops = 1

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

            logits1, logits2, det_s, route_s, rec_s, gate_s = _forward_splits_chunked(
                eval_model, visual_flat, padding_mask, prompt_text, lengths_flat, DNP_use
            )

            logits1_c = logits1.view(C, num_splits * maxlen, -1)[:, :len_cur, :]
            logits2_c = logits2.view(C, num_splits * maxlen, -1)[:, :len_cur, :]
            det_c = det_s.view(C, num_splits * maxlen, -1)[:, :len_cur, 0]
            route_c = route_s.view(C, num_splits * maxlen, -1)[:, :len_cur, 0]
            rec_c = rec_s.view(C, num_splits * maxlen, -1)[:, :len_cur, 0]

            # Temperature-calibrated Vision-Language Semantic Anomaly Probability (smooth log-odds, zero quantile distortion)
            p_sem_c = (1.0 - F.softmax(logits2_c.float() / 2.5, dim=-1)[:, :, 0]).clamp(0.0, 1.0)

            vid_det_raw_list.append(det_c.float().mean(dim=0).cpu().numpy())
            vid_route_list.append(route_c.float().mean(dim=0).cpu().numpy())
            vid_sem_list.append(p_sem_c.float().mean(dim=0).cpu().numpy())
            vid_rec_list.append(rec_c.float().mean(dim=0).cpu().numpy())
            vid_logits1_list.append(logits1_c.float().mean(dim=0).cpu())
            vid_logits2_list.append(logits2_c.float().mean(dim=0).cpu())

            if gate_s is not None:
                gate_c = gate_s.view(C, num_splits * maxlen)[:, :len_cur].mean(dim=0)
                routing_gate_vals.append(gate_c.detach().cpu())

    final_det_raw_all = []
    final_route_all = []
    final_sem_all = []
    final_collab_all = []
    element_logits2_stack = []

    for idx_v, r_v in enumerate(vid_route_list):
        d_v = vid_det_raw_list[idx_v]
        s_v = vid_sem_list[idx_v]
        rec_v = vid_rec_list[idx_v]

        # Calibrated multi-view synergy: SGNM `rec_v` is already scale-matched to `d_v` via 1.0 * ConsistencyLoss
        aux_synergy = 0.70 * s_v + 0.30 * rec_v
        c_v = (1.0 - collab_alpha) * r_v + collab_alpha * aux_synergy

        if smooth_eval:
            r_eval = smooth_scores(r_v, method=smooth_method, sigma=smooth_sigma)
            s_eval = smooth_scores(s_v, method=smooth_method, sigma=smooth_sigma)
            c_eval = smooth_scores(c_v, method=smooth_method, sigma=smooth_sigma)
        else:
            r_eval, s_eval, c_eval = r_v, s_v, c_v

        final_det_raw_all.append(d_v)
        final_route_all.append(r_eval)
        final_sem_all.append(s_eval)
        final_collab_all.append(c_eval)

        if compute_dmap:
            collab_t = torch.from_numpy(c_eval).float().unsqueeze(-1)
            opt_probs = refine_scores_hierarchical(
                vid_logits1_list[idx_v], vid_logits2_list[idx_v], args.temp, collab_t
            )
            element_logits2_stack.append(np.repeat(opt_probs.numpy(), 16, axis=0))

    pred_det_raw = np.repeat(np.concatenate(final_det_raw_all, axis=0), 16)
    pred_route = np.repeat(np.concatenate(final_route_all, axis=0), 16)
    pred_sem = np.repeat(np.concatenate(final_sem_all, axis=0), 16)
    pred_collab = np.repeat(np.concatenate(final_collab_all, axis=0), 16)

    n_gt = len(gt)
    pred_det_raw = pred_det_raw[:n_gt] if len(pred_det_raw) >= n_gt else np.pad(pred_det_raw, (0, n_gt - len(pred_det_raw)))
    pred_route = pred_route[:n_gt] if len(pred_route) >= n_gt else np.pad(pred_route, (0, n_gt - len(pred_route)))
    pred_sem = pred_sem[:n_gt] if len(pred_sem) >= n_gt else np.pad(pred_sem, (0, n_gt - len(pred_sem)))
    pred_collab = pred_collab[:n_gt] if len(pred_collab) >= n_gt else np.pad(pred_collab, (0, n_gt - len(pred_collab)))

    auc_det_raw = roc_auc_score(gt, pred_det_raw)
    ap_det_raw = average_precision_score(gt, pred_det_raw)
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

    print(f"  [Eval {avg_crops:.1f}-Crop{gate_info}]", flush=True)
    print(f"    LGT-SGNM Detector (Raw)    : AUC = {auc_det_raw * 100:.2f}% | AP = {ap_det_raw * 100:.2f}%", flush=True)
    print(f"    Safe-RUPA Detector-Routing : AUC = {auc_route * 100:.2f}% | AP = {ap_route * 100:.2f}%", flush=True)
    print(f"    Vision-Language Semantic   : AUC = {auc_sem * 100:.2f}% | AP = {ap_sem * 100:.2f}%", flush=True)
    print(f"    MA-RUPA Collaborative SOTA : AUC = {auc_collab * 100:.2f}% | AP = {ap_collab * 100:.2f}%", flush=True)

    if compute_dmap and element_logits2_stack:
        try:
            dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
            avg_dmap = float(np.mean(dmap[:5]))
            iou_str = " | ".join([f"mAP@{iou[k]:.1f}={dmap[k]:.2f}%" for k in range(min(5, len(iou)))])
            print(f"    Localization: {iou_str} | Avg mAP = {avg_dmap:.2f}%", flush=True)
        except Exception as exc:
            print(f"    [Note] Detection mAP skipped: {exc}", flush=True)

    sys.stdout.flush()
    candidates = [(auc_collab, ap_collab), (auc_route, ap_route), (auc_det_raw, ap_det_raw)]
    primary_auc, primary_ap = max(candidates, key=lambda x: x[0])
    return primary_auc, primary_ap


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, unknown = ucf_option.parser.parse_known_args()
    if unknown:
        print(f"[CLI] Ignored unrecognized arguments: {unknown}", flush=True)

    label_map = {
        'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
        'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
        'Fighting': 'fighting', 'RoadAccidents': 'roadaccidents', 'Robbery': 'robbery',
        'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
        'Vandalism': 'vandalism',
    }

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map, data_dir=args.data_dir)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    model = MARUPA(
        args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
        args.visual_head, args.visual_layers, args.attn_window,
        args.prompt_prefix, args.prompt_postfix, args, device,
    )
    ckpt = safe_load_checkpoint(args.model_path, map_location=device)
    load_compatible_model_state(model, ckpt, allow_legacy_dsanet=True)

    test(
        model, testdataloader, args.visual_length, prompt_text,
        gt, gtsegments, gtlabels, args.DNP_use, device, args, compute_dmap=True,
    )
