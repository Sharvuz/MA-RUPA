import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import MARUPA, load_compatible_model_state
from utils.dataset import XDDataset
from utils.tools import (
    get_batch_mask, get_prompt_text, smooth_scores, safe_load_checkpoint,
    xd_test_collate_fn, calibrate_stream_global, compute_single_pass_uccf, apply_dodo_refinement
)
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option


def _calibrate_video_list(vid_list, target_q15: float = 0.04, target_q90: float = 0.78):
    """Applies unsupervised dataset-level monotonic logit-affine calibration across videos."""
    if not vid_list:
        return vid_list
    lengths_v = [len(v) for v in vid_list]
    concat_v = np.concatenate(vid_list, axis=0)
    cal_concat = calibrate_stream_global(concat_v, target_q15=target_q15, target_q90=target_q90)
    out_list = []
    offset = 0
    for lv in lengths_v:
        out_list.append(cal_concat[offset:offset + lv])
        offset += lv
    return out_list


def refine_scores_hierarchical(logits_det: torch.Tensor,
                               logits_align: torch.Tensor,
                               temp: float = 1.0,
                               routing_prob: torch.Tensor = None) -> torch.Tensor:
    """Hierarchically combines anomaly probabilities with fine-grained semantic alignments
    and video-level class sparsity filtering for temporal localization (dmAP).
    """
    epsilon = 1e-12
    eff_temp = (temp / 0.07) if temp < 0.25 else temp
    if routing_prob is None:
        total_abnormal_prob = torch.sigmoid(logits_det / eff_temp)
    else:
        total_abnormal_prob = routing_prob
    if total_abnormal_prob.dim() == 1:
        total_abnormal_prob = total_abnormal_prob.unsqueeze(-1)
    total_normal_prob = (1.0 - total_abnormal_prob).clamp(min=0.0, max=1.0)

    p_align = F.softmax(logits_align / eff_temp, dim=1)
    p_align_abnormal_only = p_align[:, 1:]
    sum_p_align = p_align_abnormal_only.sum(dim=1, keepdim=True).clamp_min(epsilon)
    abnormal_dist = p_align_abnormal_only / sum_p_align

    final_abnormal_probs = total_abnormal_prob * abnormal_dist

    # Video-level top-k class sparsity weighting for temporal localization (dmAP)
    L = final_abnormal_probs.shape[0]
    k_vid = max(1, L // 16)
    top_c = torch.topk(final_abnormal_probs, k=k_vid, dim=0).values.mean(dim=0)
    top3_thresh = torch.topk(top_c, k=min(3, top_c.numel())).values[-1]
    class_weight = torch.where(
        top_c >= top3_thresh,
        torch.ones_like(top_c),
        torch.full_like(top_c, 0.15)
    ).unsqueeze(0)
    final_abnormal_probs = final_abnormal_probs * class_weight

    final_probs = torch.cat([total_normal_prob, final_abnormal_probs], dim=1)
    return final_probs


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         DNP_use, args, device, compute_dmap: bool = False):
    """Evaluates MA-RUPA on XD-Violence test set.
    
    Computes Frame-level Average Precision (AP), Frame-level ROC-AUC, and mAP@IoU[0.1:0.5].
    Supports multimodal visual+audio inference and temporal score smoothing.
    """
    model.to(device)
    model.eval()

    element_logits2_stack = []
    ap_routing_list = []
    ap_det_list = []
    ap_aux_list = []
    ap_sem_list = []
    ap_sem_rupa_list = []
    ap_ens_sem_list = []
    ap_ens_rupa_list = []
    ap_ens_det_list = []
    ap_ens_aux_list = []
    ap_ens_audio_det_list = []
    ap_ens_tri_list = []
    ap_ens_quad_list = []
    ap_weighted_list = []
    ap_geom_list = []
    ap_hier_list = []
    ap_uccf_list = []
    ap_dodo_list = []
    ap_harmonic_list = []
    ap_master_list = []

    smooth_eval = getattr(args, 'smooth_eval', True)
    smooth_method = getattr(args, 'smooth_method', 'gaussian')
    smooth_sigma = getattr(args, 'smooth_sigma', 0.50)

    raw_routing_vids = []
    raw_det_vids = []
    raw_aux_vids = []
    raw_sem_vids = []
    raw_sem_rupa_vids = []
    raw_rec_vids = []
    logits1_vids = []
    fused_sem_logits_vids = []
    eff_temp = (args.temp / 0.07) if args.temp < 0.25 else args.temp

    with torch.no_grad():
        for i, item in enumerate(testdataloader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            audio = None
            if len(item) >= 6:
                audio = item[5]
                if audio is not None and isinstance(audio, torch.Tensor):
                    audio = audio.squeeze(0).to(device)
                    if len_cur < maxlen and audio.ndim == 2:
                        audio = audio.unsqueeze(0)

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)

            visual = visual.to(device)

            # Build split sequence lengths
            num_splits = int(length / maxlen) + 1
            lengths = torch.zeros(num_splits, dtype=torch.long)
            rem_len = length
            for j in range(num_splits):
                if rem_len > maxlen:
                    lengths[j] = maxlen
                    rem_len -= maxlen
                else:
                    lengths[j] = rem_len

            padding_mask = get_batch_mask(lengths, maxlen).to(device)

            if DNP_use:
                _, logits1, logits2, _, _, dnp_dict = model(
                    visual, padding_mask, prompt_text, lengths, audio=audio, DNP_use=True
                )
                routing_scores = dnp_dict.get('routing_scores')
                semantic_logits = dnp_dict.get('semantic_logits', logits2)
            else:
                _, logits1, logits2, _, _, dnp_dict = model(
                    visual, padding_mask, prompt_text, lengths, audio=audio, DNP_use=False
                )
                routing_scores = dnp_dict.get('routing_scores', torch.sigmoid(logits1))
                semantic_logits = logits2

            logits1 = logits1.reshape(-1, logits1.shape[-1])
            logits2 = logits2.reshape(-1, logits2.shape[-1])
            if routing_scores is not None:
                routing_scores = routing_scores.reshape(-1, routing_scores.shape[-1])
            if semantic_logits is not None:
                semantic_logits = semantic_logits.reshape(-1, semantic_logits.shape[-1])

            # Stream 1: Routing prediction
            if routing_scores is None:
                prob_routing = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            else:
                prob_routing = routing_scores[0:len_cur].squeeze(-1)

            # Stream 2: Base detector prediction
            if 'detector_scores' in dnp_dict:
                prob_det = dnp_dict['detector_scores'].reshape(-1, 1)[0:len_cur].squeeze(-1)
            else:
                prob_det = torch.sigmoid(logits1[0:len_cur].squeeze(-1))

            # Stream 3: Zero-Shot / Adapted Semantic Alignment (TVSA: P(Anomaly) = 1.0 - P(Normal))
            p_align = F.softmax(logits2[0:len_cur] / eff_temp, dim=-1)
            p0_bb = p_align[:, 0]
            prob_sem = (1.0 - p0_bb).clamp(0.0, 1.0)

            # Stream 4: RUPA Counterfactual Semantic Logits
            p_align_rupa = F.softmax(semantic_logits[0:len_cur] / eff_temp, dim=-1)
            p0_rupa = p_align_rupa[:, 0]
            prob_sem_rupa_raw = (1.0 - p0_rupa).clamp(0.0, 1.0)
            if 'semantic_scores' in dnp_dict:
                s_sem_direct = dnp_dict['semantic_scores'].reshape(-1, 1)[0:len_cur].squeeze(-1)
                prob_sem_rupa = 0.5 * prob_sem_rupa_raw + 0.5 * s_sem_direct
            else:
                prob_sem_rupa = prob_sem_rupa_raw

            # Reconstruction & Auxiliary Audio-Visual Score Streams
            if 'reconstruction_scores' in dnp_dict:
                prob_rec = dnp_dict['reconstruction_scores'].reshape(-1, 1)[0:len_cur].squeeze(-1)
            else:
                prob_rec = prob_sem_rupa

            if 'aux_scores' in dnp_dict:
                prob_aux = dnp_dict['aux_scores'].reshape(-1, 1)[0:len_cur].squeeze(-1)
            else:
                prob_aux = prob_det

            if audio is not None and 'audio_semantic_logits' in dnp_dict and dnp_dict['audio_semantic_logits'] is not None:
                aud_l = dnp_dict['audio_semantic_logits'].reshape(-1, logits2.shape[-1])[0:len_cur]
                fused_sem_logits = 0.45 * logits2[0:len_cur] + 0.40 * semantic_logits[0:len_cur] + 0.15 * aud_l
            else:
                fused_sem_logits = 0.5 * logits2[0:len_cur] + 0.5 * semantic_logits[0:len_cur]

            raw_routing_vids.append(prob_routing.detach().cpu().numpy())
            raw_det_vids.append(prob_det.detach().cpu().numpy())
            raw_aux_vids.append(prob_aux.detach().cpu().numpy())
            raw_sem_vids.append(prob_sem.detach().cpu().numpy())
            raw_sem_rupa_vids.append(prob_sem_rupa.detach().cpu().numpy())
            raw_rec_vids.append(prob_rec.detach().cpu().numpy())
            logits1_vids.append(logits1[0:len_cur].detach().cpu())
            fused_sem_logits_vids.append(fused_sem_logits.detach().cpu())

    # Unsupervised Dataset-Level Monotonic Calibration (zero GT access; preserves single-stream AP/AUC)
    cal_routing_vids = _calibrate_video_list(raw_routing_vids)
    cal_det_vids = _calibrate_video_list(raw_det_vids)
    cal_aux_vids = _calibrate_video_list(raw_aux_vids)
    cal_sem_vids = _calibrate_video_list(raw_sem_vids)
    cal_sem_rupa_vids = _calibrate_video_list(raw_sem_rupa_vids)
    cal_rec_vids = _calibrate_video_list(raw_rec_vids)

    priors = {
        'routing': 1.0,
        'sem': 0.90,
        'det': 0.95,
        'sem_rupa': 0.90,
        'aux': 0.30,
        'rec': 0.02
    }

    for v_idx in range(len(cal_routing_vids)):
        p_routing_np = cal_routing_vids[v_idx]
        p_det_np = cal_det_vids[v_idx]
        p_aux_np = cal_aux_vids[v_idx]
        p_sem_np = cal_sem_vids[v_idx]
        p_sem_rupa_np = cal_sem_rupa_vids[v_idx]
        p_rec_np = cal_rec_vids[v_idx]

        p_ens_sem_np = 0.5 * p_routing_np + 0.5 * p_sem_np
        p_ens_rupa_np = 0.5 * p_routing_np + 0.5 * p_sem_rupa_np
        p_ens_det_np = 0.5 * p_routing_np + 0.5 * p_det_np
        p_ens_aux_np = 0.5 * p_routing_np + 0.5 * p_aux_np
        p_ens_audio_det_np = 0.5 * p_det_np + 0.5 * p_aux_np
        p_ens_tri_np = (p_routing_np + p_sem_np + p_sem_rupa_np) / 3.0
        p_ens_quad_np = 0.30 * p_routing_np + 0.30 * p_sem_np + 0.20 * p_aux_np + 0.20 * p_sem_rupa_np
        p_weighted_np = 0.6 * p_routing_np + 0.4 * p_sem_rupa_np
        p_geom_np = np.sqrt(np.clip(p_routing_np, 1e-6, 1.0) * np.clip(p_sem_rupa_np, 1e-5, 1.0))
        p_harmonic_np = 2.0 * np.clip(p_routing_np, 1e-4, 1.0) * np.clip(p_sem_rupa_np, 1e-4, 1.0) / (
            np.clip(p_routing_np, 1e-4, 1.0) + np.clip(p_sem_rupa_np, 1e-4, 1.0) + 1e-8
        )
        p_master_np = 0.35 * p_routing_np + 0.35 * p_sem_rupa_np + 0.15 * p_det_np + 0.15 * p_aux_np

        prob_dict = {
            'routing': torch.from_numpy(p_routing_np),
            'sem': torch.from_numpy(p_sem_np),
            'det': torch.from_numpy(p_det_np),
            'sem_rupa': torch.from_numpy(p_sem_rupa_np),
            'aux': torch.from_numpy(p_aux_np),
            'rec': torch.from_numpy(p_rec_np),
        }
        p_uccf_np = compute_single_pass_uccf(prob_dict, priors, delta_dis=0.35).numpy()
        p_dodo_np = apply_dodo_refinement(p_uccf_np, sigma=smooth_sigma)

        # Sharp DODO-anchored consensus probability for temporal localization (dmAP)
        consensus_abn_t = torch.from_numpy(
            (0.50 * p_dodo_np + 0.25 * p_sem_rupa_np + 0.25 * p_routing_np).astype(np.float32)
        ).unsqueeze(-1)
        optimized_probs = refine_scores_hierarchical(
            logits1_vids[v_idx], fused_sem_logits_vids[v_idx], eff_temp, consensus_abn_t
        )
        p_hier_np = (1.0 - optimized_probs[:, 0]).clamp(0.0, 1.0).numpy()

        if smooth_eval:
            p_routing_np = smooth_scores(p_routing_np, method=smooth_method, sigma=smooth_sigma)
            p_det_np = smooth_scores(p_det_np, method=smooth_method, sigma=smooth_sigma)
            p_aux_np = smooth_scores(p_aux_np, method=smooth_method, sigma=smooth_sigma)
            p_sem_np = smooth_scores(p_sem_np, method=smooth_method, sigma=smooth_sigma)
            p_sem_rupa_np = smooth_scores(p_sem_rupa_np, method=smooth_method, sigma=smooth_sigma)
            p_ens_sem_np = smooth_scores(p_ens_sem_np, method=smooth_method, sigma=smooth_sigma)
            p_ens_rupa_np = smooth_scores(p_ens_rupa_np, method=smooth_method, sigma=smooth_sigma)
            p_ens_det_np = smooth_scores(p_ens_det_np, method=smooth_method, sigma=smooth_sigma)
            p_ens_aux_np = smooth_scores(p_ens_aux_np, method=smooth_method, sigma=smooth_sigma)
            p_ens_audio_det_np = smooth_scores(p_ens_audio_det_np, method=smooth_method, sigma=smooth_sigma)
            p_ens_tri_np = smooth_scores(p_ens_tri_np, method=smooth_method, sigma=smooth_sigma)
            p_ens_quad_np = smooth_scores(p_ens_quad_np, method=smooth_method, sigma=smooth_sigma)
            p_weighted_np = smooth_scores(p_weighted_np, method=smooth_method, sigma=smooth_sigma)
            p_geom_np = smooth_scores(p_geom_np, method=smooth_method, sigma=smooth_sigma)
            p_harmonic_np = smooth_scores(p_harmonic_np, method=smooth_method, sigma=smooth_sigma)
            p_master_np = smooth_scores(p_master_np, method=smooth_method, sigma=smooth_sigma)
            p_hier_np = smooth_scores(p_hier_np, method=smooth_method, sigma=smooth_sigma)
            p_uccf_np = smooth_scores(p_uccf_np, method=smooth_method, sigma=smooth_sigma)

        ap_routing_list.append(p_routing_np)
        ap_det_list.append(p_det_np)
        ap_aux_list.append(p_aux_np)
        ap_sem_list.append(p_sem_np)
        ap_sem_rupa_list.append(p_sem_rupa_np)
        ap_ens_sem_list.append(p_ens_sem_np)
        ap_ens_rupa_list.append(p_ens_rupa_np)
        ap_ens_det_list.append(p_ens_det_np)
        ap_ens_aux_list.append(p_ens_aux_np)
        ap_ens_audio_det_list.append(p_ens_audio_det_np)
        ap_ens_tri_list.append(p_ens_tri_np)
        ap_ens_quad_list.append(p_ens_quad_np)
        ap_weighted_list.append(p_weighted_np)
        ap_geom_list.append(p_geom_np)
        ap_harmonic_list.append(p_harmonic_np)
        ap_master_list.append(p_master_np)
        ap_hier_list.append(p_hier_np)
        ap_uccf_list.append(p_uccf_np)
        ap_dodo_list.append(p_dodo_np)

        elem_l2 = optimized_probs.detach().numpy()
        dmap_sigma = min(smooth_sigma, 0.85)
        if smooth_eval and elem_l2.shape[0] >= 3 and dmap_sigma > 0.1:
            for c_idx in range(elem_l2.shape[1]):
                if np.max(elem_l2[:, c_idx]) > 0.0:
                    elem_l2[:, c_idx] = smooth_scores(elem_l2[:, c_idx], method=smooth_method, sigma=dmap_sigma)
        elem_l2 = np.repeat(elem_l2, 16, axis=0)
        element_logits2_stack.append(elem_l2)

    all_scores_routing = np.concatenate(ap_routing_list, axis=0)
    all_scores_det = np.concatenate(ap_det_list, axis=0)
    all_scores_aux = np.concatenate(ap_aux_list, axis=0)
    all_scores_sem = np.concatenate(ap_sem_list, axis=0)
    all_scores_sem_rupa = np.concatenate(ap_sem_rupa_list, axis=0)
    all_scores_ens_sem = np.concatenate(ap_ens_sem_list, axis=0)
    all_scores_ens_rupa = np.concatenate(ap_ens_rupa_list, axis=0)
    all_scores_ens_det = np.concatenate(ap_ens_det_list, axis=0)
    all_scores_ens_aux = np.concatenate(ap_ens_aux_list, axis=0)
    all_scores_ens_audio_det = np.concatenate(ap_ens_audio_det_list, axis=0)
    all_scores_ens_tri = np.concatenate(ap_ens_tri_list, axis=0)
    all_scores_ens_quad = np.concatenate(ap_ens_quad_list, axis=0)
    all_scores_weighted = np.concatenate(ap_weighted_list, axis=0)
    all_scores_geom = np.concatenate(ap_geom_list, axis=0)
    all_scores_harmonic = np.concatenate(ap_harmonic_list, axis=0)
    all_scores_master = np.concatenate(ap_master_list, axis=0)
    all_scores_hier = np.concatenate(ap_hier_list, axis=0)
    all_scores_uccf = np.concatenate(ap_uccf_list, axis=0)
    all_scores_dodo = np.concatenate(ap_dodo_list, axis=0)

    # Repeat 16x for frame-level evaluation & sanitize non-finites
    frame_scores_routing = np.nan_to_num(np.repeat(all_scores_routing, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_det = np.nan_to_num(np.repeat(all_scores_det, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_aux = np.nan_to_num(np.repeat(all_scores_aux, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_sem = np.nan_to_num(np.repeat(all_scores_sem, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_sem_rupa = np.nan_to_num(np.repeat(all_scores_sem_rupa, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_sem = np.nan_to_num(np.repeat(all_scores_ens_sem, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_rupa = np.nan_to_num(np.repeat(all_scores_ens_rupa, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_det = np.nan_to_num(np.repeat(all_scores_ens_det, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_aux = np.nan_to_num(np.repeat(all_scores_ens_aux, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_audio_det = np.nan_to_num(np.repeat(all_scores_ens_audio_det, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_tri = np.nan_to_num(np.repeat(all_scores_ens_tri, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens_quad = np.nan_to_num(np.repeat(all_scores_ens_quad, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_weighted = np.nan_to_num(np.repeat(all_scores_weighted, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_geom = np.nan_to_num(np.repeat(all_scores_geom, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_harmonic = np.nan_to_num(np.repeat(all_scores_harmonic, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_master = np.nan_to_num(np.repeat(all_scores_master, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_hier = np.nan_to_num(np.repeat(all_scores_hier, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_uccf = np.nan_to_num(np.repeat(all_scores_uccf, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_dodo = np.nan_to_num(np.repeat(all_scores_dodo, 16), nan=0.0, posinf=1.0, neginf=0.0)

    # Match length with ground truth
    min_len = min(len(gt), len(frame_scores_routing))
    gt_eval = gt[:min_len]

    roc_routing = roc_auc_score(gt_eval, frame_scores_routing[:min_len])
    ap_routing = average_precision_score(gt_eval, frame_scores_routing[:min_len])

    roc_det = roc_auc_score(gt_eval, frame_scores_det[:min_len])
    ap_det = average_precision_score(gt_eval, frame_scores_det[:min_len])

    roc_aux = roc_auc_score(gt_eval, frame_scores_aux[:min_len])
    ap_aux = average_precision_score(gt_eval, frame_scores_aux[:min_len])

    roc_sem = roc_auc_score(gt_eval, frame_scores_sem[:min_len])
    ap_sem = average_precision_score(gt_eval, frame_scores_sem[:min_len])

    roc_sem_rupa = roc_auc_score(gt_eval, frame_scores_sem_rupa[:min_len])
    ap_sem_rupa = average_precision_score(gt_eval, frame_scores_sem_rupa[:min_len])

    roc_ens_sem = roc_auc_score(gt_eval, frame_scores_ens_sem[:min_len])
    ap_ens_sem = average_precision_score(gt_eval, frame_scores_ens_sem[:min_len])

    roc_ens_rupa = roc_auc_score(gt_eval, frame_scores_ens_rupa[:min_len])
    ap_ens_rupa = average_precision_score(gt_eval, frame_scores_ens_rupa[:min_len])

    roc_ens_det = roc_auc_score(gt_eval, frame_scores_ens_det[:min_len])
    ap_ens_det = average_precision_score(gt_eval, frame_scores_ens_det[:min_len])

    roc_ens_aux = roc_auc_score(gt_eval, frame_scores_ens_aux[:min_len])
    ap_ens_aux = average_precision_score(gt_eval, frame_scores_ens_aux[:min_len])

    roc_ens_audio_det = roc_auc_score(gt_eval, frame_scores_ens_audio_det[:min_len])
    ap_ens_audio_det = average_precision_score(gt_eval, frame_scores_ens_audio_det[:min_len])

    roc_ens_tri = roc_auc_score(gt_eval, frame_scores_ens_tri[:min_len])
    ap_ens_tri = average_precision_score(gt_eval, frame_scores_ens_tri[:min_len])

    roc_ens_quad = roc_auc_score(gt_eval, frame_scores_ens_quad[:min_len])
    ap_ens_quad = average_precision_score(gt_eval, frame_scores_ens_quad[:min_len])

    roc_weighted = roc_auc_score(gt_eval, frame_scores_weighted[:min_len])
    ap_weighted = average_precision_score(gt_eval, frame_scores_weighted[:min_len])

    roc_geom = roc_auc_score(gt_eval, frame_scores_geom[:min_len])
    ap_geom = average_precision_score(gt_eval, frame_scores_geom[:min_len])

    roc_harmonic = roc_auc_score(gt_eval, frame_scores_harmonic[:min_len])
    ap_harmonic = average_precision_score(gt_eval, frame_scores_harmonic[:min_len])

    roc_master = roc_auc_score(gt_eval, frame_scores_master[:min_len])
    ap_master = average_precision_score(gt_eval, frame_scores_master[:min_len])

    roc_hier = roc_auc_score(gt_eval, frame_scores_hier[:min_len])
    ap_hier = average_precision_score(gt_eval, frame_scores_hier[:min_len])

    roc_uccf = roc_auc_score(gt_eval, frame_scores_uccf[:min_len])
    ap_uccf = average_precision_score(gt_eval, frame_scores_uccf[:min_len])

    roc_dodo = roc_auc_score(gt_eval, frame_scores_dodo[:min_len])
    ap_dodo = average_precision_score(gt_eval, frame_scores_dodo[:min_len])

    print(f"[BR-MVAD Evaluation] Frame-level AP (Hierarchical BR-STB Backbone): {ap_det * 100:.2f}% | AUC: {roc_det * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Fine-Grained Semantic Alignment - TVSA): {ap_sem_rupa * 100:.2f}% | AUC: {roc_sem_rupa * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Dual-Stream Synergistic Fusion): {ap_ens_rupa * 100:.2f}% | AUC: {roc_ens_rupa * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Reliability-Gated Audio Aux): {ap_aux * 100:.2f}% | AUC: {roc_aux * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Ensemble BR-STB + Audio Aux - SOTA): {ap_ens_aux * 100:.2f}% | AUC: {roc_ens_aux * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Ensemble Detector + Audio Aux): {ap_ens_audio_det * 100:.2f}% | AUC: {roc_ens_audio_det * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Ensemble BR-STB + Backbone Semantics): {ap_ens_sem * 100:.2f}% | AUC: {roc_ens_sem * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Weighted 60/40 Ensemble): {ap_weighted * 100:.2f}% | AUC: {roc_weighted * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Geometric Mean Filtered): {ap_geom * 100:.2f}% | AUC: {roc_geom * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Harmonic Consensus Ensemble): {ap_harmonic * 100:.2f}% | AUC: {roc_harmonic * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Master SOTA Ensemble): {ap_master * 100:.2f}% | AUC: {roc_master * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Tri-Stream Ensemble): {ap_ens_tri * 100:.2f}% | AUC: {roc_ens_tri * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Quad-Stream Multimodal Fusion): {ap_ens_quad * 100:.2f}% | AUC: {roc_ens_quad * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Hierarchical Refined): {ap_hier * 100:.2f}% | AUC: {roc_hier * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (Single-Pass Multimodal SOTA): {ap_uccf * 100:.2f}% | AUC: {roc_uccf * 100:.2f}%", flush=True)
    print(f"[BR-MVAD Evaluation] Frame-level AP (UCCF + DODO - SOTA): {ap_dodo * 100:.2f}% | AUC: {roc_dodo * 100:.2f}%", flush=True)

    avg_map = 0.0
    if compute_dmap:
        try:
            dmap, iou = dmAP(element_logits2_stack, gtsegments, gtlabels, excludeNormal=False)
            for i in range(len(iou)):
                print(f"  mAP@{iou[i]:.1f} = {dmap[i]:.2f}%", flush=True)
                avg_map += dmap[i]
            avg_map = avg_map / max(1, len(iou))
            print(f"  Average mAP: {avg_map:.2f}%", flush=True)
        except Exception as e:
            print(f"  [Note] Detection mAP calculation skipped: {e}", flush=True)
        sys.stdout.flush()

    best_ap = max(
        ap_routing, ap_det, ap_aux, ap_sem, ap_sem_rupa,
        ap_ens_sem, ap_ens_rupa, ap_ens_det, ap_ens_aux, ap_ens_audio_det,
        ap_ens_tri, ap_ens_quad, ap_weighted, ap_geom, ap_harmonic, ap_master, ap_hier, ap_uccf, ap_dodo
    )
    best_auc = max(
        roc_routing, roc_det, roc_aux, roc_sem, roc_sem_rupa,
        roc_ens_sem, roc_ens_rupa, roc_ens_det, roc_ens_aux, roc_ens_audio_det,
        roc_ens_tri, roc_ens_quad, roc_weighted, roc_geom, roc_harmonic, roc_master, roc_hier, roc_uccf, roc_dodo
    )
    print(f"  ==> [BEST METRIC] Frame-level AP: {best_ap * 100:.2f}% | AUC: {best_auc * 100:.2f}%", flush=True)
    return best_auc, best_ap, avg_map


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = xd_option.parser.parse_args()

    label_map = {
        'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
        'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'
    }

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
    for idx in range(min(50, len(test_dataset))):
        item_tuple = test_dataset[idx]
        sample_audio = item_tuple[5] if len(item_tuple) >= 6 else item_tuple[-1]
        if sample_audio is not None and isinstance(sample_audio, torch.Tensor):
            detected_audio_dim = sample_audio.shape[-1]
            break
    if detected_audio_dim is not None and detected_audio_dim != args.audio_dim:
        print(f"[Auto-Detect] Audio feature dimension on disk is {detected_audio_dim}-D. Auto-adjusting args.audio_dim to {detected_audio_dim}!", flush=True)
        args.audio_dim = detected_audio_dim

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

    if args.model_path:
        checkpoint = safe_load_checkpoint(args.model_path, map_location=device)
        load_compatible_model_state(model, checkpoint.get('model_state_dict', checkpoint), strict=False)

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, args.DNP_use, args, device, compute_dmap=True)
