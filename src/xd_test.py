import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import MARUPA, load_compatible_model_state
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text, smooth_scores, safe_load_checkpoint
from utils.xd_detectionMAP import getDetectionMAP as dmAP
import xd_option


def refine_scores_hierarchical(logits_det: torch.Tensor,
                               logits_align: torch.Tensor,
                               temp: float = 1.0,
                               routing_prob: torch.Tensor = None) -> torch.Tensor:
    """Hierarchically combines anomaly probabilities with fine-grained semantic alignments."""
    epsilon = 1e-12
    if routing_prob is None:
        total_abnormal_prob = torch.sigmoid(logits_det / temp)
    else:
        total_abnormal_prob = routing_prob
    total_normal_prob = (1.0 - total_abnormal_prob).clamp(min=0.0, max=1.0)

    p_align = F.softmax(logits_align / temp, dim=1)
    p_align_abnormal_only = p_align[:, 1:]
    sum_p_align = p_align_abnormal_only.sum(dim=1, keepdim=True).clamp_min(epsilon)
    abnormal_dist = p_align_abnormal_only / sum_p_align

    final_abnormal_probs = total_abnormal_prob * abnormal_dist
    final_probs = torch.cat([total_normal_prob, final_abnormal_probs], dim=1)
    return final_probs


def test(model, testdataloader, maxlen, prompt_text, gt, gtsegments, gtlabels,
         DNP_use, args, device):
    """Evaluates MA-RUPA on XD-Violence test set.
    
    Computes Frame-level Average Precision (AP), Frame-level ROC-AUC, and mAP@IoU[0.1:0.5].
    Supports multimodal visual+audio inference and temporal score smoothing.
    """
    model.to(device)
    model.eval()

    element_logits2_stack = []
    ap_routing_list = []
    ap_det_list = []
    ap_sem_list = []
    ap_ens_list = []

    smooth_eval = getattr(args, 'smooth_eval', True)
    smooth_method = getattr(args, 'smooth_method', 'gaussian')
    smooth_sigma = getattr(args, 'smooth_sigma', 1.5)

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
                logits2 = dnp_dict.get('semantic_logits', logits2)
            else:
                _, logits1, logits2, _, _, dnp_dict = model(
                    visual, padding_mask, prompt_text, lengths, audio=audio, DNP_use=False
                )
                routing_scores = dnp_dict.get('routing_scores', torch.sigmoid(logits1))

            logits1 = logits1.reshape(-1, logits1.shape[-1])
            logits2 = logits2.reshape(-1, logits2.shape[-1])
            if routing_scores is not None:
                routing_scores = routing_scores.reshape(-1, routing_scores.shape[-1])

            optimized_probs = refine_scores_hierarchical(
                logits1[0:len_cur], logits2[0:len_cur], args.temp,
                None if routing_scores is None else routing_scores[0:len_cur],
            )
            
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

            # Stream 3: Zero-shot semantic alignment prediction
            p_align = F.softmax(logits2[0:len_cur] / args.temp, dim=-1)
            prob_sem = 1.0 - p_align[:, 0]

            # Stream 4: Dual ensemble
            prob_ens = 0.5 * prob_routing + 0.5 * prob_det

            # Detach to numpy
            p_routing_np = prob_routing.cpu().numpy()
            p_det_np = prob_det.cpu().numpy()
            p_sem_np = prob_sem.cpu().numpy()
            p_ens_np = prob_ens.cpu().numpy()

            # Optional temporal smoothing
            if smooth_eval:
                p_routing_np = smooth_scores(p_routing_np, method=smooth_method, sigma=smooth_sigma)
                p_det_np = smooth_scores(p_det_np, method=smooth_method, sigma=smooth_sigma)
                p_sem_np = smooth_scores(p_sem_np, method=smooth_method, sigma=smooth_sigma)
                p_ens_np = smooth_scores(p_ens_np, method=smooth_method, sigma=smooth_sigma)

            ap_routing_list.append(p_routing_np)
            ap_det_list.append(p_det_np)
            ap_sem_list.append(p_sem_np)
            ap_ens_list.append(p_ens_np)

            elem_l2 = optimized_probs.detach().cpu().numpy()
            elem_l2 = np.repeat(elem_l2, 16, axis=0)
            element_logits2_stack.append(elem_l2)

    all_scores_routing = np.concatenate(ap_routing_list, axis=0)
    all_scores_det = np.concatenate(ap_det_list, axis=0)
    all_scores_sem = np.concatenate(ap_sem_list, axis=0)
    all_scores_ens = np.concatenate(ap_ens_list, axis=0)

    # Repeat 16x for frame-level evaluation & sanitize non-finites
    frame_scores_routing = np.nan_to_num(np.repeat(all_scores_routing, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_det = np.nan_to_num(np.repeat(all_scores_det, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_sem = np.nan_to_num(np.repeat(all_scores_sem, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores_ens = np.nan_to_num(np.repeat(all_scores_ens, 16), nan=0.0, posinf=1.0, neginf=0.0)

    # Match length with ground truth
    min_len = min(len(gt), len(frame_scores_routing))
    gt_eval = gt[:min_len]

    roc_routing = roc_auc_score(gt_eval, frame_scores_routing[:min_len])
    ap_routing = average_precision_score(gt_eval, frame_scores_routing[:min_len])

    roc_det = roc_auc_score(gt_eval, frame_scores_det[:min_len])
    ap_det = average_precision_score(gt_eval, frame_scores_det[:min_len])

    roc_sem = roc_auc_score(gt_eval, frame_scores_sem[:min_len])
    ap_sem = average_precision_score(gt_eval, frame_scores_sem[:min_len])

    roc_ens = roc_auc_score(gt_eval, frame_scores_ens[:min_len])
    ap_ens = average_precision_score(gt_eval, frame_scores_ens[:min_len])

    print(f"[Evaluation] Frame-level AP (Routing): {ap_routing * 100:.2f}% | AUC: {roc_routing * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AP (Detector): {ap_det * 100:.2f}% | AUC: {roc_det * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AP (Semantics): {ap_sem * 100:.2f}% | AUC: {roc_sem * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AP (Ensemble): {ap_ens * 100:.2f}% | AUC: {roc_ens * 100:.2f}%", flush=True)

    avg_map = 0.0
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

    best_ap = max(ap_routing, ap_det, ap_sem, ap_ens)
    best_auc = max(roc_routing, roc_det, roc_sem, roc_ens)
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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    # Auto-detect audio feature dimension from dataset if available
    detected_audio_dim = None
    for idx in range(min(50, len(test_dataset))):
        sample_audio = test_dataset[idx][3]
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

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, args.DNP_use, args, device)
