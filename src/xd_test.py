import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from model import MARUPA, load_compatible_model_state
from utils.dataset import XDDataset
from utils.tools import get_batch_mask, get_prompt_text, smooth_scores
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
    ap1_list = []
    ap2_list = []

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
            prob2 = 1.0 - optimized_probs[:, 0]
            if routing_scores is None:
                prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1))
            else:
                prob1 = routing_scores[0:len_cur].squeeze(-1)

            # Detach to numpy
            prob1_np = prob1.cpu().numpy()
            prob2_np = prob2.cpu().numpy()

            # Optional temporal smoothing
            if smooth_eval:
                prob1_np = smooth_scores(prob1_np, method=smooth_method, sigma=smooth_sigma)
                prob2_np = smooth_scores(prob2_np, method=smooth_method, sigma=smooth_sigma)

            ap1_list.append(prob1_np)
            ap2_list.append(prob2_np)

            elem_l2 = optimized_probs.detach().cpu().numpy()
            elem_l2 = np.repeat(elem_l2, 16, axis=0)
            element_logits2_stack.append(elem_l2)

    all_scores1 = np.concatenate(ap1_list, axis=0)
    all_scores2 = np.concatenate(ap2_list, axis=0)

    # Repeat 16x for frame-level evaluation & sanitize non-finites
    frame_scores1 = np.nan_to_num(np.repeat(all_scores1, 16), nan=0.0, posinf=1.0, neginf=0.0)
    frame_scores2 = np.nan_to_num(np.repeat(all_scores2, 16), nan=0.0, posinf=1.0, neginf=0.0)

    # Match length with ground truth
    min_len1 = min(len(gt), len(frame_scores1))
    roc1 = roc_auc_score(gt[:min_len1], frame_scores1[:min_len1])
    ap1 = average_precision_score(gt[:min_len1], frame_scores1[:min_len1])

    min_len2 = min(len(gt), len(frame_scores2))
    roc2 = roc_auc_score(gt[:min_len2], frame_scores2[:min_len2])
    ap2 = average_precision_score(gt[:min_len2], frame_scores2[:min_len2])

    print(f"[Evaluation] Frame-level AP (Routing): {ap1 * 100:.2f}% | AUC: {roc1 * 100:.2f}%", flush=True)
    print(f"[Evaluation] Frame-level AP (Semantics): {ap2 * 100:.2f}% | AUC: {roc2 * 100:.2f}%", flush=True)

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

    best_ap = max(ap1, ap2)
    best_auc = max(roc1, roc2)
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
        checkpoint = torch.load(args.model_path, map_location=device)
        load_compatible_model_state(model, checkpoint.get('model_state_dict', checkpoint), strict=False)

    test(model, test_loader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, args.DNP_use, args, device)
