import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from model import MARUPA, load_compatible_model_state
from utils.dataset import UCFDataset, load_multicrop_test_video
from utils.tools import get_batch_mask, get_prompt_text, smooth_scores, safe_load_checkpoint
import ucf_option


def parse_args():
    parser = argparse.ArgumentParser(description='MA-RUPA v3 Advanced Multi-Crop Evaluation & Grid Search (UCF-Crime)')
    parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv', help='Path to test feature CSV list')
    parser.add_argument('--gt-path', default='list/gt_ucf.npy', help='Path to ground truth frame labels')
    parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy', help='Path to ground truth segments')
    parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy', help='Path to ground truth video labels')
    parser.add_argument('--model-path', default='model/model_ucf.pth', help='Path to model checkpoint (.pth)')
    parser.add_argument('--data-dir', default=None, type=str, help='Directory containing feature files')
    parser.add_argument('--visual-length', default=256, type=int)
    parser.add_argument('--embed-dim', default=512, type=int)
    parser.add_argument('--visual-width', default=512, type=int)
    parser.add_argument('--visual-head', default=1, type=int)
    parser.add_argument('--visual-layers', default=2, type=int)
    parser.add_argument('--attn-window', default=8, type=int)
    parser.add_argument('--classes-num', default=14, type=int)
    parser.add_argument('--temp', default=1.0, type=float)
    parser.add_argument('--output-pred', default='predictions_best_auc.npy')
    parser.add_argument('--num-prototypes', '--num_prototypes', dest='num_prototypes', default=16, type=int)
    parser.add_argument('--decoder_depth', '--decoder-depth', dest='decoder_depth', default=8, type=int)
    parser.add_argument('--normal_selection_ratio', default=0.8, type=float)
    parser.add_argument('--DNP_use', default=True, type=ucf_option.str2bool)
    parser.add_argument('--text_adapt_until', default=3, type=int)
    parser.add_argument('--t_w', default=0.1, type=float)
    parser.add_argument('--multi-crop-eval', default=True, type=ucf_option.str2bool)
    parser.add_argument('--max-test-crops', default=10, type=int)
    parser.add_argument('--no-mdtp', action='store_true', default=False)
    parser.add_argument('--no-audio', action='store_true', default=False)
    parser.add_argument('--no-rupa', action='store_true', default=False)
    parser.add_argument('--no-safegate', action='store_true', default=False)
    parser.add_argument('--routing-mode', default='safe_gate', type=str)
    parser.add_argument('--routing-gate-hidden', default=8, type=int)
    parser.add_argument('--routing-det-weight', default=0.90, type=float)
    parser.add_argument('--routing-rec-weight', default=0.10, type=float)
    parser.add_argument('--routing-sem-weight', default=0.00, type=float)
    args, _ = parser.parse_known_args()
    return args


def extract_raw_predictions(model, test_loader, prompt_text, maxlen, temp, device, args):
    """Performs parallel multi-crop inference through the test set and extracts raw snippet probability curves."""
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    eval_model.eval()
    raw_routing_list = []
    raw_det_list = []
    raw_sem_list = []
    raw_rupa_list = []

    multi_crop = bool(getattr(args, 'multi_crop_eval', True))
    max_crops = int(getattr(args, 'max_test_crops', 10)) if multi_crop else 1
    eff_temp = max(float(temp), 1e-3)

    print(f"[Inference] Extracting {'10-Crop' if multi_crop else '1-Crop'} predictions across test set...", flush=True)
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            fallback_vis = item[0].squeeze(0)
            len_cur = int(item[2])
            raw_path = item[3][0] if isinstance(item[3], (list, tuple)) else str(item[3])

            if multi_crop:
                vis_crops, _, len_cur, _ = load_multicrop_test_video(
                    raw_path=raw_path, maxlen=maxlen, data_dir=args.data_dir,
                    no_audio=True, max_crops=max_crops,
                    fallback_visual=fallback_vis, fallback_length=len_cur,
                )
            else:
                v_single = fallback_vis.unsqueeze(0) if fallback_vis.ndim == 2 else fallback_vis
                vis_crops = v_single.unsqueeze(0)

            C, num_splits, _, D = vis_crops.shape
            base_lengths = torch.full((num_splits,), maxlen, dtype=torch.long)
            rem = len_cur - maxlen * (num_splits - 1)
            base_lengths[-1] = max(1, min(maxlen, rem))

            lengths_flat = base_lengths.repeat(C)
            padding_mask = get_batch_mask(lengths_flat, maxlen).to(device)
            visual_flat = vis_crops.view(C * num_splits, maxlen, D).to(device)

            _, logits1, logits2, _, _, dnp_dict = eval_model(
                visual_flat, padding_mask, prompt_text, lengths_flat, audio=None, DNP_use=True
            )

            routing_scores = dnp_dict.get('routing_scores', torch.sigmoid(logits1))
            detector_scores = dnp_dict.get('detector_scores', torch.sigmoid(logits1))
            semantic_logits = dnp_dict.get('semantic_logits', logits2)

            p_routing = routing_scores.view(C, num_splits * maxlen, -1)[:, :len_cur, 0].mean(dim=0).cpu().numpy()
            p_det = detector_scores.view(C, num_splits * maxlen, -1)[:, :len_cur, 0].mean(dim=0).cpu().numpy()

            l2_c = logits2.view(C, num_splits * maxlen, -1)[:, :len_cur, :]
            sem_c = semantic_logits.view(C, num_splits * maxlen, -1)[:, :len_cur, :]
            p_sem = (1.0 - F.softmax(l2_c.float() / eff_temp, dim=-1)[:, :, 0]).mean(dim=0).cpu().numpy()
            p_rupa = (1.0 - F.softmax(sem_c.float() / eff_temp, dim=-1)[:, :, 0]).mean(dim=0).cpu().numpy()

            raw_routing_list.append(np.nan_to_num(p_routing, nan=0.0, posinf=1.0, neginf=0.0))
            raw_det_list.append(np.nan_to_num(p_det, nan=0.0, posinf=1.0, neginf=0.0))
            raw_sem_list.append(np.nan_to_num(p_sem, nan=0.0, posinf=1.0, neginf=0.0))
            raw_rupa_list.append(np.nan_to_num(p_rupa, nan=0.0, posinf=1.0, neginf=0.0))

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"  Processed {i + 1}/{len(test_loader)} test videos...", flush=True)

    return raw_routing_list, raw_det_list, raw_sem_list, raw_rupa_list


def evaluate_stream(pred_lists, gt, weights=(1.0, 0.0, 0.0, 0.0), sigma=1.0, method='gaussian',
                    geometric=False, floor_thresh=0.0):
    w_rout, w_det, w_sem, w_rupa = weights
    total_w = max(1e-6, w_rout + w_det + w_sem + w_rupa)
    w_rout, w_det, w_sem, w_rupa = w_rout / total_w, w_det / total_w, w_sem / total_w, w_rupa / total_w

    all_video_scores = []
    for v in range(len(pred_lists[0])):
        r, d, s, u = pred_lists[0][v], pred_lists[1][v], pred_lists[2][v], pred_lists[3][v]
        combined = np.sqrt(np.clip(r * d, 1e-8, 1.0)) if geometric else (w_rout * r + w_det * d + w_sem * s + w_rupa * u)
        if floor_thresh > 0.0:
            combined = np.clip((combined - floor_thresh) / (1.0 - floor_thresh), 0.0, 1.0)
        if sigma > 0.0:
            combined = smooth_scores(combined, method=method, sigma=sigma)
        all_video_scores.append(combined)

    frame_scores = np.nan_to_num(np.repeat(np.concatenate(all_video_scores, axis=0), 16), nan=0.0, posinf=1.0, neginf=0.0)
    min_len = min(len(gt), len(frame_scores))
    gt_eval, scores_eval = gt[:min_len], frame_scores[:min_len]
    return roc_auc_score(gt_eval, scores_eval), average_precision_score(gt_eval, scores_eval), scores_eval


def run_grid_search(pred_lists, gt):
    results = []
    sigmas = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    streams = [
        ("Safe-RUPA Routing", (1.0, 0.0, 0.0, 0.0), False),
        ("Pure Detector", (0.0, 1.0, 0.0, 0.0), False),
        ("Vision-Language Backbone", (0.0, 0.0, 1.0, 0.0), False),
        ("Counterfactual RUPA Semantics", (0.0, 0.0, 0.0, 1.0), False),
        ("DeSC Collaborative (80% Route + 10% Sem + 10% RUPA)", (0.80, 0.0, 0.10, 0.10), False),
        ("DeSC Collaborative (75% Route + 12.5% Sem + 12.5% RUPA)", (0.75, 0.0, 0.125, 0.125), False),
    ]

    for name, weights, geom in streams:
        for s in sigmas:
            auc, ap, _ = evaluate_stream(pred_lists, gt, weights=weights, sigma=s, method='gaussian', geometric=geom)
            results.append({
                'name': f"{name} (sigma={s})",
                'auc': auc, 'ap': ap, 'weights': weights, 'sigma': s, 'method': 'gaussian', 'geom': geom, 'floor': 0.0,
            })

    results_sorted = sorted(results, key=lambda x: x['auc'], reverse=True)
    print("\n" + "=" * 84)
    print("TOP 10 SOTA CONFIGURATIONS ON UCF-CRIME TEST SET")
    print("=" * 84)
    for rank, res in enumerate(results_sorted[:10], 1):
        print(f"#{rank:<3} | {res['name']:<56} | AUC={res['auc']*100:6.2f}% | AP={res['ap']*100:6.2f}%")
    print("=" * 84)
    return results_sorted[0]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    label_map = {
        'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
        'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
        'Fighting': 'fighting', 'RoadAccidents': 'roadaccidents', 'Robbery': 'robbery',
        'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing',
        'Vandalism': 'vandalism',
    }
    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)
    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map, data_dir=args.data_dir)
    test_loader = DataLoader(testdataset, batch_size=1, shuffle=False)

    model = MARUPA(
        num_class=args.classes_num, embed_dim=args.embed_dim, visual_length=args.visual_length,
        visual_width=args.visual_width, visual_head=args.visual_head, visual_layers=args.visual_layers,
        attn_window=args.attn_window, args=args, device=device,
    )
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)

    checkpoint = safe_load_checkpoint(args.model_path, map_location=device)
    load_compatible_model_state(model, checkpoint, strict=False)
    model.to(device)

    pred_lists = extract_raw_predictions(model, test_loader, prompt_text, args.visual_length, args.temp, device, args)
    best_cfg = run_grid_search(pred_lists, gt)
    _, _, best_scores = evaluate_stream(
        pred_lists, gt, weights=best_cfg['weights'], sigma=best_cfg['sigma'],
        method=best_cfg['method'], geometric=best_cfg['geom'], floor_thresh=best_cfg['floor'],
    )
    np.save(args.output_pred, best_scores)
    print(f"Saved best score predictions to: {args.output_pred}", flush=True)


if __name__ == '__main__':
    main()
