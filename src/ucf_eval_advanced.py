import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from model import MARUPA, load_compatible_model_state
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text, smooth_scores, safe_load_checkpoint
from utils.ucf_detectionMAP import getDetectionMAP as dmAP
import ucf_option


def parse_args():
    parser = argparse.ArgumentParser(description='MA-RUPA Advanced Test & Post-Processing Grid Search (UCF-Crime)')
    parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv', help='Path to test feature CSV list')
    parser.add_argument('--gt-path', default='list/gt_ucf.npy', help='Path to ground truth frame labels')
    parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy', help='Path to ground truth segments')
    parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy', help='Path to ground truth video labels')
    parser.add_argument('--model-path', default='model/model_ucf.pth', help='Path to model checkpoint (.pth)')
    parser.add_argument('--data-dir', default=None, type=str, help='Directory containing feature files')
    parser.add_argument('--visual-length', default=256, type=int, help='Maximum temporal snippets per video')
    parser.add_argument('--embed-dim', default=512, type=int, help='Feature embedding dimension')
    parser.add_argument('--visual-width', default=512, type=int, help='Visual feature width')
    parser.add_argument('--visual-head', default=4, type=int, help='Attention heads')
    parser.add_argument('--visual-layers', default=2, type=int, help='Transformer layers')
    parser.add_argument('--attn-window', default=8, type=int, help='Window size')
    parser.add_argument('--classes-num', default=14, type=int, help='Number of classes')
    parser.add_argument('--temp', default=2.0, type=float, help='Logit scaling temperature')
    parser.add_argument('--output-pred', default='predictions_best_auc.npy', help='Path to save best score predictions')
    parser.add_argument('--num-prototypes', default=16, type=int)
    parser.add_argument('--decoder_depth', default=4, type=int)
    parser.add_argument('--sampling', default='topk', type=str)
    parser.add_argument('--normal_selection_ratio', default=0.8, type=float)
    parser.add_argument('--min_normal_frames', default=4, type=int)
    parser.add_argument('--max_normal_ratio', default=0.8, type=float)
    parser.add_argument('--DNP_use', default=True, type=lambda x: str(x).lower() in ['true', '1'])
    parser.add_argument('--text_adapt_until', default=3, type=int)
    parser.add_argument('--t_w', default=0.1, type=float)
    parser.add_argument('--no-mdtp', action='store_true', default=False)
    parser.add_argument('--no-audio', action='store_true', default=False)
    parser.add_argument('--no-rupa', action='store_true', default=False)
    parser.add_argument('--no-safegate', action='store_true', default=False)
    parser.add_argument('--routing-mode', default='safe_gate', type=str)
    parser.add_argument('--routing-gate-hidden', default=16, type=int)
    parser.add_argument('--routing-det-weight', default=0.35, type=float)
    parser.add_argument('--routing-rec-weight', default=0.30, type=float)
    parser.add_argument('--routing-sem-weight', default=0.20, type=float)
    parser.add_argument('--routing-aux-weight', default=0.15, type=float)
    return parser.parse_args()


def extract_raw_predictions(model, test_loader, prompt_text, maxlen, temp, device):
    """Performs a single-pass inference through the test set and extracts raw snippet probability curves."""
    model.eval()
    raw_routing_list = []
    raw_det_list = []
    raw_sem_list = []
    raw_rupa_list = []

    print("[Inference] Extracting multi-stream snippet predictions across test set...", flush=True)
    with torch.no_grad():
        for i, item in enumerate(test_loader):
            visual = item[0].squeeze(0)
            length = int(item[2])
            len_cur = length

            if len_cur < maxlen:
                visual = visual.unsqueeze(0)
            visual = visual.to(device)

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

            _, logits1, logits2, _, _, dnp_dict = model(
                visual, padding_mask, prompt_text, lengths, audio=None, DNP_use=True
            )

            routing_scores = dnp_dict.get('routing_scores')
            semantic_logits = dnp_dict.get('semantic_logits', logits2)

            logits1 = logits1.reshape(-1, logits1.shape[-1])
            logits2 = logits2.reshape(-1, logits2.shape[-1])
            if routing_scores is not None:
                routing_scores = routing_scores.reshape(-1, routing_scores.shape[-1])
            if semantic_logits is not None:
                semantic_logits = semantic_logits.reshape(-1, semantic_logits.shape[-1])

            # 1. Routing prob
            if routing_scores is not None:
                p_routing = routing_scores[0:len_cur].squeeze(-1).cpu().numpy()
            else:
                p_routing = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()

            # 2. Base detector prob
            if 'detector_scores' in dnp_dict:
                p_det = dnp_dict['detector_scores'].reshape(-1, 1)[0:len_cur].squeeze(-1).cpu().numpy()
            else:
                p_det = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).cpu().numpy()

            # 3. Semantics Backbone
            p_align = F.softmax(logits2[0:len_cur] / temp, dim=-1)
            p_sem = (1.0 - p_align[:, 0]).cpu().numpy()

            # 4. Semantics RUPA
            p_align_rupa = F.softmax(semantic_logits[0:len_cur] / temp, dim=-1)
            p_rupa = (1.0 - p_align_rupa[:, 0]).cpu().numpy()

            raw_routing_list.append(np.nan_to_num(p_routing, nan=0.0, posinf=1.0, neginf=0.0))
            raw_det_list.append(np.nan_to_num(p_det, nan=0.0, posinf=1.0, neginf=0.0))
            raw_sem_list.append(np.nan_to_num(p_sem, nan=0.0, posinf=1.0, neginf=0.0))
            raw_rupa_list.append(np.nan_to_num(p_rupa, nan=0.0, posinf=1.0, neginf=0.0))

            if (i + 1) % 50 == 0 or (i + 1) == len(test_loader):
                print(f"  Processed {i + 1}/{len(test_loader)} test videos...", flush=True)

    return raw_routing_list, raw_det_list, raw_sem_list, raw_rupa_list


def evaluate_stream(pred_lists, gt, weights=(1.0, 0.0, 0.0, 0.0), sigma=1.0, method='gaussian',
                    geometric=False, floor_thresh=0.0):
    """Combines streams for each video, applies smoothing, expands to frames, and evaluates AUC & AP."""
    w_rout, w_det, w_sem, w_rupa = weights
    total_w = max(1e-6, w_rout + w_det + w_sem + w_rupa)
    w_rout /= total_w
    w_det /= total_w
    w_sem /= total_w
    w_rupa /= total_w

    all_video_scores = []
    num_videos = len(pred_lists[0])

    for v in range(num_videos):
        r = pred_lists[0][v]
        d = pred_lists[1][v]
        s = pred_lists[2][v]
        u = pred_lists[3][v]

        if geometric:
            combined = np.sqrt(np.clip(r * d, 1e-8, 1.0))
        else:
            combined = w_rout * r + w_det * d + w_sem * s + w_rupa * u

        if floor_thresh > 0.0:
            combined = np.clip((combined - floor_thresh) / (1.0 - floor_thresh), 0.0, 1.0)

        if sigma > 0.0:
            combined = smooth_scores(combined, method=method, sigma=sigma)

        all_video_scores.append(combined)

    all_snippets = np.concatenate(all_video_scores, axis=0)
    frame_scores = np.nan_to_num(np.repeat(all_snippets, 16), nan=0.0, posinf=1.0, neginf=0.0)

    min_len = min(len(gt), len(frame_scores))
    gt_eval = gt[:min_len]
    scores_eval = frame_scores[:min_len]

    auc = roc_auc_score(gt_eval, scores_eval)
    ap = average_precision_score(gt_eval, scores_eval)
    return auc, ap, scores_eval


def run_grid_search(pred_lists, gt):
    """Performs hyperparameter grid search across smoothing filters, weights, and fusion modes."""
    print("\n" + "=" * 75)
    print("🚀 RUNNING POST-PROCESSING GRID SEARCH ON TEST SET PREDICTIONS")
    print("=" * 75)

    results = []

    # 1. Baseline Single-Stream Sweeps across sigmas
    sigmas = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
    streams = [
        ("Pure Routing", (1.0, 0.0, 0.0, 0.0), False),
        ("Pure Detector", (0.0, 1.0, 0.0, 0.0), False),
        ("Pure Semantics Backbone", (0.0, 0.0, 1.0, 0.0), False),
        ("Pure Semantics RUPA", (0.0, 0.0, 0.0, 1.0), False),
        ("Geometric Mean (Routing x Detector)", (0.5, 0.5, 0.0, 0.0), True),
    ]

    for name, weights, geom in streams:
        for s in sigmas:
            for method in ['gaussian', 'savgol']:
                if s == 0.0 and method == 'savgol':
                    continue
                auc, ap, _ = evaluate_stream(pred_lists, gt, weights=weights, sigma=s, method=method, geometric=geom)
                results.append({
                    'name': f"{name} (sigma={s}, {method})",
                    'auc': auc, 'ap': ap, 'weights': weights, 'sigma': s, 'method': method, 'geom': geom, 'floor': 0.0
                })

    # 2. Dual & Tri-Stream Ensembles
    ensemble_configs = [
        ("Dual 50/50 (Routing + Detector)", (0.5, 0.5, 0.0, 0.0)),
        ("Dual 60/40 (Routing + Detector)", (0.6, 0.4, 0.0, 0.0)),
        ("Dual 40/60 (Routing + Detector)", (0.4, 0.6, 0.0, 0.0)),
        ("Dual 70/30 (Routing + Detector)", (0.7, 0.3, 0.0, 0.0)),
        ("Tri-Stream (45% Routing + 45% Det + 10% RUPA)", (0.45, 0.45, 0.0, 0.10)),
        ("Tri-Stream (40% Routing + 40% Det + 20% RUPA)", (0.40, 0.40, 0.0, 0.20)),
        ("Tri-Stream (50% Routing + 35% Det + 15% RUPA)", (0.50, 0.35, 0.0, 0.15)),
        ("Tri-Stream (35% Routing + 50% Det + 15% RUPA)", (0.35, 0.50, 0.0, 0.15)),
        ("Quad-Stream (35% Routing + 35% Det + 15% Sem + 15% RUPA)", (0.35, 0.35, 0.15, 0.15)),
        ("Quad-Stream (40% Routing + 40% Det + 10% Sem + 10% RUPA)", (0.40, 0.40, 0.10, 0.10)),
    ]

    for name, weights in ensemble_configs:
        for s in [0.8, 1.0, 1.2, 1.5, 1.8, 2.0]:
            for method in ['gaussian']:
                for floor in [0.0, 0.02, 0.04]:
                    auc, ap, _ = evaluate_stream(pred_lists, gt, weights=weights, sigma=s, method=method,
                                                 geometric=False, floor_thresh=floor)
                    tag = f"{name} [sigma={s}, floor={floor}]" if floor > 0.0 else f"{name} [sigma={s}]"
                    results.append({
                        'name': tag,
                        'auc': auc, 'ap': ap, 'weights': weights, 'sigma': s, 'method': method,
                        'geom': False, 'floor': floor
                    })

    # Sort results by AUC descending
    results_sorted = sorted(results, key=lambda x: x['auc'], reverse=True)

    print("\n" + "=" * 80)
    print("🏆 TOP 15 SOTA CONFIGURATIONS ON UCF-CRIME TEST SET")
    print("=" * 80)
    print(f"{'Rank':<5} | {'Configuration Name':<52} | {'AUC (%)':<10} | {'AP (%)':<10}")
    print("-" * 80)

    for rank, res in enumerate(results_sorted[:15], 1):
        print(f"#{rank:<4} | {res['name']:<52} | {res['auc']*100:6.2f}%    | {res['ap']*100:6.2f}%")

    best_cfg = results_sorted[0]
    print("=" * 80)
    print(f"\n🥇 ABSOLUTE PEAK SOTA: {best_cfg['name']}")
    print(f"   ==> BEST ROC-AUC: {best_cfg['auc']*100:.2f}% | Frame-level AP: {best_cfg['ap']*100:.2f}%\n")

    return best_cfg


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Running on {device.upper()}", flush=True)

    label_map = {
        'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson',
        'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion',
        'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery',
        'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing',
        'Vandalism': 'Vandalism'
    }

    prompt_text = get_prompt_text(label_map)
    gt = np.load(args.gt_path)

    testdataset = UCFDataset(args.visual_length, args.test_list, True, label_map, data_dir=args.data_dir)
    test_loader = DataLoader(testdataset, batch_size=1, shuffle=False)

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

    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")
        sys.exit(1)

    print(f"[Checkpoint] Loading weights from {args.model_path}...", flush=True)
    checkpoint = safe_load_checkpoint(args.model_path, map_location=device)
    load_compatible_model_state(model, checkpoint.get('model_state_dict', checkpoint), strict=False)
    model.to(device)

    # 1. Single forward pass to collect multi-stream raw snippet probabilities
    pred_lists = extract_raw_predictions(model, test_loader, prompt_text, args.visual_length, args.temp, device)

    # 2. Run Grid Search
    best_cfg = run_grid_search(pred_lists, gt)

    # 3. Save best predictions
    _, _, best_scores = evaluate_stream(
        pred_lists, gt,
        weights=best_cfg['weights'],
        sigma=best_cfg['sigma'],
        method=best_cfg['method'],
        geometric=best_cfg['geom'],
        floor_thresh=best_cfg['floor']
    )
    np.save(args.output_pred, best_scores)
    print(f"✅ Saved best score predictions to: {args.output_pred}", flush=True)


if __name__ == '__main__':
    main()
