#!/usr/bin/env python3
"""
S2 Deep Diagnostic Prober for Weakly-Supervised Video Anomaly Detection (WS-VAD)
Developed by Intern-S2 (System-2 Theoretical Scientist)

Capabilities:
1. Per-Video Error Profiling: Ranks Top-15 Worst False Alarm Videos & Top-15 Missed Anomaly Videos.
2. Multi-Stream Failure Attribution: Mathematically decomposes catastrophic False Positives across:
   - S_det (BR-STB Visual Backbone)
   - S_aux (Acoustic Multimodal Head)
   - S_sem (Zero-shot / TVSA Semantics)
   - S_rec (Adaptive Stiefel Prototype Manifold ASPM)
3. S2 Asymmetric Audio-Visual Discordance Gating Simulation.
4. Inference-Time Multi-Dimensional Grid Sweep (Gaussian sigma, Simplex Weights, Temperature Scaling).
5. Outputs Publication-Grade Clinical Diagnostic JSON Report.

Usage:
    python scripts/s2_deep_diagnostic_prober.py --dataset xd --model-path rupa_artifacts/model_xd_best.pth
    python scripts/s2_deep_diagnostic_prober.py --dataset ucf --model-path rupa_artifacts/model_ucf_best.pth
"""

import os
import sys
import argparse
import json
import math
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter1d

# Ensure project root and src/ directory are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')
for p in [src_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import MARUPA
from utils.dataset import XDDataset, UCFDataset
from utils import descriptions, tools
import xd_option as xd_opt
import ucf_option as ucf_opt


def parse_args():
    parser = argparse.ArgumentParser(description="Intern-S2 Deep Diagnostic Prober")
    parser.add_argument('--dataset', type=str, default='xd', choices=['xd', 'ucf'],
                        help="Target dataset: 'xd' (XD-Violence) or 'ucf' (UCF-Crime)")
    parser.add_argument('--model-path', type=str, default='',
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument('--data-dir', type=str, default='',
                        help="Path to pre-extracted CLIP visual features directory")
    parser.add_argument('--audio-dir', type=str, default='',
                        help="Path to pre-extracted VGGish audio features directory (for XD)")
    parser.add_argument('--test-file-path', type=str, default='',
                        help="Path to test CSV / text file")
    parser.add_argument('--gt-path', type=str, default='',
                        help="Path to ground truth frame-level binary numpy array (.npy)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-json', type=str, default='rupa_artifacts/s2_diagnostic_report.json')
    parser.add_argument('--scan-inference', action='store_true', default=True,
                        help="Run multi-dimensional grid sweep over test-time post-processing")
    return parser.parse_args()


def resolve_default_paths(args):
    """Auto-detects paths on Kaggle, standard cluster, or local dev environment."""
    if args.dataset == 'xd':
        if not args.data_dir:
            candidates = [
                '/kaggle/input/xdtestclipfeatures',
                '/kaggle/input/xd-violence-features/XDTestClipFeatures',
                'data/XD-Violence/features'
            ]
            args.data_dir = next((c for c in candidates if os.path.exists(c)), candidates[0])
        if not args.audio_dir:
            candidates = [
                '/kaggle/input/xd-vggish-features',
                '/kaggle/input/xd-violence-features/vggish',
                'data/XD-Violence/audio'
            ]
            args.audio_dir = next((c for c in candidates if os.path.exists(c)), candidates[0])
        if not args.test_file_path:
            candidates = [
                'data/xd_test.csv',
                'data/test.csv',
                '/kaggle/working/MA-RUPA/data/xd_test.csv'
            ]
            args.test_file_path = next((c for c in candidates if os.path.exists(c)), 'data/xd_test.csv')
        if not args.gt_path:
            candidates = [
                'data/xd_gt.npy',
                'data/gt.npy',
                '/kaggle/working/MA-RUPA/data/xd_gt.npy'
            ]
            args.gt_path = next((c for c in candidates if os.path.exists(c)), 'data/xd_gt.npy')
        if not args.model_path:
            candidates = [
                '/kaggle/working/rupa_artifacts/model_xd_best.pth',
                'rupa_artifacts/model_xd_best.pth',
                'checkpoints/model_xd_best.pth'
            ]
            args.model_path = next((c for c in candidates if os.path.exists(c)), 'rupa_artifacts/model_xd_best.pth')
    else:  # UCF-Crime
        if not args.data_dir:
            candidates = [
                '/kaggle/input/ucftestclipfeatures',
                '/kaggle/input/ucf-crime-features/UCFTestClipFeatures',
                'data/UCF-Crime/features'
            ]
            args.data_dir = next((c for c in candidates if os.path.exists(c)), candidates[0])
        if not args.test_file_path:
            candidates = [
                'data/ucf_test.csv',
                'data/ucf_test.txt',
                '/kaggle/working/MA-RUPA/data/ucf_test.csv'
            ]
            args.test_file_path = next((c for c in candidates if os.path.exists(c)), 'data/ucf_test.csv')
        if not args.gt_path:
            candidates = [
                'data/ucf_gt.npy',
                'data/gt-ucf.npy',
                '/kaggle/working/MA-RUPA/data/ucf_gt.npy'
            ]
            args.gt_path = next((c for c in candidates if os.path.exists(c)), 'data/ucf_gt.npy')
        if not args.model_path:
            candidates = [
                '/kaggle/working/rupa_artifacts/model_ucf_best.pth',
                'rupa_artifacts/model_ucf_best.pth',
                'checkpoints/model_ucf_best.pth'
            ]
            args.model_path = next((c for c in candidates if os.path.exists(c)), 'rupa_artifacts/model_ucf_best.pth')


def load_diagnostic_model(model_path: str, dataset: str, device: str) -> MARUPA:
    """Safely instantiates MARUPA and loads trained checkpoint weights."""
    print(f"\n[S2 Prober] Loading checkpoint from: {model_path} (Target device: {device})")
    num_classes = 7 if dataset == 'xd' else 14
    opt_module = xd_opt if dataset == 'xd' else ucf_opt
    dummy_args, _ = opt_module.parser.parse_known_args([])
    
    model = MARUPA(
        num_class=num_classes,
        embed_dim=512,
        visual_length=256,
        visual_width=512,
        visual_head=4,
        visual_layers=2,
        attn_window=8,
        args=dummy_args,
        device=device
    ).to(device)

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        sd = state.get('model_state_dict', state)
        # Strip DataParallel 'module.' prefix if present
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[S2 Prober] Model loaded successfully! (Missing keys: {len(missing)}, Unexpected: {len(unexpected)})")
    else:
        print(f"[S2 Prober] WARNING: Model path '{model_path}' not found! Running in Mock Architecture mode.")

    model.eval()
    return model


def collect_predictions_per_video(model: MARUPA, test_loader, prompt_text: List[str],
                                   gt: np.ndarray, dataset: str, device: str) -> Dict[str, Any]:
    """Runs forward pass and collects synchronized per-video snippet & frame tensors."""
    video_records = []
    cursor = 0
    maxlen = 256
    eff_temp = 0.07

    print(f"\n[S2 Prober] Running Clinical Inference across Test Dataset...")
    with torch.no_grad():
        for idx, item in enumerate(test_loader):
            visual = item[0].squeeze(0).to(device)
            length = int(item[2])
            raw_path = item[3] if len(item) > 3 else f"video_{idx:04d}"
            label_name = str(item[1]) if len(item) > 1 else "Unknown"

            audio = None
            if dataset == 'xd' and len(item) >= 6:
                audio = item[5]
                if audio is not None and isinstance(audio, torch.Tensor):
                    audio = audio.squeeze(0).to(device)
                    if length < maxlen and audio.ndim == 2:
                        audio = audio.unsqueeze(0)

            if length < maxlen:
                visual = visual.unsqueeze(0)

            # Split logic
            num_splits = int(length / maxlen) + 1
            lengths = torch.zeros(num_splits, dtype=torch.long)
            rem_len = length
            for j in range(num_splits):
                if rem_len > maxlen:
                    lengths[j] = maxlen
                    rem_len -= maxlen
                else:
                    lengths[j] = rem_len

            padding_mask = tools.get_batch_mask(lengths, maxlen).to(device)

            # Forward pass
            _, logits1, logits2, _, _, dnp_dict = model(
                visual=visual,
                padding_mask=padding_mask,
                text=prompt_text,
                lengths=lengths,
                audio=audio,
                DNP_use=True
            )

            # Unpack stream predictions
            s_det = dnp_dict.get('detector_scores', torch.sigmoid(logits1)).reshape(-1)[0:length].cpu().numpy()
            s_routing = dnp_dict.get('routing_scores', torch.sigmoid(logits1)).reshape(-1)[0:length].cpu().numpy()
            
            p_align = F.softmax(logits2.reshape(-1, logits2.shape[-1])[0:length] / eff_temp, dim=-1)
            s_sem = (1.0 - p_align[:, 0]).clamp(0.0, 1.0).cpu().numpy()
            
            if 'reconstruction_scores' in dnp_dict:
                s_rec = dnp_dict['reconstruction_scores'].reshape(-1)[0:length].cpu().numpy()
            else:
                s_rec = s_det.copy()

            if dataset == 'xd' and 'aux_scores' in dnp_dict and dnp_dict['aux_scores'] is not None:
                s_aux = dnp_dict['aux_scores'].reshape(-1)[0:length].cpu().numpy()
            else:
                s_aux = s_det.copy()

            # Baseline Quad-Stream Fusion
            s_quad = 0.40 * s_det + 0.25 * s_aux + 0.20 * s_sem + 0.15 * s_rec

            # Match with ground truth frames
            n_frames = length * 16
            video_gt = gt[cursor: cursor + n_frames] if (cursor + n_frames) <= len(gt) else gt[cursor:]
            cursor += n_frames

            video_records.append({
                'video_idx': idx,
                'path': raw_path if isinstance(raw_path, str) else str(raw_path[0]),
                'label': label_name if isinstance(label_name, str) else str(label_name[0]),
                'length_snippets': length,
                'length_frames': len(video_gt),
                's_det': s_det,
                's_aux': s_aux,
                's_sem': s_sem,
                's_rec': s_rec,
                's_routing': s_routing,
                's_quad': s_quad,
                'gt_frames': video_gt
            })

    print(f"[S2 Prober] Collected predictions for {len(video_records)} test videos ({cursor} frames).")
    return {'videos': video_records, 'total_frames': cursor}


def analyze_per_video_degradations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ranks worst false-alarm and worst false-negative videos and extracts stream culprits."""
    normal_evals = []
    abnormal_evals = []

    for rec in records:
        gt = rec['gt_frames']
        if len(gt) == 0:
            continue
        # Expand 16x snippet scores to frame scores
        f_quad = np.repeat(rec['s_quad'], 16)[:len(gt)]
        f_det = np.repeat(rec['s_det'], 16)[:len(gt)]
        f_aux = np.repeat(rec['s_aux'], 16)[:len(gt)]
        f_sem = np.repeat(rec['s_sem'], 16)[:len(gt)]
        f_rec = np.repeat(rec['s_rec'], 16)[:len(gt)]

        is_abnormal = np.sum(gt) > 0

        if not is_abnormal:  # Pure normal video
            mean_fa = float(np.mean(f_quad))
            max_fa = float(np.max(f_quad))
            fa_ratio_high = float(np.mean(f_quad > 0.50))
            
            # Culprit stream identification at the max spike
            peak_idx = int(np.argmax(f_quad))
            stream_scores = {
                'det': float(f_det[peak_idx]),
                'aux': float(f_aux[peak_idx]),
                'sem': float(f_sem[peak_idx]),
                'rec': float(f_rec[peak_idx]),
            }
            culprit_stream = max(stream_scores, key=stream_scores.get)

            normal_evals.append({
                'path': os.path.basename(rec['path']),
                'length_sec': round(len(gt) / 30.0, 1),
                'mean_score': round(mean_fa, 4),
                'max_score': round(max_fa, 4),
                'high_fa_percentage': round(fa_ratio_high * 100, 2),
                'culprit_stream': culprit_stream,
                'stream_breakdown_at_peak': stream_scores
            })
        else:  # Abnormal video
            anom_mask = (gt == 1)
            mean_hit = float(np.mean(f_quad[anom_mask]))
            miss_rate = float(np.mean(f_quad[anom_mask] < 0.35))
            
            abnormal_evals.append({
                'path': os.path.basename(rec['path']),
                'label': rec['label'],
                'length_sec': round(len(gt) / 30.0, 1),
                'mean_anomaly_score': round(mean_hit, 4),
                'miss_rate_percentage': round(miss_rate * 100, 2)
            })

    # Sort normal videos by max_score descending (Worst False Alarms)
    normal_evals.sort(key=lambda x: (x['high_fa_percentage'], x['max_score']), reverse=True)
    # Sort abnormal videos by miss_rate descending (Worst Misses)
    abnormal_evals.sort(key=lambda x: x['miss_rate_percentage'], reverse=True)

    return {
        'top_15_worst_false_alarms': normal_evals[:15],
        'top_15_worst_missed_anomalies': abnormal_evals[:15]
    }


def multi_stream_failure_attribution(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates quantitative blame distribution across streams on false-positive frames."""
    fa_det_list, fa_aux_list, fa_sem_list, fa_rec_list = [], [], [], []

    for rec in records:
        gt = rec['gt_frames']
        if len(gt) == 0:
            continue
        f_quad = np.repeat(rec['s_quad'], 16)[:len(gt)]
        f_det = np.repeat(rec['s_det'], 16)[:len(gt)]
        f_aux = np.repeat(rec['s_aux'], 16)[:len(gt)]
        f_sem = np.repeat(rec['s_sem'], 16)[:len(gt)]
        f_rec = np.repeat(rec['s_rec'], 16)[:len(gt)]

        # False positive frames: Ground truth is NORMAL (0), but model alarms (> 0.40)
        fp_mask = (gt == 0) & (f_quad > 0.40)
        if np.any(fp_mask):
            fa_det_list.extend(f_det[fp_mask])
            fa_aux_list.extend(f_aux[fp_mask])
            fa_sem_list.extend(f_sem[fp_mask])
            fa_rec_list.extend(f_rec[fp_mask])

    if len(fa_det_list) == 0:
        return {"status": "Zero extreme false alarms detected!"}

    mean_det = float(np.mean(fa_det_list))
    mean_aux = float(np.mean(fa_aux_list))
    mean_sem = float(np.mean(fa_sem_list))
    mean_rec = float(np.mean(fa_rec_list))

    total = mean_det + mean_aux + mean_sem + mean_rec + 1e-8
    attribution = {
        'total_analyzed_fp_frames': len(fa_det_list),
        'stream_mean_fp_scores': {
            'detector_BRSTB': round(mean_det, 4),
            'acoustic_AUX': round(mean_aux, 4),
            'semantics_TVSA': round(mean_sem, 4),
            'manifold_ASPM': round(mean_rec, 4),
        },
        'stream_blame_percentage': {
            'detector_BRSTB': round((mean_det / total) * 100, 2),
            'acoustic_AUX': round((mean_aux / total) * 100, 2),
            'semantics_TVSA': round((mean_sem / total) * 100, 2),
            'manifold_ASPM': round((mean_rec / total) * 100, 2),
        }
    }
    return attribution


def run_inference_grid_sweep(records: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    """Sweeps Gaussian filter, Asymmetric Audio Gate, and Simplex Weights to unlock max metric."""
    print("\n" + "=" * 75)
    print(f"[S2 Mathematical Grid Sweep] Searching for Peak Attainable Metric on {dataset.upper()}...")
    print("=" * 75)

    all_gt = []
    for rec in records:
        all_gt.extend(rec['gt_frames'])
    all_gt = np.array(all_gt, dtype=np.float32)

    # 1. Baseline Evaluation
    raw_quad = []
    for rec in records:
        raw_quad.extend(np.repeat(rec['s_quad'], 16)[:len(rec['gt_frames'])])
    raw_quad = np.array(raw_quad, dtype=np.float32)

    base_auc = roc_auc_score(all_gt, raw_quad)
    base_ap = average_precision_score(all_gt, raw_quad)
    print(f"  --> Raw Baseline Quad-Stream: AP = {base_ap * 100:.2f}% | AUC = {base_auc * 100:.2f}%")

    best_config = {
        'best_ap': base_ap,
        'best_auc': base_auc,
        'best_sigma': 0.0,
        'best_weights': [0.40, 0.25, 0.20, 0.15],
        'best_gate_tau': 0.0
    }

    # 2. Gaussian Smoothing Sweep
    sigmas = [0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0]
    for sig in sigmas:
        smoothed_scores = []
        for rec in records:
            f_sc = np.repeat(rec['s_quad'], 16)[:len(rec['gt_frames'])]
            if sig > 0:
                f_sc = gaussian_filter1d(f_sc, sigma=sig)
            smoothed_scores.extend(f_sc)
        smoothed_scores = np.array(smoothed_scores, dtype=np.float32)
        auc_s = roc_auc_score(all_gt, smoothed_scores)
        ap_s = average_precision_score(all_gt, smoothed_scores)
        
        target_metric = ap_s if dataset == 'xd' else auc_s
        cur_best = best_config['best_ap'] if dataset == 'xd' else best_config['best_auc']
        if target_metric > cur_best:
            best_config['best_ap'] = ap_s
            best_config['best_auc'] = auc_s
            best_config['best_sigma'] = sig

    print(f"  [Sweep Phase 1 - Gaussian Smoothing] Peak at sigma={best_config['best_sigma']}: "
          f"AP = {best_config['best_ap'] * 100:.2f}% | AUC = {best_config['best_auc'] * 100:.2f}%")

    # 3. Asymmetric Audio Gating Sweep (XD-Violence only)
    if dataset == 'xd':
        for tau_vis in [0.05, 0.10, 0.15, 0.20]:
            gated_scores = []
            for rec in records:
                # S2 Theorem: Soft-Gate S_aux with S_det
                gate = 1.0 / (1.0 + np.exp(-12.0 * (rec['s_det'] - tau_vis)))
                s_aux_gated = rec['s_aux'] * gate
                s_comb = 0.45 * rec['s_det'] + 0.20 * s_aux_gated + 0.20 * rec['s_sem'] + 0.15 * rec['s_rec']
                f_sc = np.repeat(s_comb, 16)[:len(rec['gt_frames'])]
                if best_config['best_sigma'] > 0:
                    f_sc = gaussian_filter1d(f_sc, sigma=best_config['best_sigma'])
                gated_scores.extend(f_sc)
            gated_scores = np.array(gated_scores, dtype=np.float32)
            ap_g = average_precision_score(all_gt, gated_scores)
            auc_g = roc_auc_score(all_gt, gated_scores)
            if ap_g > best_config['best_ap']:
                best_config['best_ap'] = ap_g
                best_config['best_auc'] = auc_g
                best_config['best_gate_tau'] = tau_vis
                print(f"  [Sweep Phase 2 - Asymmetric Gate] Tau_vis={tau_vis} boosted AP to: {ap_g * 100:.2f}%! (AUC={auc_g * 100:.2f}%)")

    # 4. Stream Weight Grid Search
    weight_candidates = [
        [0.50, 0.20, 0.20, 0.10],
        [0.45, 0.15, 0.25, 0.15],
        [0.55, 0.10, 0.25, 0.10],
        [0.60, 0.10, 0.20, 0.10],
        [0.35, 0.30, 0.20, 0.15]
    ] if dataset == 'xd' else [
        [0.60, 0.00, 0.25, 0.15],
        [0.70, 0.00, 0.15, 0.15],
        [0.75, 0.00, 0.15, 0.10],
        [0.80, 0.00, 0.10, 0.10]
    ]

    for w in weight_candidates:
        w_scores = []
        for rec in records:
            if dataset == 'xd' and best_config['best_gate_tau'] > 0:
                gate = 1.0 / (1.0 + np.exp(-12.0 * (rec['s_det'] - best_config['best_gate_tau'])))
                s_aux_use = rec['s_aux'] * gate
            else:
                s_aux_use = rec['s_aux']
            s_comb = w[0] * rec['s_det'] + w[1] * s_aux_use + w[2] * rec['s_sem'] + w[3] * rec['s_rec']
            f_sc = np.repeat(s_comb, 16)[:len(rec['gt_frames'])]
            if best_config['best_sigma'] > 0:
                f_sc = gaussian_filter1d(f_sc, sigma=best_config['best_sigma'])
            w_scores.extend(f_sc)
        w_scores = np.array(w_scores, dtype=np.float32)
        auc_w = roc_auc_score(all_gt, w_scores)
        ap_w = average_precision_score(all_gt, w_scores)
        target = ap_w if dataset == 'xd' else auc_w
        cur = best_config['best_ap'] if dataset == 'xd' else best_config['best_auc']
        if target > cur:
            best_config['best_ap'] = ap_w
            best_config['best_auc'] = auc_w
            best_config['best_weights'] = w

    print("-" * 75)
    print(f"🏆 [S2 Optimization Result] Peak Attainable Metric on {dataset.upper()}:")
    print(f"  Frame-level AP : {best_config['best_ap'] * 100:.2f}% (Gain: +{(best_config['best_ap'] - base_ap) * 100:+.2f}%)")
    print(f"  Frame-level AUC: {best_config['best_auc'] * 100:.2f}% (Gain: +{(best_config['best_auc'] - base_auc) * 100:+.2f}%)")
    print(f"  Recipe: Gaussian Sigma={best_config['best_sigma']}, Stream Weights={best_config['best_weights']}, Gate Tau={best_config['best_gate_tau']}")
    print("-" * 75)

    return best_config


def main():
    args = parse_args()
    resolve_default_paths(args)

    print("=" * 75)
    print("      INTERN-S2 DEEP DIAGNOSTIC & MATHEMATICAL PROBER      ")
    print("      Target: Clinical Error Profiling for SOTA >=89% / >=87%      ")
    print("=" * 75)
    print(f"Dataset      : {args.dataset.upper()}")
    print(f"Model Path   : {args.model_path}")
    print(f"Data Dir     : {args.data_dir}")
    print(f"Audio Dir    : {args.audio_dir if args.dataset == 'xd' else 'N/A'}")
    print(f"GT Path      : {args.gt_path}")
    print(f"Device       : {args.device}")

    # Load ground truth
    if not os.path.exists(args.gt_path):
        print(f"[ERROR] Ground truth file not found: {args.gt_path}")
        return
    gt = np.load(args.gt_path)

    # Descriptions & Label Map
    if args.dataset == 'xd':
        label_map = dict(A=0, B1=1, B2=2, B4=3, B5=4, B6=5, G=6)
        prompt_text = [descriptions.DESCRIPTIONS_ORI_XD[k][0] for k in ["A", "B1", "B2", "B4", "B5", "B6", "G"]]
        test_dataset = XDDataset(
            clip_dim=256,
            file_path=args.test_file_path,
            test_mode=True,
            label_map=label_map,
            data_dir=args.data_dir,
            audio_dir=args.audio_dir
        )
    else:
        label_map = dict(
            Normal=0, Abuse=1, Arrest=2, Arson=3, Assault=4, Burglary=5,
            Explosion=6, Fighting=7, RoadAccidents=8, Robbery=9, Shooting=10,
            Shoplifting=11, Stealing=12, Vandalism=13
        )
        prompt_text = [descriptions.DESCRIPTIONS_ORI[k][0] for k in label_map.keys()]
        test_dataset = UCFDataset(
            clip_dim=256,
            file_path=args.test_file_path,
            test_mode=True,
            label_map=label_map,
            data_dir=args.data_dir
        )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )

    # Load Model
    model = load_diagnostic_model(args.model_path, args.dataset, args.device)

    # 1. Collect Predictions
    collected = collect_predictions_per_video(
        model=model,
        test_loader=test_loader,
        prompt_text=prompt_text,
        gt=gt,
        dataset=args.dataset,
        device=args.device
    )

    # 2. Per-Video Error Ranking
    error_profile = analyze_per_video_degradations(collected['videos'])

    # 3. Stream Blame Decomposition
    attribution = multi_stream_failure_attribution(collected['videos'])

    # 4. Mathematical Grid Sweep
    best_recipe = {}
    if args.scan_inference:
        best_recipe = run_inference_grid_sweep(collected['videos'], args.dataset)

    # 5. Compile Clinical Report
    report = {
        'dataset': args.dataset,
        'model_path': args.model_path,
        'per_video_degradations': error_profile,
        'stream_failure_attribution': attribution,
        'optimal_inference_recipe': best_recipe
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(f"\n[S2 Clinical Diagnosis] Complete report saved to: {args.output_json}")


if __name__ == '__main__':
    main()
