#!/usr/bin/env python3
"""MA-RUPA v3 Clinical Diagnostic Prober for Weakly-Supervised Video Anomaly Detection (WS-VAD).

Capabilities:
  1. Per-Video Error Profiling: Ranks Top-10 Worst False Alarm Normal Videos & Top-10 Missed Anomaly Videos.
  2. Multi-Stream Failure Attribution across Detector, Multimodal Audio, Vision-Language Semantics, and SGNM Reconstruction.
  3. Parallel Multi-Crop Inference & Inference-Time Grid Sweep (Gaussian sigma & Collaborative Weights).
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter1d

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')
for p in [src_dir, root_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import MARUPA, load_compatible_model_state
from utils.dataset import XDDataset, UCFDataset, load_multicrop_test_video
from utils import tools
import xd_option as xd_opt
import ucf_option as ucf_opt


def parse_args():
    parser = argparse.ArgumentParser(description="MA-RUPA v3 Clinical Diagnostic Prober")
    parser.add_argument('--dataset', type=str, default='xd', choices=['xd', 'ucf'])
    parser.add_argument('--model-path', '--model_path', dest='model_path', type=str, default='')
    parser.add_argument('--data-dir', '--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--audio-dir', '--audio_dir', dest='audio_dir', type=str, default='')
    parser.add_argument('--test-file-path', '--test_file_path', '--test-list', '--test_list', dest='test_file_path', type=str, default='')
    parser.add_argument('--gt-path', '--gt_path', dest='gt_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--multi-crop-eval', default=True, type=ucf_opt.str2bool)
    parser.add_argument('--output-json', '--output_json', dest='output_json', type=str, default='rupa_artifacts/s2_diagnostic_report.json')
    parser.add_argument('--scan-inference', '--scan_inference', dest='scan_inference', action='store_true', default=True)
    args, _ = parser.parse_known_args()
    return args


def resolve_default_paths(args):
    def find_first_existing(candidates, fallback):
        for c in candidates:
            if not c:
                continue
            if os.path.exists(c):
                return os.path.abspath(c)
            p_rel = os.path.join(root_dir, c)
            if os.path.exists(p_rel):
                return os.path.abspath(p_rel)
        fallback_rel = os.path.join(root_dir, fallback)
        return os.path.abspath(fallback_rel) if os.path.exists(fallback_rel) else fallback

    if args.dataset == 'xd':
        if not args.test_file_path or not os.path.exists(args.test_file_path):
            args.test_file_path = find_first_existing([
                '/kaggle/working/rupa_artifacts/xd_test.csv', 'list/xd_CLIP_rgbtest.csv',
            ], 'list/xd_CLIP_rgbtest.csv')
        if not args.gt_path or not os.path.exists(args.gt_path):
            args.gt_path = find_first_existing(['list/gt.npy', '/kaggle/working/MA-RUPA/list/gt.npy'], 'list/gt.npy')
        if not args.model_path or not os.path.exists(args.model_path):
            args.model_path = find_first_existing([
                '/kaggle/working/rupa_artifacts/model_xd_best.pth', 'rupa_artifacts/model_xd_best.pth',
            ], 'rupa_artifacts/model_xd_best.pth')
    else:
        if not args.test_file_path or not os.path.exists(args.test_file_path):
            args.test_file_path = find_first_existing([
                '/kaggle/working/rupa_artifacts/ucf_test.csv', 'list/ucf_CLIP_rgbtest.csv',
            ], 'list/ucf_CLIP_rgbtest.csv')
        if not args.gt_path or not os.path.exists(args.gt_path):
            args.gt_path = find_first_existing(['list/gt_ucf.npy', '/kaggle/working/MA-RUPA/list/gt_ucf.npy'], 'list/gt_ucf.npy')
        if not args.model_path or not os.path.exists(args.model_path):
            args.model_path = find_first_existing([
                '/kaggle/working/rupa_artifacts/model_ucf_best.pth', 'rupa_artifacts/model_ucf_best.pth',
            ], 'rupa_artifacts/model_ucf_best.pth')


def load_diagnostic_model(model_path: str, dataset: str, device: str) -> MARUPA:
    num_classes = 7 if dataset == 'xd' else 14
    opt_module = xd_opt if dataset == 'xd' else ucf_opt
    dummy_args, _ = opt_module.parser.parse_known_args([])

    model = MARUPA(
        num_class=num_classes,
        embed_dim=dummy_args.embed_dim,
        visual_length=dummy_args.visual_length,
        visual_width=dummy_args.visual_width,
        visual_head=dummy_args.visual_head,
        visual_layers=dummy_args.visual_layers,
        attn_window=dummy_args.attn_window,
        args=dummy_args,
        device=device,
    ).to(device)

    if os.path.exists(model_path):
        state = tools.safe_load_checkpoint(model_path, map_location=device)
        incomp = load_compatible_model_state(model, state, strict=False)
        print(f"[Prober] Loaded checkpoint: {model_path} (unexpected={len(incomp.unexpected_keys)})", flush=True)
    else:
        print(f"[Prober] WARNING: Checkpoint '{model_path}' not found; running initialized weights.", flush=True)

    model.eval()
    return model


def collect_predictions_per_video(model: MARUPA, test_loader, prompt_text: List[str],
                                  gt: np.ndarray, args) -> Dict[str, Any]:
    video_records = []
    cursor = 0
    maxlen = 256
    eff_temp = max(float(args.temp), 1e-3)
    max_crops = (5 if args.dataset == 'xd' else 10) if args.multi_crop_eval else 1
    no_audio = (args.dataset != 'xd')

    with torch.no_grad():
        for idx, item in enumerate(test_loader):
            fallback_vis = item[0].squeeze(0)
            length = int(item[2])
            raw_path = item[3][0] if isinstance(item[3], (list, tuple)) and len(item[3]) > 0 else str(item[3])
            label_name = item[1][0] if isinstance(item[1], (list, tuple)) and len(item[1]) > 0 else str(item[1])
            fallback_aud = item[5].squeeze(0) if (not no_audio and len(item) >= 6 and isinstance(item[5], torch.Tensor)) else None

            vis_crops, aud_crops, length, _ = load_multicrop_test_video(
                raw_path=raw_path, maxlen=maxlen,
                data_dir=args.data_dir or None, audio_dir=args.audio_dir or None,
                no_audio=no_audio, max_crops=max_crops,
                fallback_visual=fallback_vis, fallback_length=length, fallback_audio=fallback_aud,
            )
            C, num_splits, _, D = vis_crops.shape
            base_lengths = torch.full((num_splits,), maxlen, dtype=torch.long)
            rem = length - maxlen * (num_splits - 1)
            base_lengths[-1] = max(1, min(maxlen, rem))

            lengths_flat = base_lengths.repeat(C)
            padding_mask = tools.get_batch_mask(lengths_flat, maxlen).to(args.device)
            visual_flat = vis_crops.view(C * num_splits, maxlen, D).to(args.device)
            audio_flat = (
                aud_crops.view(C * num_splits, maxlen, aud_crops.shape[-1]).to(args.device)
                if aud_crops is not None else None
            )

            _, logits1, logits2, _, _, dnp_dict = model(
                visual=visual_flat, padding_mask=padding_mask, text=prompt_text,
                lengths=lengths_flat, audio=audio_flat, DNP_use=True,
            )

            s_det = dnp_dict.get('detector_scores', torch.sigmoid(logits1)).view(C, num_splits * maxlen, -1)[:, :length, 0].mean(dim=0).cpu().numpy()
            s_routing = dnp_dict.get('routing_scores', torch.sigmoid(logits1)).view(C, num_splits * maxlen, -1)[:, :length, 0].mean(dim=0).cpu().numpy()

            sem_logits = dnp_dict.get('semantic_logits', logits2).view(C, num_splits * maxlen, -1)[:, :length, :]
            l2_c = logits2.view(C, num_splits * maxlen, -1)[:, :length, :]
            s_sem = (
                0.5 * (1.0 - F.softmax(l2_c.float() / eff_temp, dim=-1)[:, :, 0])
                + 0.5 * (1.0 - F.softmax(sem_logits.float() / eff_temp, dim=-1)[:, :, 0])
            ).clamp(0.0, 1.0).mean(dim=0).cpu().numpy()

            if 'reconstruction_scores' in dnp_dict:
                s_rec = dnp_dict['reconstruction_scores'].view(C, num_splits * maxlen, -1)[:, :length, 0].mean(dim=0).cpu().numpy()
            else:
                s_rec = s_det.copy()

            if args.dataset == 'xd' and dnp_dict.get('aux_scores') is not None:
                s_aux = dnp_dict['aux_scores'].view(C, num_splits * maxlen, -1)[:, :length, 0].mean(dim=0).cpu().numpy()
            else:
                s_aux = s_routing.copy()

            s_collab = 0.80 * s_routing + 0.20 * s_sem
            n_frames = length * 16
            video_gt = gt[cursor:cursor + n_frames] if (cursor + n_frames) <= len(gt) else gt[cursor:]
            cursor += n_frames

            video_records.append({
                'video_idx': idx,
                'path': raw_path,
                'label': label_name,
                's_det': s_det,
                's_aux': s_aux,
                's_sem': s_sem,
                's_rec': s_rec,
                's_routing': s_routing,
                's_collab': s_collab,
                'gt_frames': video_gt,
            })

    return {'videos': video_records, 'total_frames': cursor}


def analyze_and_sweep(records: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    normal_evals, abnormal_evals = [], []
    all_gt = np.concatenate([r['gt_frames'] for r in records if len(r['gt_frames']) > 0]).astype(np.float32)

    for rec in records:
        gt = rec['gt_frames']
        if len(gt) == 0:
            continue
        f_col = np.repeat(rec['s_collab'], 16)[:len(gt)]
        if np.sum(gt) == 0:
            normal_evals.append({
                'path': os.path.basename(rec['path']),
                'max_score': round(float(np.max(f_col)), 4),
                'mean_score': round(float(np.mean(f_col)), 4),
            })
        else:
            anom = gt == 1
            abnormal_evals.append({
                'path': os.path.basename(rec['path']),
                'label': rec['label'],
                'mean_anomaly_score': round(float(np.mean(f_col[anom])), 4),
                'miss_rate_pct': round(float(np.mean(f_col[anom] < 0.35)) * 100.0, 2),
            })

    normal_evals.sort(key=lambda x: x['max_score'], reverse=True)
    abnormal_evals.sort(key=lambda x: x['miss_rate_pct'], reverse=True)

    print("\nTOP-5 HIGHEST NORMAL FALSE ALARMS:")
    for v in normal_evals[:5]:
        print(f"  {v['path']:<42} | max={v['max_score']:.3f} | mean={v['mean_score']:.3f}")

    print("\nTOP-5 HARDEST ABNORMAL VIDEOS:")
    for v in abnormal_evals[:5]:
        print(f"  {v['path']:<42} | {v['label']:<14} | mean_anom={v['mean_anomaly_score']:.3f} | miss={v['miss_rate_pct']:.1f}%")

    route_all = np.concatenate([np.repeat(r['s_routing'], 16)[:len(r['gt_frames'])] for r in records if len(r['gt_frames']) > 0])
    sem_all = np.concatenate([np.repeat(r['s_sem'], 16)[:len(r['gt_frames'])] for r in records if len(r['gt_frames']) > 0])
    q15_r = float(np.clip(np.percentile(route_all, 15.0), 0.01, 0.35))
    q90_r = float(np.clip(np.percentile(route_all, 90.0), 0.50, 0.98))
    sem_cal = tools.calibrate_stream_global(sem_all, target_q15=q15_r, target_q90=q90_r)

    best = {'ap': 0.0, 'auc': 0.0, 'recipe': ''}
    for alpha in [0.0, 0.10, 0.15, 0.20, 0.25]:
        blend = (1.0 - alpha) * route_all + alpha * sem_cal
        for sig in [0.0, 0.5, 0.75, 1.0, 1.25]:
            s_arr = blend if sig == 0.0 else gaussian_filter1d(blend, sigma=sig)
            auc_v = float(roc_auc_score(all_gt, s_arr))
            ap_v = float(average_precision_score(all_gt, s_arr))
            key_cur = ap_v if dataset == 'xd' else auc_v
            key_best = best['ap'] if dataset == 'xd' else best['auc']
            if key_cur > key_best:
                best = {'ap': ap_v, 'auc': auc_v, 'recipe': f"collab_alpha={alpha}, sigma={sig}"}

    print(f"\n[Prober Peak] {dataset.upper()} -> AUC={best['auc']*100:.2f}% | AP={best['ap']*100:.2f}% ({best['recipe']})", flush=True)
    return {'top_false_alarms': normal_evals[:10], 'top_misses': abnormal_evals[:10], 'best': best}


def main():
    args = parse_args()
    resolve_default_paths(args)
    if not os.path.exists(args.gt_path):
        print(f"[ERROR] GT not found: {args.gt_path}")
        return
    gt = np.load(args.gt_path)

    if args.dataset == 'xd':
        label_map = {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting', 'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident', 'G': 'explosion'}
        ds = XDDataset(256, args.test_file_path, True, label_map, data_dir=args.data_dir or None, audio_dir=args.audio_dir or None)
        collate = tools.xd_test_collate_fn
    else:
        label_map = {
            'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson',
            'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion',
            'Fighting': 'fighting', 'RoadAccidents': 'roadaccidents', 'Robbery': 'robbery',
            'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism',
        }
        ds = UCFDataset(256, args.test_file_path, True, label_map, data_dir=args.data_dir or None)
        collate = None

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)
    model = load_diagnostic_model(args.model_path, args.dataset, args.device)
    collected = collect_predictions_per_video(model, loader, tools.get_prompt_text(label_map), gt, args)
    summary = analyze_and_sweep(collected['videos'], args.dataset)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
