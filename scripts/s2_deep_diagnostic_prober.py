#!/usr/bin/env python3
"""MA-RUPA v4 Clinical Diagnostic Prober for Weakly-Supervised Video Anomaly Detection (WS-VAD).

Capabilities:
  1. Per-Video Error Profiling: Ranks Top-5 Highest False Alarm Normal Videos & Top-5 Hardest Abnormal Videos.
  2. Multi-Stream Attribution across Pure Visual Detector, Safe-RUPA Routing, CLIP Semantics, SGNM Prototype Error, and Decoupled Audio-Visual Head.
  3. Chunked Parallel Multi-Crop Inference & Visual-Protected Synergy Grid Sweep.
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
    parser = argparse.ArgumentParser(description="MA-RUPA v4 Clinical Diagnostic Prober")
    parser.add_argument('--dataset', type=str, default='xd', choices=['xd', 'ucf'])
    parser.add_argument('--model-path', '--model_path', dest='model_path', type=str, default='')
    parser.add_argument('--data-dir', '--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--audio-dir', '--audio_dir', dest='audio_dir', type=str, default='')
    parser.add_argument('--test-file-path', '--test_file_path', '--test-list', '--test_list', dest='test_file_path', type=str, default='')
    parser.add_argument('--gt-path', '--gt_path', dest='gt_path', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
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
        incomp = load_compatible_model_state(model, state, strict=False, allow_legacy_dsanet=True)
        print(f"[Prober] Loaded checkpoint: {model_path} (unexpected={len(incomp.unexpected_keys)})", flush=True)
    else:
        print(f"[Prober] WARNING: Checkpoint '{model_path}' not found; running initialized weights.", flush=True)

    model.eval()
    return model


def _forward_chunked(model: MARUPA, visual_flat: torch.Tensor, audio_flat: torch.Tensor,
                     padding_mask: torch.Tensor, prompt_text: List[str],
                     lengths_flat: torch.Tensor, max_chunk: int = 32):
    total_b = visual_flat.shape[0]
    l1_l, l2_l, det_l, route_l, rec_l, aux_l = [], [], [], [], [], []
    for start in range(0, total_b, max_chunk):
        end = min(start + max_chunk, total_b)
        aud_c = audio_flat[start:end] if audio_flat is not None else None
        _, l1, l2, _, _, dnp_c = model(
            visual=visual_flat[start:end], padding_mask=padding_mask[start:end],
            text=prompt_text, lengths=lengths_flat[start:end], audio=aud_c, DNP_use=True,
        )
        d_c = dnp_c.get('detector_scores', torch.sigmoid(l1))
        l1_l.append(l1)
        l2_l.append(l2)
        det_l.append(d_c)
        route_l.append(dnp_c.get('routing_scores', d_c))
        rec_l.append(dnp_c.get('reconstruction_scores', d_c))
        if dnp_c.get('aux_scores') is not None:
            aux_l.append(dnp_c['aux_scores'])

    return (
        torch.cat(l1_l, dim=0),
        torch.cat(l2_l, dim=0),
        torch.cat(det_l, dim=0),
        torch.cat(route_l, dim=0),
        torch.cat(rec_l, dim=0),
        torch.cat(aux_l, dim=0) if aux_l else None,
    )


def collect_predictions_per_video(model: MARUPA, test_loader, prompt_text: List[str],
                                  gt: np.ndarray, args) -> Dict[str, Any]:
    video_records = []
    cursor = 0
    maxlen = 256
    sem_temp = 2.0 if args.dataset == 'xd' else 2.5
    max_crops = 10 if args.multi_crop_eval else 1
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

            _, logits2, det_s, route_s, rec_s, aux_s = _forward_chunked(
                model, visual_flat, audio_flat, padding_mask, prompt_text, lengths_flat
            )

            s_det = det_s.view(C, num_splits * maxlen, -1)[:, :length, 0].float().mean(dim=0).cpu().numpy()
            s_routing = route_s.view(C, num_splits * maxlen, -1)[:, :length, 0].float().mean(dim=0).cpu().numpy()
            s_rec = rec_s.view(C, num_splits * maxlen, -1)[:, :length, 0].float().mean(dim=0).cpu().numpy()

            l2_c = logits2.view(C, num_splits * maxlen, -1)[:, :length, :]
            s_sem = (1.0 - F.softmax(l2_c.float() / sem_temp, dim=-1)[:, :, 0]).clamp(0.0, 1.0).mean(dim=0).cpu().numpy()

            has_aux = (args.dataset == 'xd' and aux_s is not None)
            if has_aux:
                s_aux = aux_s.view(C, num_splits * maxlen, -1)[:, :length, 0].float().mean(dim=0).cpu().numpy()
            else:
                s_aux = s_routing.copy()

            v_base = 0.88 * s_routing + 0.12 * (0.70 * s_sem + 0.30 * s_rec)
            if has_aux:
                g_vs = np.clip((np.maximum(s_det, s_sem) - 0.10) / 0.25, 0.0, 1.0)
                concordance = g_vs * s_aux
                s_collab = np.clip(v_base * (1.0 + 0.24 * (concordance - 0.15 * (1.0 - s_aux) * g_vs)) + 0.096 * concordance, 0.0, 1.0)
            else:
                s_collab = v_base

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
                'has_aux': has_aux,
                'gt_frames': video_gt,
            })

    return {'videos': video_records, 'total_frames': cursor}


def analyze_and_sweep(records: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    normal_evals, abnormal_evals = [], []
    valid_records = [r for r in records if len(r['gt_frames']) > 0]
    all_gt = np.concatenate([r['gt_frames'] for r in valid_records]).astype(np.float32)

    for rec in valid_records:
        gt = rec['gt_frames']
        f_col = np.repeat(gaussian_filter1d(rec['s_collab'], sigma=0.80), 16)[:len(gt)]
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

    det_all = np.concatenate([np.repeat(r['s_det'], 16)[:len(r['gt_frames'])] for r in valid_records])
    route_all = np.concatenate([np.repeat(r['s_routing'], 16)[:len(r['gt_frames'])] for r in valid_records])
    sem_all = np.concatenate([np.repeat(r['s_sem'], 16)[:len(r['gt_frames'])] for r in valid_records])
    rec_all = np.concatenate([np.repeat(r['s_rec'], 16)[:len(r['gt_frames'])] for r in valid_records])
    aux_all = np.concatenate([np.repeat(r['s_aux'], 16)[:len(r['gt_frames'])] for r in valid_records])

    g_vs_all = np.clip((np.maximum(det_all, sem_all) - 0.10) / 0.25, 0.0, 1.0)
    aux_alphas = [0.0, 0.06, 0.08, 0.12] if dataset == 'xd' else [0.0]

    best = {'ap': 0.0, 'auc': 0.0, 'recipe': ''}
    stream_choices = [('det', det_all), ('route', route_all)]
    alphas = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16] if dataset == 'ucf' else [0.0, 0.08, 0.12, 0.16, 0.20]
    sigmas = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]

    for stream_name, base_s in stream_choices:
        for alpha in alphas:
            v_base = (1.0 - alpha) * base_s + alpha * (0.70 * sem_all + 0.30 * rec_all)
            for aux_a in aux_alphas:
                if aux_a > 0.0:
                    concordance = g_vs_all * aux_all
                    boost_gain = min(0.35, 3.0 * aux_a)
                    blend = np.clip(v_base * (1.0 + boost_gain * (concordance - 0.15 * (1.0 - aux_all) * g_vs_all)) + (0.40 * boost_gain) * concordance, 0.0, 1.0)
                else:
                    blend = v_base
                for sig in sigmas:
                    s_arr = blend if sig == 0.0 else gaussian_filter1d(blend, sigma=sig * 16.0 / 16.0)
                    auc_v = float(roc_auc_score(all_gt, s_arr))
                    ap_v = float(average_precision_score(all_gt, s_arr))
                    key_cur = ap_v if dataset == 'xd' else auc_v
                    key_best = best['ap'] if dataset == 'xd' else best['auc']
                    if key_cur > key_best:
                        best = {
                            'ap': ap_v,
                            'auc': auc_v,
                            'recipe': f"stream={stream_name}, collab_alpha={alpha}, aux_eval_alpha={aux_a}, sigma={sig}",
                        }

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
