import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value.')


parser = argparse.ArgumentParser(description='MA-RUPA v3: Clean SOTA Multimodal WS-VAD (XD-Violence)')
parser.add_argument('--seed', default=234, type=int, help='Random seed for reproducibility')

# Core Backbone & Vision-Language Hyperparameters (Proven XD-Violence SOTA Configuration)
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int, help='64-snippet local context window for XD-Violence movies/CCTV')
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)
parser.add_argument('--mlp-dropout', default=0.1, type=float)

# Cross-Modal Gated Audio-Visual Branch (128-D VGGish / 512-D CLAP)
parser.add_argument('--audio-dim', default=128, type=int)
parser.add_argument('--audio-train-list', default=None, type=str)
parser.add_argument('--audio-test-list', default=None, type=str)
parser.add_argument('--audio-dir', default=None, type=str)

# Dynamic Normal Prototypes (SGNM) & Continuation Safe-Gate
parser.add_argument('--decoder_depth', '--decoder-depth', dest='decoder_depth', type=int, default=8)
parser.add_argument('--normal_selection_ratio', '--normal-selection-ratio', dest='normal_selection_ratio', type=float, default=0.8)
parser.add_argument('--num_prototypes', '--num-prototypes', dest='num_prototypes', type=int, default=16)
parser.add_argument('--DNP_use', '--dnp-use', dest='DNP_use', default=True, type=str2bool)
parser.add_argument('--rupa-use', '--rupa_use', dest='rupa_use', default=True, type=str2bool)
parser.add_argument('--routing-mode', choices=['fixed', 'safe_gate'], default='safe_gate')
parser.add_argument('--routing-det-weight', default=0.90, type=float)
parser.add_argument('--routing-rec-weight', default=0.10, type=float)
parser.add_argument('--routing-sem-weight', default=0.00, type=float)
parser.add_argument('--routing-gate-hidden', default=8, type=int)
parser.add_argument('--routing-gate-init', default=0.01, type=float)
parser.add_argument('--routing-gate-max', default=1.0, type=float)

# CLIP Text Adapter
parser.add_argument('--text_adapt_until', '--text-adapt-until', dest='text_adapt_until', default=1, type=int)
parser.add_argument('--t_w', '--t-w', dest='t_w', default=0.6, type=float)
parser.add_argument('--temp', default=1.0, type=float, help='Hierarchical class localization temperature')

# Ablation Flags
parser.add_argument('--no-mdtp', action='store_true', default=False, help='Disable Zero-Init Multi-Scale Temporal Sensitivity Adapter')
parser.add_argument('--no-audio', action='store_true', default=False, help='Disable Cross-Modal Gated Audio-Visual Adapter')
parser.add_argument('--no-rupa', action='store_true', default=False, help='Disable Counterfactual Prototype Refiner')
parser.add_argument('--no-safegate', action='store_true', default=False, help='Use fixed routing instead of SafeGate')

# Optimization & Schedule (Dual Decoupled Optimizers: AdamW Main/Audio + StableAdamW Refiner)
parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--batch-size', default=96, type=int, help='Total batch size across GPUs (48 Normal + 48 Abnormal)')
parser.add_argument('--lr', type=float, default=1.5e-5, help='SOTA base learning rate for XD-Violence visual/text backbone')
parser.add_argument('--main-lr', type=float, default=None)
parser.add_argument('--audio-lr', type=float, default=5e-5, help='Dedicated LR for newly initialized VGGish cross-modal adapter')
parser.add_argument('--refiner-lr', type=float, default=None)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--grad-clip', type=float, default=1.0)
parser.add_argument('--use-ema', default=True, type=str2bool, help='Maintain Polyak EMA shadow weights from Epoch 2+')
parser.add_argument('--ema-decay', default=0.996, type=float)

# Principled SOTA Loss Weights
parser.add_argument('--loss2-weight', '--loss2_weight', dest='loss2_weight', type=float, default=5.0, help='CLASM multi-class vision-language alignment weight for XD-Violence')
parser.add_argument('--loss-sep-weight', '--loss_sep_weight', dest='loss_sep_weight', type=float, default=1.0, help='Normal-Abnormal text orthogonality weight')
parser.add_argument('--loss-residual-weight', '--lambda-residual', dest='loss_residual_weight', type=float, default=0.20)
parser.add_argument('--loss-reconstructed-normal-weight', '--lambda-bkg', dest='loss_reconstructed_normal_weight', type=float, default=0.20)
parser.add_argument('--loss-consistency-weight', '--lambda-consistency', dest='loss_consistency_weight', type=float, default=0.10)
parser.add_argument('--loss-gather-weight', '--lambda-gather', dest='loss_gather_weight', type=float, default=0.10)
parser.add_argument('--loss-dnp-normal-weight', default=0.01, type=float)
parser.add_argument('--loss-routing-weight', default=1.0, type=float)
parser.add_argument('--loss-routing-anchor-weight', '--loss_routing_anchor_weight', dest='loss_routing_anchor_weight', type=float, default=0.10)
parser.add_argument('--lambda-aux', default=0.25, type=float, help='Audio-Guided MIL + Normal Acoustic False-Alarm Suppression weight')
parser.add_argument('--lambda-smooth', default=0.02, type=float, help='Edge-preserving Charbonnier smoothness weight')

# Dataset & Checkpoints
parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='list/gt_label.npy')
parser.add_argument('--data-dir', default=None, type=str)
parser.add_argument('--model-path', default='model/model_xd.pth')
parser.add_argument('--checkpoint-path', default='model/checkpoint_xd.pth')
parser.add_argument('--use-checkpoint', default=False, type=str2bool)
parser.add_argument('--init-model-path', default=None)
parser.add_argument('--sample-one-crop-per-video', default=False, type=str2bool)
parser.add_argument('--crops-per-video', default=1, type=int)

# Evaluation & Multi-Crop Multimodal Inference
parser.add_argument('--multi-crop-eval', default=True, type=str2bool, help='Aggregate all 5 spatial test crops (__0..__4) in parallel')
parser.add_argument('--max-test-crops', default=5, type=int)
parser.add_argument('--collab-alpha', default=0.15, type=float, help='DeSC collaborative semantic consistency fusion weight')
parser.add_argument('--aux-eval-alpha', default=0.10, type=float, help='Visual-gated acoustic auxiliary reinforcement weight')
parser.add_argument('--smooth-eval', default=True, type=str2bool)
parser.add_argument('--smooth-method', choices=['gaussian', 'savgol'], default='gaussian')
parser.add_argument('--smooth-sigma', default=0.75, type=float)
parser.add_argument('--eval-interval', default=80, type=int, help='Mid-epoch validation interval in iterations (0 = epoch-end only)')
parser.add_argument('--num-workers', default=2, type=int)
