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


parser = argparse.ArgumentParser(description='MA-RUPA v4+ (Argon-4 Gradient Conservation Edition - XD-Violence)')
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

# Decoupled Cross-Modal Gated Audio-Visual Branch (128-D VGGish / 512-D CLAP)
parser.add_argument('--audio-dim', default=128, type=int)
parser.add_argument('--audio-train-list', default=None, type=str)
parser.add_argument('--audio-test-list', default=None, type=str)
parser.add_argument('--audio-dir', default=None, type=str)

# Dynamic Normal Prototypes (SGNM) & Stop-Gradient Safe-RUPA Calibration
parser.add_argument('--decoder_depth', '--decoder-depth', dest='decoder_depth', type=int, default=8)
parser.add_argument('--normal_selection_ratio', '--normal-selection-ratio', dest='normal_selection_ratio', type=float, default=0.8)
parser.add_argument('--num_prototypes', '--num-prototypes', dest='num_prototypes', type=int, default=16)
parser.add_argument('--DNP_use', '--dnp-use', dest='DNP_use', default=True, type=str2bool)
parser.add_argument('--rupa-use', '--rupa_use', dest='rupa_use', default=True, type=str2bool)
parser.add_argument('--routing-mode', choices=['fixed', 'safe_gate'], default='safe_gate')
parser.add_argument('--routing-det-weight', default=0.80, type=float)
parser.add_argument('--routing-rec-weight', default=0.10, type=float)
parser.add_argument('--routing-sem-weight', default=0.10, type=float)
parser.add_argument('--routing-gate-hidden', default=8, type=int)
parser.add_argument('--routing-gate-init', default=0.15, type=float)
parser.add_argument('--routing-gate-max', default=0.50, type=float)

# CLIP Text Adapter (Pure Canonical Concept Space by Default)
parser.add_argument('--text_adapt_until', '--text-adapt-until', dest='text_adapt_until', default=1, type=int)
parser.add_argument('--t_w', '--t-w', dest='t_w', default=0.6, type=float)
parser.add_argument('--aat-weight', default=0.0, type=float, help='Keep 0.0 for pure canonical concept orthogonality')
parser.add_argument('--temp', default=1.0, type=float, help='Hierarchical class localization temperature')

# Legacy Optional Flags (Disabled by default under Argon-4 Gradient Conservation Law)
parser.add_argument('--iam-momentum', default=0.10, type=float)
parser.add_argument('--acc-eta', default=0.50, type=float)
parser.add_argument('--acc-tau', default=0.80, type=float)
parser.add_argument('--use-mdtp', action='store_true', default=False)
parser.add_argument('--no-mdtp', action='store_true', default=True)
parser.add_argument('--no-audio', action='store_true', default=False)
parser.add_argument('--no-rupa', action='store_true', default=False)
parser.add_argument('--no-safegate', action='store_true', default=False)
parser.add_argument('--use-iam', action='store_true', default=False)
parser.add_argument('--no-iam', action='store_true', default=True)
parser.add_argument('--use-acc', action='store_true', default=False)
parser.add_argument('--no-acc', action='store_true', default=True)

# Optimization & Schedule (Stratified 3-Crop Unique Video Epoch = ~185 steps/epoch, ending right at the Iter-150 Golden Peak!)
parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--batch-size', default=64, type=int, help='Total batch size across GPUs')
parser.add_argument('--balanced-sampling', default=False, type=str2bool, help='Use single natural shuffled DataLoader')
parser.add_argument('--lr', type=float, default=1e-5, help='SOTA base learning rate for XD-Violence visual/text backbone')
parser.add_argument('--main-lr', type=float, default=None)
parser.add_argument('--audio-lr', type=float, default=3e-5, help='Controlled LR for decoupled VGGish cross-modal auxiliary branch')
parser.add_argument('--refiner-lr', type=float, default=None)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--grad-clip', type=float, default=5.0)
parser.add_argument('--use-ema', default=True, type=str2bool, help='Maintain Polyak EMA shadow weights from Epoch 2+')
parser.add_argument('--ema-decay', default=0.997, type=float)

# Argon-4 Gradient-Conserving SOTA Loss Weights
parser.add_argument('--loss2-weight', '--loss2_weight', dest='loss2_weight', type=float, default=5.0, help='CLASM multi-class vision-language alignment weight for XD-Violence')
parser.add_argument('--loss-sep-weight', '--loss_sep_weight', dest='loss_sep_weight', type=float, default=1.0, help='Normal-Abnormal text orthogonality weight')
parser.add_argument('--loss-residual-weight', '--lambda-residual', dest='loss_residual_weight', type=float, default=1.0, help='Counterfactual Residual Event Alignment (RUPA_RESIDUAL_EVENT) weight')
parser.add_argument('--loss-reconstructed-normal-weight', '--lambda-bkg', dest='loss_reconstructed_normal_weight', type=float, default=1.0, help='Video-level Background Alignment (CLASM_BKG) weight')
parser.add_argument('--loss-consistency-weight', '--lambda-consistency', dest='loss_consistency_weight', type=float, default=1.0, help='CLIP-Cone Normalized & Peak-Conserving Consistency weight')
parser.add_argument('--loss-gather-weight', '--lambda-gather', dest='loss_gather_weight', type=float, default=1.0, help='Reliability-Weighted Dynamic Normal Prototype Compactness (g_loss) weight')
parser.add_argument('--loss-dnp-normal-weight', default=0.0, type=float)
parser.add_argument('--loss-routing-weight', default=0.20, type=float, help='Detached SafeGate calibration MIL weight')
parser.add_argument('--loss-routing-anchor-weight', '--loss_routing_anchor_weight', dest='loss_routing_anchor_weight', type=float, default=0.05)
parser.add_argument('--lambda-iam', '--loss-iam-weight', dest='lambda_iam', default=0.0, type=float)
parser.add_argument('--lambda-acc', '--loss-acc-weight', dest='lambda_acc', default=0.0, type=float)
parser.add_argument('--lambda-aux', default=0.50, type=float, help='Decoupled Audio-Visual MIL + Normal Soundtrack Suppression weight')
parser.add_argument('--lambda-smooth', default=0.0, type=float)

# Dataset & Stratified Crop Epoch Sampling (3 distinct spatial crops/video/epoch -> ~185 steps/epoch, zero late-epoch memorization!)
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
parser.add_argument('--sample-one-crop-per-video', default=True, type=str2bool, help='Group by unique video ID to prevent 10x intra-epoch bag memorization')
parser.add_argument('--crops-per-video', default=5, type=int, help='Sample 5 disjoint spatial crops per video per epoch (19770 clips = 309 steps/epoch at BS=64, standard 5-crop XD-Violence paper protocol)')

# Evaluation & 5-Crop Visual-Protected Multimodal Inference
parser.add_argument('--multi-crop-eval', default=True, type=str2bool, help='Aggregate available spatial test crops (__0..__4) in parallel')
parser.add_argument('--max-test-crops', default=10, type=int, help='Maximum spatial test crops to search (auto-discovers all 5 crops on XD-Violence)')
parser.add_argument('--collab-alpha', default=0.03, type=float, help='Peak-preserving semantic + prototype visual synergy weight')
parser.add_argument('--aux-eval-alpha', default=0.04, type=float, help='Peak-preserving concordance-gated acoustic modulation weight')
parser.add_argument('--smooth-eval', default=True, type=str2bool)
parser.add_argument('--smooth-method', choices=['gaussian', 'savgol'], default='gaussian')
parser.add_argument('--smooth-sigma', default=1.00, type=float, help='Optimal Gaussian temporal smoothing sigma verified by Prober')
parser.add_argument('--eval-interval', default=150, type=int, help='Mid-epoch validation interval in iterations (150 = half of 309-step 5-crop epoch)')
parser.add_argument('--num-workers', default=2, type=int)
