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


parser = argparse.ArgumentParser(description='MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network (UCF-Crime)')
parser.add_argument('--seed', default=234, type=int, help='Random seed for reproducibility')

# Architecture Hyperparameters
parser.add_argument('--use-brst', default=True, type=str2bool, help='Enable Bi-Relational Spatio-Temporal Backbone (BR-STB)')
parser.add_argument('--embed-dim', default=512, type=int, help='Feature embedding dimension')
parser.add_argument('--visual-length', default=256, type=int, help='Maximum temporal snippets per video')
parser.add_argument('--visual-width', default=512, type=int, help='Visual feature width')
parser.add_argument('--visual-head', default=4, type=int, help='Attention heads in backbone & cross-attention')
parser.add_argument('--visual-layers', default=2, type=int, help='Number of transformer layers in Branch A')
parser.add_argument('--attn-window', default=8, type=int, help='Window size for Layer 0 local attention (default 8; Layer 1 uses continuous distance-decay global attention)')
parser.add_argument('--classes-num', default=14, type=int, help='Number of classes (13 anomalies + 1 normal)')
parser.add_argument('--mlp-dropout', default=0.1, type=float, help='Dropout rate for MLPs')

# RUPA Normality & Prototype Hyperparameters
parser.add_argument('--num_prototypes', type=int, default=16, help='Number of Dynamic Normal Prototypes (M=16)')
parser.add_argument('--decoder_depth', type=int, default=4, help='Prototype reconstruction decoder depth')
parser.add_argument('--sampling', choices=['topk', 'otsu'], default='topk', help='Pseudo-normal segment selection method')
parser.add_argument('--normal_selection_ratio', type=float, default=0.80, help='Selection ratio for top-k normal mining (0.80 matches DSANet SOTA)')
parser.add_argument('--min_normal_frames', type=int, default=4, help='Minimum normal frames to select per video')
parser.add_argument('--max_normal_ratio', type=float, default=0.85, help='Maximum normal frames ratio')
parser.add_argument('--DNP_use', default=True, type=str2bool, help='Whether to use Dynamic Normal Prototypes')

# Adapter & Temperature
parser.add_argument('--text_adapt_until', default=3, type=int, help='CLIP text adapter depth (default 3)')
parser.add_argument('--t_w', default=0.15, type=float, help='Text adapter residual interpolation weight (default 0.15 for UCF)')
parser.add_argument('--temp', default=5.0, type=float, help='Logit scaling temperature for evaluation refinement (default 5.0 matches DSANet)')

# Ablation & MA-RUPA v2 Breakthrough Flags
parser.add_argument('--no-mdtp', action='store_true', default=False, help='Ablation: disable Multi-Scale Dilated Temporal Pyramid')
parser.add_argument('--no-audio', action='store_true', default=False, help='Ablation: disable audio branch (forces motion residuals or visual only)')
parser.add_argument('--no-rupa', action='store_true', default=False, help='Ablation: disable RUPA prototype reconstruction branch')
parser.add_argument('--no-safegate', action='store_true', default=False, help='Ablation: disable dynamic uncertainty Safe Gate')
parser.add_argument('--use-aspm', default=True, type=str2bool, help='MA-RUPA v2: enable Soft-Masked Adaptive Stiefel Prototype Manifold (SM-ASPM)')
parser.add_argument('--aspm-max-rank', default=48, type=int, help='MA-RUPA v2: maximum adaptive Stiefel subspace rank M_max (default 48)')
parser.add_argument('--use-epts', default=True, type=str2bool, help='MA-RUPA v2: enable Semantic-Affinity Edge-Preserving Temporal Sharpness & Peak Erasure')
parser.add_argument('--use-tasl', default=True, type=str2bool, help='MA-RUPA v2: enable CLIP-Text-Anchored Semantic Lifting (TASL)')
parser.add_argument('--use-pcgrad', default=True, type=str2bool, help='MA-RUPA v2: enable Per-Snippet Multi-Objective PCGrad Junctions')
parser.add_argument('--routing-mode', choices=['fixed', 'safe_gate'], default='safe_gate', type=str)
parser.add_argument('--routing-gate-hidden', default=32, type=int, help='Hidden dimension of dynamic safe gate MLP')

# Prior weights for UCF-Crime
parser.add_argument('--routing-det-weight', default=0.50, type=float, help='Prior weight for detector stream')
parser.add_argument('--routing-rec-weight', default=0.05, type=float, help='Prior weight for reconstruction stream')
parser.add_argument('--routing-sem-weight', default=0.38, type=float, help='Prior weight for semantic stream')
parser.add_argument('--routing-aux-weight', default=0.07, type=float, help='Prior weight for auxiliary motion stream')

# Training & Optimization
parser.add_argument('--max-epoch', default=10, type=int, help='Total training epochs (default 10 for full dataset SOTA schedule ~2500 gradient steps)')
parser.add_argument('--batch-size', default=64, type=int, help='Batch size (32 normal + 32 anomaly)')
parser.add_argument('--lr', type=float, default=7e-5, help='Learning rate for End-to-End AdamW optimizer (default 7e-5 for BR-STB)')
parser.add_argument('--weight-decay', type=float, default=2e-3, help='Weight decay for regularization (default 2e-3)')
parser.add_argument('--k-min', type=int, default=16, help='Minimum top-k snippets for ATS MAF-MIL (default 16)')
parser.add_argument('--k-max', type=int, default=48, help='Maximum top-k snippets for ATS MAF-MIL (default 48)')
parser.add_argument('--margin-gamma', type=float, default=0.30, help='Margin gamma for intra-bag separation (default 0.30)')
parser.add_argument('--focal-alpha', type=float, default=2.0, help='Focal loss focusing parameter alpha (default 2.0)')
parser.add_argument('--lambda-mar', type=float, default=0.50, help='Weight for intra-bag margin loss (default 0.50)')
parser.add_argument('--curriculum-epochs', type=float, default=3.0, help='Epochs to smoothly ramp up RUPA losses via cosine schedule (default 3.0)')
parser.add_argument('--max-omega', default=0.20, type=float, help='Maximum curriculum omega ceiling (default 0.20 to prevent over-regularization)')
parser.add_argument('--topk-ratio', type=int, default=16, help='Base divisor for variance-adaptive top-k (VAT-K) selection (default 16 -> ~6.25%% frames for sparse CCTV crimes)')
parser.add_argument('--topk-ratio-clasm', type=int, default=16, help='Top-k divisor for snippet multi-class classification (default 16)')
parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing factor for binary MIL loss (default 0.0; AT-BCE + OHNM prevents collapse)')
parser.add_argument('--snippet-dropout', type=float, default=0.15, help='Snippet dropout rate during training to prevent 1-frame spike collapse (default 0.15)')
parser.add_argument('--sample-one-crop-per-video', default=False, type=str2bool, help='Train on full 10 crops per video (default False for Full 10-Epoch SOTA schedule)')
parser.add_argument('--crops-per-video', default=2, type=int, help='DPSS: Number of random crops sampled per video per epoch (default 2 for 50 iters/epoch)')
parser.add_argument('--vatk', default=True, type=str2bool, help='Whether to use Variance-Adaptive Top-K selection (default True)')
parser.add_argument('--gamma-edge', default=5.0, type=float, help='Edge-preserving sensitivity for SA-EPTS smoothness loss (default 5.0)')
parser.add_argument('--lambda-entropy', default=0.02, type=float, help='Soft maximum entropy regularization weight (default 0.02)')
parser.add_argument('--lambda-ohnm', default=0.2, type=float, help='Online hard negative mining weight for normal bags (default 0.2)')

# Loss Weighting (Pareto optimal weighting for simultaneous Detector and RUPA performance)
parser.add_argument('--lambda-aux', type=float, default=0.20, help='Weight for auxiliary branch MIL loss (default 0.20 for UCF-Crime)')
parser.add_argument('--lambda-tasl', type=float, default=0.10, help='Weight for TASL cross-modal semantic alignment loss (default 0.10 for UCF-Crime)')
parser.add_argument('--lambda-smooth', type=float, default=0.08, help='Weight for Total Variation temporal smoothness loss (default 0.08)')
parser.add_argument('--lambda-orth', type=float, default=0.05, help='Weight for active-slot prototype orthogonality & repulsion loss (default 0.05)')
parser.add_argument('--lambda-gather', type=float, default=0.12, help='Weight for DNP subspace gather loss (default 0.12)')
parser.add_argument('--lambda-residual', type=float, default=0.05, help='Weight for residual fine-grained event alignment loss (default 0.05)')
parser.add_argument('--lambda-gate', type=float, default=0.01, help='Weight for safe gate KL regularization loss (default 0.01)')
parser.add_argument('--loss2_weight', type=float, default=1.2, help='Weight for snippet multi-class classification loss (default 1.2)')
parser.add_argument('--lambda-consistency', type=float, default=0.15, help='Weight for confidence-gated consistency loss between detector and RUPA (default 0.15)')
parser.add_argument('--lambda-bkg', type=float, default=0.5, help='Weight for background normal loss (default 0.5)')

# Data & Paths
parser.add_argument('--train-list', default='list/ucf_CLIP_rgb.csv', help='Path to training feature CSV list')
parser.add_argument('--test-list', default='list/ucf_CLIP_rgbtest.csv', help='Path to test feature CSV list')
parser.add_argument('--gt-path', default='list/gt_ucf.npy', help='Path to ground truth frame labels')
parser.add_argument('--gt-segment-path', default='list/gt_segment_ucf.npy', help='Path to ground truth segments')
parser.add_argument('--gt-label-path', default='list/gt_label_ucf.npy', help='Path to ground truth video labels')
parser.add_argument('--data-dir', default=None, type=str, help='Base directory to resolve/remap feature file paths')
parser.add_argument('--model-path', default='model/model_ucf.pth', help='Path to save best model checkpoint')
parser.add_argument('--checkpoint-path', default='model/checkpoint_ucf.pth', help='Path to save latest checkpoint')
parser.add_argument('--use-checkpoint', default=False, type=str2bool, help='Resume from checkpoint')

# Post-processing & Multi-crop Test Evaluation
parser.add_argument('--smooth-eval', default=True, type=str2bool, help='Apply temporal score smoothing during evaluation')
parser.add_argument('--smooth-method', choices=['gaussian', 'savgol'], default='gaussian', help='Temporal smoothing filter method')
parser.add_argument('--smooth-sigma', default=0.50, type=float, help='Gaussian filter sigma for score smoothing (default 0.50 preserves sharp CCTV event peaks)')
parser.add_argument('--eval-interval', default=1280, type=int, help='Step interval for mid-epoch evaluation')
parser.add_argument('--num-workers', default=2, type=int, help='DataLoader workers')
parser.add_argument('--grad-clip', default=1.0, type=float, help='Max gradient norm for clipping')
