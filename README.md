# MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MA-RUPA** (Modality-Adaptive Multiscale Prototype Alignment Network) is a novel, standalone, **End-to-End trainable** architecture designed for Weakly-Supervised Video Anomaly Detection (WS-VAD) targeting top-tier computer vision and multimedia journals (**IEEE TIP, IEEE TMM, Pattern Recognition**).

---

## 🚀 Key Innovations & Architectural Upgrades

Unlike legacy methods that rely on frozen two-stage warm-starting (e.g., freezing DSANet GCN checkpoints with `main-lr = 0.0`), MA-RUPA is **completely independent of legacy DSANet checkpoints** and trains from scratch in a **single stage** without gradient collapse or early-peak degradation.

```
Visual CLIP (512-D) ───► [ MDTBackbone (Branch A: Windowed Attn + Branch B: MDTP Dilations 1,2,4) ] ───► S_det
                                        │
                                        ▼
Audio/Motion ──────────► [ ModalityAdaptiveBranch (Cross-Attn + Noise Suppression Gate G_av) ] ────► S_aux
                                        │
                                        ▼
Detached Normals ──────► [ GradientSafeRUPA (M=16 Dynamic Normal Prototypes + R_sem) ] ────────► S_rec, S_sem
                                        │
                                        ▼
Quad-Stream ───────────► [ QuadSafeGate: P_routing = S_det + g * (S_fixed - S_det) ] ──────────► Final Prediction
```

### 1. Multi-Scale Dilated Temporal & Local Attention Backbone (`MDTBackbone`)
- **Branch A (Local Windowed Transformer):** Captures short-range snippet context with $O(T \cdot W)$ complexity ($W=8$ for UCF-Crime, $W=64$ for XD-Violence).
- **Branch B (Multi-Scale Dilated Temporal Pyramid - MDTP):** Employs parallel 1D Depthwise-Separable Convolutions with dilation rates $d \in \{1, 2, 4\}$ and Gated Linear Units (GLU) to jointly detect rapid transient anomalies (explosions, gunfire) and prolonged anomalies (riots, fighting).
- **Channel Attention Gate:** Dynamically aggregates Branch A and Branch B representations into $F_{base}$ and computes base detector scores $S_{det}$.

### 2. Modality-Adaptive Audio-Visual / Motion Residual Branch (`ModalityAdaptiveBranch`)
- **XD-Violence Multimodal Mode:** Projects pre-extracted audio features (128-D VGGish or 512-D CLAP/AST) to 512-D with temporal interpolation alignment.
- **UCF-Crime Motion-Residual Fallback:** For silent CCTV videos or missing audio, automatically computes 1st- and 2nd-order temporal visual residuals ($M_1 = X_t - X_{t-1}, M_2 = M_1_t - M1_{t-1}$) as an auxiliary dynamic modality, preserving a unified architecture across datasets.
- **Reliability-Aware Cross-Modal Gated Attention:** Bidirectional attention between visual $F_{base}$ and auxiliary $F_a$ modulated by a dynamic Noise Suppression Gate $G_{av} = \sigma(\text{MLP}([F_{base} \parallel F_a \parallel (F_{base} \odot F_a)]))$ to filter out ambient background audio noise.
- Generates auxiliary score stream $S_{aux} = \sigma(\text{Classifier}_{aux}(F_{fused}))$.

### 3. Gradient-Safe Prototype Alignment (`GradientSafeRUPA`)
- **Controlled Gradient Detachment:** Dynamically isolates pseudo-normal snippet representations via Top-K or Otsu thresholding and detaches their gradients (`X_norm.detach()`) before feeding the Dynamic Normal Prototype (DNP) memory ($M=16$). This prevents noisy early prototype gradients from corrupting the backbone.
- **Counterfactual Normal Reconstruction:** Reconstructs normal features $F_{rec}$ and extracts anomaly residual $R = F_{fused} - F_{rec}$ with reconstruction score $S_{rec} = (1 - \cos(F_{fused}, F_{rec})) / 2$.
- **Domain-Adaptive Semantic Alignment:** Learns fine-grained category semantics from residual $R$ with learnable temperature $\tau$ and feature variance damping to prevent hallucinated anomalies on blurry CCTV frames.

### 4. Quad-Stream Dynamic Safe Gate (`QuadSafeGate`)
- Routes all 4 distinct evidence streams ($S_{det}, S_{rec}, S_{sem}, S_{aux}$).
- **Static Prior Mixture:** Softmax-constrained weighted mixture $S_{fixed} = \alpha S_{det} + \beta S_{rec} + \gamma S_{sem} + \delta S_{aux}$.
- **Dynamic Uncertainty MLP Gate:** Evaluates an 8-D snippet uncertainty vector $[S_{det}, S_{rec}, S_{sem}, S_{aux}, |S_{rec}-S_{det}|, |S_{sem}-S_{det}|, |S_{aux}-S_{det}|, \text{Var}(S)]$.
- Negative bias initialization ($\text{bias} = -4.5 \implies g \approx 0.01$) ensures conservative gate opening during early training.

### 5. Curriculum Ramp-Up Loss Schedule
$$\mathcal{L}_{total} = \mathcal{L}_{MIL}(P_{routing}) + \lambda_{aux}\mathcal{L}_{MIL}(S_{aux}) + \mathcal{L}_{smooth} + \omega(e) \cdot \left( \lambda_1 \mathcal{L}_{Orth} + \lambda_2 \mathcal{L}_{Gather} + \lambda_3 \mathcal{L}_{Residual} + \lambda_4 \mathcal{L}_{Gate} \right)$$
where $\omega(e) = \min(1.0, \frac{e}{3.0})$ gradually introduces prototype pressure only after backbone ranking stabilizes.

---

## 🏆 Benchmark Targets & Results

| Dataset | Modality | Primary Metric | Target / Result |
|---|---|---|---|
| **UCF-Crime** | Visual + Motion Residuals | **Frame-level ROC-AUC** | **89.5% – 90.0%+** (AP ~40%+) |
| **XD-Violence** | Visual + Audio (VGGish/CLAP) | **Frame-level AP** | **86.5% – 87.5%+** (AUC ~95.5%+) |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Sharvuz/RUPA-DSA-v0.git
cd RUPA-DSA_final

# Create virtual environment (using uv or conda)
uv venv .venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Training (Single-Stage End-to-End from Scratch)

### 1. UCF-Crime
```bash
python src/ucf_train.py \
    --train-list list/ucf_CLIP_rgb.csv \
    --test-list list/ucf_CLIP_rgbtest.csv \
    --gt-path list/gt_ucf.npy \
    --gt-segment-path list/gt_segment_ucf.npy \
    --gt-label-path list/gt_label_ucf.npy \
    --batch-size 64 \
    --lr 2e-4 \
    --max-epoch 12 \
    --curriculum-epochs 3.0 \
    --sampling topk \
    --model-path model/model_ucf.pth
```

### 2. XD-Violence (with Multimodal Audio)
```bash
python src/xd_train.py \
    --train-list list/xd_CLIP_rgb.csv \
    --test-list list/xd_CLIP_rgbtest.csv \
    --gt-path list/gt.npy \
    --gt-segment-path list/gt_segment.npy \
    --gt-label-path list/gt_label.npy \
    --audio-dim 128 \
    --batch-size 64 \
    --lr 2e-4 \
    --max-epoch 15 \
    --curriculum-epochs 3.0 \
    --sampling topk \
    --model-path model/model_xd.pth
```

---

## 🔬 Evaluation & Testing

Run evaluation on test splits with automatic multi-crop handling and temporal score smoothing:

```bash
# UCF-Crime Evaluation
python src/ucf_test.py \
    --test-list list/ucf_CLIP_rgbtest.csv \
    --gt-path list/gt_ucf.npy \
    --gt-segment-path list/gt_segment_ucf.npy \
    --gt-label-path list/gt_label_ucf.npy \
    --model-path model/model_ucf.pth \
    --smooth-eval True \
    --smooth-method gaussian \
    --smooth-sigma 1.5

# XD-Violence Evaluation
python src/xd_test.py \
    --test-list list/xd_CLIP_rgbtest.csv \
    --gt-path list/gt.npy \
    --gt-segment-path list/gt_segment.npy \
    --gt-label-path list/gt_label.npy \
    --model-path model/model_xd.pth \
    --smooth-eval True \
    --smooth-method gaussian \
    --smooth-sigma 1.5
```

---

## 🧪 Ablation Studies (For Q1 Journal Submission)

Every component can be toggled via command-line flags to reproduce paper ablation tables:

| Experiment / Table | Command Flags |
|---|---|
| **w/o MDTP (Branch B disabled)** | `--no-mdtp` |
| **w/o Audio Branch (Motion only)** | `--no-audio` |
| **w/o RUPA (Detector only)** | `--no-rupa` |
| **w/o Dynamic Safe Gate (Static mixture)** | `--no-safegate` |
| **Otsu vs. Top-K Normal Selection** | `--sampling otsu` vs. `--sampling topk` |
| **w/o Temporal Smoothing** | `--smooth-eval False` |

Example ablation command:
```bash
python src/xd_train.py --no-audio --no-mdtp --sampling otsu
```

---

## 📁 Repository Structure

```
├── src/
│   ├── model.py              # Full MA-RUPA model (MDTBackbone, ModalityAdaptiveBranch, GradientSafeRUPA, QuadSafeGate)
│   ├── ucf_train.py          # End-to-End training script for UCF-Crime
│   ├── ucf_test.py           # Evaluation script for UCF-Crime
│   ├── ucf_option.py         # Argument parser and configs for UCF-Crime
│   ├── xd_train.py           # End-to-End multimodal training script for XD-Violence
│   ├── xd_test.py            # Evaluation script for XD-Violence
│   ├── xd_option.py          # Argument parser and configs for XD-Violence
│   ├── clip/                 # CLIP ViT-B/16 text model and tokenizers
│   └── utils/
│       ├── dataset.py        # Multimodal UCF & XD dataset loaders with audio alignment
│       ├── tools.py          # Processing utilities, xd_collate_fn, and temporal smoothing
│       ├── descriptions.py   # Text prompts for anomaly categories
│       └── ...
├── tests/
│   ├── test_marupa.py        # Unit tests for all modules and backward gradient flow
│   └── test_train_pipeline.py# Integration tests for complete train and eval loops
├── list/                     # Ground truth annotations and CSV file lists
└── README.md
```

---

## 📜 Citation
If you use MA-RUPA in your research, please cite:
```bibtex
@article{marupa2026,
  title={MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network for Weakly-Supervised Video Anomaly Detection},
  author={Huynh, Quang-Vu and Collaborators},
  journal={IEEE Transactions on Image Processing (under review)},
  year={2026}
}
```
