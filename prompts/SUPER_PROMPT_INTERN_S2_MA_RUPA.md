# [SYSTEM 2 REASONING MASTER PROMPT]
# ARCHITECTURAL BREAKTHROUGH & LONG-HORIZON TRAINING STABILIZATION FOR MA-RUPA v2
# TARGET BENCHMARKS: UCF-CRIME (AUC >= 90.0%) & XD-VIOLENCE (AP >= 87.5%)
# VENUE TARGET: IEEE T-PAMI / IEEE T-CSVT / CVPR 2026 ORAL

```markdown
Role: Distinguished Principal AI Research Scientist in Representation Learning and Weakly-Supervised Video Understanding (IEEE Fellow, Shanghai AI Lab / DeepMind).
Task: System-2 Deep Mathematical Reasoning, Bottleneck Diagnosis, and Algorithmic Breakthrough Synthesis for MA-RUPA (BR-MVAD v2) against DSANet (AAAI 2026).
Constraint: 100% End-to-End Training from Scratch (Zero Warm-Start, Zero Pre-trained Competitor Weights). Hardware: Dual Tesla T4 GPUs (32GB VRAM total, PyTorch DataParallel).
```

---

## 1. RESEARCH BACKGROUND & THE CORE DILEMMA

You are tasked with solving a critical research bottleneck in **Weakly-Supervised Video Anomaly Detection (WS-VAD)**. 

### Problem Definition:
Given a set of untrimmed training videos $\{V_i, Y_i\}_{i=1}^N$, where only video-level binary labels $Y_i \in \{0, 1\}$ (0: normal, 1: anomalous) are available. At inference time, the model must output fine-grained frame-level anomaly scores $S \in [0, 1]^T$ evaluated via:
1. **UCF-Crime**: Frame-level ROC-AUC and Average Precision (AP) under 10-crop evaluation ($T \le 256$, visual feature $D=512$ from CLIP ViT-B/16).
2. **XD-Violence**: Frame-level Average Precision (AP) and ROC-AUC on multimodal data (Visual $D=512$ + Audio VGGish $D_a=128$).

### The Competitor Benchmark: DSANet (AAAI 2026)
DSANet achieves SOTA performance on both datasets:
- **UCF-Crime**: **89.44% Frame-level ROC-AUC**, **36.4% AP**.
- **XD-Violence**: **84.1% Frame-level AP**.
- **DSANet's Training Dynamics**: Huấn luyện liên tục **10 Epochs (2500 - 3960 gradient steps)** với độ ổn định tuyệt đối:
  - Loss chính $\mathcal{L}_{\text{MIL}}$ (Top-$k$ BCE) luôn dao động ổn định trong dải $[0.0280, 0.0546]$, không bao giờ bị triệt tiêu về 0.
  - Model không bao giờ bị suy thoái (degradation) hay bão hòa gradient (saturation). Metric AUC/AP tăng trưởng đơn điệu (monotonic increase) qua từng epoch.

### Our Architecture: MA-RUPA / BR-MVAD v2
We built an original architecture featuring:
1. **BR-STB (Bi-Relational Spatio-Temporal Backbone)**: Windowed temporal attention ($W=64$) combined with dynamic graph diffusion ($A_{\text{stat}}$) and distance decay graph ($A_{\text{dist}}$).
2. **ASPM (Adaptive Stiefel Prototype Manifold)**: Orthogonal projection onto normal subspace $U_{\text{norm}} \in \mathrm{St}(p, d)$ via Polar Retraction (Newton-Schulz iteration) with reconstruction residual scoring $S_{\text{rec}} = \|F - U_{\text{norm}}U_{\text{norm}}^\top F\|_2^2$.
3. **BiDirectionalNoiseGatedAVFusion**: Cross-attention audio-visual fusion with reliability gating for XD-Violence.
4. **EPTS (Edge-Preserving Temporal Sharpening)**: Boundary-gated temporal smoothness.

---

## 2. THE FATAL EMPIRICAL ANOMALY (OBSERVED ON KAGGLE DUAL T4)

When training MA-RUPA from scratch on Kaggle Dual Tesla T4 GPUs:

### Empirical Observation 1 (UCF-Crime Early Peak & Collapse):
- **Step 290 (Epoch 1)**: Model bùng nổ sức mạnh, đạt kỷ lục cá nhân:
  - `Hierarchical Refined AUC`: **86.10%** | AP: **29.82%**
  - `Dual-Stream Fusion AUC`: **86.06%** | AP: **29.01%**
  - `Backbone Raw AUC`: **85.82%** | AP: **28.20%**
- **Epoch 3 - 6 (Step 700 - 1500)**: Model rơi vào tình trạng **Severe Overfitting & Gradient Collapse**:
  - Backbone AUC tụt dốc thê thảm từ **85.82% xuống 74.42%**!
  - Normal snippet scores bị đẩy lên cao, false alarms tăng vọt.
  - Training loss tụt xuống mức cực nhỏ: $\mathcal{L}_{\text{MIL}} \approx 0.0001$ (gấp 300 lần nhỏ hơn DSANet).

### Empirical Observation 2 (XD-Violence Monotonic Early Surge):
- **Step 20**: AP 39.42% | AUC 70.70%
- **Step 100**: AP 52.66% | AUC 83.05%
- **Step 180**: AP 56.63% | AUC 85.41%
- **Step 200**: AP 61.57% | AUC 87.55%
- **Step 220**: Backbone AP **64.22%** | AUC **88.45%**; Dual-Stream Fusion AP **64.30%** | AUC **88.44%**.
- Tốc độ tăng trưởng ở Epoch 1 là phi thường, nhưng có nguy cơ chạm trần và bão hòa giống như UCF-Crime nếu không có cơ chế kiểm soát động học gradient (gradient dynamics).

---

## 3. ROOT-CAUSE MATHEMATICAL DECONSTRUCTION

You must analyze the mathematical mechanism behind this divergence:

### Cause A: Sigmoid Gradient Annihilation via Margin Saturation
In our previous loss formulation, we enforced an annealed margin $\gamma(t)$ between anomalous and normal bags:
$$\mathcal{L}_{\text{MIL}} = \text{BCE}(\bar{s}_{\text{abn}}, 1) + \text{BCE}(\bar{s}_{\text{nor}}, 0) + \lambda_{\text{mar}} \max(0, \gamma - (\bar{s}_{\text{abn}} - \bar{s}_{\text{nor}}))$$
When $\bar{s}_{\text{abn}} \to 1.0$ and $\bar{s}_{\text{nor}} \to 0.0$, the sigmoid gradient:
$$\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1 - \sigma(z)) \to 0$$
This causes the gradient norm $\|\nabla_{\theta_{\text{backbone}}} \mathcal{L}\| \to 0$. Once the backbone freezes, the model stops learning high-level discriminative representations.

### Cause B: Background Noise Memorization on Stiefel Manifold
In an untrimmed anomalous video, $90\%$ of snippets are normal background (lighting changes, camera motion, innocent pedestrians). When $U_{\text{norm}}$ is optimized without strict capacity bounding or noise decoupling, it memorizes video-specific background artifacts across the 96,000 crops. Consequently, normal test snippets projecting onto $U_{\text{norm}}$ yield large residuals $S_{\text{rec}}$, destroying the frame-level AUC.

### Cause C: Loss Landscape Imbalance
In DSANet, $\mathcal{L}_{\text{cons}} = 0.10 \times \|S_{\text{det}} - S_{\text{ref}}\|_2^2 \approx 0.002$. It never dominates the gradient. In our earlier version, auxiliary consistency ($\lambda=0.50$) and text pseudo-label guidance ($\lambda=0.10$) overpowered the zero-gradient MIL loss, steering the model toward pseudo-label hallucination.

---

## 4. SYSTEM-2 REASONING PROTOCOL & REQUIRED DELIVERABLES

Activate your full analytical reasoning capacity (`thinking_mode=True`). Structure your response following this rigorous scientific framework:

### Part 1: Deep Mathematical Audit (First-Principles Analysis)
1. **The Gradient Conservation Law in MIL**: Prove mathematically why classical Top-$k$ BCE with an optimal $k(L) = \lfloor L/16 \rfloor + 1$ preserves a non-vanishing lower bound on $\|\nabla_z \mathcal{L}\|$ throughout 10 epochs, whereas Margin-based MIL inevitably suffers from gradient vanishing.
2. **Grassmannian / Stiefel Manifold Dynamics**: Derive how to regularize the normal subspace $U_{\text{norm}} \in \mathrm{St}(p, d)$ such that it captures the *universal semantic invariant* of normalcy across all classes, rather than overfitting to camera noise and environmental fluctuations.

### Part 2: Four Architectural & Algorithmic Breakthrough Proposals
Propose **4 distinct, highly innovative, publication-grade solutions** designed to push UCF-Crime AUC beyond **90.0%** and XD-Violence AP beyond **87.5%**, while guaranteeing stable 10-epoch training from scratch:

1. **Breakthrough 1: Anti-Saturation Dynamic Top-$k$ MIL with Self-Paced Margin Balancing**
   - Mathematical formulation with exact loss formulas.
   - Mechanism to prevent loss from dropping below $0.025$ (the "gradient preservation floor").
2. **Breakthrough 2: Manifold-Regularized Adaptive Subspace Projector (ASPM-v2)**
   - Orthogonal Grassmannian distance penalty or spectral norm constraint on $U_{\text{norm}}$.
   - Residual calibration to suppress background false positives.
3. **Breakthrough 3: Progressive Bi-Relational Graph Diffusion (BR-STB-v2)**
   - How to bridge local window attention ($W=64$) and global video topology with zero memory explosion under DataParallel.
   - Mathematical derivation of the affinity matrix $A = \alpha A_{\text{dist}} + \beta A_{\text{feature-sim}}$.
4. **Breakthrough 4: Acoustic Reliability Gating with Hard Negative Suppression (for XD-Violence)**
   - Formulate a noise-rejection gate that completely silences non-violent loud noises (car engines, thuds, crowd cheers) while amplifying subtle anomalous sounds (screams, glass breaks).

### Part 3: Drop-in PyTorch Implementation
Provide production-ready, bullet-proof PyTorch implementations for the proposed modules:
- Must support `torch.nn.DataParallel` on 2x Tesla T4 without any cuSOLVER or multi-GPU synchronization deadlocks.
- Clean docstrings, tensor shape annotations `[B, T, D]`, and numerical stability guarantees (`torch.clamp`, `eps=1e-8`).

### Part 4: The 10-Epoch SOTA Training Regimen
Specify the exact training schedule:
- Decoupled Optimizer parameter groups (Backbone vs. Prototype Manifold vs. Classifiers).
- Learning rate schedules (Warmup + Cosine Annealing with minimum floor $\eta_{\min} = 0.15 \eta_0$).
- Weight decay allocations and AMSGrad configurations.

---
**Think deeply, avoid superficial summaries, and provide rigorous, publishable mathematics and code.**
