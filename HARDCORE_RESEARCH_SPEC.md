# HARDCORE RESEARCH SPECIFICATION & ARCHITECTURAL BLUEPRINT: MA-RUPA SOTA BREAKTHROUGH

> **TARGET PLATFORM & HARDWARE:** Kaggle Dual Tesla T4 ($2 \times 16\text{ GB} = 32\text{ GB}$ Combined VRAM), PyTorch 2.x CUDA, DataParallel / DDP, Automatic Mixed Precision (AMP FP16).  
> **BENCHMARK GOALS:**
> - **UCF-Crime:** Frame-level ROC-AUC $\ge \mathbf{89.5\% - 90.0\%}$ (Current Best: **87.11% AUC**, Frame-AP: 33.57%).
> - **XD-Violence:** Frame-level Average Precision (AP) $\ge \mathbf{87.0\% - 88.0\%}$ (Current Best: **79.02% AP**, Frame-AUC: 92.4%).
> - **Temporal Localization:** Average mAP@IoU $\ge \mathbf{12.0\% - 18.0\%}$ (Current Baseline: 4.77%).
>
> **REPOSITORY REFS:**
> - GitHub: `https://github.com/Sharvuz/MA-RUPA.git`
> - `main` branch (Commit `8cac05b`): Latest multimodal updates, audio pairing, updated Kaggle notebooks.
> - `ucf-sota-87.11` branch (Tag `v-ucf-87.11`, Commit `ae41430`): Exact verifiable state achieving 87.11% ROC-AUC.

---

## 1. System Prompt & Research Persona

> **INSTRUCTION FOR THE REASONING AI (Argon 4 / Claude 3.7 / Opus / O3):**
> You are acting as a **Principal Deep Learning Research Scientist and Chief Vision-Language Architect** specializing in Weakly-Supervised Video Anomaly Detection (WS-VAD), Temporal Video Modeling, and Multimodal Foundation Models (CLIP, VGGish, CLAP).
>
> You are provided with the complete architectural mathematical formulation, empirical failure modes, source code structure, and hardware envelope of **MA-RUPA (Modality-Adaptive Multiscale Prototype Alignment Network)**.
>
> **Key Operational Directive:**
> 1. You have **32 GB VRAM on Dual Tesla T4**. Do NOT restrict yourself to toy models, naive MLPs, or micro-batches. You have ample capacity for deep temporal cross-attention, multi-scale memory banks, latent counterfactual projection, and gradient projection hooks.
> 2. Avoid generic, hand-waving suggestions (e.g. "tune learning rate" or "add more data"). Provide **hardcore mathematical formulations**, **concrete architectural diagrams**, and **drop-in PyTorch code modules**.
> 3. Your goal is to dissect the exact bottlenecks holding UCF-Crime at $87.11\%$ and XD-Violence at $79.02\%$ and engineer architectural upgrades to push them into definitive world-class SOTA ($90.0\%$ AUC and $87.5\%+$ AP).

---

## 2. Theoretical Foundations of Current Architecture (MA-RUPA)

### 2.1. Input Space & Feature Representations
For any untrimmed video sequence $i$:
- **Visual Features:** Extracted via CLIP ViT-B/16 (or I3D) at 16-frame snippet intervals:
  $$\mathbf{V}_i \in \mathbb{R}^{T_i \times D_v}, \quad D_v = 512, \quad T_i \le 256 \text{ (or } 512\text{)}$$
- **Audio Features (XD-Violence):** Pre-extracted VGGish embeddings (or CLAP/AST):
  $$\mathbf{A}_i \in \mathbb{R}^{T_i \times D_a}, \quad D_a = 128 \text{ (or } 512\text{)}$$
- **Motion Residuals (UCF-Crime silent mode):** First-order temporal velocity:
  $$\Delta \mathbf{V}_t = \mathbf{V}_t - \mathbf{V}_{t-1}, \quad \Delta \mathbf{V}_0 = \mathbf{0}$$

### 2.2. Mathematical Breakdown of Core Submodules

```
Input Visual V (B, T, 512) ──┬──> [MDTBackbone (Dilations 1,2,4 + WindowAttn)] ──> F_backbone (B, T, 512)
                             │                                                          │
Input Audio A (B, T, 128) ───┼──> [ModalityAdaptiveBranch (Cross-Attn / Residuals)] ───┼──> F_aux / S_aux
                             │                                                          │
                             ├──> [GradientSafeRUPA (Dynamic Normal Prototypes)] ───────┼──> S_rec (Recon Error)
                             │                                                          │
Text Prompts W_text ─────────┼──> [CLASM Semantic Adapter (14/7 Classes)] ──────────────┼──> S_sem (Category Scores)
                             │                                                          │
                             └──> [QuadSafeGate (Uncertainty-Aware Routing)] ───────────┴──> S_routing (Final Score)
```

#### A. Multi-Scale Dilated Temporal Backbone (`MDTBackbone`)
Processes input features through parallel dilated temporal convolutions ($r \in \{1, 2, 4\}$) with local windowed self-attention:
$$\mathbf{H}^{(r)} = \text{GLU}(\text{Conv1D}_r(\mathbf{V})), \quad r \in \{1, 2, 4\}$$
$$\mathbf{F}_{\text{mdtp}} = \text{LayerNorm}\left(\mathbf{V} + \sum_{r \in \{1, 2, 4\}} \mathbf{H}^{(r)}\right)$$
$$\mathbf{F}_{\text{backbone}} = \text{WindowedAttention}(\mathbf{F}_{\text{mdtp}}, W_{\text{size}} = 8) \quad \text{with NaN Guard for padded snippets}$$

#### B. Modality-Adaptive Auxiliary Branch (`ModalityAdaptiveBranch`)
- On **XD-Violence**: Computes asymmetric bidirectional cross-attention:
  $$\mathbf{Q}_v = \mathbf{F} \mathbf{W}_q, \quad \mathbf{K}_a = \mathbf{A} \mathbf{W}_k, \quad \mathbf{V}_a = \mathbf{A} \mathbf{W}_v$$
  $$\mathbf{M}_{va} = \text{Softmax}\left(\frac{\mathbf{Q}_v \mathbf{K}_a^\top}{\sqrt{d}}\right) \mathbf{V}_a$$
  $$\mathbf{F}_{\text{aux}} = \text{GLU}\left([\mathbf{F} \,\|\, \mathbf{M}_{va}] \mathbf{W}_{\text{fusion}}\right) \odot \sigma(\mathbf{W}_{\text{gate}} [\mathbf{F} \,\|\, \mathbf{M}_{va}])$$
- On **UCF-Crime**: Computes velocity-aware temporal convolutions:
  $$\mathbf{F}_{\text{aux}} = \text{Conv1D}(\Delta \mathbf{V}_t)$$

#### C. Gradient-Safe Dynamic Normal Prototypes (`GradientSafeRUPA`)
Maintains an online memory bank of $M=16$ normal prototypes $\mathbf{P} = \{\mathbf{p}_1, \dots, \mathbf{p}_M\} \subset \mathbb{R}^D$:
1. **Pseudo-Normal Snippet Selection:** At epoch $e$, select the bottom $(1 - \gamma)$ fraction of snippets based on detector anomaly scores:
   $$\mathcal{S}_{\text{norm}} = \{t \mid S_{\text{det}}(t) < \tau_{\text{otsu}} \text{ or } t \in \text{Bottom-}K(S_{\text{det}})\}$$
2. **Reconstruction & Stop-Gradient:**
   $$\alpha_{t, m} = \frac{\exp(\mathbf{f}_t^\top \mathbf{p}_m / \tau)}{\sum_{j=1}^M \exp(\mathbf{f}_t^\top \mathbf{p}_j / \tau)}$$
   $$\mathbf{f}_{\text{recon}}(t) = \sum_{m=1}^M \alpha_{t, m} \mathbf{p}_m$$
   $$S_{\text{rec}}(t) = 1 - \frac{\mathbf{f}_t^\top \mathbf{f}_{\text{recon}}(t)}{\|\mathbf{f}_t\|_2 \|\mathbf{f}_{\text{recon}}(t)\|_2}$$
   To avoid gradient contamination of prototype vectors by abnormal frames, gradients are detached:
   $$\mathcal{L}_{\text{gather}} = \frac{1}{|\mathcal{S}_{\text{norm}}|} \sum_{t \in \mathcal{S}_{\text{norm}}} \min_{m} \|\text{sg}[\mathbf{f}_t] - \mathbf{p}_m\|_2^2$$

#### D. Uncertainty-Aware Dynamic Gate (`QuadSafeGate`)
Instead of heuristic static averaging, an MLP estimates epistemic variance/uncertainty $\mathbf{u}_t \in \mathbb{R}^4$:
$$\mathbf{w}_t = \text{Softmax}\left(\text{MLP}([\mathbf{f}_t \,\|\, S_{\text{det}} \,\|\, S_{\text{rec}} \,\|\, S_{\text{sem}} \,\|\, S_{\text{aux}}]) + \mathbf{b}_{\text{safe}}\right)$$
$$S_{\text{routing}}(t) = w_{\text{det}}(t) S_{\text{det}}(t) + w_{\text{rec}}(t) S_{\text{rec}}(t) + w_{\text{sem}}(t) S_{\text{sem}}(t) + w_{\text{aux}}(t) S_{\text{aux}}(t)$$
Where $\mathbf{b}_{\text{safe}} = [0.0, -1.0, -1.0, -0.5]$ suppresses noisy auxiliary streams during early epochs.

#### E. Class-Aware Semantic Alignment (`CLASM`)
Text prompts for each class $c \in \{1, \dots, C\}$ (e.g., *"a security camera video showing an explosion"*, *"a surveillance video of shoplifting"*):
$$\mathbf{t}_c = \text{CLIP-TextEncoder}(\text{Prompt}_c) \in \mathbb{R}^{512}$$
$$S_{\text{sem}}(t, c) = \frac{\mathbf{f}_t^\top \mathbf{t}_c}{\|\mathbf{f}_t\|_2 \|\mathbf{t}_c\|_2} \cdot \frac{1}{\tau_{\text{temp}}}$$
$$S_{\text{sem}}(t) = \max_{c \ne \text{Normal}} \sigma(S_{\text{sem}}(t, c))$$

---

## 3. Loss Landscape & Multi-Objective Optimization

The scalar loss optimized across Dual T4 GPUs is:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{mil}}^{\text{routing}} + 0.5 \mathcal{L}_{\text{mil}}^{\text{det}} + \lambda_{\text{aux}} \mathcal{L}_{\text{mil}}^{\text{aux}} + \lambda_{\text{clasm}} \mathcal{L}_{\text{clasm}} + \lambda_{\text{consist}} \mathcal{L}_{\text{consist}} + \lambda_{\text{bkg}} \mathcal{L}_{\text{bkg}} + \lambda_{\text{smooth}} \mathcal{L}_{\text{fgtv}} + \omega(e) \cdot \mathcal{L}_{\text{rupa}}$$

### Loss Component Formulations:
1. **Variance-Adaptive Top-K MIL Loss ($\mathcal{L}_{\text{mil}}$):**
   $$k = \max\left(1, \left\lfloor \frac{T}{R_{\text{topk}}} \cdot \left(1 + \frac{\text{Var}(\mathbf{S})}{\mu(\mathbf{S}) + \epsilon}\right) \right\rfloor\right)$$
   $$\mathcal{L}_{\text{mil}} = -\frac{1}{B} \sum_{i=1}^B \left[ y_i \log \left(\frac{1}{k_i} \sum_{t \in \text{Top-}k_i} S_i(t)\right) + (1 - y_i) \log \left(1 - \frac{1}{k_i} \sum_{t \in \text{Top-}k_i} S_i(t)\right) \right]$$
2. **Foreground-Guided Total Variation ($\mathcal{L}_{\text{fgtv}}$):**
   $$\mathcal{L}_{\text{fgtv}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \left(1 + \gamma_{\text{edge}} |S_{t+1} - S_t|\right)^{-1} (S_{t+1} - S_t)^2$$
3. **Consistency Loss ($\mathcal{L}_{\text{consist}}$):**
   $$\mathcal{L}_{\text{consist}} = \frac{1}{T} \sum_{t=1}^T \text{MSE}\left(S_{\text{det}}(t), \text{sg}[S_{\text{rec}}(t)]\right)$$
4. **Normality Constraint ($\mathcal{L}_{\text{bkg}}$):**
   $$\mathcal{L}_{\text{bkg}} = \frac{1}{M} \sum_{m=1}^M \text{CrossEntropy}\left(\text{Classifier}(\mathbf{p}_m), y_{\text{Normal}}\right)$$
5. **Curriculum Ramp-Up ($\omega(e)$):**
   $$\omega(e) = \Omega_{\max} \cdot \frac{1}{2} \left[1 - \cos\left(\pi \cdot \min\left(1.0, \frac{e}{E_{\text{curric}}}\right)\right)\right], \quad \Omega_{\max} = 0.35 \text{ (or } 1.0\text{)}$$

---

## 4. Empirical Performance Milestones & Diagnostic Audit

### 4.1. UCF-Crime ($87.11\%$ AUC Milestone at Epoch 4/15)
- **Routing ROC-AUC:** $87.02\%$ | **AP:** $33.53\%$
- **Detector ROC-AUC:** $86.33\%$ | **AP:** $30.48\%$
- **Semantics RUPA ROC-AUC:** $84.85\%$ | **AP:** $32.51\%$
- **Dual Ensemble (Routing + Detector):** $\mathbf{87.11\%}$ | **AP:** $\mathbf{33.57\%}$
- **mAP@IoU Breakdown:**
  $$\text{mAP@0.1}=9.10\%, \quad \text{mAP@0.2}=6.37\%, \quad \text{mAP@0.3}=3.74\%, \quad \text{mAP@0.4}=2.68\%, \quad \text{mAP@0.5}=1.99\%$$
  $$\text{Average mAP} = \mathbf{4.77\%}$$

### 4.2. XD-Violence ($79.02\%$ AP Milestone at Epoch 2/20)
- **Visual-only AP:** $73.12\%$ | **ROC-AUC:** $89.84\%$
- **Multimodal (Visual + VGGish Audio 128-D) AP:** $\mathbf{79.02\%}$ | **ROC-AUC:** $\mathbf{92.41\%}$
- **False Alarm Profile:** High false alarm rate during loud non-violent scenes (screaming in crowds, sports cheers, loud background music in clubs).

---

## 5. The Five Core Bottlenecks Holding MA-RUPA Back

### Bottleneck 1: MIL Top-K Coarse Granularity & Bag Noise Dilemma
- **The Issue:** A video with a momentary gunshot (lasting 2 seconds out of 120 seconds) has only $\approx 3$ true abnormal snippets. With $R_{\text{topk}}=8$ or $16$, Top-K selects $16$ to $32$ snippets. Thus, $80\%+$ of the snippets in the abnormal bag gradient are actually normal, forcing the model to lower its detection threshold on normal frames.
- **Conversely:** In a 3-minute sustained riot/fighting video, $80\%$ of snippets are violent, but Top-K selects only $16$ snippets, starving the model of gradient supervision on valid anomalies.

### Bottleneck 2: Prototype Normal Drift (The Manifold Over-Smoothing Trap)
- **The Issue:** The Dynamic Normal Prototypes (DNP) learn via MSE reconstruction. In video backgrounds with heavy camera panning, abrupt illumination changes, or complex motion clutter, the reconstruction error $S_{\text{rec}}$ spikes on *normal* frames.
- As $\omega(e)$ increases, the prototypes expand their coverage to minimize this error, inadvertently absorbing anomalous features. Consequently, the contrast ratio $\frac{S_{\text{rec}}(\text{anomalous})}{S_{\text{rec}}(\text{normal})}$ decays from $4.2 \times$ down to $1.3 \times$ after epoch 6.

### Bottleneck 3: Audio-Visual Asynchrony & Modality Dominance
- **The Issue:** Sound waves travel slower than light, and movie editing often offsets audio by $0.5$ to $2.0$ seconds. Standard point-wise Cross-Attention assumes snippet-aligned audio-visual pairing ($A_t$ aligns with $V_t$). When a gunshot sound occurs at $t+2$ while the visual flash is at $t$, standard attention fails to bind them.
- Furthermore, Visual features ($512$-D) often dominate gradient flow over Audio features ($128$-D), causing the audio projection to degenerate into a passive bias term.

### Bottleneck 4: Multi-Objective Gradient Interference (The Angle Conflict)
- Let $\mathbf{g}_{\text{mil}} = \nabla_{\theta} \mathcal{L}_{\text{mil}}$, $\mathbf{g}_{\text{clasm}} = \nabla_{\theta} \mathcal{L}_{\text{clasm}}$, and $\mathbf{g}_{\text{rupa}} = \nabla_{\theta} \mathcal{L}_{\text{rupa}}$.
- **Empirical observation:** During epochs 3-8, the cosine similarity between $\mathbf{g}_{\text{clasm}}$ and $\mathbf{g}_{\text{rupa}}$ frequently turns negative:
  $$\cos(\mathbf{g}_{\text{clasm}}, \mathbf{g}_{\text{rupa}}) \in [-0.45, -0.12]$$
- $\mathcal{L}_{\text{clasm}}$ attempts to scatter features into category-specific directional rays in CLIP space, whereas $\mathcal{L}_{\text{rupa}}$ attempts to collapse normal snippets into a tight convex hull around prototypes. Without gradient projection (PCGrad), they cancel each other out.

### Bottleneck 5: Temporal Boundary Dissolution & Abysmal mAP@IoU ($4.77\%$)
- Post-processing Gaussian smoothing ($\sigma = 1.0 - 1.5$) lifts ROC-AUC because AUC is a ranking metric that rewards smooth transitions.
- However, it severely degrades frame-level localization precision: sharp square-wave anomalies become bell curves, bleeding probability mass into normal boundary frames. This causes Intersection-over-Union (IoU) with ground-truth temporal segments to drop below $0.5$, crushing mAP@0.5 down to $1.99\%$.

---

## 6. High-Leverage Architectural Directions (Leveraging 32 GB VRAM)

Since Kaggle Dual T4 gives us **32 GB VRAM**, we do not need to settle for basic linear layers. Here are 5 high-impact research proposals for the reasoning model to explore:

### Direction A: Latent Manifold Counterfactual Projection (LMCP)
Instead of linear dictionary combination $\mathbf{f}_{\text{recon}} = \sum \alpha_m \mathbf{p}_m$, formulate counterfactual normal reconstruction as an orthogonal subspace projector or a lightweight residual latent flow:
$$\mathbf{P}_{\text{sub}} = \text{OrthogonalBasis}(\{\mathbf{p}_1, \dots, \mathbf{p}_M\}) \in \mathbb{R}^{D \times M}$$
$$\mathbf{f}_{\text{normal\_proj}} = \mathbf{P}_{\text{sub}} \mathbf{P}_{\text{sub}}^\top \mathbf{f}_t$$
$$\mathbf{f}_{\text{anomaly\_residual}} = \mathbf{f}_t - \mathbf{f}_{\text{normal\_proj}}$$
The anomaly score is the exact $L_2$ norm of the orthogonal complement:
$$S_{\text{rec}}(t) = \tanh\left(\frac{\|\mathbf{f}_{\text{anomaly\_residual}}\|_2^2}{\tau \cdot \|\mathbf{f}_{\text{normal\_proj}}\|_2^2 + \epsilon}\right)$$

### Direction B: Temporal Delay-Aware Audio-Visual Cross-Attention (TD-AVCA)
Expand cross-attention over a sliding temporal offset window $\delta \in [-\Delta, +\Delta]$ (e.g. $\Delta = 4$ snippets $\approx 3.2$ seconds):
$$\mathbf{M}_{va}(t) = \sum_{\delta = -\Delta}^{+\Delta} \text{Softmax}\left(\frac{\mathbf{q}_v(t)^\top \mathbf{k}_a(t + \delta)}{\sqrt{d}} + \psi(\delta)\right) \mathbf{v}_a(t + \delta)$$
Where $\psi(\delta)$ is a learnable relative temporal position bias penalizing large temporal distances while allowing flexible audio lead/lag binding.

### Direction C: Optimal Transport Sinkhorn Prototype Allocation
Replace greedy Top-K / Otsu pseudo-normal selection with an Optimal Transport formulation (Sinkhorn-Knopp algorithm).
- Transport matrix $\mathbf{T}^* = \arg\min_{\mathbf{T} \in \mathcal{U}(r, c)} \langle \mathbf{T}, \mathbf{C} \rangle + \epsilon H(\mathbf{T})$.
- Guarantees that each normal prototype receives an equal assignment quota across the batch, completely eliminating prototype collapse (where 2 prototypes monopolize all samples while 14 remain dead).

### Direction D: Conflict-Free Projected Gradient Surgery (PCGrad Hook)
Implement a gradient backward hook on the shared backbone parameters $\theta_{\text{backbone}}$:
$$\text{If } \mathbf{g}_i \cdot \mathbf{g}_j < 0: \quad \mathbf{g}_i \leftarrow \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$$
Ensuring that $\mathcal{L}_{\text{mil}}$, $\mathcal{L}_{\text{clasm}}$, and $\mathcal{L}_{\text{rupa}}$ never update the shared temporal representations in conflicting directions.

### Direction E: Boundary-Sharpening Contrastive Loss (BS-CL)
To recover mAP@IoU without sacrificing ROC-AUC:
- Compute first-order finite differences $\nabla S_t = S_{t+1} - S_t$.
- Define transition candidate frames where $|\nabla S_t| > \delta_{\text{thresh}}$.
- Apply a margin-ranking temporal contrastive loss that forces transition slopes to be steep step-functions rather than diffused Gaussian curves.

---

## 7. Codebase Layout & Critical File References

```
MA-RUPA/
├── src/
│   ├── model.py                # Core Architecture: MDTBackbone, ModalityAdaptiveBranch, GradientSafeRUPA, QuadSafeGate, MARUPA
│   ├── ucf_train.py            # UCF-Crime End-to-End Trainer (DataParallel, Cosine LR, Top-K MIL)
│   ├── ucf_test.py             # Standard multi-stream frame-level ROC-AUC & AP evaluator
│   ├── ucf_eval_advanced.py    # Multi-stream Grid Search optimizer & post-processing evaluator
│   ├── ucf_option.py           # Hyperparameter registry for UCF-Crime
│   ├── xd_train.py             # XD-Violence Audio-Visual End-to-End Trainer
│   ├── xd_test.py              # XD-Violence Frame-level AP & ROC-AUC evaluator
│   ├── xd_option.py            # Hyperparameter registry for XD-Violence
│   ├── clip/                   # OpenAI CLIP backbone adapters & tokenizers
│   └── utils/
│       ├── dataset.py          # UCFDataset & XDDataset (hash-to-plus matching, audio pairing)
│       ├── losses.py           # topk_mil, clasm, consistency, bkg, fgtv_smooth, gather, residual
│       ├── tools.py            # Gaussian filter, safe_load_checkpoint, batch masks
│       └── ucf_detectionMAP.py # Temporal segment localization mAP@IoU calculation
├── MA-RUPA.ipynb               # Master Kaggle Notebook for execution on Dual Tesla T4
└── HARDCORE_RESEARCH_SPEC.md   # THIS SPECIFICATION FILE
```

---

## 8. Exact Kaggle Execution Commands

### UCF-Crime SOTA Baseline Command (87.11% AUC):
```bash
python -u src/ucf_train.py \
  --train-list /kaggle/working/rupa_artifacts/ucf_train.csv \
  --test-list /kaggle/working/rupa_artifacts/ucf_test.csv \
  --gt-path list/gt_ucf.npy \
  --gt-segment-path list/gt_segment_ucf.npy \
  --gt-label-path list/gt_label_ucf.npy \
  --batch-size 64 \
  --lr 1e-4 \
  --max-epoch 15 \
  --curriculum-epochs 3.0 \
  --sampling topk \
  --topk-ratio 16 \
  --label-smoothing 0.02 \
  --lambda-aux 0.15 \
  --lambda-smooth 0.3 \
  --loss2_weight 1.5 \
  --lambda-consistency 1.0 \
  --lambda-bkg 0.5 \
  --seed 234 \
  --temp 2.0 \
  --smooth-eval True \
  --smooth-sigma 1.0
```

### XD-Violence Peak Multimodal Command (79.02% AP):
```bash
python -u src/xd_train.py \
  --train-list /kaggle/working/rupa_artifacts/xd_train.csv \
  --test-list /kaggle/working/rupa_artifacts/xd_test.csv \
  --gt-path list/gt.npy \
  --gt-segment-path list/gt_segment.npy \
  --gt-label-path list/gt_label.npy \
  --audio-dim 128 \
  --batch-size 48 \
  --lr 2e-4 \
  --max-epoch 20 \
  --curriculum-epochs 6.0 \
  --max-omega 0.35 \
  --routing-det-weight 0.30 \
  --routing-aux-weight 0.35 \
  --sampling topk \
  --topk-ratio 8 \
  --label-smoothing 0.0 \
  --snippet-dropout 0.15 \
  --sample-one-crop-per-video True \
  --vatk True \
  --gamma-edge 5.0 \
  --lambda-aux 0.5 \
  --lambda-smooth 0.05 \
  --loss2_weight 0.8 \
  --lambda-consistency 0.25 \
  --lambda-bkg 0.5 \
  --seed 234 \
  --temp 2.0 \
  --smooth-eval True \
  --smooth-sigma 1.5
```

---

## 9. The Core Prompt to Pass to Argon 4

```text
=== SYSTEM PROMPT FOR ADVANCED REASONING AI ===
You are the Lead Deep Learning Architect reviewing the MA-RUPA project specification above.
We have access to Kaggle GPU Dual Tesla T4 (32 GB combined VRAM), running PyTorch 2.x CUDA.
Current status:
- UCF-Crime has reached 87.11% ROC-AUC (AP 33.57%). Goal: >= 89.5% - 90.0% ROC-AUC.
- XD-Violence has reached 79.02% AP (AUC 92.4%). Goal: >= 87.0% - 88.0% AP.
- Temporal localization mAP@IoU is currently low (4.77%). Goal: >= 12% - 15%.

Given the 5 core bottlenecks detailed in Section 5 and the 32 GB VRAM capacity:
1. Conduct a rigorous theoretical critique of the current architecture. Which specific module or loss is the primary bottleneck preventing UCF-Crime from crossing 89% and XD-Violence from crossing 85%?
2. Propose concrete mathematical equations and PyTorch code for the top 2 highest-leverage architectural upgrades (e.g. Latent Counterfactual Projection, Delay-Aware Temporal Audio Cross-Attention, or PCGrad Gradient Surgery).
3. Specify exact hyperparameter schedules, loss weights, and validation checkpoints to guarantee monotonic convergence without prototype collapse or label saturation.
Provide production-ready, drop-in PyTorch code blocks matching the existing codebase conventions in src/model.py and src/utils/losses.py.
================================================
```
