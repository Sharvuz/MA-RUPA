# S2-DEEPSEEK COLLABORATIVE SURGERY RESULT

## 1. S2 Diagnosis & Mathematical Formulation (Intern-S2)



# Mathematical Prescriptions for MA-RUPA Rectification

## 1. Dynamic Density Top-k MIL (Pathology 1 Remedy)
To mitigate signal dilution of transient anomalies, replace fixed $k$ with density-adaptive selection. Let $\mathbf{s} = \{s_t\}_{t=1}^T$ denote snippet-level anomaly scores for a video of $T$ snippets.

**Otsu Thresholding:** Compute optimal threshold $\tau^*$ maximizing inter-class variance between high-score (anomaly) and low-score (normal) distributions:
$$
\tau^* = \arg\max_{\tau} \left[ \omega_0(\tau)\omega_1(\tau)\left(\mu_0(\tau) - \mu_1(\tau)\right)^2 \right]
$$
where $\omega_{0,1}$ are class probabilities and $\mu_{0,1}$ are means induced by $\tau$.

**Adaptive Window Size:**
$$
k(\mathbf{s}) = \text{clamp}\left( \sum_{t=1}^T \mathbb{I}(s_t > \tau^*), \; k_{\min}=2, \; k_{\max}=\left\lfloor \frac{T}{12} \right\rfloor \right)
$$

**Dynamic MIL Loss:** Let $\{s_{(j)}\}_{j=1}^{k}$ be the top-$k$ scores sorted descending. The bag-level probability is aggregated via:
$$
\mathcal{L}_{\text{DD-MIL}} = -\frac{1}{k(\mathbf{s})} \sum_{j=1}^{k(\mathbf{s})} \log \sigma(s_{(j)})
$$
This ensures transient events (2 snippets) dominate the gradient rather than being averaged into silence.

## 2. Stiefel Normality Anti-FA Gate (Pathology 2 Remedy)
To suppress semantic hallucinations ($S_{\text{sem}}$) when manifold geometry ($S_{\text{rec}}$) indicates normality, define a confidence gate based on ASPM reconstruction error.

**Gate Function:**
$$
G_{\text{Stiefel}} = \tanh\left( \frac{S_{\text{rec}}}{\beta} \right), \quad \beta = 0.50
$$
Since $S_{\text{rec}} \approx 0$ for normal geometry, $G_{\text{Stiefel}} \to 0$, nulling semantic noise. For anomalies, $S_{\text{rec}}$ rises, allowing $G_{\text{Stiefel}} \to 1$.

**Corrected Semantic Score:**
$$
S_{\text{sem}}^{\prime} = S_{\text{sem}} \cdot G_{\text{Stiefel}}
$$

**Consistency Regularization:** Enforce alignment between reconstruction and semantic branches during training:
$$
\mathcal{L}_{\text{geo-sem}} = \left\| S_{\text{rec}} - S_{\text{sem}}^{\prime} \right\|_2^2 \cdot \mathbb{I}(y_{\text{normal}})
$$

## 3. Asymmetric Audio-Visual Discordance Gate (Pathology 3 Remedy)
To prevent acoustic transients from triggering false alarms in the absence of visual evidence, introduce an asymmetric suppression mechanism. Let $S_{\text{vis}}$ be the visual anomaly score and $S_{\text{aux}}$ be the audio anomaly score.

**Discordance Metric:**
$$
D_{\text{av}} = \max\left(0, \; S_{\text{aux}} - S_{\text{vis}} - \delta \right), \quad \delta = 0.15
$$

**Audio Suppression Factor:**
$$
\gamma = \sigma\left( -\eta \cdot D_{\text{av}} \right), \quad \eta = 10
$$
If audio exceeds visual evidence by margin $\delta$, $\gamma \to 0$, suppressing the audio contribution.

**Multimodal Fusion:**
$$
S_{\text{final}} = S_{\text{vis}} + \gamma \cdot S_{\text{aux}}
$$

## 4. Composite Objective Function
The total training objective integrates the adaptive MIL, geometric consistency, and standard reconstruction losses:
$$
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{DD-MIL}} + \lambda_2 \mathcal{L}_{\text{rec}} + \lambda_3 \mathcal{L}_{\text{geo-sem}} + \lambda_4 \mathcal{L}_{\text{rank}}
$$
where $\mathcal{L}_{\text{rank}}$ enforces anomaly score ordering between abnormal and normal bags. Hyperparameters $\{\lambda_i\}$ are tuned via Pareto frontier analysis on validation AUC/AP.

## 2. DeepSeek Code Surgery (DeepSeek v4 Pro)

