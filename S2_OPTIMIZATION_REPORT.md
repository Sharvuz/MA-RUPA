# INTERN-S2 ARCHITECTURAL & TRAINING OPTIMIZATION REPORT

## 1. System 2 Thinking Process

The user is asking for a deep technical solution to a Weakly-Supervised Video Anomaly Detection (WS-VAD) problem. They want me to act as a Distinguished Principal AI Research Scientist and provide:

1. Mathematical audit of gradient preservation law (comparing Top-k BCE vs Margin-based MIL)
2. Grassmannian/Stiefel manifold regularization for ASPM-v2
3. Bi-relational graph diffusion enhancement for BR-STB-v2
4. Acoustic hard negative suppression for XD-Violence
5. Complete PyTorch implementation

This is a complex ML research task. Let me think through each component carefully.

First, let me understand the problem:
- WS-VAD: Only video-level labels available (bag-level), need frame-level predictions
- MA-RUPA architecture has issues with gradient collapse
- Need to beat DSANet which uses Top-k BCE MIL

Key insight: The user's previous loss (Margin-Aware Focal MIL) forced scores to 0/1 with margin, causing sigmoid saturation and gradient collapse. DSANet's Top-k BCE stays in [0.028, 0.054] which keeps gradients alive.

Let me work through the math:

**1. Gradient Preservation Law**

For BCE loss with sigmoid:
L = -[y log(σ(z)) + (1-y) log(1-σ(z))]
∂L/∂z = σ(z) - y

For Top-k BCE MIL: We select top-k snippets from each bag.
For abnormal bag: top-k predicted scores, y=1
∂L/∂z = σ(z) - 1, which is negative but bounded away from 0 if σ(z) < 1
For normal bag: all snippets y=0
∂L/∂z = σ(z) - 0 = σ(z) > 0

Actually, the key is that BCE never saturates completely because:
- If we predict p=0.99 for abnormal, gradient = 0.99 - 1 = -0.01
- But with Top-k, we're always selecting uncertain samples, so p stays away from 0 and 1

For margin-based MIL:
L = max(0, s_n - s_a + margin)
If we force s_n → 0 and s_a → 1 with margin, once satisfied, gradient = 0.

Actually, if using sigmoid on scores and margin loss:
The issue is that when we push scores to extremes, sigmoid derivative → 0.

Let me formalize:

Classical Top-k BCE:
For abnormal bag B^a, select top-k scores {z_i}:
L_a = -(1/k) Σ_{i∈top-k} log σ(z_i)
∂L_a/∂z_i = -(1/k)(1 - σ(z_i)) = (1/k)(σ(z_i) - 1)

For normal bag B^n:
L_n = -(1/L) Σ_{j} log(1 - σ(z_j))
∂L_n/∂z_j = (1/L) σ(z_j)

The gradient magnitude is bounded:
|∂L/∂z| ≥ min(1/k * (1 - σ(z_max)), 1/L * σ(z_min))

If we can ensure σ(z) ∈ [ε, 1-ε], then |grad| ≥ ε/k.

The key insight: BCE with Top-k on abnormal bags means we're always looking at the highest-scoring snippets. If the model hasn't perfectly classified them, σ(z) < 1, so gradient is nonzero. And for normal bags, if any snippet has σ(z) > 0, gradient is nonzero.

For margin-based loss with sigmoid saturation:
If we use a loss like:
L = max(0, margin - (σ(z_a) - σ(z_n)))
Once σ(z_a) - σ(z_n) ≥ margin, gradient is exactly 0.

Or if using focal-like pushing to 0/1:
L_focal = -(1-p_t)^γ log(p_t)
As p_t → 1, (1-p_t)^γ → 0, so gradient → 0.

So the Anti-Saturation Self-Paced MIL should:
- Maintain a gradient floor
- Use a self-paced weighting that prevents any sample from having weight too small
- Ensure loss never drops below 0.030

Let me design:
L_ASSP = Σ_i w_i · ℓ_CE(p_i, y_i)
where w_i = max(w_min, (1 - p_i)^{γ_i}) with γ_i adaptively chosen so that w_i ≥ w_min

Actually, better: use a loss that has bounded gradient away from zero.

Consider the logit-space BCE with gradient clipping in loss formulation:
We want |∂L/∂z| ≥ c for some c > 0.

One approach: use a combination of BCE and a gradient-preserving term.

Let me define:
L_preserve = BCE + λ · L_GP

where L_GP ensures the gradient magnitude stays bounded.

Actually, a cleaner approach: use a loss whose derivative with respect to logit is bounded away from zero.

Consider: L = -y log σ(z) - (1-y) log(1-σ(z)) + α|z|
The |z| term gives gradient α·sign(z) which is never zero (except at z=0, subgradient).

But that might be too crude. 

Better: Self-paced weighting that ensures each selected sample contributes at least a minimum gradient.

Let me think about the gradient floor requirement: loss ≥ 0.030.

For a bag with L snippets, if we use Top-k BCE:
If all top-k abnormal snippets have p = 0.97, then L_a = -log(0.97) ≈ 0.0305 per sample.
So if k samples, total ≈ k * 0.0305.

Actually, DSANet's loss stays in [0.028, 0.054]. So per-sample BCE loss around that range means p ∈ [e^{-0.054}, e^{-0.028}] ≈ [0.947, 0.972] for positive samples.

So to ensure loss ≥ 0.030, we need -log(p) ≥ 0.030, i.e., p ≤ e^{-0.030} ≈ 0.9704 for positives.
And for negatives: -log(1-p) ≥ 0.030, i.e., 1-p ≤ 0.9704, p ≥ 0.0296.

So we need to prevent p from going above 0.97 for positives and below 0.03 for negatives.

Anti-Saturation Self-Paced MIL:
L = - Σ_i w_i [y_i log p_i + (1-y_i) log(1-p_i)]

with w_i chosen adaptively. But we want the gradient w.r.t. logit to be bounded.

∂L/∂z_i = w_i (p_i - y_i)

For y_i = 1: |∂L/∂z_i| = w_i (1 - p_i)
For y_i = 0: |∂L/∂z_i| = w_i p_i

To ensure |∂L/∂z_i| ≥ c, we need:
w_i ≥ c / (1 - p_i) for positives
w_i ≥ c / p_i for negatives

So set w_i = max(1, c / min(p_i, 1-p_i)) ... but this could blow up.

Alternative: use a modified loss function directly.

Let me propose the **Anti-Saturating BCE (AS-BCE)**:

L_AS(p, y) = -y log(p + ε_safe) - (1-y) log(1 - p + ε_safe) + λ_sat · H(p)

where H(p) is an entropy-like barrier that prevents p → 0 or 1.

Actually, a cleaner mathematical formulation:

Define the **Gradient-Floor BCE**:
For a target gradient floor c > 0, define:
φ_c(p) = p if c ≤ p ≤ 1-c
and extend linearly/smoothly outside.

Hmm, let me think differently.

The key mathematical insight: 
For standard BCE, ∂L/∂z = p - y.
If y=1 and p→1, then ∂L/∂z → 0.
To prevent this, we can add a term that ensures the derivative is bounded.

Consider: L = -y log p - (1-y) log(1-p) + c·|z| where z = σ^{-1}(p).
Then ∂L/∂z = (p - y) + c·sign(z).
So |∂L/∂z| ≥ c - |p - y|.

Hmm, not quite bounded below by c.

Better: L = -y log p - (1-y) log(1-p) - c·y·z + c·(1-y)·z
Wait: ∂/∂z [-c y z] = -c y, ∂/∂z [c(1-y)z] = c(1-y)
So total: (p - y) - c y + c(1-y) = p - y - c(2y - 1)

For y=1: p - 1 - c = -(1 - p + c), magnitude ≥ c.
For y=0: p - 0 + c = p + c, magnitude ≥ c.

Wait let me redo:
L = BCE - c·y·z + c·(1-y)·z = BCE + c(1 - 2y)z

∂L/∂z = (p - y) + c(1 - 2y)

y=1: p - 1 - c = -(1 - p + c). Since p ∈ [0,1], |∂L/∂z| ≥ c.
y=0: p + c. |∂L/∂z| ≥ c.

Yes! This works. The gradient magnitude is always at least c.

But we need to be careful: adding c(1-2y)z is like adding a linear term in logit space. This is equivalent to shifting the target in a sense.

Actually, let's verify: for y=1, L = -log p + c·z + const... wait.

L = -log p + c(1-2)z = -log p - c z
p = σ(z) = 1/(1+e^{-z})
log p = -log(1+e^{-z})
So -log p - c z = log(1+e^{-z}) - c z

As z → ∞: log(1+e^{-z}) → 0, so L → -c z → -∞. That's bad - it encourages z → ∞.

Hmm, I need the sign right.

For y=1, we want gradient ∂L/∂z to be negative (to increase z) but bounded away from 0.
Standard BCE: ∂L/∂z = p - 1 ≤ 0. As p→1, this → 0.
We want ∂L/∂z ≤ -c < 0 always.

So: ∂L/∂z = (p - 1) - c = p - (1+c). This is always ≤ -c since p ≤ 0... wait p ≥ 0, so p - (1+c) ∈ [-1-c, -c]. Yes! Always ≤ -c in magnitude... |p - 1 - c| = 1 + c - p ≥ c since p ≤ 1. 

So for y=1: L = -log p + c z
∂L/∂z = -(1-p) + c = p - 1 + c. Wait:
d/dz[-log p] = d/dz[log(1+e^{-z})] = -e^{-z}/(1+e^{-z}) = -(1-p) = p - 1. Yes.
d/dz[c z] = c.
Total: p - 1 + c.

Since p ∈ [0,1]: p - 1 + c ∈ [c-1, c]. 
For this to always be negative (so z increases), we need c < 1.
And |p - 1 + c| = 1 - c - p... no wait.
If c < 1 and p ∈ [0,1]: p - 1 + c ranges from c-1 (when p=0) to c (when p=1).
When p > 1-c, this becomes positive, which means z decreases. That's wrong - we want z to keep increasing.

Hmm, I want the gradient to push p toward 1 but never stop. Let me reconsider.

Actually, the issue is: we don't want p to go to 1. We want to prevent saturation. So maybe we want p to stay bounded away from 1.

If we want p ≤ p_max < 1, then we want the gradient to become positive (decreasing z) when p > p_max.

Let me reframe: The goal is to keep p ∈ [p_min, p_max] where p_min = 0.03, p_max = 0.97 approximately (so that -log(p) ≥ 0.03 and -log(1-p) ≥ 0.03).

So we want a loss that has a stable equilibrium in this range.

A simple way: use BCE but with label smoothing. If we smooth labels to y' = 1 - δ for positives and y' = δ for negatives, then:
∂L/∂z = p - (1-δ) for positives.
At equilibrium p = 1-δ, gradient = 0. But near equilibrium, gradient is small.

Hmm, that doesn't give a gradient floor either.

Let me think about what DSANet does that works. Top-k BCE with k = L//16 + 1. The key is that by selecting top-k, you're always training on the hardest samples. As the model improves, the top-k samples are those that are still uncertain. This naturally creates a self-paced effect.

But the user says the problem is their Margin-Aware Focal MIL collapses to 0.0001. So we need to prevent that.

**Proposal: Anti-Saturation Self-Paced MIL (AS-SP-MIL)**

Let me define it properly.

For each bag with snippets {z_i}, predictions p_i = σ(z_i).

Abnormal bag (y=1):
Select top-k indices T = indices of top-k p_i.
L_a = -(1/|T|) Σ_{i∈T} log p_i

Normal bag (y=0):
L_n = -(1/|B|) Σ_{i∈B} log(1 - p_i)

To add anti-saturation:
Add a gradient-floor penalty that activates when p_i gets too close to 0 or 1.

L_as = L_bce + λ_as · L_af

where L_af = Σ_i [max(0, p_i - p_max)^2 + max(0, p_min - p_i)^2]

This is a barrier. But it's not exactly a gradient floor.

Alternatively, define the **clamped log-likelihood**:
L_clip = -y log(clamp(p, p_min, 1-p_min)) - (1-y) log(clamp(1-p, p_min, 1-p_min))

Then -log(p) ≤ -log(p_min) = -log(0.03) ≈ 3.5, and -log(p) ≥ -log(0.97) ≈ 0.0305.
So the per-sample loss is bounded in [0.0305, 3.5].

And the gradient:
∂L_clip/∂z = (clamp(p, p_min, 1-p_min) - y) · 1_{p ∈ (p_min, 1-p_min)} 
plus the barrier derivatives at boundaries.

Actually with clamp, when p hits p_min, the gradient from the BCE part is p_min - 0 = p_min > 0 for y=0, which pushes p up. For y=1 at p=1-p_min, gradient = (1-p_min) - 1 = -p_min < 0, pushes p up further... wait that's wrong.

For y=1, p at upper bound 1-p_min: gradient = (1-p_min) - 1 = -p_min. Since ∂L/∂z < 0, z increases, p increases, but it's clamped. So we have a constant gradient -p_min trying to push it. That's a gradient floor! |grad| = p_min = 0.03.

For y=0, p at lower bound p_min: gradient = p_min - 0 = p_min > 0. So z decreases, p decreases, clamped. Constant gradient p_min.

So clamping gives exactly a gradient floor of p_min when saturated.

But inside the range, gradient can still be small (e.g., p=0.5, y=1: gradient = -0.5).

Hmm, but the requirement is that the *loss* doesn't drop below 0.030, not necessarily the gradient. Let me re-read.

"Formulate an Anti-Saturation Self-Paced MIL Loss with a mathematical gradient floor that prevents loss from ever dropping below 0.030."

So both: gradient floor AND loss floor ≥ 0.030.

With clamping at p_min = 0.03:
- For positive sample at p = 0.97: loss = -log(0.97) ≈ 0.0305. ✓
- For negative sample at p = 0.03: loss = -log(1-0.03) = -log(0.97) ≈ 0.0305. ✓

So clamping ensures per-sample loss ≥ 0.0305.

But we also want self-pacing. Self-paced learning typically weights samples by their difficulty.

Let me combine:

**AS-SP-MIL Loss:**

For bag with label y ∈ {0,1}, snippets {z_i}_{i=1}^L, p_i = σ(z_i).

Define hard-sample selection:
If y=1 (abnormal): T = top-k indices by p_i, where k = ⌈L/16⌉ + 1 (matching DSANet).
If y=0 (normal): T = all indices (or bottom-k, but typically all for normal).

Define clamped probability: 
p̃_i = clamp(p_i, ε, 1-ε) where ε = 0.03.

Self-paced weight:
w_i = 1 + α · max(0, τ - ℓ_i) where ℓ_i = -y log p̃_i - (1-y) log(1-p̃_i)
Wait, that weights easy samples higher. Self-paced usually weights hard samples.

Actually, standard self-paced: w_i = 1 if ℓ_i < τ (easy), 0 otherwise, or smooth version.

For anti-saturation, we want to focus on samples that are NOT saturated. Let me define:

w_i = 1 / (ℓ_i + ε_sp) normalized... 

Hmm, let me keep it simpler and more principled.

**The Gradient Floor Theorem:**

Let me state and prove:

**Theorem (Gradient Collapse in Margin-MIL):** Consider a margin-based MIL loss L_margin = max(0, m - (s_a - s_n)) where s_a = max_{i∈B^a} σ(z_i), s_n = max_{j∈B^n} σ(z_j). Once s_a - s_n ≥ m, ∇L_margin = 0 identically. Since s_a, s_n ∈ (0,1) via sigmoid, and the backbone only receives gradients through these scores, the backbone gradient vanishes. Furthermore, if focal modulation (1-p)^γ is applied, the effective gradient is scaled by (1-p)^γ → 0 as p→1, accelerating collapse.

**Theorem (Gradient Preservation in Top-k BCE):** For Top-k BCE on abnormal bags, L = -(1/k)Σ_{i∈T} log σ(z_i). The gradient w.r.t. any z_i in T is ∂L/∂z_i = -(1/k)(1 - σ(z_i)). Since σ(z_i) < 1 for all finite z_i, ∂L/∂z_i < 0 strictly. Moreover, if the model has not perfectly ranked the top-k, there exists at least one i with σ(z_i) ≤ 1-δ, giving |∂L/∂z_i| ≥ δ/k.

Wait, but if the model DOES perfectly classify, σ(z_i) → 1 and gradient → 0. So Top-k BCE doesn't strictly guarantee a gradient floor either!

The key difference: In practice with Top-k, you're selecting different samples each iteration (the top-k change). So even if some samples saturate, new hard samples enter the top-k. This dynamic selection is what preserves gradients.

Whereas in margin loss, once the margin is satisfied, the loss is 0 and no samples are selected - it's a hard switch.

So the mathematical audit should highlight:
1. Top-k BCE: gradient is always nonzero for finite logits, AND the sample set T is dynamic (changes every batch/iteration), creating a moving target that prevents permanent saturation.
2. Margin MIL: hard thresholding at margin creates a dead zone where gradient is exactly 0, and once entered, no recovery.

For our Anti-Saturation loss, we need:
- No dead zones (smooth, strictly positive gradient contribution)
- Dynamic sample re-weighting
- Explicit floor on per-sample loss

**Proposed AS-SP-MIL:**

L_as-sp = y · L_a + (1-y) · L_n

L_a = -(1/Σ_{i∈T} w_i) Σ_{i∈T} w_i · log(p̃_i)
L_n = -(1/L) Σ_{j=1}^L log(1 - p̃_j)

where:
- p̃_j = clamp(p_j, ε, 1-ε), ε = 0.03
- w_i = 1 + β · (1 - p_i) / (p_i + ε)  [hardness weight: higher for smaller p_i]
- T = top-k by p_i, k = ⌈L/16⌉ + 1

Wait, the weight should be such that harder samples (lower p_i for abnormal) get higher weight.

Actually, let me use a cleaner formulation.

Let me define the **Anti-Saturation Self-Paced (AS-SP) Loss**:

For abnormal bag:
L_a = -(1/k) Σ_{i∈T} [log(p̃_i) + λ_as · (1 - p_i)^2]

Hmm, the extra term (1-p_i)^2 has gradient -2(1-p_i) which is negative, pushing p_i up. But at p_i=1, gradient is 0. Not good.

Let me use: λ_as · (log(p_max) - log(p_i)) for p_i > p_max... 

Actually, the simplest mathematically clean approach:

**Definition (ε-Clamped BCE with Dynamic Top-k):**
For ε = 0.03, define the clamped BCE for a single sample:
ℓ_ε(p, y) = -y log(p ∨ ε) - (1-y) log((1-p) ∨ ε)
where a ∨ b = max(a, b).

Then for the bag:
L = y · [-(1/k) Σ_{i∈T} log(p_i ∨ ε)] + (1-y) · [-(1/L) Σ_j log((1-p_j) ∨ ε)]

**Theorem (Loss Floor):** For any p ∈ [0,1] and y ∈ {0,1}, ℓ_ε(p,y) ≥ -log(1-ε) ≈ ε + ε²/2. With ε=0.03, ℓ_ε ≥ 0.03045 > 0.030.

Proof: If y=1, ℓ = -log(p ∨ ε) ≥ -log(1) ... no wait. p ∨ ε ∈ [ε, 1]. So -log(p ∨ ε) ∈ [0, -log(ε)]. The minimum is 0 when p∨ε = 1, i.e., p=1. Hmm, that gives 0.

Oops. If p=1 and y=1, then p∨ε = 1, -log(1) = 0. That's below 0.030.

I need a different clamping. Let me clamp p away from 1 for positives and away from 0 for negatives.

Actually: for y=1, I want to clamp p ≤ 1-ε. For y=0, clamp p ≥ ε.

So: p̃ = y · (p ∧ (1-ε)) + (1-y) · (p ∨ ε)

Then for y=1: p̃ ∈ [0, 1-ε], so -log(p̃) ≥ -log(1-ε) ≈ 0.03045. ✓
For y=0: p̃ ∈ [ε, 1], so -log(1-p̃) ≥ -log(1-ε) ≈ 0.03045. ✓

Great! This gives the loss floor.

And gradient:
For y=1, p < 1-ε: ∂ℓ/∂z = p - 1 (standard)
For y=1, p ≥ 1-ε: p̃ = 1-ε constant, so ∂ℓ/∂z = 0 from BCE part. But we need a gradient floor!

So clamping alone gives loss floor but not gradient floor. I need both.

To get gradient floor, I add a linear term in z:

For y=1: ℓ = -log(p̃) + c · z  ... wait, earlier I showed this has issues.

Let me reconsider. The gradient floor we want: |∂ℓ/∂z| ≥ c.

For y=1: We want ∂ℓ/∂z ≤ -c always (pushing z up, but never stopping).
But if we always push z up, z → ∞ and p → 1. That's saturation.

Hmm, there's a tension: gradient floor means we never stop pushing, which means we never converge. But we want convergence to a region.

I think the right interpretation: the gradient floor applies to the *selected* samples and prevents the gradient from vanishing to machine precision. It doesn't mean we push forever; it means the loss landscape always has a slope.

Actually, re-reading: "mathematical gradient floor that prevents loss from ever dropping below 0.030". The primary requirement is the loss floor. The gradient floor is the mechanism.

Let me think about what "gradient floor" means practically. In the collapsed state, grad ≈ 0 (like 1e-7). A gradient floor of c = 1e-3 or 1e-4 would prevent collapse.

So: design a loss where |∂L/∂z| ≥ c > 0 for all z.

Consider: ℓ(p, y) = -y log p - (1-y) log(1-p) + c · |z|

∂ℓ/∂z = (p - y) + c · sign(z)

For y=1, z>0: ∂ℓ/∂z = p - 1 + c. For this to be ≤ -c', we need p - 1 + c ≤ -c', i.e., p ≤ 1 - c - c'. If p is close to 1, this could be positive.

Not working cleanly.

Alternative: use a loss based on softplus that has bounded derivative.

Actually, here's a clean construction:

**Lemma:** For any c ∈ (0, 1/2), define
ℓ_c(p, y) = -y log(p + c(1-p)) - (1-y) log(1 - p + cp)

Hmm, complicated.

Simpler: Use the **logit-margin BCE**:

ℓ(p, y) = -y log σ(z - δ) - (1-y) log(1 - σ(z + δ))

For y=1: target is effectively z = δ, so p = σ(δ). Gradient ∂ℓ/∂z = σ(z-δ) - 1. At z=δ, gradient = σ(0) - 1 = -0.5. As z→∞, gradient → 0. Still vanishes.

OK here's the thing: ANY loss that converges to a finite optimum will have gradient → 0 at the optimum. So a "gradient floor" in the strict sense is impossible for a convergent system.

What we really want is:
1. No dead zones (regions where gradient is exactly 0 before reaching optimum)
2. Loss bounded away from 0 (so we detect non-convergence)
3. Dynamic sample selection that keeps providing gradient signal

So let me reframe the deliverable:

**Anti-Saturation Self-Paced MIL** has three components:
1. **Clamped likelihood** ensuring per-sample loss ≥ ε_0 = 0.030
2. **Gradient pathway preservation** via residual logit connection (never let the gradient path be blocked by a saturated gate)
3. **Dynamic hardness re-weighting** ensuring the effective batch always contains samples with |∂ℓ/∂z| ≥ η

Let me formalize component 2: Add a skip-gradient term.

ℓ_total = ℓ_bce + λ_g · (z - z_0)^2 / 2

No, that's just L2.

Actually, the real issue in the user's problem: the backbone gradient is killed because the loss is applied only through the classification head, and when the head saturates, no gradient flows to backbone. 

Solution: Add an auxiliary gradient path. But that's architectural.

For the loss itself, let me propose:

**AS-SP-MIL (final formulation):**

Let z_i be snippet logits, p_i = σ(z_i).

**Step 1 (Clamping for loss floor):**
For positive samples (y=1): p̃_i = min(p_i, 1-ε)
For negative samples (y=0): p̃_i = max(p_i, ε)
with ε = 0.03.

**Step 2 (Self-paced hardness weighting):**
For abnormal bag, top-k selection T = argtop-k(p_i).
Hardness score h_i = 1 - p_i for i ∈ T.
Weight w_i = (h_i + ε_sp) / (mean(h_T) + ε_sp), sp_normalization.

Actually simpler: w_i = log(1 + 1/(p_i + ε)) - this diverges as p_i → 0.

**Step 3 (Gradient-preserving auxiliary):**
Add a term λ_gp · Σ_i (p_i - y_i)^2 / 2 ... no, that also saturates.

Let me just go with the cleanest mathematically defensible formulation and prove the properties.

**Final AS-SP-MIL:**

L = y · L^a + (1-y) · L^n + λ_cons · L_cons

L^a = -(1/Σ_{i∈T} w_i) Σ_{i∈T} w_i · log(min(p_i, 1-ε))
L^n = -(1/L) Σ_j log(max(1-p_j, ε))

w_i = 1 + α · (1 - p_i)  (hard samples get more weight)

T = top-k(p), k = ⌈L/16⌉ + 1

**Theorem 1 (Loss Floor):** L^a ≥ -log(1-ε) = 0.03045, L^n ≥ -log(1-ε) = 0.03045. Hence L ≥ 0.03045 (ignoring consistency which is small).

**Theorem 2 (No Dead Zone):** For any finite z_i, the gradient ∂L^a/∂z_i is strictly negative for i∈T with p_i < 1-ε, and for p_i ≥ 1-ε, the clamping creates a subgradient that includes -w_i/(1-ε)·(1-p_i)·p_i... 

Hmm, min(p_i, 1-ε) is not differentiable at p_i = 1-ε. The subgradient at the boundary includes values from the left derivative.

Actually, for p_i ≥ 1-ε, log(min(p_i,1-ε)) = log(1-ε) which is constant, so gradient = 0. That's a dead zone!

To fix: use a smooth clamping or add a term that pulls back.

Better: use a **barrier loss** instead of hard clamping.

ℓ_barrier(p, y=1) = -log(p) + ε_barrier · (p - (1-ε))^2_+ 

The barrier term has gradient 2ε_barrier(p - (1-ε)) which is positive when p > 1-ε, pushing p back. So there's a restoring force. At p = 1-ε + δ, gradient from barrier = 2ε_barrier δ. This is small when δ is small.

Hmm, still can be small.

OK, I think the cleanest way to state this:

**The gradient preservation comes from the combination of:**
(a) Top-k dynamic selection (sample set T changes, so even if some saturate, new unsaturated ones enter)
(b) Clamping that keeps per-sample loss ≥ ε
(c) A small linear-in-logit term that ensures |∂L/∂z| ≥ c

For (c), let me use: λ_lin · z for negatives and -λ_lin · z for positives... but unbounded.

Actually, in practice, with weight decay on weights, z can't grow unbounded. So adding -λ_lin · z for positives is fine - it creates a constant gradient -λ_lin, and weight decay creates a restoring force -λ_wd · z. Equilibrium at z = λ_lin / λ_wd, giving p = σ(λ_lin/λ_wd) < 1.

So: For positive samples, add -λ_lin · z_i to the loss. Then:
∂L/∂z_i = (p_i - 1) - λ_lin.
At equilibrium: p_i = 1 - λ_lin. So if λ_lin = 0.03, equilibrium p = 0.97. And gradient magnitude is at least... well, near equilibrium it's small.

Ugh. Let me just step back and write a rigorous answer that makes sense.

I'll structure it as:

**1. Mathematical Audit**

**Proposition 1 (DSANet Gradient Preservation):** Top-k BCE maintains ||∇_z L||_1 ≥ (1/k)·min_{i∈T}(1-p_i) > 0 for all finite z. Because T is recomputed every forward pass and contains samples with p_i bounded away from 1 (otherwise they wouldn't be the top-k if all are saturated... wait, if ALL are saturated, top-k still has p_i close to 1).

Actually the key insight is: in a bag, if ALL snippets have p_i → 1, then the bag is definitely abnormal and we've learned. The issue is the backbone gradient. But in WS-VAD, the backbone extracts features, and if the head saturates, the backbone gets no gradient.

The real reason DSANet works: the consistency loss (weight 0.1) provides a secondary gradient path that never vanishes. And the Top-k selection ensures that within a bag, there's always variance - some snippets have lower scores.

So for the audit:

**Classical Top-k BCE:**
L = -(1/k)Σ_{i∈T} log σ(z_i) - (1/L)Σ_{j∉abnormal} log(1-σ(z_j))

∇_{z_i} L = (1/k)(σ(z_i) - 1) for i∈T (abnormal bag)
∇_{z_j} L = (1/L) σ(z_j) for

## 2. Formal Optimization Proposals & Code

