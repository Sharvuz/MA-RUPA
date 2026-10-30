# INTERN-S2 SYSTEM-2 AUDIT REPORT: EPOCHS 1-4 LOG ANALYSIS

## 1. System-2 Internal Reasoning

The user wants me to perform a comprehensive mathematical and empirical audit of MA-RUPA training log across Epochs 1-4 on UCF-Crime. Let me analyze the log carefully.

First, let me parse the key observations:

1. **CLI Argument Warning**: `--loss2-weight`, `--loss-sep-weight`, `--loss-routing-anchor-weight` were ignored. This means the argparse parser doesn't recognize these arguments, so the custom weighting for these loss components was never applied. They fell back to default values (likely 1.0 or 0.0). This is a critical bug because the intended loss balancing (loss2-weight=1.10, loss-sep-weight=1.00, loss-routing-anchor-weight=0.10) was silently discarded.

2. **Gradient Dynamics & Vanishing MIL Loss**: 
   - Epoch 1 Step 50: Det=0.1500, Route=0.1503
   - Epoch 2 Step 300: Det=0.0329, Route=0.0332
   - Epoch 3 Step 550: Det=0.0013, Route=0.0016 (100x collapse!)
   - Epoch 4 Step 800: Det=0.0014, Route=0.0018
   
   Compare to DSANet where MIL stays in [0.0280, 0.0546]. The MIL loss collapsing means the detection and routing branches are no longer receiving meaningful gradients. This is likely because:
   - The ignored CLI args mean loss weights defaulted incorrectly
   - The MIL loss (Det/Route) is being dominated by other terms (CLASM, DCSA, Event)
   - Without proper weighting, the gradients from MIL vanish relative to other terms
   - Also possible: the MIL loss uses max/min pooling over bags and as the model gets better, the anomaly scores separate, making the hinge/margin loss saturate to near-zero

3. **Stream Discrepancy & Fusion Failure**:
   - TVSA reaches 85.84% AUC
   - ASPM AP reaches 30.93%
   - But Backbone Raw SOTA oscillates 83.5%-85.3%
   - Routing never breaks 85.84%
   
   This suggests the fusion mechanism isn't properly leveraging the complementary strengths. TVSA (temporal visual-semantic alignment?) and ASPM (adaptive semantic prototype module?) are doing well individually, but the routing/fusion isn't combining them effectively. The routing loss collapsing means the router isn't learning to weight streams properly.

4. **Why plateau/drop at Step 310**:
   - Step 310 (Epoch 2, Iter 60): Peak 85.84%
   - Then Step 410: sudden collapse to 84.43%
   - Epoch 3 best drops to 84.78%
   - The vanishing MIL loss means the model stops learning discriminative features and starts overfitting to the auxiliary losses (CLASM, DCSA, Event)

Let me think about the mathematical structure. In MIL-based weakly-supervised anomaly detection (like MA-RUPA, which seems related to MIST, UGCT, or similar frameworks):

The typical loss structure:
- MIL loss (Det): hinge loss or BCE on bag-level labels
- Route loss: routing consistency or entropy minimization
- CLASM: contrastive local semantic alignment?
- ASPM: adaptive semantic prototype module
- Sep: separation loss
- EPTS: event temporal smoothness?
- DCSA: dynamic cross-stream alignment?
- Orth: orthogonality regularization
- Event: event-level loss

The total loss seems to be:
Total = w1*Det + w2*Route + w3*CLASM + w4*ASPM + w5*Sep + w6*EPTS + w7*DCSA + w8*Orth + ...

From the log, at Epoch 1 Step 50: Total=2.8534, Det=0.1500, Route=0.1503, CLASM=1.0516, ASPM=0.1179, Sep=0.7205, EPTS=0.0027, DCSA=2.5364

Wait, the sum of components doesn't equal Total. Let me check: 0.1500+0.1503+1.0516+0.1179+0.7205+0.0027+2.5364 = 4.7294, but Total=2.8534. So there must be weighting applied, or Total is a weighted sum, or some terms are subtracted.

Actually looking at Epoch 1 Summary: Loss Total: 2.3325 | MIL: 0.1727 | CLASM: 0.9560 | Consist: 0.1276 | Aux: 0.0000 | Gather: 0.0000 | Orth: 0.5071 | Event: 1.6814 | Smooth: 0.0025 | Omega: 0.05

Sum: 0.1727+0.9560+0.1276+0+0+0.5071+1.6814+0.0025+0.05 = 3.4973, but Total=2.3325. So clearly there are weights.

The ignored CLI args are: `--loss2-weight 1.10`, `--loss-sep-weight 1.00`, `--loss-routing-anchor-weight 0.10`.

So the intended weights were not applied. If `--loss-routing-anchor-weight` was supposed to be 0.10 but defaulted to something else (maybe 1.0 or 0.0), that would drastically change the routing gradient magnitude.

Actually, looking at the per-step breakdown vs summary breakdown, they use different naming:
- Step-level: Det, Route, CLASM, ASPM, Sep, EPTS, DCSA
- Summary-level: MIL, CLASM, Consist, Aux, Gather, Orth, Event, Smooth

So Det+Route ≈ MIL (0.1500+0.1503=0.3003 vs summary MIL 0.1727 - different steps though).

The key issue: the CLI arguments were ignored, so the loss weighting schema is wrong. Let me think about what this means mathematically.

If the intended loss was:
L_total = w_det * L_det + w_route * L_route + w_clasm * L_clasm + w_sep * L_sep + w_dcsa * L_dcsa + ...

And `--loss-routing-anchor-weight 0.10` was ignored, then if the default was 1.0, the routing loss is 10x too strong. If default was 0.0, routing gets no gradient.

Given that Route loss tracks Det loss very closely (0.1500 vs 0.1503, 0.0329 vs 0.0332, 0.0013 vs 0.0016, 0.0014 vs 0.0018), it seems Route = Det + small constant, or they're coupled. Actually the ratio is roughly 1:1, suggesting Route loss might be an anchor-based regularization tied to Det.

The collapse to 0.0013 suggests the model has saturated the MIL objective. In standard MIL for anomaly detection:
L_det = max(0, margin - s_normal + s_abnormal) or similar hinge
Or L_det = -[y log(p) + (1-y)log(1-p)] where p = max(s_i) for bag

As the model learns, s_abnormal → 1 and s_normal → 0, so the loss → 0. This is expected BUT the problem is:
1. It collapses too fast (by Epoch 3, it's 100x smaller than Epoch 1)
2. DSANet keeps it in [0.028, 0.0546] which suggests DSANet uses a different formulation (maybe with hard negative mining, or margin that doesn't saturate, or label smoothing)
3. When MIL loss vanishes, gradients vanish, and the model stops learning discriminative features - it then optimizes auxiliary losses which may not correlate with AUC

The plateau at 85.84% and subsequent drop is classic "gradient starvation" - the auxiliary losses (CLASM, DCSA, Event) dominate and pull the representation away from what's good for detection.

Now for the exact mathematical solution to push past 89.5% AUC:

**Issue 1: CLI Argument Parsing Bug**
The argparse likely uses `parse_known_args()` or the arguments aren't defined in the parser. Need to add explicit argument definitions:
```python
parser.add_argument('--loss2-weight', type=float, default=1.10)
parser.add_argument('--loss-sep-weight', type=float, default=1.00)
parser.add_argument('--loss-routing-anchor-weight', type=float, default=0.10)
```
Or better, use a structured config.

**Issue 2: Vanishing MIL Gradients**
The MIL loss needs to be reformulated to prevent saturation. Standard fix:
- Use focal loss instead of BCE/hinge
- Use hard negative mining
- Use adaptive margin
- Use gradient clipping per-loss-term
- Use loss balancing (GradNorm, or uncertainty weighting)

Mathematically, if we use the standard MIL loss:
L_det = -log(σ(max_i s_i)) for abnormal bags
L_det = -log(1 - σ(max_i s_i)) for normal bags

As max_i s_i → 1 for abnormal, gradient ∝ (1 - σ(max_i s_i)) → 0.

Fix: use a margin-based hinge that doesn't saturate, or use focal loss:
L_FL = -(1 - p_t)^γ log(p_t)

Or better: use the MIL loss with a dynamic margin that scales with the difficulty:
L_det = max(0, m(t) - s_abnormal + s_normal)
where m(t) is annealed.

Actually, looking at DSANet keeping MIL in [0.028, 0.0546], they probably use a fixed margin hinge loss with hard negative mining.

**Issue 3: Loss Balancing**
The total loss is dominated by DCSA (2.5364 at Epoch 1 Step 50) and CLASM (1.0516), while Det is only 0.15. Even with weights, if DCSA and CLASM are large, they dominate gradients.

We need to normalize the loss terms so each contributes comparable gradient magnitude. Use:
- Uncertainty weighting (Kendall et al., 2018): L_total = (1/(2σ_1^2))L_1 + log σ_1 + ...
- GradNorm: balance gradient magnitudes across tasks
- Or simply normalize each loss by its moving average

**Issue 4: Fusion/Routing Failure**
Routing never breaks 85.84% because the routing loss collapsed. The router needs to learn to weight TVSA and ASPM. If routing loss vanishes, the fusion becomes static (e.g., uniform averaging).

The fix: add routing entropy regularization, use Gumbel-softmax for differentiable routing, or add a routing consistency loss that doesn't vanish.

**Exact equations for the fix:**

Let me propose a concrete mathematical reformulation:

1. **Reformulated MIL with Hard Negative Mining and Focal Modulation:**
For a bag B with label y ∈ {0,1}:
p = max_{i∈B} σ(s_i)  (standard MIL max pooling)
L_det = -y ⋅ (1-p)^γ log(p) - (1-y) ⋅ p^γ log(1-p)

With γ=2, this keeps gradients larger for easy examples.

But better: use min-max MIL with margin:
L_det = max(0, Δ + max_{i∈B_normal} s_i - max_{j∈B_abnormal} s_j)

And add instance-level hard negative mining:
L_det^hard = max(0, Δ + s_normal^top-k - s_abnormal^top-k)

2. **Dynamic Loss Balancing:**
Let g_i(t) = ||∇_{θ_shared} L_i|| be the gradient norm of task i at step t.
Target: g_i(t) ≈ r_i ⋅ g_avg(t) where r_i are desired ratios.
Use GradNorm or simply:
w_i(t+1) = w_i(t) ⋅ exp(η ⋅ (log(g_i(t)/g_avg(t)) - log r_i))

3. **Routing Loss Fix:**
If routing uses anchor-based loss:
L_route = Σ_i ||R(x_i) - a_i||^2
where a_i are anchors. If weight was ignored (default 1.0 vs intended 0.1), then L_route dominates. Set correct weight.

Actually, looking at the numbers: Det and Route are almost identical (0.1500 vs 0.1503). This suggests Route might be defined as:
L_route = L_det + λ||θ_route - θ_anchor||^2
or Route loss is computed on the same scores.

Wait - 0.1500 and 0.1503 are suspiciously close. Difference is 0.0003. Similarly 0.0329 vs 0.0332 (diff 0.0003), 0.0013 vs 0.0016 (diff 0.0003), 0.0014 vs 0.0018 (diff 0.0004). 

So Route = Det + 0.0003 approximately? That's strange. Or maybe Route = Det + ε where ε is a small anchor term. If `--loss-routing-anchor-weight` was supposed to scale that ε but was ignored (default 1.0 instead of 0.1), then the anchor term is 10x too large. But the anchor term is only 0.0003, so that doesn't explain much.

Hmm, actually maybe the values shown are already weighted. Let me reconsider.

Actually, looking at Epoch 1 Step 50: Total=2.8534. If we sum the components shown: Det(0.1500) + Route(0.1503) + CLASM(1.0516) + ASPM(0.1179) + Sep(0.7205) + EPTS(0.0027) + DCSA(2.5364) = 4.7294. This is much larger than Total=2.8534. So either:
- The displayed values are unweighted, and Total applies weights
- Or some terms are subtracted
- Or Total is a running average while step values are instantaneous

Given the summary shows Orth=0.5071 and Event=1.6814 which aren't in the step breakdown, the step breakdown is a subset.

Let me try: if weights are w_det, w_route, w_clasm, w_sep, w_dcsa... and the intended weights from CLI were loss2-weight=1.10 (maybe for DCSA?), loss-sep-weight=1.00, loss-routing-anchor-weight=0.10.

If the default for loss-routing-anchor-weight was 0.0 (not 1.0), then routing anchor contributes nothing. But we see Route ≈ Det, so maybe Route includes both the routing term and the anchor term.

I think the key insight is: **the CLI arguments were ignored, so the loss weights defaulted to whatever the code has as default, which is likely an unbalanced configuration.** This causes:
1. MIL loss to be under-weighted relative to auxiliary losses → gradients dominated by CLASM/DCSA/Event
2. When auxiliary losses optimize, they may hurt detection AUC
3. The model plateaus because the detection signal is too weak

**Why Step 310 was the peak:**
At Step 310 (Epoch 2, Iter 60), the model had learned good features from the initial strong MIL gradient (Det=0.15 in Epoch 1). But as MIL collapsed to 0.0013 by Epoch 3, the model started optimizing auxiliary objectives that don't improve AUC, causing the plateau and drop.

**Why TVSA=85.84% but Routing < 85.84%:**
TVSA is one stream. ASPM is another. Routing is supposed to combine them. If routing isn't learning (loss collapsed), the fusion is suboptimal. The routing should ideally exceed both streams, but it's stuck at the best single stream (85.84%), meaning it's essentially just copying TVSA.

**Production-Ready Plan:**

1. **Fix CLI parser** - add the missing arguments with proper defaults
2. **Rebalance loss weights** - use GradNorm or uncertainty weighting
3. **Prevent MIL vanishing** - use focal MIL or hard-negative mining with margin
4. **Add routing entropy regularization** - prevent collapse to uniform/one-hot
5. **Gradient clipping per component** - ensure no single term dominates
6. **Learning rate warmup + cosine annealing** - the current schedule may be too aggressive

Let me write the exact equations:

**Reformulated Loss:**
L_total = w₁L_det^focal + w₂L_route + w₃L_clasm + w₄L_sep + w₅L_dcsa + w₆L_orth + w₇L_event

where L_det^focal uses instance-level focal modulation:
For abnormal bag B: p = max_i σ(s_i)
L_det^focal = -(1-p)^γ log(p), γ=2

For normal bag: q = 1 - max_i σ(s_i)  
L

---

## 2. Principal Scientist Diagnosis & Prescription


