import os
import sys
import json
import urllib.request
from pathlib import Path

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 1. Load from .env
root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

API_KEY = os.environ.get("INTERN_AI_API_KEY")
base_endpoint = os.environ.get("INTERN_AI_BASE_URL", "https://discovery-api.intern-ai.org.cn/v1").rstrip("/")
BASE_URL = f"{base_endpoint}/chat/completions" if not base_endpoint.endswith("/chat/completions") else base_endpoint
MODEL = os.environ.get("INTERN_AI_MODEL", "intern-s2")

SYSTEM_PROMPT = """You are a Distinguished Principal AI Research Scientist in Representation Learning and Weakly-Supervised Video Understanding (IEEE Fellow, Shanghai AI Lab & DeepMind).
CRITICAL FORMAT REQUIREMENT:
- Keep your internal thinking concise (< 800 tokens) so that 100% of the diagnostics, mathematical reasoning, and exact code modifications are completely written in the final response without being cut off!
- Output rigorous, actionable, publication-grade solutions and exact PyTorch code modifications."""

USER_PROMPT = """[TASK: SYSTEM-2 AUDIT & REMEDY FOR UCF-CRIME AUC PLATEAU AT 85.92% (TARGET: >= 89.5% - 90.0%)]

We just ran the full 10-crop training of MA-RUPA (commit c9a45a2) on Dual Tesla T4 GPUs on Kaggle.
Here are the actual empirical results:

======================================================================
1. EMPIRICAL TRAINING & EVALUATION TRAJECTORY ON KAGGLE DUAL T4:
======================================================================
- Step 20: AUC 71.63% | AP 15.91%
- Step 40: AUC 77.86% | AP 18.20%
- Step 60: AUC 80.53% | AP 19.17%
- Step 100: AUC 82.88% | AP 22.45%
- Step 140: AUC 84.95% | AP 25.54%
- Step 180: AUC 85.68% | AP 28.03%
- Step 200: AUC 85.88% | AP 29.06%
- Step 250 (End Epoch 1): AUC 84.91% | AP 26.69%
- Step 270: AUC 85.61% | AP 29.36%
- Step 370: AUC 85.61% | AP 28.87%
- Step 390: AUC 85.85% | AP 29.36% (Raw Backbone: 85.63%, TVSA Semantics: 85.09%, ASPM Rec: 85.59%)
- Step 410: AUC 85.92% | AP 28.87% (PEAK SOTA)
- Step 430: AUC 85.51% | AP 28.41%
- Step 450: AUC 84.81% | AP 28.27% (Drops back down!)

======================================================================
2. DETAILED LOSS BREAKDOWN ACROSS TRAINING:
======================================================================
- Iter 50:  Total: 3.8030 | MIL: 0.5499 | CLASM: 1.8528 | ASPM: 0.0148 | Sep: 0.8767 | DCSA: 3.7191
- Iter 100: Total: 2.2811 | MIL: 0.1704 | CLASM: 1.1105 | ASPM: 0.0837 | Sep: 0.7612 | DCSA: 2.5412
- Iter 150: Total: 2.2214 | MIL: 0.2015 | CLASM: 1.1253 | ASPM: 0.0574 | Sep: 0.6610 | DCSA: 2.1916
- Iter 200: Total: 1.8328 | MIL: 0.2775 | CLASM: 0.8764 | ASPM: 0.0484 | Sep: 0.5190 | DCSA: 1.5771
- Iter 250: Total: 1.6281 | MIL: 0.2004 | CLASM: 0.7831 | ASPM: 0.0832 | Sep: 0.4703 | DCSA: 1.5068
- Iter 300: Total: 1.3279 | MIL: 0.0971 | CLASM: 0.7937 | ASPM: 0.0520 | Sep: 0.3513 | DCSA: 1.6001
- Iter 350: Total: 0.9165 | MIL: 0.0747 | CLASM: 0.5102 | ASPM: 0.0624 | Sep: 0.3101 | DCSA: 1.0571
- Iter 450: Total: 0.9323 | MIL: 0.1154 | CLASM: 0.5229 | ASPM: 0.0665 | Sep: 0.2134 | DCSA: 0.9985

======================================================================
3. COMPARISON WITH COMPETITOR DSANet (AAAI 2026, AUC 89.44%, AP 36.4%):
======================================================================
We examined DSANet's actual codebase and training logs on UCF-Crime:
1. DSANet's losses:
   - Primary loss1 (CLAS2): F.binary_cross_entropy on top-k scores (k = L//16 + 1).
   - loss2 (CLASM): Multi-class top-k CE on logits2 (weight = 1.1).
   - loss3 (L_sep): text separation (weight = 1.0 / 13 ~ 0.077).
   - loss4 (RUPA_RESIDUAL_EVENT): CLASM on anomaly residual features (R = F_vid - F_rec), ONLY on anomaly videos (labels[:, 1:].sum() > 0), with top-1 pooling (k=1), weight = 0.2.
   - loss5 (CLASM_BKG): CLASM on reconstructed normal features F_rec matching class 0 (Normal) with label smoothing (eps=0.1), weight = 0.2.
   - consistency_loss: MSE(sigmoid(logits1), (1 - cos(F_vid, F_rec))/2), weight = 0.10.
   - route_loss: CLAS2 on routing logits, weight = 1.0.
   - route_anchor: MSE(routing_scores, detector_scores.detach()), weight = 0.10.
2. How DSANet encodes video:
   - images + frame_position_embeddings -> Temporal Attention -> GraphConvolution(adj4, threshold=0.7) + GraphConvolution(disadj).
3. In MA-RUPA:
   - DCSA (Decoupled Contrastive Alignment) loss has weight 0.20, but its magnitude is HUGE (3.72 down to 1.00), which is 5x to 10x larger than MIL!
   - In DCSA, abn_weights and nor_weights are computed using exp(scale * p) with scale = 10. When p is noisy, exp(10 * p) can saturate or cause high gradient variance.
   - CLASM loss (0.52 - 1.85) is also dominating MIL (0.07 - 0.20).
   - In evaluation, why does AUC peak at 85.92% (Step 410) and then degrade to 84.81% by Step 450?

======================================================================
4. REQUIRED SYSTEM-2 DELIVERABLES:
======================================================================
Conduct deep first-principles reasoning and explain:
1. THE ROOT CAUSE OF THE 85.92% PLATEAU:
   - Why did the model climb rapidly to 85.92% in Epoch 1, but then get stuck and degrade in Epoch 2?
   - Is there a gradient imbalance where auxiliary losses (DCSA at 1.0 - 3.7, CLASM at 0.5 - 1.8) overpower the primary binary MIL loss (0.07 - 0.20)?
   - Analyze the effect of DCSA pooling (scale=10 exponentiation vs top-k selection) and whether it causes gradient noise.
2. ARCHITECTURAL & LOSS REGULATION REMEDIES TO REACH >= 89.5% - 90.0%:
   - Exact loss re-balancing formula: What weights should MIL, CLASM, DCSA (or separate Residual Event + Reconstructed BKG), and Consistency have?
   - Should DCSA be replaced or regularized with DSANet's exact residual-event + reconstructed-normal formulation (which only applies residual loss to anomaly videos)?
   - Learning rate & optimizer schedule: Does the cosine annealing decay too quickly or is the learning rate ratio optimal?
   - Evaluation fusion: At test time, which fusion formula gives the optimal test AUC (Backbone vs ASPM vs TVSA vs Dual-Stream)?
3. EXACT CODE MODIFICATIONS:
   - Provide exact, drop-in PyTorch code modifications for `src/ucf_train.py`, `src/utils/losses.py`, and `src/model.py` that will push UCF-Crime from 85.92% to >= 89.5% AUC.
"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ],
    "temperature": 0.4,
    "top_p": 0.95,
    "max_tokens": 16384,
    "stream": True
}

print(f"[*] Ket noi toi mo hinh {MODEL} qua API InternLM Discovery...")
print(f"[*] API Endpoint: {BASE_URL}")
print(f"[*] API Key: {API_KEY[:6]}...{API_KEY[-4:]}")
print("=" * 70)

req = urllib.request.Request(
    BASE_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MA-RUPA/2.0"
    },
    data=json.dumps(payload).encode("utf-8")
)

output_text = []
thinking_text = []
in_thinking = False

with urllib.request.urlopen(req, timeout=600) as resp:
    while True:
        raw_line = resp.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            
            # Thinking Process
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                if not in_thinking:
                    print("\n[THINKING PROCESS / TU DUY SUY LUAN S2]:\n", flush=True)
                    in_thinking = True
                sys.stdout.write(r)
                sys.stdout.flush()
                thinking_text.append(r)
                
            # Content
            c = delta.get("content")
            if c:
                if in_thinking:
                    print("\n\n" + "="*70 + "\n[CHAN DOAN & GIAI PHAP TOI UU S2 CHO UCF-CRIME]:\n" + "="*70 + "\n", flush=True)
                    in_thinking = False
                sys.stdout.write(c)
                sys.stdout.flush()
                output_text.append(c)
        except Exception:
            continue

report_path = root_dir / "S2_UCF_PLATEAU_DIAGNOSIS.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# INTERN-S2 UCF-CRIME PLATEAU DIAGNOSIS & REMEDIES (>=89.5% AUC)\n\n")
    f.write("## 1. System 2 Reasoning Summary\n\n")
    f.write("".join(thinking_text))
    f.write("\n\n## 2. Definitive Remedies & Code Modifications\n\n")
    f.write("".join(output_text))

print(f"\n\n[+] Da luu toan bo chan doan va giai phap vao file: {report_path}")
