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

API_KEY = os.environ.get("INTERN_AI_API_KEY", "sk-5446505a47fe3b9af487c18ec6be490b4388490319e8ffc8e2675faa2afaa0d8")
base_endpoint = os.environ.get("INTERN_AI_BASE_URL", "https://discovery-api.intern-ai.org.cn/v1").rstrip("/")
BASE_URL = f"{base_endpoint}/chat/completions" if not base_endpoint.endswith("/chat/completions") else base_endpoint
MODEL = os.environ.get("INTERN_AI_MODEL", "intern-s2")

SYSTEM_PROMPT = """You are a World-Renowned Principal Research Scientist and IEEE Fellow specializing in Weakly-Supervised Video Anomaly Detection (WS-VAD), Representation Learning, and Vision-Language Architectures (DeepMind / Shanghai AI Lab / CVPR & NeurIPS Area Chair).
You have authored landmark papers on WS-VAD (including DSANet AAAI, LAS-VAD CVPR, LEC-VAD ICML, VadCLIP, UR-DMU).
You think deeply using first-principles mathematical rigor, adversarial gradient flow auditing, and empirical SOTA benchmarking before formulating publication-grade, breakthrough solutions.
CRITICAL INSTRUCTION:
- Keep internal thinking (< 600 tokens) so that 100% of your detailed technical diagnosis, module-by-module audit, mathematical proofs, and concrete PyTorch implementations fit completely into the output without truncation.
- Be uncompromisingly direct, rigorous, and exhaustive. Highlight exact lines of code, mathematical formulas, and architectural upgrades."""

USER_PROMPT = """[TASK: COMPREHENSIVE SYSTEM-2 MASTER AUDIT & ARCHITECTURAL OVERHAUL FOR MA-RUPA WS-VAD]

======================================================================
1. THE CONTEXT & PROBLEM STATEMENT
======================================================================
We are building MA-RUPA (BR-MVAD v2) for Weakly-Supervised Video Anomaly Detection from scratch (zero pretrained competitor weights) on Dual Tesla T4 GPUs on Kaggle:
- UCF-Crime: CLIP ViT-B/16 (D=512, L=256, 14 crime classes, 16,000 samples/epoch, 10 crops). Target: Frame-level ROC-AUC >= 89.5% - 90.0%, AP >= 35.0%.
- XD-Violence: CLIP Visual (D=512) + VGGish Audio (D=128), 6 violence classes. Target: Frame-level AP >= 87.5%, ROC-AUC >= 95.0%.

THE COMPETITOR BENCHMARK: DSANet (AAAI)
DSANet reaches 89.44% AUC / 36.4% AP on UCF-Crime and 84.1% AP on XD-Violence with a smooth, monotonic training curve:
- Epoch 1: climbs to 87.35%
- Epoch 2: reaches 88.21%
- Epoch 3: reaches 88.89%
- Epoch 10: stabilizes at 89.14% - 89.44% AUC, 36.5% AP.

OUR CURRENT EMPIRICAL OBSERVATION (Commit 3b759a5 on Kaggle T4x2):
The model achieves remarkable initial progress:
- Total loss drops cleanly from 4.1221 (step 50) down to 0.8585 (epoch 3).
- AP surges from 15.93% up to 30.08%!
- Adaptive Stiefel Prototype Manifold (ASPM S_rec) alone reaches 85.82% AUC!
HOWEVER, THE MODEL HITS AN IMPENETRABLE GLASS CEILING AT 85.82%:
- Epoch 1: Peaks at 85.54% (Step 180) -> drops to 84.25% at epoch end.
- Epoch 2: Peaks at 85.82% (Step 390) -> drops to 84.97% at epoch end.
- Epoch 3: Peaks at 85.68% (Step 560) -> drops to 84.48% at epoch end.
- Epoch 4: Climbs to 85.12% (Step 770) -> drops to 84.36% at Step 790.
The user notes: "tôi cảm giác nó luôn ở mức 84-85%. Không bao giờ bứt phá được" (It is permanently stuck at 84-85%, oscillating and never breaking through).

======================================================================
2. RELEVANT SOTA METHODOLOGY LANDSCAPE (2025-2027)
======================================================================
Consider key design principles from top-tier literature:
- DSANet (AAAI): Coarse-grained Self-Guided Normality Modeling (SGNM with 64 dynamic prototypes + 8-layer cross-attention decoder) + Fine-grained Decoupled Contrastive Semantic Alignment (DCSA) via event residual (R = F_vid - F_rec) and background (F_rec) decomposition.
  Routing probability: p_routing = 0.5*s_det + 0.3*s_rec + 0.2*s_sem.
  CLAS2: pure top-k mean (k = L//16 + 1) without self-paced squashing.
- LAS-VAD (CVPR): Anomaly-connected components & intention reasoning (kinematics: velocity, acceleration).
- LEC-VAD (ICML): Event completeness with category-aware & category-agnostic semantics + anomaly-aware GMM.
- Segment Exploration (CVPR): Dual exploration strategy preventing over-fitting to easy modes.

======================================================================
3. COMPREHENSIVE AUDIT QUESTIONS FOR SYSTEM-2
======================================================================
Perform a ruthless, module-by-module audit of MA-RUPA and provide concrete, mathematically proven remedies:

1. [CODEBASE AUDIT: WHAT IS WEAK, HARMFUL, OR DEAD WEIGHT?]
   - Inspect our MIL loss: In AntiSaturationMILLoss, does `use_self_paced=True` (hardness = |0.5 - s|) actively penalize the model when it predicts confident anomalies (s -> 1.0), pulling scores into [0.5, 0.7] and creating the 85.82% ceiling?
   - Inspect our routing score: In `src/model.py`, why was `p_routing` set to `s_det` (ignoring ASPM's 85.82% reconstruction signal)?
   - Inspect our multi-class semantic logits: In `src/model.py`, `logits2` is computed from raw `f_main` for all 14 classes. Does background clutter in crime videos constantly trigger false Normal activations? How should `semantic_logits` be decomposed using `f_rec` (for Normal) and `r_anom = f_main - f_rec` (for Crime)?
   - Identify any other redundant or counter-productive components (e.g. dummy losses, hyperbolic curvature terms if under-constrained, etc.).

2. [ARCHITECTURAL UPGRADE: HOW TO MATCH/BEAT DSANET'S 89.44%?]
   - Normality Modeling: ASPM (Stiefel manifold projection via Newton-Schulz polar retraction) vs DSANet SGNM (prototype decoder). How can ASPM be made superior or synergized so its reconstruction score exceeds 88%?
   - Backbone: 4-layer BR-STB vs DSANet GCN (adj4 dynamic similarity + disadj temporal decay). What is causing the backbone `s_det` to cap at 84.8%? How to unblock it to reach 88%+?
   - Dynamic Adaptive Routing: How should `p_routing` dynamically fuse `s_det`, `s_rec`, and `s_sem` with a learnable gate or optimal convex combination?

3. [EXACT CODE MODIFICATIONS & TRAINING RECIPE]
   - Provide exact, ready-to-run PyTorch code modifications for `src/model.py`, `src/utils/losses.py`, `src/ucf_train.py`, and `src/ucf_test.py`.
   - Loss combination equation with proven weights.
   - Exact test-time inference pipeline that guarantees the reported ROC-AUC and AP match the true capabilities of the trained model.
"""

def main():
    print(f"[*] Connecting to: {BASE_URL}")
    print(f"[*] Model: {MODEL} (Thinking Mode + Streaming Enabled)")
    masked_key = API_KEY[:7] + "..." + API_KEY[-4:] if len(API_KEY) > 12 else "***"
    print(f"[*] API Key: {masked_key}")
    print("[*] Dispatching Comprehensive System-2 Codebase & Architectural Audit (streaming)...")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT}
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "thinking_mode": True,
        "stream": True
    }

    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MA-RUPA/2.0"
        },
        method="POST"
    )

    thinking_text = []
    output_text = []
    in_thinking = False

    try:
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
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    # Reasoning / Thinking tokens
                    r = delta.get("reasoning_content") or delta.get("reasoning")
                    if r:
                        if not in_thinking:
                            print("\n[THINKING PROCESS / SYSTEM 2 REASONING]:\n", flush=True)
                            in_thinking = True
                        sys.stdout.write(r)
                        sys.stdout.flush()
                        thinking_text.append(r)

                    # Output content tokens
                    c = delta.get("content")
                    if c:
                        if in_thinking:
                            print("\n\n" + "=" * 80 + "\n[SYSTEM-2 MASTER AUDIT & ARCHITECTURAL BLUEPRINT]:\n" + "=" * 80 + "\n", flush=True)
                            in_thinking = False
                        sys.stdout.write(c)
                        sys.stdout.flush()
                        output_text.append(c)

                except Exception:
                    continue

        full_thinking = "".join(thinking_text)
        full_content = "".join(output_text)

        print("\n\n" + "=" * 80)
        print("[+] SYSTEM-2 MASTER AUDIT COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        output_file = root_dir / "S2_MASTER_CODEBASE_AUDIT.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# SYSTEM-2 (INTERN-S2) MASTER CODEBASE AUDIT & ARCHITECTURAL OVERHAUL\n\n")
            if full_thinking:
                f.write("## 1. System 2 Internal Thinking & First-Principles Deduction\n\n")
                f.write(full_thinking)
                f.write("\n\n---\n\n")
            f.write("## 2. Actionable Master Audit & Blueprint\n\n")
            f.write(full_content)

        print(f"[+] Full Diagnostic & Master Blueprint saved to: {output_file}")
        print(f"[+] Output length: {len(full_content)} characters | Thinking length: {len(full_thinking)} characters")

    except Exception as e:
        print(f"\n[-] FAILED TO EXECUTE AUDIT: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
