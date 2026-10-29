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
- Keep your internal thinking process compact (< 800 tokens) so that 100% of the deliverables, equations, and full runnable PyTorch code are completely printed in the final response without being cut off!
- Output publication-grade mathematical formulations and production-ready PyTorch modules with full numerical stability safeguards (eps=1e-8, torch.clamp) and torch.nn.DataParallel thread safety."""

USER_PROMPT = """[TASK: SYSTEM-2 ARCHITECTURAL & TRAINING CODE IMPLEMENTATION FOR MA-RUPA]
You previously analyzed the gradient collapse problem where Margin-Aware Focal MIL caused loss to collapse to 0.0001, freezing the backbone and overfitting to background noise.

Now, directly provide the definitive solutions and full drop-in PyTorch code for the following 4 pillars:

Pillar 1: Anti-Saturation Self-Paced MIL Loss (AS-SP-MIL)
- Exact mathematical equation with clamped likelihood (eps=0.03, ensuring loss >= 0.030) and self-paced hardness weighting.
- Clean PyTorch nn.Module `AntiSaturationSelfPacedMILLoss` with top-k dynamic selection k = L//16 + 1.

Pillar 2: ASPM-v2: Grassmannian / Stiefel Manifold Regularization
- How to constrain U_norm in St(p, d) so it does NOT memorize video-specific background artifacts (camera motion, lighting variations), forcing it to learn universal class-invariant normalcy across videos.
- Provide the mathematical regularization loss L_grassmann / L_stiefel and PyTorch code.

Pillar 3: BR-STB-v2 & Acoustic Hard Negative Suppression (AHNS for XD-Violence)
- Closed-form dynamic bi-relational graph diffusion combining continuous temporal decay and adaptive feature similarity in O(T) or sparse O(T*W) without O(T^2) memory explosion.
- Mathematically grounded noise-rejection acoustic gate for XD-Violence that rejects ambient noise (car engines, thuds) while preserving explosive anomaly signals (screams, gunshots).

Pillar 4: Complete PyTorch Code & Training Hyperparameters for Kaggle Dual Tesla T4
- Exact optimizer grouping (different lr and weight decay for backbone vs prototype manifold vs classifier heads).
- Exact learning rate schedule (warmup + cosine floor).
- Loss weights recipe for UCF-Crime and XD-Violence.

Write out the COMPLETE, runnable PyTorch code without placeholders or shortcuts.
"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ],
    "temperature": 0.5,
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
                    print("\n\n" + "="*70 + "\n[GIAI PHAP VA MA NGUON PYTORCH TU INTERN-S2]:\n" + "="*70 + "\n", flush=True)
                    in_thinking = False
                sys.stdout.write(c)
                sys.stdout.flush()
                output_text.append(c)
        except Exception:
            continue

# Append or save to file
report_path = root_dir / "S2_CODE_SOLUTIONS.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# INTERN-S2 DEFINITIVE ARCHITECTURAL & TRAINING CODE SOLUTIONS\n\n")
    f.write("## 1. System 2 Reasoning Summary\n\n")
    f.write("".join(thinking_text))
    f.write("\n\n## 2. Complete Deliverables & PyTorch Modules\n\n")
    f.write("".join(output_text))

print(f"\n\n[+] Da luu toan bo giai phap va ma nguon vao file: {report_path}")
