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

# 1. Cấu hình API InternLM Discovery (intern-s2)
API_KEY = os.environ.get("INTERN_AI_API_KEY", "sk-5446505a47fe3b9af487c18ec6be490b4388490319e8ffc8e2675faa2afaa0d8")
base_endpoint = os.environ.get("INTERN_AI_BASE_URL", "https://discovery-api.intern-ai.org.cn/v1").rstrip("/")
BASE_URL = f"{base_endpoint}/chat/completions" if not base_endpoint.endswith("/chat/completions") else base_endpoint
MODEL = os.environ.get("INTERN_AI_MODEL", "intern-s2")

# 2. System & User Messages tối ưu cho tư duy Hệ thống 2
SYSTEM_PROMPT = """You are a Distinguished Principal AI Research Scientist in Representation Learning and Weakly-Supervised Video Understanding (IEEE Fellow, Shanghai AI Lab & DeepMind).
You think deeply using first-principles mathematical rigor, adversarial analysis, and gradient dynamics auditing before outputting actionable, publication-grade solutions.
Always provide explicit mathematical proofs, exact loss equations, and clean PyTorch code compatible with torch.nn.DataParallel."""

USER_PROMPT = """[TASK: SYSTEM-2 ARCHITECTURAL & TRAINING REGULATION BREAKTHROUGH FOR WS-VAD]
We are developing MA-RUPA (BR-MVAD v2) for Weakly-Supervised Video Anomaly Detection (WS-VAD) on two benchmarks:
1. UCF-Crime (Visual CLIP ViT-B/16 D=512, L=256, 14 classes): Target Frame-level ROC-AUC >= 90.0%, AP >= 35.0%.
2. XD-Violence (Visual D=512 + Audio VGGish D=128, 6 violence classes): Target Frame-level AP >= 87.5%, ROC-AUC >= 95.0%.

Constraint: 100% Training From Scratch (Zero Warm-Start, Zero Pretrained Competitor Weights). Hardware: Dual Tesla T4 GPUs (32GB VRAM total, torch.nn.DataParallel).

======================================================================
1. THE COMPETITOR BENCHMARK: DSANet (AAAI 2026)
======================================================================
DSANet achieves SOTA (UCF-Crime AUC 89.44%, AP 36.4%; XD-Violence AP 84.1%) by training smoothly for 10 full epochs (2,500 - 3,960 gradient steps):
- Its primary Top-k BCE MIL loss (k = L//16 + 1) stays strictly within [0.0280, 0.0546] throughout all 10 epochs without vanishing.
- Consistency loss weight is exactly 0.10 (effective loss ~ 0.002).
- Separate StableAdamW optimizer (weight_decay = 1e-4, amsgrad=True) for prototype/refinement heads.

======================================================================
2. THE EMPIRICAL BOTTLENECK IN OUR ARCHITECTURE (MA-RUPA)
======================================================================
Our architecture features:
- BR-STB: Bi-Relational Spatio-Temporal Backbone (Local Window Attention W=64 + Dynamic Graph Diffusion A_stat + Distance Graph A_dist).
- ASPM: Adaptive Stiefel Prototype Manifold projecting normal snippets onto U_norm in St(p, d) via Polar Retraction (Newton-Schulz iteration), scoring reconstruction residual S_rec = ||F - U U^T F||^2.
- BiDirectionalNoiseGatedAVFusion: Cross-attention audio-visual fusion with reliability gating.

Empirical Observation on Dual Tesla T4:
- UCF-Crime: Surges rapidly in Epoch 1 to PEAK SOTA at Step 290 (Refined AUC: 86.10%, AP: 29.82%, Raw AUC: 85.82%).
  HOWEVER, by Epoch 6 (Step 1200+), it suffers severe overfitting & collapse: AUC plummets from 85.82% down to 74.42%!
  Investigation revealed: In our previous loss, Margin-Aware Focal MIL forced normal/abnormal bag scores to 0.0 and 1.0 with a margin gap, causing L_MIL to collapse to 0.0001 (300x smaller than DSANet). Sigmoid derivative d(sigma)/dz -> 0, killing backbone gradients. The frozen backbone then allowed auxiliary consistency and unconstrained U_norm to memorize background noise across 96,000 crops.
- XD-Violence: In Epoch 1, AP rapidly climbed from 30.69% (Step 0) -> 56.63% (Step 180) -> 61.57% (Step 200) -> 64.30% AP / 88.45% AUC (Step 220). It shows strong potential, but needs long-horizon regularization to cross 87.5% AP.

======================================================================
3. REQUIRED SYSTEM-2 REASONING DELIVERABLES
======================================================================
Conduct deep reasoning (<thinking> protocol) and deliver:

1. MATHEMATICAL AUDIT OF THE GRADIENT PRESERVATION LAW:
   - Prove why classical Top-k BCE preserves a non-vanishing lower bound ||grad_z L|| >= c > 0 throughout 10 epochs, while Margin-based MIL inevitably collapses into sigmoid saturation.
   - Formulate an Anti-Saturation Self-Paced MIL Loss with a mathematical gradient floor that prevents loss from ever dropping below 0.030.

2. GRASSMANNIAN / STIEFEL MANIFOLD REGULARIZATION (ASPM-v2):
   - Formulate a Riemannian manifold regularizer or spectral constraint on U_norm in St(p, d) that prevents it from memorizing video-specific background artifacts (camera motion, lighting variations), forcing it to learn universal class-invariant normalcy.

3. BI-RELATIONAL GRAPH DIFFUSION ENHANCEMENT (BR-STB-v2):
   - Propose a closed-form dynamic affinity formulation combining continuous temporal decay and adaptive feature similarity that captures long-range video dependencies without O(T^2) memory explosion under DataParallel.

4. ACOUSTIC HARD NEGATIVE SUPPRESSION & GATING (FOR XD-VIOLENCE):
   - Formulate a mathematically grounded noise-rejection gate that suppresses acoustic false alarms (car engines, thuds, crowd noise) while preserving subtle anomaly signals (screams, gunshots).

5. COMPLETE DROP-IN PYTORCH IMPLEMENTATION:
   - Provide complete, verified PyTorch nn.Module implementations for the proposed modules with numerical stability safeguards (eps=1e-8, torch.clamp) and DataParallel thread-safety.
   - Specify the exact optimizer parameter grouping, learning rate warm-up + cosine schedule floor, and weight decay ratios to train 10 full epochs up to >=90.0% UCF-Crime AUC and >=87.5% XD-Violence AP.
"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 8192,
    "stream": True
}

print(f"[*] Đang kết nối tới mô hình {MODEL} qua API InternLM Discovery...")
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
            
            # Xử lý Thinking Process (Hệ thống 2)
            r = delta.get("reasoning_content") or delta.get("reasoning")
            if r:
                if not in_thinking:
                    print("\n🧠 [INTERN-S2 THINKING PROCESS / TƯ DUY SUY LUẬN]:\n", flush=True)
                    in_thinking = True
                sys.stdout.write(r)
                sys.stdout.flush()
                thinking_text.append(r)
                
            # Xử lý nội dung câu trả lời chính thức
            c = delta.get("content")
            if c:
                if in_thinking:
                    print("\n\n" + "="*70 + "\n🎯 [BÁO CÁO GIẢI PHÁP TỐI ƯU CỦA S2]:\n" + "="*70 + "\n", flush=True)
                    in_thinking = False
                sys.stdout.write(c)
                sys.stdout.flush()
                output_text.append(c)
        except Exception:
            continue

# Lưu toàn bộ báo cáo vào file markdown
report_path = root_dir / "S2_OPTIMIZATION_REPORT.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# INTERN-S2 ARCHITECTURAL & TRAINING OPTIMIZATION REPORT\n\n")
    f.write("## 1. System 2 Thinking Process\n\n")
    f.write("".join(thinking_text))
    f.write("\n\n## 2. Formal Optimization Proposals & Code\n\n")
    f.write("".join(output_text))

print(f"\n\n✅ Đã lưu toàn bộ báo cáo vào file: {report_path}")
