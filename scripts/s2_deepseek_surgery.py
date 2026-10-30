import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
BASE_URL = "https://discovery-api.intern-ai.org.cn/v1/chat/completions"

# 1. Configuration: S2 is Diagnostician, DeepSeek is Surgeon
DIAGNOSTICIAN_MODEL = "intern-s2"
SURGEON_MODEL = "deepseek-v4-flash-0731"


def call_api(model: str, messages: list, temperature: float = 0.2, stream: bool = True, max_tokens: int = 4096, max_retries: int = 4):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream
    }
    if "s2" in model.lower():
        payload["thinking_mode"] = True

    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
            "User-Agent": "MA-RUPA-S2-DeepSeek/1.0"
        },
        method="POST"
    )

    for attempt in range(max_retries):
        try:
            if stream:
                output_chunks = []
                thinking_chunks = []
                in_thinking = False
                with urllib.request.urlopen(req, timeout=300) as resp:
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
                            # Thinking
                            r = delta.get("reasoning_content") or delta.get("reasoning")
                            if r:
                                if not in_thinking:
                                    print(f"\n[{model.upper()} - THINKING PROCESS]:\n", flush=True)
                                    in_thinking = True
                                sys.stdout.write(r)
                                sys.stdout.flush()
                                thinking_chunks.append(r)
                            # Content
                            c = delta.get("content")
                            if c:
                                if in_thinking:
                                    in_thinking = False
                                    print(f"\n\n[{model.upper()} - OUTPUT]:\n", flush=True)
                                sys.stdout.write(c)
                                sys.stdout.flush()
                                output_chunks.append(c)
                        except Exception:
                            pass
                print("\n", flush=True)
                return "".join(output_chunks), "".join(thinking_chunks)
            else:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    return res["choices"][0]["message"]["content"], ""
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait_sec = 15 * (attempt + 1)
                print(f"\n[!] Rate limit (HTTP 429) encountered. Backing off for {wait_sec}s before retry...", flush=True)
                time.sleep(wait_sec)
                continue
            raise e
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n[!] Network error: {e}. Retrying in 10s...", flush=True)
                time.sleep(10)
                continue
            raise e


def main():
    import argparse
    parser = argparse.ArgumentParser(description="S2-DeepSeek Collaborative Surgery Pipeline")
    parser.add_argument("--use-cache", action="store_true", default=False, help="Reuse cached S2 prescription")
    args = parser.parse_args()

    print("=" * 70)
    print("🔬 S2-DEEPSEEK COLLABORATIVE SURGERY PIPELINE FOR MA-RUPA")
    print(f"   - Diagnostician (Bác sĩ bắt bệnh): {DIAGNOSTICIAN_MODEL}")
    print(f"   - Surgeon       (Bác sĩ mổ xẻ code): {SURGEON_MODEL}")
    print("=" * 70)

    cache_path = root_dir / "scripts" / "s2_prescription_cache.txt"
    if cache_path.exists():
        print(f"[*] Reusing high-grade S2 prescription from cache: {cache_path}")
        s2_diagnosis = cache_path.read_text(encoding="utf-8")
    else:
        s2_system = """You are Intern-S2, Principal Research Scientist in Video Anomaly Detection (WS-VAD).
Your role is SOLELY THE THEORETICAL BRAIN:
- Formulate exact mathematical theorems, loss functions, and architectural mechanisms.
- Do NOT write boilerplate code. Write concise, rigorous mathematical prescriptions with LaTeX equations for DeepSeek v4 Pro (the Code Surgeon) to implement.
- Keep internal thinking under 350 tokens, focused on mathematical first principles."""

        s2_user = """[CLINICAL DIAGNOSTIC REPORT FROM EMPIRICAL PROBER ON DUAL T4]
We ran the Intern-S2 Deep Diagnostic Prober on our trained MA-RUPA model:
Baseline scores:
- Standalone Hierarchical BR-STB (S_routing): 84.46% AUC (peak 86.26% AUC)
- Standalone Detector Backbone (S_det): 84.45% AUC
- Standalone ASPM Stiefel Manifold (S_rec): 74.24% AUC | 23.10% AP
- Standalone TVSA Semantics (S_sem): 81.63% AUC | 19.66% AP
- DSANet Benchmark Blend (90/10 Det + Rec): 84.46% AUC | 26.37% AP

CRITICAL PATHOLOGY 1: Subtle Anomaly Blindspot (100% Miss Rate on Shoplifting & RoadAccidents)
- Top-10 missed abnormal videos: 40% are Shoplifting (Shoplifting005, 010, 015, 022) and 30% are RoadAccidents (RoadAccidents011, 127, 132).
- Diagnosis: Standard MIL with fixed k = T/16 averages 16 snippets. For subtle actions or instantaneous crashes lasting only 1-2 snippets, fixed Top-k averages 2 anomaly snippets with 14 normal snippets, diluting the abnormal signal by 87.5% and causing complete miss!
- Prescribe: Dynamic Density Top-k MIL Loss (Otsu-anchored or entropy-density adaptive k) that dynamically shrinks k to k_min=2 for sparse/transient anomalies while expanding for prolonged events:
  k_adaptive = clamp(sum_t I(s_t > tau_otsu), k_min=2, k_max=floor(T/12)).

CRITICAL PATHOLOGY 2: High-Dynamic False Alarm Cluster on Night/Traffic Scenes
- Top-10 worst false alarms on normal videos (e.g. Normal_Videos_925, 878, 896, 887):
  S_sem explodes to 1.00, S_det reaches 0.97 - 1.00.
  HOWEVER, ASPM Reconstruction (S_rec) stays clean and calm at 0.38 - 0.44!
- Diagnosis: Night headlight glare and shadows cause CLIP visual embeddings to drift toward crime prompts (fire, accident). TVSA hallucinates crime. ASPM on the Stiefel Grassmannian correctly identifies normal manifold geometry.
- Prescribe: Stiefel Normality Anti-FA Gate that dynamically suppresses TVSA semantic hallucination using ASPM S_rec confidence: S_sem_clean = S_sem * tanh(S_rec / 0.50).

CRITICAL PATHOLOGY 3: XD-Violence Audio Discordance False Alarms (Crushing AP to 83.08% vs 94.72% AUC)
- Normal scenes with loud acoustic transients (music, shouts, engines) spike S_aux > 0.9 while visual is calm.
- Prescribe: Asymmetric Audio-Visual Discordance Gating for the ModalityAdaptiveBranch.

Provide the exact mathematical formulation and equations for DeepSeek v4 Pro to implement."""

        print("\n" + "=" * 70)
        print("🏥 [STAGE 1] INTERN-S2 ĐANG BẮT BỆNH VÀ THIẾT LẬP CÔNG THỨC TOÁN HỌC...")
        print("=" * 70)

        s2_diagnosis, _ = call_api(
            model=DIAGNOSTICIAN_MODEL,
            messages=[
                {"role": "system", "content": s2_system},
                {"role": "user", "content": s2_user}
            ],
            stream=True
        )
        cache_path.write_text(s2_diagnosis, encoding="utf-8")
        print("[*] S2 prescription cached successfully.")

    # Polite rate limit delay between S2 and DeepSeek calls
    print("[*] Pausing 10s to respect API rate limits before invoking DeepSeek Surgeon...", flush=True)
    time.sleep(10)

    print("\n" + "=" * 70)
    print("🔪 [STAGE 2] DEEPSEEK V4 PRO ĐANG MỔ XẺ VÀ HIỆN THỰC HÓA MÃ NGUỒN PYTORCH...")
    print("=" * 70)

    deepseek_system = """You are DeepSeek v4 Pro, Elite Principal PyTorch Research Engineer & Code Surgeon.
You receive the theoretical diagnosis and mathematical formulas from Intern-S2.
Your role is SOLELY TO PERFORM CODE SURGERY:
- Implement complete, production-grade, bug-free, fully vectorized PyTorch code matching S2's prescription.
- Strict numerical stability guards: clamp, eps=1e-8, nan_to_num, and no division by zero.
- Compatible with torch.nn.DataParallel on Dual Tesla T4 GPUs.
- Output clean, drop-in replacement classes ready to integrate directly:
  1. `DynamicDensityTopKMILLoss`
  2. `StiefelGatedSemanticLoss`
  3. `AsymmetricAVDiscordanceGate`
  4. Integration guide with exact variable names for `src/ucf_train.py` and `src/xd_train.py`."""

    deepseek_user = f"""[CODE SURGERY REQUEST]
Intern-S2 has formulated the exact mathematical blueprint:

{s2_diagnosis}

Please perform the code surgery: output the complete, production-grade PyTorch implementation satisfying this prescription."""

    deepseek_code, _ = call_api(
        model=SURGEON_MODEL,
        messages=[
            {"role": "system", "content": deepseek_system},
            {"role": "user", "content": deepseek_user}
        ],
        stream=True,
        max_tokens=8192
    )

    # Save output artifact
    artifact_path = root_dir / "S2_DEEPSEEK_SURGERY_RESULT.md"
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("# S2-DEEPSEEK COLLABORATIVE SURGERY RESULT\n\n")
        f.write("## 1. S2 Diagnosis & Mathematical Formulation (Intern-S2)\n\n")
        f.write(s2_diagnosis)
        f.write("\n\n## 2. DeepSeek Code Surgery (DeepSeek v4 Pro)\n\n")
        f.write(deepseek_code)

    print(f"\n[+] Surgery completed successfully! Result saved to: {artifact_path}")


if __name__ == "__main__":
    main()
