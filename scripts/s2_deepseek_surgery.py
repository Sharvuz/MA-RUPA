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
SURGEON_MODEL = "deepseek-v4-pro-0813"


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
    print("=" * 70)
    print("🔬 S2-DEEPSEEK COLLABORATIVE SURGERY PIPELINE FOR MA-RUPA")
    print(f"   - Diagnostician (Bác sĩ bắt bệnh): {DIAGNOSTICIAN_MODEL}")
    print(f"   - Surgeon       (Bác sĩ mổ xẻ code): {SURGEON_MODEL}")
    print("=" * 70)

    # Check for cached diagnosis or query S2 fresh
    cache_path = root_dir / "scripts" / "s2_prescription_cache.txt"
    if cache_path.exists():
        print(f"[*] Reusing high-grade S2 prescription from cache: {cache_path}")
        s2_diagnosis = cache_path.read_text(encoding="utf-8")
    else:
        s2_system = """You are Intern-S2, Principal Research Scientist in Video Anomaly Detection.
Your role is SOLELY TO DIAGNOSE & PRESCRIBE:
- Identify mathematical bottlenecks, gradient conflicts, and missing SOTA mechanisms.
- Provide exact mathematical formulas and clear architectural prescriptions.
- Keep internal thinking (<300 tokens) so diagnosis is concise, rigorous, and actionable. Do NOT write full code, write prescriptions for the DeepSeek coder."""

        s2_user = """[DIAGNOSIS TASK FOR UCF-CRIME & XD-VIOLENCE]
Review our current SOTA architecture:
1. Backbone: BR-STB (Bi-Relational Spatio-Temporal Backbone)
2. Normality Modeling: ASPM (Adaptive Stiefel Prototype Manifold on Grassmannian)
3. Routing: 50% Det + 30% ASPM Rec + 20% DCSA Semantics
4. Loss: Top-K Anti-Saturation MIL without self-paced sink.

What is the single most critical remaining refinement in PyTorch for the ModalityAdaptiveBranch or Loss to guarantee XD-Violence AP >= 87.5%? Prescribe the exact mathematical formulation for DeepSeek to implement."""

        print("\n" + "=" * 70)
        print("🏥 [STAGE 1] S2 ĐANG BẮT BỆNH VÀ KÊ ĐƠN THUỐC...")
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
    print("🔪 [STAGE 2] DEEPSEEK ĐANG MỔ XẺ VÀ HIỆN THỰC HÓA MÃ NGUỒN PYTORCH...")
    print("=" * 70)

    deepseek_system = """You are DeepSeek v4 Pro, Elite Principal PyTorch Research Engineer & Code Surgeon.
You receive the theoretical diagnosis and mathematical prescription from Intern-S2.
Your role is SOLELY TO PERFORM CODE SURGERY:
- Implement production-grade, bug-free, fully vector-parallel PyTorch code matching S2's prescription.
- Compatible with torch.nn.DataParallel on Dual Tesla T4 GPUs.
- Output clean, drop-in replacement classes/functions ready to integrate directly."""

    deepseek_user = f"""[CODE SURGERY REQUEST]
Intern-S2 has diagnosed the system and provided the following prescription:

{s2_diagnosis}

Please perform the code surgery: output the complete, production-grade PyTorch implementation satisfying this prescription."""

    deepseek_code, _ = call_api(
        model=SURGEON_MODEL,
        messages=[
            {"role": "system", "content": deepseek_system},
            {"role": "user", "content": deepseek_user}
        ],
        stream=True
    )

    # Save output artifact
    artifact_path = root_dir / "S2_DEEPSEEK_SURGERY_RESULT.md"
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("# S2-DEEPSEEK COLLABORATIVE SURGERY RESULT\n\n")
        f.write("## 1. S2 Diagnosis (Bác sĩ bắt bệnh: Intern-S2)\n\n")
        f.write(s2_diagnosis)
        f.write("\n\n## 2. DeepSeek Code Surgery (Bác sĩ mổ xẻ code: DeepSeek v4 Pro)\n\n")
        f.write(deepseek_code)

    print(f"\n[+] Surgery completed successfully! Result saved to: {artifact_path}")


if __name__ == "__main__":
    main()
