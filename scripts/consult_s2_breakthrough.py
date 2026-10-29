"""
Automated Breakthrough Consultation with InternLM Discovery (intern-s2 / Step-2)
Loads the Master Super Prompt and queries the Chinese System-2 reasoning model.

Usage:
    python scripts/consult_s2_breakthrough.py [--api-key sk-xxxx] [--model intern-s2]
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from utils.intern_s2_client import InternS2Client


def load_env_file(env_path: Path):
    """Load key-value pairs from .env file if it exists."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("\"'")


def main():
    parser = argparse.ArgumentParser(description="Query InternLM intern-s2 for MA-RUPA Architectural Breakthroughs")
    parser.add_argument("--api-key", type=str, default=None, help="InternLM Discovery API Key")
    parser.add_argument("--base-url", type=str, default="https://discovery-api.intern-ai.org.cn/v1", help="API Base URL")
    parser.add_argument("--model", type=str, default="intern-s2", help="Model name (intern-s2 / intern-latest)")
    parser.add_argument("--output-report", type=str, default=str(root_dir / "docs" / "INTERN_S2_SCIENTIFIC_REPORT.md"), help="Path to save report")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    args = parser.parse_args()

    # Load from .env if present
    load_env_file(root_dir / ".env")

    prompt_path = root_dir / "prompts" / "SUPER_PROMPT_INTERN_S2_MA_RUPA.md"
    if not prompt_path.exists():
        print(f"[!] Khong tim thay file prompt tai: {prompt_path}")
        sys.exit(1)

    with open(prompt_path, "r", encoding="utf-8") as fp:
        prompt_content = fp.read()

    print("=" * 70)
    print("   MA-RUPA v2: CONSULTING INTERNLM DISCOVERY INTERN-S2 (SYSTEM 2)   ")
    print("=" * 70)
    print(f"[*] Prompt File: {prompt_path} ({len(prompt_content)} bytes)")

    api_key = args.api_key or os.environ.get("INTERN_AI_API_KEY")
    if not api_key:
        print("\n[!] CHUA CO API KEY CHO INTERNLM DISCOVERY (intern-s2).")
        print("    Ban co the:")
        print("    1. Mo web https://discovery.intern-ai.org.cn/ hoac https://chat.intern-ai.org.cn/")
        print("    2. Copy toan bo noi dung file prompts/SUPER_PROMPT_INTERN_S2_MA_RUPA.md va paste vao chat.")
        print("    3. Hoac lay API Key va chay:")
        print("       python scripts/consult_s2_breakthrough.py --api-key sk-xxxx")
        sys.exit(0)

    print(f"[*] Dang ket noi toi: {args.base_url}")
    print(f"[*] Su dung mo hinh: {args.model} voi thinking_mode=True...")
    print("[*] Dang gui Super Prompt toi mo hinh suy luan he thong 2 (co the mat 60-120 giay)...")

    client = InternS2Client(api_key=api_key, base_url=args.base_url, model=args.model, timeout=180)
    res = client.chat_completion(
        messages=[
            {"role": "user", "content": prompt_content}
        ],
        thinking_mode=True,
        temperature=args.temperature,
        max_tokens=8192
    )

    if res.get("success"):
        content = res["content"]
        out_path = Path(args.output_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write(f"# INTERNLM DISCOVERY (intern-s2) SCIENTIFIC BREAKTHROUGH REPORT\n\n")
            fp.write(f"- Model: {args.model}\n")
            fp.write(f"- Timestamp: {res.get('timestamp')}\n\n")
            fp.write(content)
        
        print("\n" + "=" * 70)
        print(f"[+] TUYET VOI! DA NHAN DUOC BAO CAO KHOA HOC TU INTERN-S2!")
        print(f"[+] Bao cao duoc luu tai: {out_path}")
        print("=" * 70)
        print("\nTrích đoạn phân tích đầu tiên:")
        print(content[:1000] + "...\n")
    else:
        print("\n[-] LOI KET NOI INTERN-S2:")
        print(f"Status Code: {res.get('status_code')}")
        print("Chi tiet:", res.get("error"))


if __name__ == "__main__":
    main()
