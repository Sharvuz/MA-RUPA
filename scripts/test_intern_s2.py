"""
Script to test connection with InternLM Discovery intern-s2 reasoning model.
Usage:
    python scripts/test_intern_s2.py --api-key sk-xxxx
    or
    python scripts/test_intern_s2.py (uses INTERN_AI_API_KEY from environment or .env)
"""

import os
import sys
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
    parser = argparse.ArgumentParser(description="Test Intern-S2 API connection")
    parser.add_argument("--api-key", type=str, default=None, help="InternLM Discovery API Key")
    parser.add_argument("--base-url", type=str, default="https://discovery-api.intern-ai.org.cn/v1", help="API Base URL")
    parser.add_argument("--model", type=str, default="intern-s2", help="Model name (intern-s2 / intern-latest)")
    parser.add_argument("--prompt", type=str, default="Xin chao, hay gioi thieu ve kha nang suy luan cua mo hinh intern-s2 trong 2 cau.", help="Test prompt")
    args = parser.parse_args()

    # Load from .env if present
    load_env_file(root_dir / ".env")

    api_key = args.api_key or os.environ.get("INTERN_AI_API_KEY")
    if not api_key:
        print("[!] Khong tim thay API key. Vui long truyen --api-key hoac dat trong file .env (INTERN_AI_API_KEY=sk-...)")
        sys.exit(1)

    print(f"[*] Dang ket noi toi: {args.base_url}")
    print(f"[*] Su dung mo hinh: {args.model} (thinking_mode=True)")
    print(f"[*] API Key: {api_key[:6]}...{api_key[-4:]}")

    client = InternS2Client(api_key=api_key, base_url=args.base_url, model=args.model)
    res = client.chat_completion(
        messages=[
            {"role": "user", "content": args.prompt}
        ],
        thinking_mode=True
    )

    if res.get("success"):
        print("\n[+] KET NOI THANH CONG!")
        print("-" * 50)
        print(res["content"])
        print("-" * 50)
    else:
        print("\n[-] KET NOI THAT BAI!")
        print(f"Status Code: {res.get('status_code')}")
        print("Chi tiet loi:", res.get("error"))
        if res.get("status_code") == 401:
            print("\n[CHU Y] Loi 401 thuong do:")
            print("1. Token/API key da het han (msg: 'user token expired').")
            print("2. Vui long dang nhap https://discovery.intern-ai.org.cn/ hoac https://chat.intern-ai.org.cn/ de lay Key moi.")


if __name__ == "__main__":
    main()
