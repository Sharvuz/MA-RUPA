"""
Script to test connection with InternLM Discovery intern-s2 reasoning model.
Supports real-time streaming of System 2 Thinking Process and final response.

Usage:
    python scripts/test_intern_s2.py
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
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main():
    parser = argparse.ArgumentParser(description="Test Intern-S2 API connection")
    parser.add_argument("--api-key", type=str, default=None, help="InternLM Discovery API Key")
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL")
    parser.add_argument("--model", type=str, default="intern-s2", help="Model name (intern-s2 / intern-latest)")
    parser.add_argument("--prompt", type=str, default="Xin chao! Hay gioi thieu ngan gon trong 2 cau ve kha nang suy luan cua mo hinh intern-s2.", help="Test prompt")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    args = parser.parse_args()

    load_env_file(root_dir / ".env")

    api_key = args.api_key or os.environ.get("INTERN_AI_API_KEY")
    base_url = args.base_url or os.environ.get("INTERN_AI_BASE_URL", "https://discovery-api.intern-ai.org.cn/v1")
    model = args.model or os.environ.get("INTERN_AI_MODEL", "intern-s2")

    if not api_key:
        print("[!] Khong tim thay API key. Vui long truyen --api-key hoac dat trong file .env (INTERN_AI_API_KEY=sk-...)")
        sys.exit(1)

    print(f"[*] Dang ket noi toi: {base_url}")
    print(f"[*] Su dung mo hinh: {model} (System 2 Deep Reasoning)")
    print(f"[*] API Key: {api_key[:6]}...{api_key[-4:]}")
    print(f"[*] Prompt: \"{args.prompt}\"")
    print("-" * 65)

    client = InternS2Client(api_key=api_key, base_url=base_url, model=model, timeout=180)

    first_reasoning = [True]
    first_content = [True]

    def on_reasoning(chunk: str):
        if first_reasoning[0]:
            print("\n🧠 [TƯ DUY SUY LUẬN / THINKING PROCESS - SYSTEM 2]:")
            first_reasoning[0] = False
        print(chunk, end="", flush=True)

    def on_content(chunk: str):
        if first_content[0]:
            print("\n\n💬 [CÂU TRẢ LỜI CHÍNH THỨC]:")
            first_content[0] = False
        print(chunk, end="", flush=True)

    res = client.chat_completion(
        messages=[{"role": "user", "content": args.prompt}],
        stream=not args.no_stream,
        on_reasoning_chunk=on_reasoning,
        on_content_chunk=on_content,
        max_tokens=2048
    )

    print("\n" + "-" * 65)
    if res.get("success"):
        print("✅ KẾT NỐI THÀNH CÔNG! MÔ HÌNH S2 ĐÃ HOẠT ĐỘNG HOÀN HẢO.")
    else:
        print("❌ KẾT NỐI THẤT BẠI!")
        print("Chi tiết lỗi:", res.get("error"))


if __name__ == "__main__":
    main()
