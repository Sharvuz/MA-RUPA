"""
Automated Semantic Description Enhancement using InternLM Discovery intern-s2
Generates fine-grained, vision-aligned text descriptions for UCF-Crime and XD-Violence.

Usage:
    python scripts/enhance_descriptions_s2.py [--dataset ucf|xd|both] [--dry-run]
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from utils.intern_s2_client import InternS2Client


def load_env(env_path: Path):
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


UCF_CLASSES = [
    "normal", "abuse", "arrest", "arson", "assault", "burglary",
    "explosion", "fighting", "roadaccidents", "robbery",
    "shooting", "shoplifting", "stealing", "vandalism"
]

XD_CLASSES = [
    "normal", "fighting", "shooting", "riot", "abuse", "car accident", "explosion"
]


def enhance_dataset_prompts(client: InternS2Client, dataset_name: str, classes: list, prompts_per_class: int = 6):
    print(f"\n{'='*60}")
    print(f"[*] Bat dau sinh descriptions cho dataset: {dataset_name.upper()} ({len(classes)} classes)")
    print(f"{'='*60}")

    enhanced = {}
    for cls in classes:
        print(f"\n[+] Dang suy luan cho category: '{cls}'...")
        prompt = (
            f"You are a computer vision and video anomaly detection researcher. "
            f"We are using CLIP ViT-B/16 text encoder to detect anomaly class '{cls}' in the {dataset_name} dataset. "
            f"Please generate exactly {prompts_per_class} distinctive, diverse, and concise surveillance video descriptions "
            f"that capture the visual appearance, action dynamics, motion, and scene context of '{cls}'.\n"
            f"Rules:\n"
            f"1. The first item must simply be '{cls}'.\n"
            f"2. Items 2 to {prompts_per_class} must be single clear English sentences describing realistic CCTV/surveillance footage.\n"
            f"3. Focus on visible human actions, objects, and environmental changes.\n"
            f"Output strictly a JSON list of strings, for example:\n"
            f'["{cls}", "a surveillance video of ...", "..."]'
        )

        res = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a precise computer vision reasoning assistant. Return strictly JSON."},
                {"role": "user", "content": prompt}
            ],
            thinking_mode=True,
            temperature=0.6
        )

        if not res.get("success"):
            print(f"[-] Loi khi goi S2 cho '{cls}': {res.get('error')}")
            continue

        content = res["content"].strip()
        # Parse JSON
        try:
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end != 0:
                descs = json.loads(content[start:end])
                if isinstance(descs, list) and len(descs) > 0:
                    enhanced[cls] = descs
                    print(f"    [OK] Da sinh {len(descs)} prompts cho '{cls}'")
                    for d in descs[:2]:
                        print(f"         - {d}")
                    continue
        except Exception as e:
            print(f"    [!] Parsing JSON that bai: {e}")

        # Fallback split
        lines = [line.strip("- *0123456789.\" ") for line in content.splitlines() if line.strip()]
        if lines:
            enhanced[cls] = [cls] + [l for l in lines if l != cls][:prompts_per_class - 1]
            print(f"    [Fallback] Lay duoc {len(enhanced[cls])} prompts")

    return enhanced


def main():
    parser = argparse.ArgumentParser(description="Enhance MA-RUPA descriptions with intern-s2 reasoning model")
    parser.add_argument("--dataset", type=str, default="both", choices=["ucf", "xd", "both"], help="Dataset to enhance")
    parser.add_argument("--dry-run", action="store_true", help="Print results without saving")
    parser.add_argument("--output", type=str, default=str(root_dir / "src" / "utils" / "descriptions_s2.py"), help="Output python file")
    args = parser.parse_args()

    load_env(root_dir / ".env")
    api_key = os.environ.get("INTERN_AI_API_KEY")
    base_url = os.environ.get("INTERN_AI_BASE_URL", "https://discovery-api.intern-ai.org.cn/v1")
    model = os.environ.get("INTERN_AI_MODEL", "intern-s2")

    if not api_key:
        print("[!] Thieu API Key trong file .env hoac bien moi truong INTERN_AI_API_KEY.")
        sys.exit(1)

    print(f"[*] Khoi tao InternS2Client voi Base URL: {base_url}")
    print(f"[*] Mo hinh suy luan: {model}")

    client = InternS2Client(api_key=api_key, base_url=base_url, model=model)

    # First test ping
    print("[*] Kiem tra ket noi den Intern-S2...")
    test_res = client.chat_completion(
        messages=[{"role": "user", "content": "Ping test"}],
        thinking_mode=False
    )
    if not test_res.get("success"):
        print(f"[-] Ket noi that bai: {test_res.get('error')}")
        print("\n[!] Luu y:")
        print("1. Kiem tra xem tai khoan tren discovery.intern-ai.org.cn da nhan diem (claim Ink Points) chua.")
        print("2. Kiem tra xem key co bi han che IP / so dien thoai khong.")
        sys.exit(1)

    print("[+] Ket noi thanh cong! Bat dau sinh prompts...")

    results = {}
    if args.dataset in ["ucf", "both"]:
        results["DESCRIPTIONS_S2_UCF"] = enhance_dataset_prompts(client, "UCF-Crime", UCF_CLASSES)

    if args.dataset in ["xd", "both"]:
        results["DESCRIPTIONS_S2_XD"] = enhance_dataset_prompts(client, "XD-Violence", XD_CLASSES)

    if not args.dry_run and results:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("# Automatically generated by InternLM Discovery intern-s2 reasoning model\n\n")
            for var_name, data in results.items():
                f.write(f"{var_name} = {json.dumps(data, indent=4, ensure_ascii=False)}\n\n")
        print(f"\n[+] Da luu toan bo descriptions moi vao: {out_path}")


if __name__ == "__main__":
    main()
