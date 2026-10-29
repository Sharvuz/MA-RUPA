import os
import sys
import json
import urllib.request
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
env_path = root_dir / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))

api_key = os.environ.get("INTERN_AI_API_KEY")
base_url = "https://discovery-api.intern-ai.org.cn/v1/chat/completions"

for test_tokens in [16384, 8192, 4096]:
    payload = {
        "model": "intern-s2",
        "messages": [{"role": "user", "content": "Ping"}],
        "max_tokens": test_tokens
    }
    req = urllib.request.Request(
        base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"max_tokens={test_tokens}: HTTP {r.status} OK!")
            break
    except urllib.error.HTTPError as e:
        print(f"max_tokens={test_tokens}: HTTP {e.code} - {e.read().decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"max_tokens={test_tokens}: Error: {e}")
