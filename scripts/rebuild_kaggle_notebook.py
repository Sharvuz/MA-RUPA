import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def build_notebook():
    nb = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "accelerator": "GPU",
        },
        "nbformat_minor": 5,
        "nbformat": 4,
        "cells": [],
    }

    def add_md(source_text):
        lines = [l + "\n" for l in source_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        })

    def add_code(source_text):
        lines = [l + "\n" for l in source_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": lines,
        })

    # Cell 1: Overview
    add_md(r"""# 🔬 MA-RUPA v3: Clean SOTA Multimodal & Vision-Language WS-VAD Pipeline
### Dual Tesla T4 GPUs | UCF-Crime ($\ge 89.0\%$ ROC-AUC) & XD-Violence ($\ge 87.0\%$ AP)

Kiến trúc **MA-RUPA v3** được xây dựng mới hoàn toàn từ gốc dựa trên tổng hợp toán học từ các bài báo SOTA WS-VAD hiện đại (**VadCLIP**, **DSANet/Safe-RUPA**, **DeSC**, **MAVD**):
1. **LGT + Zero-Init Multi-Scale Dilated Sensitivity Backbone (`MSTSensitivityAdapter`):** Kết hợp Local-Window Transformer + Dual Similarity/Distance GCN với nhánh tích chập giãn nở đa tỷ lệ $d \in \{1, 2, 4\}$ khởi tạo $\gamma = 0$ ($\tanh(\gamma)=0$ tại Step 0, bảo toàn 100% không gian đặc trưng CLIP ViT-B/16).
2. **Memory-Light Orthogonal-Free CLIP Text Adapter (`CLIPTextAdapter`):** Loại bỏ 86M tham số ảnh thừa của CLIP (tiết kiệm $\sim 350\text{ MB}$ VRAM/GPU), giải phóng toàn bộ không gian $\mathbb{R}^{512}$ để tối ưu trực giao $L_{\text{orth}}(\mathbf{t}_{\text{normal}}, \mathbf{t}_{\text{abn}}) \to 0$ kết hợp giám sát đa lớp `CLASM` ($w_2 = 1.1$ cho UCF, $w_2 = 5.0$ cho XD).
3. **Visual-Guided Cross-Modal Gated Audio-Visual Adapter (`GatedCrossModalAudioAdapter`):** Đồng bộ chéo đặc trưng âm thanh VGGish 128-D và CLIP 512-D với cổng tin cậy $[V, A, V \odot A]$ và hàm phạt triệt tiêu báo động giả âm thanh trên video bình thường (`audio_guided_mil_loss`).
4. **Parallel Multi-Crop Test Aggregation & DeSC Collaborative Scoring:** Tự động gom song song toàn bộ **10 spatial crops** trên UCF-Crime (`2,900` files) và **5 spatial crops** trên XD-Violence (`4,000` files) kèm bộ đệm RAM `float16` siêu tốc.""")

    # Cell 2: Setup Environment & GPU Preflight
    add_code("""# ========================= 1. SETUP ENVIRONMENT & GPU PREFLIGHT =========================
import os, sys, shutil, subprocess
from pathlib import Path
import torch

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORK / "MA-RUPA"
ARTIFACTS = WORK / "rupa_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

os.chdir(WORK)

REPO_URL = "https://github.com/Sharvuz/MA-RUPA.git"
REPO_REF = "main"

if (REPO_DIR / ".git").exists():
    print("🔄 Đang đồng bộ commit MA-RUPA v3 mới nhất từ GitHub...", flush=True)
    subprocess.run(["git", "fetch", "origin", REPO_REF], cwd=REPO_DIR, check=False)
    subprocess.run(["git", "reset", "--hard", f"origin/{REPO_REF}"], cwd=REPO_DIR, check=False)
else:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    print("📥 Cloning MA-RUPA repository...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO_DIR)], cwd=WORK, check=True)

os.chdir(REPO_DIR)

commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR).decode().strip()
print(f"🚀 [MA-RUPA v3 Codebase] Git Commit: {commit_hash}", flush=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "ftfy", "regex", "einops==0.8.0", "packaging", "scikit-learn", "pandas", "scipy"
], check=True)

print("\\n" + "=" * 64)
print(f"PyTorch Version : {torch.__version__} | CUDA: {torch.cuda.is_available()}")
gpu_count = torch.cuda.device_count()
for i in range(gpu_count):
    print(f"  GPU [{i}]       : {torch.cuda.get_device_name(i)}")
if gpu_count >= 2:
    print("🔥 Dual Tesla T4 x 2 sẵn sàng! Interleaved DataParallel 50/50 Normal-Abnormal active.")
print("=" * 64)""")

    # Cell 3: Feature Radar & Path Resolution
    add_code("""# ========================= 2. RADAR: QUÉT DỮ LIỆU, ĐỒNG BỘ '#' <-> '+' & KIỂM TRA MULTI-CROP =========================
import os, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

print("🔍 Đang kiểm tra bản đồ đặc trưng CLIP (.npy) & VGGish Audio...")

ucf_train_csv = ARTIFACTS / "ucf_train.csv"
ucf_test_csv = ARTIFACTS / "ucf_test.csv"
xd_train_csv = ARTIFACTS / "xd_train.csv"
xd_test_csv = ARTIFACTS / "xd_test.csv"

skip_scan = False
if all((ARTIFACTS / f).exists() for f in ["ucf_train.csv", "ucf_test.csv", "xd_train.csv", "xd_test.csv"]):
    try:
        s_df = pd.read_csv(ucf_train_csv, nrows=5)
        if all(Path(p).exists() for p in s_df['path']):
            print("⚡ Fast-Pass: Danh sách CSV trong rupa_artifacts đã hợp lệ!")
            skip_scan = True
    except Exception:
        skip_scan = False

if not skip_scan:
    zip_files = []
    if INPUT.exists():
        for root, dirs, files in os.walk(INPUT):
            if len(Path(root).relative_to(INPUT).parts) >= 2:
                dirs.clear()
            for f in files:
                if f.lower().endswith(".zip"):
                    zip_files.append(Path(root) / f)

    if zip_files:
        extract_dir = WORK / "extracted_features"
        extract_dir.mkdir(parents=True, exist_ok=True)
        for zf in zip_files:
            dest = extract_dir / zf.stem
            if not dest.exists() or not any(dest.iterdir()):
                dest.mkdir(parents=True, exist_ok=True)
                print(f"  Unzipping {zf.name} -> {dest.name}...")
                with zipfile.ZipFile(zf, 'r') as archive:
                    archive.extractall(dest)

    file_map = {}
    total_npy = 0
    for s_root in [INPUT, WORK]:
        if not s_root.exists():
            continue
        for root, dirs, files in os.walk(s_root):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.ipynb_checkpoints', '.cache'}]
            for fname in files:
                if fname.endswith(".npy"):
                    total_npy += 1
                    full_path = os.path.join(root, fname)
                    file_map[fname] = full_path
                    if '+' in fname:
                        file_map[fname.replace('+', '#')] = full_path
                    if '#' in fname:
                        file_map[fname.replace('#', '+')] = full_path

    print(f"📡 Đã lập chỉ mục {total_npy:,} tệp .npy trên hệ thống!")

    def sync_csv_list(csv_rel_path, output_name):
        df = pd.read_csv(REPO_DIR / csv_rel_path)
        resolved, missing = [], []
        for orig_path in df['path'].astype(str):
            bname = Path(orig_path).name
            match = None
            for cand in [bname, bname.replace('#', '+'), bname.replace('+', '#'), bname.replace('#', '_')]:
                if cand in file_map:
                    match = file_map[cand]
                    break
            if match:
                resolved.append(match)
            else:
                missing.append(bname)
                resolved.append(orig_path)
        print(f"  {'✅' if not missing else '⚠️'} {output_name}: {len(df) - len(missing)}/{len(df)} đường dẫn khớp thành công.")
        df['path'] = resolved
        out_p = ARTIFACTS / output_name
        df.to_csv(out_p, index=False)
        return out_p

    ucf_train_csv = sync_csv_list("list/ucf_CLIP_rgb.csv", "ucf_train.csv")
    ucf_test_csv = sync_csv_list("list/ucf_CLIP_rgbtest.csv", "ucf_test.csv")
    xd_train_csv = sync_csv_list("list/xd_CLIP_rgb.csv", "xd_train.csv")
    xd_test_csv = sync_csv_list("list/xd_CLIP_rgbtest.csv", "xd_test.csv")

sys.path.insert(0, str(REPO_DIR / "src"))
from utils.dataset import resolve_multicrop_paths
ucf_sample_test = pd.read_csv(ucf_test_csv).iloc[0]['path']
xd_sample_test = pd.read_csv(xd_test_csv).iloc[0]['path']
ucf_crops_found = len(resolve_multicrop_paths(ucf_sample_test, max_crops=10))
xd_crops_found = len(resolve_multicrop_paths(xd_sample_test, max_crops=10))
print(f"🎯 Multi-Crop Check: UCF test video có {ucf_crops_found}/10 crops | "
      f"XD test video có {xd_crops_found} crops (tối đa 10).")""")

    # Cell 4: UCF-Crime Markdown
    add_md(r"""## 3. Huấn luyện SOTA trên UCF-Crime (Dual GPU Tesla T4 | Batch Size 64)

* **Mục tiêu:** **$\ge 89.0\%$ Frame-level ROC-AUC** (vượt VadCLIP $88.02\%$ và DSANet 1-crop $89.17\%$).
* **Cấu hình chuẩn thực nghiệm:**
  * `LR = 7e-5` với **Dual Decoupled Optimizers**: `AdamW` cho LGT + Multi-Scale Sensitivity + CLIP Text Adapter và `StableAdamW` cho SGNM Dynamic Normal Prototypes + Continuation Safe-Gate.
  * `loss2_weight = 1.1` (`CLASM` 14 lớp) + `loss_sep_weight = 1.0` (`Text Orthogonality`) + `Polyak ModelEMA` ($\text{decay}=0.996$).
  * **10-Crop Parallel Evaluation (`__0..__9`):** Đánh giá đồng thời cả 10 góc cắt không gian trên toàn bộ 290 video test.""")

    # Cell 5: Train UCF-Crime Code
    add_code("""# ========================= 3. TRAIN UCF-CRIME (MA-RUPA v3 | BATCH 64 | LR 7e-5) =========================
import sys, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "MA-RUPA"
ARTIFACTS = WORK / "rupa_artifacts"

subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO_DIR, check=False)
subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO_DIR, check=False)

ucf_train_csv = ARTIFACTS / "ucf_train.csv" if (ARTIFACTS / "ucf_train.csv").exists() else REPO_DIR / "list/ucf_CLIP_rgb.csv"
ucf_test_csv = ARTIFACTS / "ucf_test.csv" if (ARTIFACTS / "ucf_test.csv").exists() else REPO_DIR / "list/ucf_CLIP_rgbtest.csv"
ucf_model_path = ARTIFACTS / "model_ucf_best.pth"
ucf_checkpoint_path = ARTIFACTS / "checkpoint_ucf.pth"

cmd_train_ucf = [
    sys.executable, "-u", "src/ucf_train.py",
    "--train-list", str(ucf_train_csv),
    "--test-list", str(ucf_test_csv),
    "--gt-path", "list/gt_ucf.npy",
    "--gt-segment-path", "list/gt_segment_ucf.npy",
    "--gt-label-path", "list/gt_label_ucf.npy",
    "--batch-size", "64",
    "--lr", "7e-5",
    "--max-epoch", "10",
    "--visual-layers", "2",
    "--visual-head", "1",
    "--attn-window", "8",
    "--decoder-depth", "8",
    "--num-prototypes", "16",
    "--text-adapt-until", "3",
    "--t-w", "0.1",
    "--loss2-weight", "1.1",
    "--loss-sep-weight", "1.0",
    "--loss-residual-weight", "0.20",
    "--loss-reconstructed-normal-weight", "0.20",
    "--loss-consistency-weight", "0.10",
    "--loss-gather-weight", "0.10",
    "--loss-routing-anchor-weight", "0.10",
    "--lambda-smooth", "0.02",
    "--multi-crop-eval", "True",
    "--max-test-crops", "10",
    "--collab-alpha", "0.20",
    "--smooth-eval", "True",
    "--smooth-sigma", "0.75",
    "--eval-interval", "50",
    "--use-ema", "True",
    "--seed", "234",
    "--model-path", str(ucf_model_path),
    "--checkpoint-path", str(ucf_checkpoint_path),
]

print("🚀 Bắt đầu huấn luyện MA-RUPA v3 trên UCF-Crime (10-Crop Eval, Target ≥89.0% AUC)...\\n", flush=True)
proc = subprocess.Popen(cmd_train_ucf, cwd=REPO_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait() != 0:
    raise RuntimeError("❌ UCF-Crime training exited with non-zero status.")
print(f"\\n✅ Hoàn tất! Checkpoint tốt nhất đã lưu tại: {ucf_model_path}")""")

    # Cell 6: Evaluate UCF-Crime Code
    add_code("""# ========================= 4. EVALUATE UCF-CRIME (10-CROP + LOCALIZATION mAP) =========================
import sys, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "MA-RUPA"
ARTIFACTS = WORK / "rupa_artifacts"
ucf_test_csv = ARTIFACTS / "ucf_test.csv" if (ARTIFACTS / "ucf_test.csv").exists() else REPO_DIR / "list/ucf_CLIP_rgbtest.csv"
ucf_model_path = ARTIFACTS / "model_ucf_best.pth"

subprocess.run([
    sys.executable, "-u", "src/ucf_test.py",
    "--test-list", str(ucf_test_csv),
    "--gt-path", "list/gt_ucf.npy",
    "--gt-segment-path", "list/gt_segment_ucf.npy",
    "--gt-label-path", "list/gt_label_ucf.npy",
    "--model-path", str(ucf_model_path),
    "--multi-crop-eval", "True",
    "--max-test-crops", "10",
    "--collab-alpha", "0.20",
    "--smooth-eval", "True",
    "--smooth-sigma", "0.75",
], cwd=REPO_DIR, check=True)""")

    # Cell 7: XD-Violence Markdown
    add_md(r"""## 5. Huấn luyện Đa phương thức SOTA trên XD-Violence (Visual CLIP + Audio VGGish)

* **Mục tiêu:** **$\ge 87.0\%$ Frame-level Average Precision (AP)** (vượt xa bản visual-only 1-crop $84.99\%$).
* **Cấu hình chuẩn thực nghiệm:**
  * `Batch Size = 96` (48 Normal + 48 Abnormal chia đều 2 GPU Tesla T4).
  * **Tốc độ học phân tầng:** `main_lr = 1.5e-5` (giữ ổn định đặc trưng CLIP/LGT với cửa sổ ngữ cảnh dài `attn_window = 64`) và `audio_lr = 5e-5` (giúp nhánh `GatedCrossModalAudioAdapter` 128-D VGGish hội tụ nhanh).
  * `loss2_weight = 5.0` (`CLASM` đa nhãn bạo lực) + `loss_sep_weight = 1.0` + `lambda_aux = 0.25` (`Audio-Guided MIL + Normal Acoustic Suppression`).
  * **5-Crop Parallel Multimodal Evaluation (`__0..__4`):** Gom 5 góc cắt không gian kết hợp VGGish Audio cho toàn bộ 800 video test.""")

    # Cell 8: Train XD-Violence Code
    add_code("""# ========================= 5. TRAIN XD-VIOLENCE (MA-RUPA v3 MULTIMODAL | BATCH 96) =========================
import sys, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "MA-RUPA"
ARTIFACTS = WORK / "rupa_artifacts"

subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO_DIR, check=False)
subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO_DIR, check=False)

xd_train_csv = ARTIFACTS / "xd_train.csv" if (ARTIFACTS / "xd_train.csv").exists() else REPO_DIR / "list/xd_CLIP_rgb.csv"
xd_test_csv = ARTIFACTS / "xd_test.csv" if (ARTIFACTS / "xd_test.csv").exists() else REPO_DIR / "list/xd_CLIP_rgbtest.csv"
xd_model_path = ARTIFACTS / "model_xd_best.pth"
xd_checkpoint_path = ARTIFACTS / "checkpoint_xd.pth"

cmd_train_xd = [
    sys.executable, "-u", "src/xd_train.py",
    "--train-list", str(xd_train_csv),
    "--test-list", str(xd_test_csv),
    "--gt-path", "list/gt.npy",
    "--gt-segment-path", "list/gt_segment.npy",
    "--gt-label-path", "list/gt_label.npy",
    "--audio-dim", "128",
    "--batch-size", "96",
    "--lr", "1.5e-5",
    "--audio-lr", "5e-5",
    "--max-epoch", "10",
    "--visual-layers", "1",
    "--visual-head", "1",
    "--attn-window", "64",
    "--decoder-depth", "8",
    "--num-prototypes", "16",
    "--text-adapt-until", "1",
    "--t-w", "0.6",
    "--temp", "1.0",
    "--loss2-weight", "5.0",
    "--loss-sep-weight", "1.0",
    "--loss-residual-weight", "0.20",
    "--loss-reconstructed-normal-weight", "0.20",
    "--loss-consistency-weight", "0.10",
    "--loss-gather-weight", "0.10",
    "--loss-routing-anchor-weight", "0.10",
    "--lambda-aux", "0.25",
    "--lambda-smooth", "0.02",
    "--multi-crop-eval", "True",
    "--max-test-crops", "10",
    "--collab-alpha", "0.15",
    "--aux-eval-alpha", "0.10",
    "--smooth-eval", "True",
    "--smooth-sigma", "0.75",
    "--eval-interval", "80",
    "--use-ema", "True",
    "--seed", "234",
    "--model-path", str(xd_model_path),
    "--checkpoint-path", str(xd_checkpoint_path),
]

print("🚀 Bắt đầu huấn luyện MA-RUPA v3 Multimodal trên XD-Violence (5-Crop + VGGish Audio, Target ≥87.0% AP)...\\n", flush=True)
proc = subprocess.Popen(cmd_train_xd, cwd=REPO_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait() != 0:
    raise RuntimeError("❌ XD-Violence training exited with non-zero status.")
print(f"\\n✅ Hoàn tất! Checkpoint XD-Violence tốt nhất đã lưu tại: {xd_model_path}")""")

    # Cell 9: Evaluate XD-Violence + Clinical Diagnostic Prober
    add_code("""# ========================= 6. EVALUATE XD-VIOLENCE & CLINICAL DIAGNOSTIC PROBER =========================
import sys, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "MA-RUPA"
ARTIFACTS = WORK / "rupa_artifacts"
xd_test_csv = ARTIFACTS / "xd_test.csv" if (ARTIFACTS / "xd_test.csv").exists() else REPO_DIR / "list/xd_CLIP_rgbtest.csv"
xd_model_path = ARTIFACTS / "model_xd_best.pth"

subprocess.run([
    sys.executable, "-u", "src/xd_test.py",
    "--test-list", str(xd_test_csv),
    "--gt-path", "list/gt.npy",
    "--gt-segment-path", "list/gt_segment.npy",
    "--gt-label-path", "list/gt_label.npy",
    "--model-path", str(xd_model_path),
    "--multi-crop-eval", "True",
    "--max-test-crops", "10",
    "--collab-alpha", "0.15",
    "--aux-eval-alpha", "0.10",
    "--smooth-eval", "True",
    "--smooth-sigma", "0.75",
], cwd=REPO_DIR, check=True)

subprocess.run([
    sys.executable, "-u", "scripts/s2_deep_diagnostic_prober.py",
    "--dataset", "xd",
    "--model-path", str(xd_model_path),
    "--test-list", str(xd_test_csv),
    "--gt-path", "list/gt.npy",
], cwd=REPO_DIR, check=False)""")

    # Cell 10: Ablation Studies Table
    add_md(r"""## 7. Bảng Cờ Ablation Studies (Phục vụ Bài báo Khoa học)

| Cấu hình Ablation | Cờ dòng lệnh (CLI Flag) | Thành phần được kiểm chứng |
|---|---|---|
| **w/o Multi-Scale Temporal Sensitivity** | `--no-mdtp` | Vai trò của nhánh tích chập giãn nở đa tỷ lệ Zero-Init (`MSTSensitivityAdapter`) |
| **w/o Cross-Modal Audio Fusion (XD)** | `--no-audio` | Vai trò của nhánh âm thanh VGGish có cổng dẫn đường thị giác (`GatedCrossModalAudioAdapter`) |
| **w/o Counterfactual Normal Prototypes** | `--no-rupa` | Vai trò của tái tạo mẫu chuẩn động `SGNM` và đối sánh phần dư sự kiện |
| **w/o Safe-Gate Continuation** | `--no-safegate` | So sánh cổng `Safe-Gate` thích ứng với trọng số trộn cố định |
| **1-Crop vs Multi-Crop Test Eval** | `--multi-crop-eval False` | Đóng góp của gom 10-crop (UCF) / 5-crop (XD) không gian khi suy luận |
| **w/o DeSC Collaborative Semantics** | `--collab-alpha 0.0` | Đóng góp của luồng đối sánh ngôn ngữ - thị giác CLIP tại pha test |""")

    nb_path = Path("MA-RUPA.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✨ [SUCCESS] Regenerated {nb_path.resolve()} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    build_notebook()
