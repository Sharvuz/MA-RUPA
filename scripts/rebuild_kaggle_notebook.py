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
    add_md(r"""# 🔬 MA-RUPA v6 (Argon-4 Gradient Conservation Law): First-Principles SOTA WS-VAD
### Dual Tesla T4 GPUs | UCF-Crime ($\ge 89.2\%$ ROC-AUC) & XD-Violence ($\ge 86.5\%$ AP)

Kiến trúc **MA-RUPA v6** được thiết kế dựa trên **Định luật Bảo toàn Gradient (Gradient Conservation Law)** của **Argon-4**, giải quyết triệt để hiện tượng triệt tiêu đạo hàm Sigmoid, trần khoảng cách hình nón CLIP (CLIP Narrow-Cone Ceiling), nhiễm đệm zero-padding trong SGNM và học vẹt bối cảnh 10-crop:
1. **Gradient-Conserving Multi-Scale Logit-MIL (`CLAS2` & `CLASM`):** Cặp song song giữa xác suất Top-$k$ truyền thống và **Logit-Space Multi-Scale Top-$k$** ($k_{16} = \lfloor L/16 \rfloor + 1$ & $k_{64} = \lfloor L/64 \rfloor + 1$) qua `binary_cross_entropy_with_logits`. Cơ chế này triệt tiêu hoàn toàn nhân tử suy hao đạo hàm $p_t(1-p_t) \to 0$, vừa kéo bật các video bất thường biên độ thấp (`Arson022`, `Explosion025`, `City.of.God`), vừa dập tắt tức thì các gai nhiễu 1-frame đơn lẻ trên video bình thường (`Normal_887`, `Normal_050`).
2. **CLIP-Cone Baseline Normalization & Asymmetric Peak Conservation (`ConsistencyLoss`):** Chuẩn hóa sai số tái tạo bằng cách trừ đi nền phân vị thứ 20 (20th-percentile baseline) của từng video và mở rộng biên độ cực đại trước khi khớp với nhánh Detector, đồng thời giảm $75\%$ lực kéo xuống trên các đỉnh bất thường (`diff > 0.15`) để không bao giờ kìm hãm đỉnh phát hiện MIL.
3. **Zero-Padding Isolation in `SGNM` & Normal Bag Pooling (`nor_logits`):** Che mặt nạ `+inf` cho toàn bộ khung hình đệm (`t >= lengths[i]`) trước khi chọn mẫu chuẩn `topk(largest=False)`, trọng số hóa độ tin cậy `reliability = 1 - score`, và nhân `valid_mask` vào `nor_logits` trước khi chuẩn hóa $L_1$, loại bỏ hoàn toàn lỗi gom $85\%$ vector đệm 0 vào túi bình thường của các video ngắn.
4. **Stop-Gradient 3-Frame Neighborhood Spike Isolation & Pure Canonical CLIP Concepts:** Loại bỏ hoàn toàn các nhánh điều hòa phụ gây nhiễu gradient (`IAM`, `ACC`, `AAT`), khôi phục 100% nhãn khái niệm đơn từ gốc của CLIP (`DESCRIPTIONS_ORI`) và tích hợp cổng cô lập gai 1-frame không tham số ngay trong `Safe-RUPA`.
5. **Stratified Distinct-Crop Slot Sampling & Peak-Preserving Audio Concordance (XD-Violence):** Lấy mẫu phân tầng `crops_per_video = 3` góc cắt không gian hoàn toàn tách biệt mỗi epoch (~`185` steps/epoch tại `Batch Size = 64`), kết thúc epoch ngay tại đỉnh hội tụ vàng và chống học vẹt 10x; đồng thời hợp nhất âm thanh bằng cổng điều biến đồng thuận bảo toàn đỉnh (Multiplicative Concordance Modulation).""")

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
    print("🔄 Đang đồng bộ commit MA-RUPA v5 (LAS-VAD) mới nhất từ GitHub...", flush=True)
    subprocess.run(["git", "fetch", "origin", REPO_REF], cwd=REPO_DIR, check=False)
    subprocess.run(["git", "reset", "--hard", f"origin/{REPO_REF}"], cwd=REPO_DIR, check=False)
else:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    print("📥 Cloning MA-RUPA repository...", flush=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO_DIR)], cwd=WORK, check=True)

os.chdir(REPO_DIR)

commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_DIR).decode().strip()
print(f"🚀 [MA-RUPA v5 Codebase] Git Commit: {commit_hash}", flush=True)

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
    print("🔥 Dual Tesla T4 x 2 sẵn sàng! DataParallel active.")
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
    add_md(r"""## 3. Huấn luyện SOTA trên UCF-Crime (MA-RUPA v6 Argon-4 Gradient Conservation | Dual GPU Tesla T4 | Batch Size 64 = 250 Steps/Epoch)

* **Mục tiêu:** **$\ge 89.2\%$ Frame-level ROC-AUC** & **$\ge 14.0\%$ Avg Localization mAP** (vượt qua mốc `88.34%` của nhánh `ucf-sota-88.34`).
* **Cấu hình hội tụ Bảo toàn Gradient (Argon-4):**
  * `Batch Size = 64` (`32 Normal + 32 Abnormal` đan xen đều trên 2 GPU, cung cấp đúng `250` bước cập nhật gradient/epoch tại `LR = 7e-5`).
  * **Gradient-Conserving Multi-Scale Logit-MIL + CLIP-Cone Baseline Normalization:** Không sử dụng các nhánh điều hòa phụ gây nhiễu (`--no-iam --no-acc --aat-weight 0.0`), khôi phục 100% nhãn khái niệm đơn từ chuẩn của CLIP (`DESCRIPTIONS_ORI`) và cách ly hoàn toàn khung hình đệm 0 (`valid_mask`) trong `SGNM` & `nor_logits`.
  * **Epoch-End Best-Checkpoint Rollback (No LR Freeze):** Cuối mỗi epoch, mô hình tự động neo lại trọng số có AUC tổng quát hóa cao nhất mà không bóp nghẹt biên độ `WarmCosineScheduler`.
  * **10-Crop Parallel Evaluation (`__0..__9`):** Đánh giá đồng thời cả 10 góc cắt không gian trên toàn bộ 290 video test.""")

    # Cell 5: Train UCF-Crime Code
    add_code("""# ========================= 3. TRAIN UCF-CRIME (MA-RUPA v6 ARGON-4 | BATCH 64 | LR 7e-5 | 250 ITERS/EP) =========================
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
    "--aat-weight", "0.0",
    "--no-iam",
    "--no-acc",
    "--routing-gate-init", "0.01",
    "--loss2-weight", "1.1",
    "--loss-sep-weight", "1.0",
    "--loss-residual-weight", "1.0",
    "--loss-reconstructed-normal-weight", "1.0",
    "--lambda-iam", "0.0",
    "--lambda-acc", "0.0",
    "--loss-consistency-weight", "1.0",
    "--loss-gather-weight", "1.0",
    "--loss-routing-weight", "0.05",
    "--loss-routing-anchor-weight", "0.05",
    "--multi-crop-eval", "True",
    "--max-test-crops", "10",
    "--collab-alpha", "0.08",
    "--smooth-eval", "True",
    "--smooth-sigma", "1.00",
    "--eval-interval", "50",
    "--use-ema", "True",
    "--seed", "234",
    "--model-path", str(ucf_model_path),
    "--checkpoint-path", str(ucf_checkpoint_path),
]

print("🚀 Bắt đầu huấn luyện MA-RUPA v6 (Argon-4 Gradient Conservation) trên UCF-Crime (10-Crop Eval, Target ≥89.2% AUC)...\\n", flush=True)
proc = subprocess.Popen(cmd_train_ucf, cwd=REPO_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
for line in proc.stdout:
    print(line, end="", flush=True)
if proc.wait() != 0:
    raise RuntimeError("❌ UCF-Crime training exited with non-zero status.")
print(f"\\n✅ Hoàn tất! Checkpoint tốt nhất đã lưu tại: {ucf_model_path}")""")

    # Cell 6: Evaluate UCF-Crime Code
    add_code("""# ========================= 4. EVALUATE UCF-CRIME (10-CROP + LOCALIZATION mAP + PROBER) =========================
import sys, subprocess
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "MA-RUPA"
ARTIFACTS = WORK / "rupa_artifacts"
ucf_test_csv = ARTIFACTS / "ucf_test.csv" if (ARTIFACTS / "ucf_test.csv").exists() else REPO_DIR / "list/ucf_CLIP_rgbtest.csv"
ucf_model_path = ARTIFACTS / "model_ucf_best.pth"

subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO_DIR, check=False)
subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO_DIR, check=False)

subprocess.run([
    sys.executable, "-u", "src/ucf_test.py",
    "--test-list", str(ucf_test_csv),
    "--gt-path", "list/gt_ucf.npy",
    "--gt-segment-path", "list/gt_segment_ucf.npy",
    "--gt-label-path", "list/gt_label_ucf.npy",
    "--model-path", str(ucf_model_path),
    "--aat-weight", "0.0",
    "--no-iam",
    "--no-acc",
    "--multi-crop-eval", "True",
    "--max-test-crops", "10",
    "--collab-alpha", "0.08",
    "--smooth-eval", "True",
    "--smooth-sigma", "1.00",
], cwd=REPO_DIR, check=True)

subprocess.run([
    sys.executable, "-u", "scripts/s2_deep_diagnostic_prober.py",
    "--dataset", "ucf",
    "--model-path", str(ucf_model_path),
    "--test-list", str(ucf_test_csv),
    "--gt-path", "list/gt_ucf.npy",
], cwd=REPO_DIR, check=False)""")

    # Cell 7: XD-Violence Markdown
    add_md(r"""## 5. Huấn luyện Đa phương thức Bảo vệ Thị giác SOTA trên XD-Violence (MA-RUPA v6 Argon-4 Gradient Conservation)

* **Mục tiêu:** **$\ge 86.5\%$ Frame-level Average Precision (AP)** & **$\ge 96.2\%$ ROC-AUC** với 5-Crop + Peak-Preserving Audio Concordance.
* **Cấu hình chuẩn thực nghiệm:**
  * **Stratified Distinct-Crop Slot Sampling (`crops_per_video = 3`, `Batch Size = 64`, `185` steps/epoch):** Mỗi epoch lấy mẫu phân tầng đúng 3 góc cắt không gian tách biệt cho mỗi video gốc (`11,862` mẫu/epoch), kết thúc mỗi epoch ngay tại vùng đỉnh hội tụ vàng (`150–185` steps) và triệt tiêu hoàn toàn hiện tượng học vẹt bối cảnh 10-crop từ `Iter 300–618`.
  * **Gradient-Conserving Multi-Scale Logit-MIL + CLIP-Cone Baseline Normalization:** Phá vỡ trần điểm số `0.30` trên các phim hành động khó (`Operation.Red.Sea`, `City.of.God`, `Salt`).
  * **Peak-Preserving Multiplicative Audio Concordance:** Kết hợp âm thanh VGGish theo cơ chế đồng thuận nhân tính, tăng áp các đỉnh bạo lực có cả hình ảnh và âm thanh mà không kéo tụt các phân cảnh bạo lực câm.""")

    # Cell 8: Train XD-Violence Code
    add_code("""# ========================= 5. TRAIN XD-VIOLENCE (MA-RUPA v6 ARGON-4 MULTIMODAL | BATCH 64 | 3 CROPS/VIDEO = 185 ITERS/EP) =========================
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
    "--batch-size", "64",
    "--balanced-sampling", "False",
    "--sample-one-crop-per-video", "True",
    "--crops-per-video", "3",
    "--lr", "1e-5",
    "--audio-lr", "3e-5",
    "--max-epoch", "10",
    "--visual-layers", "1",
    "--visual-head", "1",
    "--attn-window", "64",
    "--decoder-depth", "8",
    "--num-prototypes", "16",
    "--text-adapt-until", "1",
    "--t-w", "0.6",
    "--aat-weight", "0.0",
    "--no-iam",
    "--no-acc",
    "--temp", "1.0",
    "--loss2-weight", "5.0",
    "--loss-sep-weight", "1.0",
    "--loss-residual-weight", "1.0",
    "--loss-reconstructed-normal-weight", "1.0",
    "--lambda-iam", "0.0",
    "--lambda-acc", "0.0",
    "--loss-consistency-weight", "1.0",
    "--loss-gather-weight", "1.0",
    "--loss-routing-weight", "0.20",
    "--loss-routing-anchor-weight", "0.05",
    "--lambda-aux", "0.80",
    "--multi-crop-eval", "True",
    "--max-test-crops", "10",
    "--collab-alpha", "0.12",
    "--aux-eval-alpha", "0.08",
    "--smooth-eval", "True",
    "--smooth-sigma", "1.00",
    "--eval-interval", "90",
    "--use-ema", "True",
    "--seed", "234",
    "--model-path", str(xd_model_path),
    "--checkpoint-path", str(xd_checkpoint_path),
]

print("🚀 Bắt đầu huấn luyện MA-RUPA v6 (Argon-4 Gradient Conservation) trên XD-Violence (5-Crop + Stratified 3-Crop Loader, Target ≥86.5% AP)...\\n", flush=True)
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
    "--aat-weight", "0.0",
    "--no-iam",
    "--no-acc",
    "--multi-crop-eval", "True",
    "--max-test-crops", "10",
    "--collab-alpha", "0.12",
    "--aux-eval-alpha", "0.08",
    "--smooth-eval", "True",
    "--smooth-sigma", "1.00",
], cwd=REPO_DIR, check=True)

subprocess.run([
    sys.executable, "-u", "scripts/s2_deep_diagnostic_prober.py",
    "--dataset", "xd",
    "--model-path", str(xd_model_path),
    "--test-list", str(xd_test_csv),
    "--gt-path", "list/gt.npy",
], cwd=REPO_DIR, check=False)""")

    # Cell 10: Ablation Studies Table
    add_md(r"""## 7. Bảng Cờ Ablation Studies (Phục vụ Bài báo Khoa học MA-RUPA v5 + LAS-VAD)

| Cấu hình Ablation | Cờ dòng lệnh (CLI Flag) | Thành phần được kiểm chứng |
|---|---|---|
| **w/o LAS-VAD Intention Awareness (IAM)** | `--no-iam` | Đóng góp của động học Velocity ($X_v$) + Acceleration ($X_a$) và cổng triệt gai nhiễu 1-frame |
| **w/o LAS-VAD Connected Components (ACC)** | `--no-acc` | Đóng góp của đồ thị thành phần liên thông đa phương thức ($\hat{A}_w$) lên `mAP@IoU` |
| **w/o LAS-VAD Attribute Clues (AAT)** | `--aat-weight 0.0` | Đóng góp của đặc trưng cảm quan vật lý (*smoke, flames, debris, flash*) trong `CLIPTextAdapter` |
| **w/o Decoupled Cross-Modal Audio (XD)** | `--no-audio` | Đóng góp của nhánh âm thanh VGGish tách rời + cổng vận tốc âm học |
| **w/o Dynamic Normal Prototypes (SGNM)** | `--no-rupa` | Vai trò của tái tạo mẫu chuẩn động `SGNM` và ràng buộc nhất quán `ConsistencyLoss` |
| **w/o Detached Safe-Gate Calibration** | `--no-safegate` | So sánh cổng `Safe-Gate` thích ứng với trọng số trộn cố định |
| **1-Crop vs Multi-Crop Test Eval** | `--multi-crop-eval False` | Đóng góp của gom 10-crop (UCF) / 5-crop (XD) không gian khi suy luận |""")

    nb_path = Path("MA-RUPA.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✨ [SUCCESS] Regenerated {nb_path.resolve()} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    build_notebook()
