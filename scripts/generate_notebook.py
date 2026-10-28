import json
from pathlib import Path


def create_notebook():
    nb = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py"
            },
            "accelerator": "GPU"
        },
        "nbformat_minor": 5,
        "nbformat": 4,
        "cells": []
    }

    def md(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else source.strip().splitlines(keepends=True)
        }

    def code(source):
        return {
            "cell_type": "code",
            "metadata": {"trusted": True},
            "execution_count": None,
            "outputs": [],
            "source": source if isinstance(source, list) else source.strip().splitlines(keepends=True)
        }

    # ---------------------------------------------------------
    # Cell 0: Header
    # ---------------------------------------------------------
    nb["cells"].append(md("""# MA-RUPA: Modality-Adaptive Multiscale Prototype Alignment Network
### Weakly-Supervised Video Anomaly Detection (WS-VAD) on Kaggle GPU T4 x 2

**Architectural Innovations:**
1. **End-to-End Trainable from Scratch:** Completely removes dependency on pre-trained DSANet checkpoints and frozen two-stage warm-starting (`main-lr=0.0` is removed).
2. **Dual GPU T4 x 2 Acceleration:** Built-in `torch.nn.DataParallel` distributing batch workloads across both GPUs.
3. **Automatic Dataset Discovery with `# -> +` Resolution:** Automatically finds pre-extracted visual and audio `.npy` features, unzips archives if needed, and maps all Kaggle `# -> +` renamed files.
4. **Multi-Scale Dilated Temporal Backbone (`MDTBackbone`):** Windowed Attention + MDTP Dilations (1, 2, 4) with GLU.
5. **Modality-Adaptive Branch (`ModalityAdaptiveBranch`):** Multimodal Audio-Visual fusion for XD-Violence and automatic Motion-Residual mode ($M_1, M_2$) for silent UCF-Crime.
6. **Gradient-Safe Prototype Alignment (`GradientSafeRUPA`):** Controlled gradient detachment preventing prototype gradient collapse.
7. **Quad-Stream Dynamic Safe Gate (`QuadSafeGate`):** Dynamically routes $S_{det}, S_{rec}, S_{sem}, S_{aux}$ with safe bias initialization (-4.5).
"""))

    # ---------------------------------------------------------
    # Cell 1: Environment Setup & Hotfixes
    # ---------------------------------------------------------
    nb["cells"].append(code("""# ========================= 1. SETUP ENVIRONMENT & GPU PREFLIGHT =========================
import os, sys, shutil, subprocess, time
from pathlib import Path
import torch

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORK / "MA-RUPA"
ARTIFACTS = WORK / "rupa_artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Luôn luôn chuyển về /kaggle/working trước để tránh lỗi Linux "Unable to read current working directory" khi re-run cell
os.chdir(WORK)

# Clone or synchronize repository
REPO_URL = "https://github.com/Sharvuz/MA-RUPA.git"
REPO_REF = None  # Đặt tên branch nếu cần (ví dụ "main")

if (REPO_DIR / ".git").exists():
    print("🔄 Repository đã tồn tại. Đang đồng bộ commit mới nhất từ GitHub...", flush=True)
    subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO_DIR, check=False)
    subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO_DIR, check=False)
else:
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    print("📥 Cloning repository from GitHub...", flush=True)
    clone_cmd = ["git", "clone", "--depth", "1"]
    if REPO_REF:
        clone_cmd += ["--branch", REPO_REF]
    clone_cmd += [REPO_URL, str(REPO_DIR)]
    subprocess.run(clone_cmd, cwd=WORK, check=True)

os.chdir(REPO_DIR)

# Install required dependencies (bao gồm ipdb và packaging để tránh lỗi)
print("📦 Installing dependencies...", flush=True)
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "ftfy", "regex", "einops==0.8.0", "ipdb", "packaging", "scikit-learn", "pandas", "scipy"
], check=True)

# Auto-fix bất kỳ lỗi import cũ nào nếu repo chưa kịp push commit mới nhất
adapter_file = REPO_DIR / "src/utils/adapter_modules.py"
if adapter_file.exists():
    txt = adapter_file.read_text(encoding="utf-8")
    if "import ipdb" in txt:
        txt = txt.replace("import ipdb\\n", "").replace("import ipdb", "")
        adapter_file.write_text(txt, encoding="utf-8")
        print("🔧 Auto-hotfixed: Removed unused 'import ipdb' from adapter_modules.py")

clip_file = REPO_DIR / "src/clip/clip.py"
if clip_file.exists():
    txt = clip_file.read_text(encoding="utf-8")
    if "from pkg_resources import packaging" in txt:
        txt = txt.replace(
            "from pkg_resources import packaging",
            "try:\\n    import packaging.version as _pkg_version\\nexcept ImportError:\\n    _pkg_version = None"
        )
        clip_file.write_text(txt, encoding="utf-8")
        print("🔧 Auto-hotfixed: Patched pkg_resources warning in clip.py")

# Hardware & GPU Verification
print("\\n" + "=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
gpu_count = torch.cuda.device_count()
print(f"Detected GPUs  : {gpu_count}")
for i in range(gpu_count):
    print(f"  GPU [{i}]: {torch.cuda.get_device_name(i)}")

if gpu_count >= 2:
    print("🔥 Dual GPU T4 x 2 detected! DataParallel will be automatically activated.")
elif gpu_count == 1:
    print("⚡ Single GPU detected.")
else:
    print("⚠️ No GPU detected. Please set Accelerator = GPU T4 x 2 in Notebook Settings.")
print("=" * 60)
"""))

    # ---------------------------------------------------------
    # Cell 2: Data Radar with # -> + mapping
    # ---------------------------------------------------------
    nb["cells"].append(code("""# ========================= 2. RADAR: QUÉT DỮ LIỆU & XỬ LÝ '#' -> '+' =========================
import zipfile
from collections import defaultdict
import numpy as np
import pandas as pd

print("🔍 Đang quét toàn bộ /kaggle/input và /kaggle/working để lập bản đồ feature (.npy)...")

# 1. Tự động giải nén file .zip nếu có
zip_files = list(INPUT.rglob("*.zip"))
if zip_files:
    extract_dir = WORK / "extracted_features"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"📦 Tìm thấy {len(zip_files)} file zip. Đang kiểm tra giải nén...")
    for zf in zip_files:
        dest = extract_dir / zf.stem
        if not dest.exists() or not any(dest.iterdir()):
            dest.mkdir(parents=True, exist_ok=True)
            print(f"  Unzipping {zf.name} -> {dest.name}...")
            with zipfile.ZipFile(zf, 'r') as archive:
                archive.extractall(dest)

# 2. Quét tất cả các file .npy trên hệ thống
search_roots = [INPUT, WORK]
all_npy_files = []
for root in search_roots:
    if root.exists():
        all_npy_files.extend(root.rglob("*.npy"))

print(f"📡 Radar phát hiện tổng cộng {len(all_npy_files)} tệp tin .npy trên hệ thống!")

# 3. Lập bản đồ tra cứu đa chiều (Hỗ trợ đổi tên '#' -> '+' trên Kaggle)
file_map = {}
for p in all_npy_files:
    fname = p.name
    full_path = str(p.resolve())
    file_map[fname] = full_path
    
    # Kaggle không cho phép ký tự '#' nên đổi thành '+'
    if '+' in fname:
        file_map[fname.replace('+', '#')] = full_path
    if '#' in fname:
        file_map[fname.replace('#', '+')] = full_path
    if '_' in fname and ('+' in fname or '#' in fname):
        file_map[fname.replace('+', '_').replace('#', '_')] = full_path


def sync_csv_list(dataset_name, csv_rel_path, output_name):
    \"\"\"Đồng bộ đường dẫn CSV gốc sang đường dẫn thực tế trên Kaggle.\"\"\"
    input_csv = REPO_DIR / csv_rel_path
    output_csv = ARTIFACTS / output_name
    
    if not input_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy file danh sách mẫu: {input_csv}")
        
    df = pd.read_csv(input_csv)
    resolved_paths = []
    missing_files = []
    
    for orig_path in df['path'].astype(str):
        base_name = Path(orig_path).name
        
        # Danh sách các biến thể tên khả dĩ
        candidates = [
            base_name,
            base_name.replace('#', '+'),
            base_name.replace('+', '#'),
            base_name.replace('#', '_'),
            base_name.replace('#', '')
        ]
        
        match = None
        for cand in candidates:
            if cand in file_map:
                match = file_map[cand]
                break
                
        if match:
            resolved_paths.append(match)
        else:
            missing_files.append(base_name)
            resolved_paths.append(orig_path)
            
    if missing_files:
        print(f"⚠️ Cảnh báo: Thiếu {len(missing_files)}/{len(df)} files cho {output_name}. Ví dụ: {missing_files[:3]}")
    else:
        print(f"✅ Đồng bộ 100% {len(df)} file cho {output_name} thành công!")
        
    df['path'] = resolved_paths
    df.to_csv(output_csv, index=False)
    return output_csv

# Đồng bộ danh sách cho cả 2 tập dữ liệu
ucf_train_csv = sync_csv_list("ucf", "list/ucf_CLIP_rgb.csv", "ucf_train.csv")
ucf_test_csv = sync_csv_list("ucf", "list/ucf_CLIP_rgbtest.csv", "ucf_test.csv")
xd_train_csv = sync_csv_list("xd", "list/xd_CLIP_rgb.csv", "xd_train.csv")
xd_test_csv = sync_csv_list("xd", "list/xd_CLIP_rgbtest.csv", "xd_test.csv")

# Kiểm tra thử nghiệm 1 file bất kỳ
sample_p = Path(pd.read_csv(ucf_train_csv).iloc[0]['path'])
if sample_p.exists():
    sample_feat = np.load(sample_p, mmap_mode='r')
    print(f"\\n🎯 Feature Preflight Test: {sample_p.name} | Shape: {sample_feat.shape}")
"""))

    # ---------------------------------------------------------
    # Cell 3: Markdown UCF
    # ---------------------------------------------------------
    nb["cells"].append(md("""## 3. UCF-Crime Training (Single-Stage End-to-End on GPU T4 x 2)

- **Mục tiêu hiệu năng:** **89.5% – 90.0% Frame-level ROC-AUC** (~40% AP).
- **Cơ chế:** Huấn luyện End-to-End từ đầu (không dùng DSANet cũ). Branch tự động kích hoạt **Motion-Residual Mode** ($M_1, M_2$) bổ sung thông tin động học cho CCTV không có audio.
- **Multi-GPU:** Tự động kích hoạt `DataParallel` chia đều batch 64 (32 mẫu/GPU) trên 2x GPU T4.
"""))

    # ---------------------------------------------------------
    # Cell 4: Train UCF
    # ---------------------------------------------------------
    nb["cells"].append(code("""# ========================= 3. TRAIN UCF-CRIME (GPU T4 x 2) =========================
import subprocess, sys

# Tự động đồng bộ commit mới nhất từ GitHub
subprocess.run(["git", "fetch", "origin", "main"], cwd=REPO_DIR, check=False)
subprocess.run(["git", "reset", "--hard", "origin/main"], cwd=REPO_DIR, check=False)

UCF_BATCH_SIZE = 64
UCF_EPOCHS = 25
UCF_CROPS_PER_VIDEO = 2  # DPSS: 2 distinct crops/video/epoch -> 50 iters/epoch -> 1,250 gradient steps!
UCF_LR = 2.5e-5
SEED = 234

ucf_model_path = ARTIFACTS / "model_ucf_best.pth"
ucf_checkpoint_path = ARTIFACTS / "checkpoint_ucf.pth"

cmd_train_ucf = [
    sys.executable, "-u", "src/ucf_train.py",
    "--train-list", str(ucf_train_csv),
    "--test-list", str(ucf_test_csv),
    "--gt-path", "list/gt_ucf.npy",
    "--gt-segment-path", "list/gt_segment_ucf.npy",
    "--gt-label-path", "list/gt_label_ucf.npy",
    "--batch-size", str(UCF_BATCH_SIZE),
    "--lr", str(UCF_LR),
    "--max-epoch", str(UCF_EPOCHS),
    "--sample-one-crop-per-video", "True",
    "--crops-per-video", str(UCF_CROPS_PER_VIDEO),
    "--attn-window", "0",          # 0 = Full Global Temporal Attention (256x256 all-pairs relational reasoning)
    "--curriculum-epochs", "3.0",
    "--sampling", "topk",
    "--topk-ratio", "8",          # L / 8 (~12.5% frames) bao quát trọn vẹn diễn biến tội phạm nhiều phút
    "--label-smoothing", "0.02",  # Ngăn BCE loss tụt về 0, giữ gradient sống qua các epoch sau
    "--lambda-aux", "0.15",       # Giảm trọng số nhánh Aux (Motion Residual) chống overfit sớm
    "--lambda-smooth", "0.06",     # Tăng làm phẳng biến thiên cục bộ giữa frame t và t-1, giảm False Positives
    "--seed", str(SEED),
    "--model-path", str(ucf_model_path),
    "--checkpoint-path", str(ucf_checkpoint_path),
    "--smooth-eval", "True",
    "--smooth-sigma", "0.50"
]

print("🚀 Bắt đầu huấn luyện End-to-End MA-RUPA trên UCF-Crime (Tối ưu SOTA 88-90% AUC) với GPU T4x2...\\n", flush=True)

process = subprocess.Popen(
    cmd_train_ucf, cwd=REPO_DIR, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1
)

for line in process.stdout:
    print(line, end="")
    sys.stdout.flush()

ret_code = process.wait()
if ret_code != 0:
    raise RuntimeError(f"❌ Training UCF-Crime thất bại với mã lỗi {ret_code}")
else:
    print(f"\\n✅ Huấn luyện UCF-Crime thành công! Model tốt nhất lưu tại: {ucf_model_path}")
"""))

    # ---------------------------------------------------------
    # Cell 5: Eval UCF
    # ---------------------------------------------------------
    nb["cells"].append(code("""# ========================= 4. EVALUATE UCF-CRIME =========================
cmd_eval_ucf = [
    sys.executable, "-u", "src/ucf_test.py",
    "--test-list", str(ucf_test_csv),
    "--gt-path", "list/gt_ucf.npy",
    "--gt-segment-path", "list/gt_segment_ucf.npy",
    "--gt-label-path", "list/gt_label_ucf.npy",
    "--model-path", str(ucf_model_path),
    "--smooth-eval", "True",
    "--smooth-sigma", "0.50"
]

print("🔍 Đang đánh giá model UCF-Crime tốt nhất với Multi-Stream Evaluation & Gaussian Smoothing (sigma=2.0)...", flush=True)
subprocess.run(cmd_eval_ucf, cwd=REPO_DIR, check=True)
"""))

    # ---------------------------------------------------------
    # Cell 6: Markdown XD
    # ---------------------------------------------------------
    nb["cells"].append(md("""## 4. XD-Violence Training (Multimodal Audio-Visual on GPU T4 x 2)

- **Mục tiêu hiệu năng:** **86.5% – 87.5%+ Frame-level Average Precision (AP)**.
- **Cơ chế:** Huấn luyện End-to-End đa phương thức kết hợp Visual CLIP (512-D) và Audio (128-D VGGish hoặc 512-D CLAP/AST) qua cổng lọc nhiễu âm thanh $G_{av}$. Nếu không có file audio, nhánh tự động chạy Motion Residual.
- **Multi-GPU:** Tự động kích hoạt `DataParallel` trên 2x GPU T4.
"""))

    # ---------------------------------------------------------
    # Cell 7: Train XD
    # ---------------------------------------------------------
    nb["cells"].append(code("""# ========================= 5. TRAIN XD-VIOLENCE (GPU T4 x 2) =========================
XD_BATCH_SIZE = 64
XD_EPOCHS = 25
XD_CROPS_PER_VIDEO = 2  # DPSS: 2 distinct crops/video/epoch -> ~59 iters/epoch -> 1,475 gradient steps!
XD_LR = 1.5e-5
SEED = 234

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
    "--batch-size", str(XD_BATCH_SIZE),
    "--lr", str(XD_LR),
    "--max-epoch", str(XD_EPOCHS),
    "--sample-one-crop-per-video", "True",
    "--crops-per-video", str(XD_CROPS_PER_VIDEO),
    "--attn-window", "0",          # 0 = Full Global Temporal Attention
    "--curriculum-epochs", "3.0",
    "--sampling", "topk",
    "--seed", str(SEED),
    "--model-path", str(xd_model_path),
    "--checkpoint-path", str(xd_checkpoint_path),
    "--smooth-eval", "True",
    "--smooth-sigma", "0.50"
]

print("🚀 Bắt đầu huấn luyện End-to-End MA-RUPA trên XD-Violence với GPU T4x2...\\n", flush=True)

process = subprocess.Popen(
    cmd_train_xd, cwd=REPO_DIR, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1
)

for line in process.stdout:
    print(line, end="")
    sys.stdout.flush()

ret_code = process.wait()
if ret_code != 0:
    raise RuntimeError(f"❌ Training XD-Violence thất bại với mã lỗi {ret_code}")
else:
    print(f"\\n✅ Huấn luyện XD-Violence thành công! Model tốt nhất lưu tại: {xd_model_path}")
"""))

    # ---------------------------------------------------------
    # Cell 8: Eval XD
    # ---------------------------------------------------------
    nb["cells"].append(code("""# ========================= 6. EVALUATE XD-VIOLENCE =========================
cmd_eval_xd = [
    sys.executable, "-u", "src/xd_test.py",
    "--test-list", str(xd_test_csv),
    "--gt-path", "list/gt.npy",
    "--gt-segment-path", "list/gt_segment.npy",
    "--gt-label-path", "list/gt_label.npy",
    "--model-path", str(xd_model_path),
    "--smooth-eval", "True",
    "--smooth-sigma", "0.50"
]

print("🔍 Đang đánh giá model XD-Violence tốt nhất...", flush=True)
subprocess.run(cmd_eval_xd, cwd=REPO_DIR, check=True)
"""))

    # ---------------------------------------------------------
    # Cell 9: Ablation Studies
    # ---------------------------------------------------------
    nb["cells"].append(md("""## 5. Ablation Studies (Phục vụ bài báo Q1)

Các cờ dòng lệnh cho phép bạn thử nghiệm và tái tạo đầy đủ bảng Ablation trong bài báo khoa học:

| Cấu hình nghiên cứu | Cờ dòng lệnh | Ý nghĩa nghiên cứu |
|---|---|---|
| **w/o MDTP** | `--no-mdtp` | Đánh giá vai trò của Multi-Scale Dilated Pyramid |
| **w/o Audio / Motion** | `--no-audio` | Đánh giá vai trò của nhánh phụ trợ đa phương thức |
| **w/o RUPA** | `--no-rupa` | Đánh giá vai trò của tái tạo counterfactual và DNP |
| **w/o Safe Gate** | `--no-safegate` | Đánh giá vai trò của cổng định tuyến bất định động |
| **Otsu Selection** | `--sampling otsu` | So sánh phân đoạn ngưỡng Otsu với Top-K cố định |
| **w/o Score Smoothing** | `--smooth-eval False` | Đánh giá hiệu quả của bộ lọc Gauss làm mượt điểm |
"""))

    nb["cells"].append(code("""# Ví dụ chạy 1 thí nghiệm Ablation Study:
# !python src/xd_train.py --train-list /kaggle/working/rupa_artifacts/xd_train.csv --test-list /kaggle/working/rupa_artifacts/xd_test.csv --no-audio --no-mdtp
"""))

    # Write notebook file
    out_path = Path("MA-RUPA.ipynb")
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Generated {out_path} successfully!")


if __name__ == '__main__':
    create_notebook()
