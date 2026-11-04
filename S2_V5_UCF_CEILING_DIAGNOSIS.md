# [SYSTEM 2 CLINICAL AUDIT] MA-RUPA v5: CHẨN ĐOÁN SỰ THOÁI HÓA -0.95% AUC

## PHẦN 1: NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE ANALYSIS)

### 🔴 NGUYÊN NHÂN SỐ 1: MẤT CÂN ĐẰNG NGHIÊM TRỌNG TRONG HÀM MẤT MÁT — "CÁI CHẾT NGẠT THỞ" CỦA MIL
Đây là **nguyên nhân số 1**, quyết định 60% thất bại.

Tại **Epoch 1, Iter 1**:
- `loss_MIL = 1.31` (λ=1.0 → đóng góp = **1.31**)
- `loss_IAM = 2.98` (λ=0.4 → đóng góp = **1.192**)
- `loss_ACC` (λ=0.3, giả sử magnitude ~1.5) → ~**0.45**
- `loss_CLASM` (λ=1.1) + `Orth` (λ=1.0) + `Cons` (λ=1.0) + `Gather` (λ=1.0) → tổng phụ trợ ≈ **4.0–4.5**

**Tổng trọng số phụ trợ = 4.8 vs MIL = 1.0.**

Mô hình không còn là "bộ dò dị thường được hỗ trợ bởi phụ trợ" nữa — nó trở thành **một mô hình tối ưu hóa IAM/ACC/Consistency với MIL làm nhiễu**. Gradient từ IAM (đơn nhiệm vụ intention) và ACC (kinetic continuity) **đánh bật** gradient từ MIL (nhiệm vụ chính: phân biệt abnormal/normal segment). Kết quả: không gian biểu diễn bị kéo về hướng tối ưu phụ, làm giảm khả năng phân ly ROC-AUC.

**Bằng chứng thực nghiệm**: v5 đạt đỉnh 87.39% rồi **không bao giờ vượt qua**, trong khi v4 với MIL thống trị duy trì >88% qua 8 epoch. v5 đang tối ưu một hàm mục tiêu khác v4.

---

### 🔴 NGUYÊN NHÂN SỐ 2: ROLLBACK-DAMP TỰ TẠO VÒNG LẶP CHẾT (DEADLOCK LOOP)
Quan sát chuỗi sự kiện:
- Epoch 4: LR 7.0e-5 → **3.44e-5** (damp ~2x)
- Epoch 7: → **8.12e-6** (damp ~4x so với Ep4)
- Epoch 8: → **3.50e-6**
- Epoch 9: → **3.50e-6** (không giảm thêm vì đã chạm đáy)

**Cơ chế tự hủy**: Rollback-Damp phát hiện AUC không cải thiện sau Epoch 6 (87.39%) → restore checkpoint + giảm LR. Nhưng LR quá nhỏ (3.5e-6 ≈ **0** so với scale 7e-5 ban đầu) → mô hình **không thể dịch chuyển** khỏi cực trị địa phương 87.39% → epoch sau vẫn không cải thiện → Rollback-Damp lại trigger → restore cùng một checkpoint.

Đây là **self-fulfilling prophecy**. v4 dùng CosineAnnealing: LR tại Epoch 8 vẫn đủ lớn (~1e-5 đến 5e-5 tùy cấu hình) để mô hình tiếp tục leo dốc và đạt 88.08%. v5 bị "đóng băng sớm" (premature freezing) hoàn toàn do damp factor quá agresive và **không có floor LR**.

---

### 🔴 NGUYÊN NHÂN SỐ 3: GRADIENT SHORT-CIRCUIT TỪ IAM VÀO VISUAL FEATURES
Đoạn code:
```python
f_iam_in = 0.15 * visual_features + 0.85 * visual_features.detach()
```

看似 85% detached là an toàn, nhưng **0.15 gradient vẫn rò rỉ vào `visual_features`** — chính là đầu vào của **LGT Dual-GCN**. Vấn đề:
1. `loss_IAM` khởi đầu bằng **2.98** (gấp 2.3× MIL). Gradient của IAM tại early stage cực lớn và nhiễu.
2. Dual-GCN dựa vào sự ổn định của visual features để xây dựng đồ thị quan hệ temporal. Khi 15% feature bị kéo bởi objective intention (có thể không tương quan với anomaly), ma trận adjacency của GCN bị **nhiễu hệ thống**.
3. Hiệu ứng lan truyền: GCN bị nhiễu → segment-level representation bị sai → MIL loss không giảm được → AUC bị ghìm.

**Solution**: IAM phải là **read-only probe**. Không được phép có bất kỳ gradient nào chảy ngược vào visual backbone.

---

### 🟡 NGUYÊN NHÂN SỐ 4: DET_ANCROSS BỊ CAN THIẾP QUÁ MỨC
`det_anchor` trong Safe-RUPA bị can thiệp bởi `0.15 * acc_d` và `kin_continuity glitch dampening`. 

- `acc_d` (acceleration derivative) là tín hiệu vật lý, không phải xác suất calibrated. Khi trộn 15% vào anchor probability, nó **làm lệch calibration** của detector.
- Trong v4, `det_anchor` thuần túy học từ MIL + CLASM. v5 thêm thành phần vật lý chưa được normalize đúng → xác suất đầu ra bị dịch chuyển hệ thống (systematic shift), làm giảm AUC vì AUC phụ thuộc vào thứ tự xếp hạng (ranking), không phải threshold.

---

### 🟡 NGUYÊN NHÂN SỐ 5: COLLAB-ALPHA TĂNG QUÁ MỨC (0.05 → 0.08)
Tăng 60% trọng số collaboration có thể gây **over-regularization**. Nếu `collab-alpha` điều khiển sự đồng nhất giữa hai nhánh (hoặc teacher-student), giá trị