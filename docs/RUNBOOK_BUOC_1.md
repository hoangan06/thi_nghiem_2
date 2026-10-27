# Runbook bước 1 — kiểm pipeline trên Llama-3-8B-Instruct (GPU thuê)

Mục tiêu: chạy ba script của bước 1 trên một GPU thuê, kéo kết quả về, đọc kết quả. Ước tổng thời gian
khoảng 1,5–2 giờ máy (con số ước, chưa đo).

## 0. Trước khi thuê (máy cá nhân, WSL)

1. Commit code của kho `thi_nghiem_2` (Ân tự chạy git). Ghi lại commit:
   ```bash
   cd /mnt/d/thesis/thi_nghiem_2
   git status --porcelain        # phải rỗng
   git rev-parse HEAD            # GHI LẠI
   ```
2. Token Hugging Face có quyền đọc `meta-llama/Meta-Llama-3-8B-Instruct`. Repo này được cấp quyền RIÊNG
   (khác Llama-3.1): mở trang repo trên Hugging Face, đồng ý giấy phép Llama 3 nếu chưa làm.

## 1. Chọn máy (Vast.ai)

| Tiêu chí | Giá trị |
|---|---|
| GPU | 1 GPU, **ưu tiên 80 GB** (A100 80GB, H100 80GB); tối thiểu 48 GB (L40S, A6000). Phần 2 nạp lại mô hình ở float32 (≈ 32 GB), ép lên một GPU |
| Đĩa | ≥ 60 GB |
| Ảnh | Ubuntu có CUDA ≥ 12.8, Python 3.12 |
| Mạng | Direct SSH |

Đặt bí danh `may-thue` như các phiên trước (IP, PORT của máy).

## 2. Đưa code lên, dựng môi trường

Trên máy cá nhân:
```bash
rsync -rlt --exclude results --exclude __pycache__ /mnt/d/thesis/thi_nghiem_2/ may-thue:/workspace/thi_nghiem_2/
```
(Chép cả `.git` để `meta.json` ghi được commit của kho.)

Trên máy thuê, vào tmux trước (`tmux new -s b1`, hoặc dùng phiên tmux Vast mở sẵn):
```bash
cd /workspace/thi_nghiem_2
python3 -m venv /workspace/venv && . /workspace/venv/bin/activate
pip install -U pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128     # ⏱ 3–8 phút
pip install -r requirements.txt
python -c "import torch,transformers,tokenizers; print(torch.__version__, transformers.__version__, tokenizers.__version__, torch.cuda.get_device_name(0))"
python -m pytest -q -p no:cacheprovider tests/          # ⏱ 2–4 phút; phải: 50 passed
```
Phải thấy `5.17.0` và `0.23.2`, tên GPU, và `50 passed`. (Bộ test đã chạy xanh trên CPU với torch 2.14.1 và
2.11.0; dòng `50 passed` trên máy thuê xác nhận lại với đúng bản torch đã cài.) Có test hỏng thì dừng, chép log về, không chạy tiếp.

## 3. Tải mô hình

```bash
export HF_HOME=/workspace/hf
read -rsp "HF token: " HF_TOKEN; echo; export HF_TOKEN
hf download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*"     # ⏱ 3–10 phút, khoảng 16 GB
```
(`original/` là bản trọng số định dạng Meta, không dùng.)

## 4. Kiểm trước (vài giây)

```bash
python scripts/buoc1_kiem_truoc.py 2>&1 | tee -a /workspace/log_b1.txt; echo "MÃ THOÁT: ${PIPESTATUS[0]}"
```
Phải thấy: `0/520 câu lệch`, khối `['<|eot_id|>', '<|start_header_id|>', 'assistant', '<|end_header_id|>', 'ĊĊ']`
(ký hiệu `ĊĊ` là "\n\n" trong tokenizer Llama-3), `id của 'I' = 40`, cấu hình hiệu lực greedy đúng ở cả hai,
`ĐẠT: True`, `MÃ THOÁT: 0`. (Các dòng `MÃ THOÁT` dùng `${PIPESTATUS[0]}` để in mã của python, không phải của tee.) Mã khác 0 thì dừng, chép `results/` về, không chạy tiếp.

Nếu khuôn chat lệch (`0/520` không đạt) vì Hugging Face đã đổi chat template ở bản mới của repo: **không sửa
code**. Chạy lại cả ba script với `--revision <commit của một snapshot cũ>` (xem lịch sử commit của repo trên
Hugging Face, chọn bản trước lần đổi template), ghi lý do vào nhật ký.

## 5. Phần 1 — tái lập Arditi et al.

```bash
python scripts/buoc1_tai_lap_arditi.py 2>&1 | tee -a /workspace/log_b1.txt; echo "MÃ THOÁT: ${PIPESTATUS[0]}"     # ⏱ ước 20–40 phút
```
Mã thoát: 0 = đạt cả bốn tiêu chí; 1 = có tiêu chí không đạt (vẫn đủ sản phẩm, xem mục 8); 2 = khuôn chat lệch
(dừng). Lỗi Python (traceback) thì chép log về và dừng.

## 6. Phần 2 — kiểm đường can thiệp

```bash
python scripts/buoc1_kiem_hook.py 2>&1 | tee -a /workspace/log_b1.txt; echo "MÃ THOÁT: ${PIPESTATUS[0]}"     # ⏱ ước 30–60 phút (bf16 rồi float32)
```
Mã thoát: 0 = đạt; 1 = không đạt; 2 = khuôn chat lệch; 3 = mô hình không nằm trọn trên một GPU; 4 = **cần xem**
(KHÔNG phải đạt, xem mục 8). Nếu lỗi hết bộ nhớ khi nạp bản float32 (bản này được ép lên một GPU nên thiếu bộ nhớ
là báo lỗi ngay): thuê máy 80 GB và chạy lại riêng mục 6.

## 7. Kéo kết quả về, rồi huỷ máy

Trên máy thuê:
```bash
cp /workspace/log_b1.txt results/buoc1/
pip freeze > results/buoc1/pip_freeze.txt
cd /workspace/thi_nghiem_2 && find results -type f -print0 | sort -z | xargs -0 sha256sum > /workspace/sha256_b1.txt
```
Trên máy cá nhân:
```bash
mkdir -p /mnt/d/thesis/luu_tru/buoc1
rsync -rlt may-thue:/workspace/thi_nghiem_2/results/ /mnt/d/thesis/thi_nghiem_2/results/; echo "rsync: $?"
rsync -rlt may-thue:/workspace/sha256_b1.txt /mnt/d/thesis/luu_tru/buoc1/
(cd /mnt/d/thesis/thi_nghiem_2 && sha256sum -c --quiet /mnt/d/thesis/luu_tru/buoc1/sha256_b1.txt) && echo "SHA ĐỦ"
cp -r /mnt/d/thesis/thi_nghiem_2/results/buoc1 /mnt/d/thesis/luu_tru/buoc1/
```
Phải in `rsync: 0` và `SHA ĐỦ`. Rồi mới huỷ máy. (`results/` nằm ngoài Git; bản lưu ở `luu_tru/buoc1`.)

## 8. Đọc kết quả

**Phần 1** — `results/buoc1/arditi_*/`:

| File | Nội dung |
|---|---|
| `kiem_khuon.json` | khuôn của ta so với cách tác giả mã hoá (phải 0 câu lệch) |
| `loc.json` | số câu trước/sau khi lọc theo điểm từ chối, cho 4 tập |
| `bang_chon_huong.json` | 160 ứng viên (vị trí −5…−1 × lớp 0…31): điểm sau chiếu bỏ, điểm sau cộng, KL |
| `huong_chon.json` | ứng viên được chọn |
| `sinh/*.jsonl` | câu trả lời của 4 lượt sinh (JailbreakBench gốc/chiếu bỏ, Alpaca gốc/cộng) |
| `ket_qua.json` | tỉ lệ từ chối 4 lượt, **bốn tiêu chí**, số câu dừng ở `<|end_of_text|>`, và (chỉ chẩn đoán) tỉ lệ từ chối tính trên phần trước `<|eot_id|>` đầu tiên |
| `chan_doan.json` | so với sản phẩm tác giả (cosine với `mean_diffs.pt`, lệch bảng ứng viên, trùng ứng viên tác giả không) — chỉ chẩn đoán |

Bốn tiêu chí (Nhap): (1) chỉ số lớp của ứng viên được chọn (theo code tác giả: đầu vào khối ℓ, tức luồng dư
sau ℓ khối) nằm trong 11–21, như Cổng 2a; (2) tỉ lệ từ chối
JailbreakBench sau chiếu bỏ < 10 %; (3) tỉ lệ từ chối Alpaca sau khi cộng > 80 %; (4) hai số đó lệch số của bài
(0,01 và 1,00) không quá 10 điểm %. Để so: ứng viên tác giả là vị trí −5, lớp 12; số tác giả công bố là
JailbreakBench gốc 0,95, sau chiếu bỏ 0,00, Alpaca gốc 0,01, sau khi cộng 1,00 (sản phẩm tác giả tạo bằng khuôn
chat lỗi, nên không bắt buộc trùng).

**Phần 2** — `results/buoc1/kiem_hook_*/`:

| File | Nội dung |
|---|---|
| `ket_luan.json` | **kết luận**: (ii) ở bf16 và float32; (i), (iii), kiểm âm ở float32; `trang_thai` = `dat` / `can_xem` / `khong_dat` |
| `ket_qua_bf16.json`, `ket_qua_fp32.json` | chi tiết từng phép: nhiễu, ngưỡng, lệch theo câu và theo vị trí, kiểm âm 4 lượt, tỉ số lệch/nhiễu, thang logit, `chan_doan_vi_tri` |
| `huong_kiem.pt` | r̂, ‖r‖, s ở lớp 8, 16, 24 (chỉ để kiểm hook) |

Luật ba mức (viết trước, chốt 27/10; chi tiết trong docstring của `src/tn2/kiem_hook.py`):

- `dat`: (ii) khớp tuyệt đối ở cả hai kiểu số; (i), (iii) ở float32 ≤ 2 × nhiễu; bốn kiểm âm > 5 × ngưỡng.
- `can_xem` (mã 4, **không phải đạt**): như trên trừ việc (i) hoặc (iii) vượt 2 × nhiễu nhưng ≤ 10 × nhiễu, và cả
  bốn kiểm âm > 100 × nhiễu. Đọc `ket_qua_fp32.json` → `tong_ket.chan_doan_vi_tri`:
  - `nghi_loi_co_che: true` (có vị trí lệch > 10 × đường bao nhiễu α = 0 tại cùng vị trí): dấu hiệu nghi lỗi cơ
    chế. **Dừng**, tìm lỗi trong hook, sửa, rồi chạy lại riêng mục 6.
  - `nghi_loi_co_che: false`: chẩn đoán chưa thấy dấu hiệu lỗi, **chưa xác nhận đạt về cơ chế**, kết quả vẫn là
    "cần xem". Mang kết quả về; Ân, sau khi hai reviewer xem, quyết: đi tiếp sang bước 2 và ghi riêng quyết định
    cùng câu hạn chế, hoặc tìm lỗi như trên.
  - Vị trí có `thieu_du_lieu_nhieu: true` (không câu đo nhiễu nào sinh tới đó) không có tỉ số; xem kèm
    `so_vi_tri_thieu_du_lieu_nhieu` và `so_cau_nhieu_theo_vi_tri` (vị trí thưa mẫu).
  Không trình bày lượt `can_xem` như đã đạt ngưỡng 2×.
- `khong_dat`: các trường hợp còn lại. Mang kết quả về, chưa kết luận, chẩn đoán trước khi làm gì tiếp.

Riêng (i), (iii) ở bf16 trượt không làm phần 2 trượt (số bf16 chỉ để tham khảo).
