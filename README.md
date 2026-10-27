# thi_nghiem_2 — code cho thí nghiệm refusal steering hai ngôn ngữ (en, vi)

Thiết kế đã chốt: `doc_thi_nghiem_2/Nhap.docx` (bản dễ đọc: `Thiet_ke_thi_nghiem_27_10_2026.md`).
Kho này viết mới hoàn toàn; chỉ dữ liệu của Arditi et al. được mang sang (xem `data/NGUON.md`).

Hiện có: **bước 1 của trình tự chạy** — kiểm pipeline trên Llama-3-8B-Instruct.

| Lệnh | Việc | Ghi ra |
|---|---|---|
| `python scripts/buoc1_kiem_truoc.py` | kiểm tokenizer, khuôn chat, cấu hình sinh thật (vài giây, không nạp trọng số) | `results/buoc1/kiem_truoc_*/` |
| `python scripts/buoc1_tai_lap_arditi.py` | phần 1: tái lập Arditi et al. (2024), bốn tiêu chí | `results/buoc1/arditi_*/` |
| `python scripts/buoc1_kiem_hook.py` | phần 2: kiểm đường can thiệp của ta, (i), (ii), (iii), kiểm âm | `results/buoc1/kiem_hook_*/` |

Cách chạy trên GPU thuê: `docs/RUNBOOK_BUOC_1.md`. Cấu trúc code và kế hoạch cho các bước sau: `docs/CAU_TRUC_CODE.md`.

Đã viết, chưa chạy: **bước 3** — với Qwen3.5-9B và Gemma 4 12B: nạp và kiểm, dựng hướng và s_ℓ, bản thu nhỏ phần 2,
đo tốc độ (`--mo_hinh_dac_ta qwen35_9b | gemma4_12b`).

| Lệnh | Việc | Ghi ra |
|---|---|---|
| `python scripts/buoc3_kiem_truoc.py` | khối 9 token sau câu hỏi, vị trí lấy −5, token kết thúc, layer_types, greedy (không nạp trọng số) | `results/buoc3/<mô hình>/kiem_truoc_*/` |
| `python scripts/buoc3_nap_kiem.py` | một GPU, cấu trúc lớp, bộ đếm vị trí, sinh thử, rò thinking | `…/nap_kiem_*/` |
| `python scripts/buoc3_dung_huong.py` | chẩn đoán position_ids / đệm; dựng hướng điểm 1–8 trên 4 × 128 câu dựng (luôn truyền position_ids) | `…/dung_huong_*/` |
| `python scripts/buoc3_kiem_hook.py --huong …` | bản thu nhỏ phần 2, bf16 rồi float32 | `…/kiem_hook_*/` |
| `python scripts/buoc3_do_toc_do.py --huong …` | câu/giờ, bộ nhớ đỉnh theo cỡ lô | `…/toc_do_*/` |

Cách chạy: `docs/RUNBOOK_BUOC_3.md`.

## Test (CPU, không tải gì từ mạng)

```bash
pip install -r requirements.txt     # cộng torch, xem docs/RUNBOOK_BUOC_1.md
python -m pytest -q -p no:cacheprovider tests/
```

Test dùng mô hình nhỏ khởi tạo ngẫu nhiên cùng kiến trúc (Llama, Qwen3.5, Gemma 4) và tokenizer đồ chơi có id
đặc biệt trùng id thật của Llama-3, Qwen3.5 và Gemma 4. Chúng kiểm logic của code, không thay được lượt chạy trên mô hình thật.

## Quy ước

- Tên hàm, biến bằng tiếng Việt không dấu; docstring ghi mục tương ứng của Nhap.
- Lớp đánh số từ 0 trong code (chỉ số khối). Tiêu chí (1) của phần 1 so thẳng chỉ số lớp của tác giả
  (đầu vào khối ℓ = luồng dư sau ℓ khối) với dải 11–21, như Cổng 2a.
- Mỗi lượt chạy ghi vào thư mục mới trong `results/` (không ghi đè), kèm `meta.json` (phiên bản thư viện, GPU,
  commit của kho đọc từ `.git`, SHA-256 của dữ liệu) và `nhat_ky.txt`.
- Lượt có tham số khác mặc định (`--gioi_han`, `--max_new_tokens`, …) được ghi `chinh_thuc: false`.
