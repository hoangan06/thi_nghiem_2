# Cấu trúc code của thí nghiệm

Tài liệu này mô tả cách code bám vào Nhap. Phần "đã có" là code đã viết và đã test. Phần "dự kiến" là đề xuất
của người viết code (Claude) cho các bước sau: chưa chốt, viết ra để thấy bước 1 không chặn đường cho bước sau.

## 1. Nguyên tắc

1. **Một pipeline chính, dùng chung.** Template, lấy kích hoạt, tính hướng, hook, sinh, điểm chỉ viết một lần
   trong `src/tn2/`. Phần tái lập Arditi et al. và các bước sau gọi lại đúng các hàm đó (Nhap, Kiểm pipeline,
   nguyên tắc viết code cho phần 1).
2. **Mỗi bước của trình tự chạy có một gói con và các script mỏng.** Gói con `src/tn2/buocN/` chứa phần riêng
   của bước, còn script trong `scripts/` chỉ đọc tham số, gọi hàm và ghi kết quả.
3. **Không ghi đè, có nguồn gốc.** Mỗi lượt chạy ghi vào thư mục mới, kèm `meta.json` và nhật ký.
4. **Đơn giản trước.** Chưa làm cơ chế nối tiếp sau khi ngắt, không có manifest khoá hay sổ nguồn kiểu thí nghiệm
   cũ. Lượt nào hỏng thì chạy lại. Nếu bước sau cần thêm, sẽ thêm khi đến bước đó.

## 2. Đã có (bước 1)

```
src/tn2/
  mo_hinh.py      đặc tả 3 mô hình (Llama-3-8B-Instruct, Qwen3.5-9B, Gemma 4 12B), nạp mô hình, tìm lớp, eos
  khuon.py        chat template; ĐO số token khối sau câu hỏi (K) và vị trí lấy kích hoạt từ tokenizer; đệm trái
  kich_hoat.py    trung bình kích hoạt (float64) ở đầu ra / đầu vào khối, và độ lớn trung bình s_ℓ
  huong.py        hiệu trung bình, chuẩn hoá, cosine, chia đôi có seed
  can_thiep.py    hook: cộng vector hoặc chiếu bỏ; vùng = khối K token + token sinh (phương án 1 / 3) hoặc mọi vị trí
  sinh.py         sinh greedy theo lô, mọi tham số giải mã đặt rõ; hai cách giải mã ("đến eos", "như Arditi")
  diem.py         điểm từ chối theo logit, KL, so khớp chuỗi (đúng công thức của tác giả)
  kiem_hook.py    phần 2 của kiểm pipeline: (i), (ii), (iii), kiểm âm; dùng lại ở bước 3 cho hai mô hình chính
  nguon_goc.py    meta của lượt chạy, ghi JSON/JSONL, thư mục kết quả mới
  buoc1/tai_lap_arditi.py   phần riêng của bài Arditi: lọc, chọn hướng theo KL và điểm từ chối, 4 tiêu chí, chẩn đoán
scripts/
  buoc1_kiem_truoc.py       kiểm tokenizer, template, cấu hình sinh thật trước khi nạp trọng số
  buoc1_tai_lap_arditi.py   phần 1
  buoc1_kiem_hook.py        phần 2 (chạy bf16 rồi float32)
tests/                      50 test trên CPU (mô hình nhỏ, tokenizer đồ chơi; tests/tac_gia: hook của tác giả để đối chiếu)
data/arditi/                dữ liệu Arditi et al. và sản phẩm của tác giả cho Llama-3 (xem data/NGUON.md)
```

Đối chiếu với Nhap:

| Nhap | Code |
|---|---|
| Dựng hướng điểm 2: vị trí lấy = token cuối phần mở đầu lượt trả lời, không tính khối thinking rỗng | `khuon.phan_tich_mot_cau` (đo từ tokenizer; Llama −1) |
| Dựng hướng điểm 3: hiệu trung bình ở đầu ra khối ℓ | `kich_hoat.trung_binh_kich_hoat(diem="ra_khoi")`, `huong.hieu_trung_binh` |
| 3.2: vùng = khối K token sau câu hỏi + token sinh; phương án 1 (mọi token sinh) và 3 (token sinh đầu) | `can_thiep.vung_phuong_an`, `mat_na_vung` |
| 3.3: h ← h ± α·s_ℓ·r̂_ℓ, dấu + kiểu (a), − kiểu (b) | `can_thiep.vector_can_thiep`, `can_thiep_chinh` |
| 3.5: greedy, đệm trái, eos đặt rõ | `sinh.cau_hinh_greedy`, `sinh_tham_lam`; `mo_hinh.eos_tu_mo_hinh` |
| Kiểm pipeline phần 1 (bảng chi tiết theo Cổng 2a, 4 tiêu chí) | `buoc1/tai_lap_arditi.py`, `scripts/buoc1_tai_lap_arditi.py` |
| Kiểm pipeline phần 2 ((i), (ii), (iii), kiểm âm, α = 0,4, 8 câu × 3 lớp) | `kiem_hook.py`, `scripts/buoc1_kiem_hook.py` |

## 3. Dự kiến cho các bước sau (đề xuất, chưa chốt)

| Bước (Nhap, Trình tự chạy) | Gói / script dự kiến | Dùng lại |
|---|---|---|
| 2. Dự đoán bản 1 | không có code; file văn bản trong `doc_thi_nghiem_2` | — |
| 3. Nạp, kiểm, dựng hướng và s_ℓ, phần 2 thu nhỏ, đo tốc độ | `buoc3/` + `scripts/buoc3_nap_kiem.py`, `buoc3_dung_huong.py`, `buoc3_kiem_hook.py` | `khuon` (K, thinking), `kich_hoat`, `huong`, `kiem_hook` |
| 4. Hiệu chỉnh (sinh toàn bộ) | `sinh_luot.py` dùng chung cho 4, 7, 8, 10: một lượt = (mô hình, tập câu, ngôn ngữ, chiều, lớp, α, phương án, kiểu) → JSONL | `sinh`, `can_thiep` |
| 5. Tập nhãn tham chiếu, chọn trọng tài | `trong_tai/` (gọi OpenRouter, lời nhắc trọng tài, Se/Sp/kappa) | — |
| 6. Chấm, chọn α và phương án | `trong_tai/cham.py`, `phan_tich/chon_alpha.py` | — |
| 7–8. Pilot Qwen, kiểm α Gemma | `sinh_luot.py`, `phan_tich/` | như trên |
| 9. Dự đoán bản 2, giữ/bỏ kiểu (b) | `phan_tich/` | — |
| 10. Chạy chính, hướng ngẫu nhiên | `sinh_luot.py`, `huong` (hướng ngẫu nhiên cùng độ lớn) | như trên |
| Phân tích mục 7.1–7.7 | `phan_tich/` (bootstrap, R*, Rogan–Gladen, …), chạy trên CPU | — |

Hai điểm bước 1 đã chuẩn bị cho bước 3: `khuon.py` đã xử lý khối thinking rỗng của Qwen3.5 và Gemma 4 (đã test
trên template kiểu Qwen), và `can_thiep.py` / `kiem_hook.py` đã chạy đúng trên bản nhỏ của Qwen3.5 (attention
tuyến tính + đầy đủ) và Gemma 4 (cửa sổ trượt).

Điểm quyết ở bước 3 (chốt 27/10, hoãn từ bước 1): các lượt forward lấy kích hoạt hiện không truyền
`position_ids`, nên với lô đệm trái, vị trí RoPE đếm cả token đệm, còn `generate` đánh vị trí theo mặt nạ.
Về lý thuyết điều này chỉ đổi sai số làm tròn (RoPE chỉ dùng khoảng cách giữa các token thật; lớp DeltaNet của
Qwen3.5 không dùng vị trí; mặt nạ cửa sổ trượt của Gemma 4 theo vị trí trong cache), nhưng chưa đo trên mô hình
thật. Phần 1 giữ đúng cách của tác giả. Luật cho bước 3, khi nạp từng mô hình chính: trên cùng một lô của tập
dựng hướng, tính trung bình kích hoạt ở vị trí lấy, mọi lớp, hai cách (có và không có
`position_ids = cumsum(mask) − 1`). Lệch ở mức làm tròn (sai khác tương đối cỡ nhiễu kiểu số, cosine ≈ 1) → giữ
cách hiện tại; lệch lớn hơn → thêm `position_ids` cho pipeline chính và ghi lại.
