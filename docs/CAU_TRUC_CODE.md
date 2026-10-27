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

## 2. Đã có (bước 1 và code bước 3)

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

Thêm cho bước 3 (code viết 27/10/2026, chưa chạy trên mô hình thật):

```
src/tn2/
  du_lieu.py      nạp 4 tập dựng hướng (128 câu mỗi tập, ghép cặp en–vi theo id); kiểm trùng (khớp, Jaccard ≥ 0,6, nguon_idx)
  kich_hoat.py    + kich_hoat_tung_cau: kích hoạt từng câu ở các vị trí, mọi lớp (cho chia đôi, cosine hai vị trí, bootstrap C3)
  mo_hinh.py      + đặc tả khối 9 token, vị trí lấy −5, dấu hiệu thinking, lớp kiểm hook, eos dự kiến của Qwen3.5-9B, Gemma 4 12B;
                  phan_bo_thiet_bi viết lại (tham số và buffer); backend_lop_tuyen_tinh, ban_tham_chieu_pytorch
                  (lượt float32 của Qwen3.5 chạy DeltaNet / conv bằng bản PyTorch tham chiếu, tắt TF32 thật)
  nguon_goc.py    + NaN/inf → null; ghi phiên bản flash-linear-attention, causal-conv1d, flash_attn
  buoc3/nap_kiem.py    kiểm khối sau câu hỏi, eos, cấu trúc lớp, bộ đếm vị trí (embedding một lần mỗi forward), rò thinking, chẩn đoán position_ids / đệm
  buoc3/dung_huong.py  dựng hướng điểm 1–8: r, r̂, ‖r‖, s_ℓ (512 câu gộp), ‖r‖/s, chia đôi, cosine hai vị trí, cosine en–vi
  buoc3/toc_do.py      đo thời gian, câu/giờ, token/giây, bộ nhớ đỉnh theo cỡ lô; ước giờ GPU theo mô hình
scripts/
  buoc3_kiem_truoc.py   tokenizer và config, không nạp trọng số (mã 2: khối / template / eos lệch → DỪNG)
  buoc3_nap_kiem.py     nạp một GPU, cấu trúc, bộ đếm vị trí, sinh thử 32 câu, rò thinking
  buoc3_dung_huong.py   chẩn đoán đệm, rồi dựng hướng (luôn truyền position_ids); ghi huong.pt và kích hoạt từng câu
  buoc3_kiem_hook.py    bản thu nhỏ phần 2 (bf16 rồi float32), lớp theo đặc tả; lượt không hook lặp lại (chẩn đoán)
  buoc3_do_toc_do.py    tốc độ và bộ nhớ: lô 16, 32, 64, 128; nền và can thiệp
docs/RUNBOOK_BUOC_3.md      trình tự chạy trên máy thuê
tests/test_buoc3.py         30 test (tổng 80 test)
data/dung_huong/            D_harm_build, D_harmless_build, en và vi (xem data/NGUON.md)
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
| Trình tự bước 3: nạp và kiểm (template, tắt thinking, token kết thúc) | `buoc3/nap_kiem.py`, `scripts/buoc3_kiem_truoc.py`, `buoc3_nap_kiem.py` |
| Dựng hướng điểm 1–8 (s_ℓ trên 512 câu gộp, chia đôi, cosine hai vị trí) | `buoc3/dung_huong.py`, `scripts/buoc3_dung_huong.py` |
| Bản thu nhỏ phần 2 trên hai mô hình chính | `kiem_hook.py` (dùng lại), `scripts/buoc3_kiem_hook.py` |
| Đo tốc độ sinh để tính lại chi phí | `buoc3/toc_do.py`, `scripts/buoc3_do_toc_do.py` |

## 3. Dự kiến cho các bước sau (đề xuất, chưa chốt)

| Bước (Nhap, Trình tự chạy) | Gói / script dự kiến | Dùng lại |
|---|---|---|
| 2. Dự đoán bản 1 | không có code; file văn bản trong `doc_thi_nghiem_2` | — |
| 3. Nạp, kiểm, dựng hướng và s_ℓ, phần 2 thu nhỏ, đo tốc độ | **đã viết** (mục 2), chưa chạy | `khuon`, `kich_hoat`, `huong`, `kiem_hook` |
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

**Luật trên đã thay (Ân duyệt Q1 ngày 27/10/2026, sau rà soát code bước 3 của hai reviewer).** Pipeline chính
LUÔN truyền `position_ids = cumsum(mask) − 1` khi lấy kích hoạt, đúng cách `generate` đánh vị trí; không còn "đo rồi
tự chọn". Lý do: ở bf16, ngưỡng "lệch cỡ nhiễu" quá rộng để chứng minh bỏ `position_ids` là an toàn (lỗi vài phần
trăm lọt qua), còn truyền `position_ids` thì không thể sai. Phần 1 (tái lập Arditi) giữ đúng cách của tác giả.
`buoc3.nap_kiem.do_position_ids` chỉ còn là phép chẩn đoán: so A (không truyền), B (truyền, cách dùng) với C (từng câu
không đệm), theo lớp và theo từng câu. Cảnh báo đệm (cosine B–C nhỏ nhất của trung bình < 0,999, Ân duyệt Q2) đưa
hướng về trạng thái "cần xem" (mã 4), không dùng cho bước 4 trước khi Ân xem; cờ tắt không chứng nhận không có lỗi
đệm (cosine không đổi khi nhân hệ số).

Việc thêm cho bước 3, chốt 27/10 sau rà soát kết quả bước 1 (chi tiết trong báo cáo kết quả bước 1, mục 7–8). **Tất cả đã làm trong code bước 3** (báo cáo code bước 3, 27/10/2026):

- `mo_hinh.phan_bo_thiet_bi`: kiểm thiết bị của mọi tensor tham gia forward, gồm cả `parameters()` và
  `buffers()` (`layer_scalar` của Gemma 4 là tham số hay buffer: kiểm cả hai), báo tên tensor lệch; bản đồ thiết bị rỗng không còn là đạt.
- Tóm tắt cosine trong `chan_doan`: bỏ ô có độ lớn 0, ghi số ô bị bỏ, xuất `null` thay cho `NaN`.
- Bản thu nhỏ phần 2 (`buoc3_kiem_hook.py`): bf16 và float32 như bước 1; quy mô như bước 1 (8 câu mỗi kiểu,
  64 token, α = 0,4, nhiễu trên 128 câu); câu từ D_harmless_build / D_harm_build tiếng Anh; hướng tiếng Anh của
  bước 3; lớp Qwen3.5-9B 8, 15, 24 và Gemma 4 12B 12, 23, 36 (lớp giữa là full attention, chạy kiểm âm). Không
  dùng `lop_mac_dinh()` của bước 1 cho hai mô hình này.
- Tách và khoá hai vị trí khi nạp: token lấy kích hoạt, và toàn bộ khối can thiệp sau câu hỏi (kể cả khối
  thinking rỗng); `so_token_khoi_du_kien` nay là 9 cho cả hai mô hình, vị trí lấy −5, kèm danh sách token của khối.
- Dựng hướng đủ điểm 1–8 của Nhap, gồm chia đôi và cosine giữa hai vị trí lấy kích hoạt.
- Dữ liệu D_harm_build, D_harmless_build vào `data/` kèm SHA và nguồn; kiểm trùng ba cặp tập.

Việc đã hẹn cho bước 4 (từ rà soát và đối chiếu code bước 3, 27/10/2026):

- Nhập tập test vào `data/`, chạy kiểm trùng (`tn2.du_lieu.kiem_trung`) cho dựng–chính, dựng–hiệu chỉnh,
  hiệu chỉnh–chính **trước** khi rút mẫu; loại cả cặp en–vi D_plain_test/0221 khỏi nguồn rút (314 cặp; 40 cặp
  hiệu chỉnh rút từ 70 cặp còn lại), ghi danh sách id và seed (Nhap mục D_plain_test, Q3).
- Code của bước 4 chỉ nhận `huong.pt` có `trang_thai == "dat"` và `chinh_thuc == True`; hướng `can_xem` (cảnh
  báo đệm, Q2) phải được Ân xem và quyết trước.
- Nạp đúng `commit_hf` đã ghi ở bước 3 (`--revision`), cùng backend kernel (`backend_lop_tuyen_tinh` trong meta).
