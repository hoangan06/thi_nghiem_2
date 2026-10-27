# Runbook bước 3 — nạp, kiểm, dựng hướng, bản thu nhỏ phần 2, đo tốc độ (Qwen3.5-9B, Gemma 4 12B; GPU thuê)

Mục tiêu (Nhap, Trình tự chạy bước 3): với từng mô hình chính, nạp và kiểm (template, tắt thinking, token kết
thúc); dựng hướng và s_ℓ (mục Dựng hướng, điểm 1–8); chạy bản thu nhỏ phần 2 của kiểm pipeline (bf16 và float32);
đo tốc độ sinh để tính lại chi phí. Chạy lần lượt hai mô hình trên cùng một máy. Thời gian dưới đây là ước, chưa đo.

## 0. Trước khi thuê (máy cá nhân, WSL)

1. Commit code của kho `thi_nghiem_2` (Ân tự chạy git); `git status --porcelain` phải rỗng; ghi lại `git rev-parse HEAD`.
2. Trên Hugging Face, bằng tài khoản có token: mở `google/gemma-4-12B-it` và đồng ý giấy phép (Gemma cần đồng ý
   trước khi tải). `Qwen/Qwen3.5-9B` là Apache 2.0, không cần đồng ý.
3. (Nên làm, không bắt buộc; cần môi trường Python có `requirements.txt` và token HF.) Chạy bước 4a cho cả hai mô
   hình ngay trên máy cá nhân: chỉ tải tokenizer và config (vài MB), không cần GPU. Mã 2 thì biết trước khi trả tiền
   máy; mã 0 thì `kiem_truoc.json` có luôn `commit_hf`.
   ```bash
   cd /mnt/d/thesis/thi_nghiem_2
   read -rsp "HF token: " HF_TOKEN; echo; export HF_TOKEN
   python scripts/buoc3_kiem_truoc.py --mo_hinh_dac_ta qwen35_9b; echo "MÃ THOÁT: $?"
   python scripts/buoc3_kiem_truoc.py --mo_hinh_dac_ta gemma4_12b; echo "MÃ THOÁT: $?"
   unset HF_TOKEN
   ```

## 1. Chọn máy (Vast.ai)

| Tiêu chí | Giá trị |
|---|---|
| GPU | 1 GPU **80 GB** (A100 80GB hoặc H100 80GB). Báo cáo kết quả bước 1, mục 7: bản float32 của Gemma 4 12B khoảng 44,5 GiB chỉ riêng trọng số, chọn 80 GB để tránh hết bộ nhớ |
| Đĩa | ≥ 120 GB (hai mô hình khoảng 45 GB, môi trường, kết quả khoảng 1 GB) |
| Ảnh | Ubuntu có CUDA ≥ 12.8, Python 3.12, có `nvcc` (để thử cài kernel ở mục 2) |
| Mạng | Direct SSH |

## 2. Đưa code lên, dựng môi trường

Trên máy cá nhân:
```bash
rsync -rlt --exclude results --exclude __pycache__ /mnt/d/thesis/thi_nghiem_2/ may-thue:/workspace/thi_nghiem_2/
```
Trên máy thuê, trong tmux (`tmux new -s b3`):
```bash
cd /workspace/thi_nghiem_2
python3 -m venv /workspace/venv && . /workspace/venv/bin/activate
pip install -U pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128     # ⏱ 3–8 phút
pip install -r requirements.txt
python -c "import torch,transformers;print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
CUDA_VISIBLE_DEVICES="" python -m pytest -q -p no:cacheprovider tests/ 2>&1 | tee -a /workspace/log_b3.txt; echo "MÃ THOÁT: ${PIPESTATUS[0]}"     # ⏱ 5–8 phút; phải: 80 passed
```
Có test hỏng thì dừng, chép log về, không chạy tiếp. `CUDA_VISIBLE_DEVICES=""` chỉ áp cho tiến trình test: test
chạy đúng như đã kiểm trên CPU, còn các lệnh ở mục 4 vẫn thấy GPU.

**Chạy test TRƯỚC khi cài kernel.** Test chạy mô hình nhỏ trên CPU; khi đã cài `flash-linear-attention`, transformers
dùng kernel Triton cho lớp DeltaNet bất kể thiết bị, nên test trên CPU sẽ lỗi. Sau đó mới cài kernel (Ghi chú kỹ
thuật hai mô hình chính, mục 1). Cài lỗi thì bỏ qua: transformers tự dùng bản PyTorch thuần (cùng phép toán, chậm
và tốn bộ nhớ hơn). Backend thực dùng được ghi vào `meta.json` (`goi_flash-linear-attention`, `goi_causal-conv1d`).
Các lượt về sau (hiệu chỉnh, pilot, chạy chính) phải dùng cùng backend với lượt này. Bản `flash-linear-attention`
ghim ở 0.6.0 (bản đã đọc mã khi rà soát, 27/10/2026); `causal-conv1d` ghi phiên bản vào log.
```bash
pip install flash-linear-attention==0.6.0 2>&1 | tail -2
pip install causal-conv1d --no-build-isolation 2>&1 | tail -2                      # ⏱ có thể 10–20 phút biên dịch
pip list 2>/dev/null | grep -i -E "flash-linear|causal-conv" | tee -a /workspace/log_b3.txt
```
Kernel Triton của `flash-linear-attention` dùng TF32 bất kể cờ của PyTorch. Vì vậy lượt float32 ở mục 4d của
Qwen3.5-9B tự chuyển lớp DeltaNet / conv về bản PyTorch tham chiếu của transformers (không gỡ kernel; bf16 và các
lượt sau vẫn dùng kernel). Backend thực dùng ghi trong `meta.json` (`backend_lop_tuyen_tinh*`) và
`ket_qua_fp32.json` (`backend_lop_tuyen_tinh`). Lượt float32 của Qwen vì thế chậm hơn; chưa đo được bao nhiêu.

## 3. Tải hai mô hình

```bash
export HF_HOME=/workspace/hf
read -rsp "HF token: " HF_TOKEN; echo; export HF_TOKEN
hf download Qwen/Qwen3.5-9B              # ⏱ khoảng 19 GB
hf download google/gemma-4-12B-it        # ⏱ khoảng 24 GB
unset HF_TOKEN
# Từ đây chỉ dùng bản đã tải (không tải lại, không đổi commit). Mỗi cửa sổ tmux mới: source /workspace/env_b3.sh
printf 'export HF_HOME=/workspace/hf\nexport HF_HUB_OFFLINE=1\nexport USE_HUB_KERNELS=NO\n. /workspace/venv/bin/activate\ncd /workspace/thi_nghiem_2\n' > /workspace/env_b3.sh
. /workspace/env_b3.sh
```
Commit của bản đã tải được ghi trong `kiem_truoc.json` (`cau_truc_config.commit_hf`) và trong `meta.json` của mọi
lượt. Các bước sau (hiệu chỉnh, pilot, chạy chính) phải nạp đúng commit này (`--revision`).

## 4. Chạy cho từng mô hình

Dán định nghĩa hàm `b3` dưới đây một lần (mỗi cửa sổ tmux), rồi chạy `b3 qwen35_9b`, xong thì `b3 gemma4_12b`.
Hàm dừng ngay ở bước đầu tiên có mã thoát khác 0. Riêng mã 4 ("cần xem") ở 4c và 4d thì chạy tiếp: phần 2 kiểm cơ
chế hook (đúng với hướng nào cũng được), đo tốc độ không phụ thuộc hướng. Mã 4 vẫn **không phải đạt** (mục 6): hướng
có mã 4 ở 4c không được dùng cho bước 4 trước khi Ân xem. Khi hàm dừng: chép `results/` về, không chạy tiếp mô
hình đó, không sửa code trên máy thuê.

```bash
b3() {
  local M=$1 L=/workspace/log_b3.txt rc H r4c r4d
  _b() { local ten=$1; shift; echo "=== $M $ten: $*" | tee -a $L; python "$@" 2>&1 | tee -a $L; rc=${PIPESTATUS[0]}; echo "=== $M $ten MÃ THOÁT: $rc" | tee -a $L; return $rc; }
  # 4a. Kiểm trước (vài giây, không nạp trọng số): khối 9 token, vị trí lấy −5, token kết thúc, layer_types, greedy.
  _b 4a scripts/buoc3_kiem_truoc.py --mo_hinh_dac_ta $M || return
  # 4b. Nạp và kiểm (⏱ 3–6 phút): một GPU, cấu trúc, bộ đếm vị trí, sinh thử 32 câu, rò thinking, token kết thúc.
  _b 4b scripts/buoc3_nap_kiem.py --mo_hinh_dac_ta $M || return
  # 4c. Dựng hướng và s_ℓ (⏱ 3–8 phút): chẩn đoán position_ids / đệm, 512 câu dựng, mọi lớp.
  _b 4c scripts/buoc3_dung_huong.py --mo_hinh_dac_ta $M; r4c=$rc; [ $rc -eq 0 ] || [ $rc -eq 4 ] || return $rc
  # Lượt dựng hướng chính thức mới nhất (tên dung_huong_<ngày>_<giờ>; lượt thử tên dung_huong_thu_… bị loại).
  H=$(ls -td results/buoc3/$M/dung_huong_2*/ | head -1)huong.pt; echo "Hướng: $H" | tee -a $L
  [ -f "$H" ] || { echo "Không thấy $H" | tee -a $L; return 1; }
  # 4d. Bản thu nhỏ phần 2, bf16 rồi float32 (⏱ ước 40–120 phút, Qwen có thể lâu hơn vì float32 dùng bản tham chiếu).
  _b 4d scripts/buoc3_kiem_hook.py --mo_hinh_dac_ta $M --huong $H; r4d=$rc; [ $rc -eq 0 ] || [ $rc -eq 4 ] || return $rc
  # 4e. Đo tốc độ và bộ nhớ (⏱ 10–25 phút): lô 16, 32, 64, 128; nền và can thiệp α = 0,8.
  _b 4e scripts/buoc3_do_toc_do.py --mo_hinh_dac_ta $M --huong $H || return
  echo "=== $M: chạy xong bước 3; mã 4c = $r4c, mã 4d = $r4d (0 đạt, 4 cần xem: KHÔNG phải đạt)" | tee -a $L
}
```

Mã thoát:

| Script | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| buoc3_kiem_truoc | đạt | cấu trúc config hoặc cấu hình sinh lệch | **khối sau câu hỏi / template / token kết thúc lệch dự kiến: DỪNG** | — | — |
| buoc3_nap_kiem | đạt | có phép kiểm không đạt (xem `nap_kiem.json`) | — | không nằm trọn một GPU | — |
| buoc3_dung_huong | xong | lỗi | — | không nằm trọn một GPU | **cần xem**: cảnh báo đệm bật, hướng chưa dùng cho bước 4 |
| buoc3_kiem_hook | đạt | không đạt | khối lệch | không nằm trọn một GPU | **cần xem** (không phải đạt) |
| buoc3_do_toc_do | xong | lỗi | — | không nằm trọn một GPU | — |

Nếu `buoc3_kiem_truoc` trả mã 2: **không sửa code trên máy thuê**. Chép `results/` về. Trước khi kết luận "khối
lệch", xem `khoi_sau_cau_hoi` trong `kiem_truoc.json`: nếu `K_theo_so_cau` vẫn là 9 và vị trí lấy vẫn −5, chỉ chuỗi
giải mã của token khác chuỗi dự kiến (ví dụ có dấu cách đầu) thì đó là lệch ở đặc tả, không phải ở khối. Dù trường hợp
nào cũng quyết lại cùng Ân và hai reviewer trước khi chạy tiếp (Nhap: "cách tách token cần xác nhận khi nạp
tokenizer").

## 5. Kéo kết quả về, rồi huỷ máy

Trên máy thuê:
```bash
cp /workspace/log_b3.txt results/buoc3/
pip freeze > results/buoc3/pip_freeze.txt
cd /workspace/thi_nghiem_2 && find results/buoc3 -type f -print0 | sort -z | xargs -0 sha256sum > /workspace/sha256_b3.txt
```
Trên máy cá nhân:
```bash
mkdir -p /mnt/d/thesis/luu_tru/buoc3
rsync -rlt may-thue:/workspace/thi_nghiem_2/results/buoc3/ /mnt/d/thesis/thi_nghiem_2/results/buoc3/; echo "rsync: $?"
rsync -rlt may-thue:/workspace/sha256_b3.txt /mnt/d/thesis/luu_tru/buoc3/
(cd /mnt/d/thesis/thi_nghiem_2 && sha256sum -c --quiet /mnt/d/thesis/luu_tru/buoc3/sha256_b3.txt) && echo "SHA ĐỦ"
cp -r /mnt/d/thesis/thi_nghiem_2/results/buoc3 /mnt/d/thesis/luu_tru/buoc3/
```
Phải in `rsync: 0` và `SHA ĐỦ`. Rồi mới huỷ máy. File `kich_hoat_vi_tri_lay.pt` khoảng 134 MB (Qwen) và 189 MB
(Gemma); cần cho bootstrap của C3 ở bước phân tích.

## 6. Đọc kết quả (`results/buoc3/<mô hình>/`)

| Lượt | File chính | Đọc gì |
|---|---|---|
| `kiem_truoc_*` | `kiem_truoc.json` | `khoi_sau_cau_hoi` (9 token trên 512 câu, vị trí −5), `token_ket_thuc`, `cau_truc_config`, `cau_hinh_sinh.la_greedy_dung`, `chat_template` đã dùng |
| `nap_kiem_*` | `nap_kiem.json`, `sinh_thu.jsonl` | `dat_thanh_phan`; `phan_bo_thiet_bi`, `cau_truc` (đường dẫn lớp, full attention, `layer_scalar` của Gemma), `goi_embedding`, `sinh_thu.ro_thinking`, `sinh_thu.so_cau_theo_eos` (đạt khi có câu dừng ở id chính; 0 câu → xem lại eos); đọc vài câu trong `sinh_thu.jsonl` bằng mắt |
| `dung_huong_*` | `tom_tat.json`, `do_position_ids.json`, `huong.pt` | **`trang_thai`** / **`canh_bao_dem`** (True → mã 4: không dùng hướng cho bước 4 trước khi Ân xem); `do_position_ids.json` chỉ chẩn đoán: lệch tương đối B–C theo lớp và theo câu (cờ tắt không chứng nhận không có lỗi đệm); ‖r‖, ‖r‖/s, s theo lớp; cosine chia đôi; cosine hai vị trí (chỉ chẩn đoán, Nhap điểm 7, null ở lớp có hướng bằng 0); cosine en–vi (chỉ mô tả) |
| `kiem_hook_*` | `ket_luan.json`, `ket_qua_fp32.json` | như bước 1: (ii) ở hai kiểu số; (i), (iii), kiểm âm ở float32; `trang_thai` `dat` / `can_xem` / `khong_dat`, luật ba mức (docstring `tn2.kiem_hook`). Chẩn đoán: `tong_ket.khong_hook_lap_lai_khop` (False: đường nền không tất định, nên khi (ii) trượt chưa thể quy riêng độ lệch cho hook; cũng chưa loại trừ lỗi hook: dừng, xem cùng Ân); `backend_lop_tuyen_tinh` của lượt float32 phải toàn `la_ban_tham_chieu: true` với Qwen |
| `toc_do_*` | `toc_do.json` | câu/giờ, bộ nhớ đỉnh theo cỡ lô; ước giờ GPU theo số câu của Nhap cho riêng mô hình này (chạy chính theo số lớp) |

Lưu ý khi đọc tốc độ: số đo trên máy 80 GB không áp sang loại GPU khác. Trước khi thuê máy cho lượt hiệu chỉnh và
chạy chính, đo lại vài phút trên đúng loại máy đó, cùng môi trường và backend (báo cáo kết quả bước 1, mục 7).
