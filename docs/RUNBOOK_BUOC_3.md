# Runbook bước 3 — nạp, kiểm, dựng hướng, bản thu nhỏ phần 2, đo tốc độ (Qwen3.5-9B, Gemma 4 12B; GPU thuê)

Mục tiêu (Nhap, Trình tự chạy bước 3): với từng mô hình chính, nạp và kiểm (template, tắt thinking, token kết
thúc); dựng hướng và s_ℓ (mục Dựng hướng, điểm 1–8); chạy bản thu nhỏ phần 2 của kiểm pipeline (bf16 và float32);
đo tốc độ sinh để tính lại chi phí. Chạy lần lượt hai mô hình trên cùng một máy. Thời gian dưới đây là ước, chưa đo.

**Cập nhật 29/10/2026 (sau kết quả bước 3 ngày 28/10; Ân duyệt Q-E1 đến Q-E6).** Lượt thuê tiếp theo làm theo
`RUNBOOK_LUOT_THUE_TIEP.md`; runbook này giữ cho tra cứu và cho máy chết giữa chừng. Các thay đổi:
- Gemma 4 12B luôn nạp bằng attention `eager`, Qwen3.5-9B bằng `sdpa` (đặc tả trong `src/tn2/mo_hinh.py`); kiểu thực
  dùng ghi vào `meta.json`, `huong.pt`, và bước 4 so lại.
- 4b và 4d có thêm phép quét attention theo độ dài (`src/tn2/kiem_attention.py`).
- `b3 <mô hình> kiem_nap` chỉ chạy 4a và 4b.
- `b3` truyền `--revision <commit>` cho 4a, 4b, 4c, lấy từ `meta.json` của lượt dựng hướng đã xong mới nhất: trên máy
  thứ hai trở đi, bản tải theo commit không có nhãn `main` nên nạp offline không kèm revision sẽ lỗi.
- Cách xử lý khi phép quét bf16 ra `can_kiem` hay `khong_dat`: luật Ân duyệt ngày 29/10/2026 ở
  `RUNBOOK_LUOT_THUE_TIEP.md`, mục 4 (Qwen) và mục 5 (Gemma).
- GPU: bất kỳ A100 nào là cùng loại (Ân quyết 29/10/2026). Lượt có float32 ở 4d cần 80 GB. `ho_gpu` không coi lát
  MIG là A100 thường.
- `env_b3.sh` đặt `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16`; không nâng torch; máy thứ hai trở đi không cài
  `causal-conv1d` khi lượt bước 3 không có gói này.

## 0. Trước khi thuê (máy cá nhân, WSL)

1. Commit code của kho `thi_nghiem_2` (Ân tự chạy git); `git status --porcelain` phải rỗng; ghi lại `git rev-parse HEAD`.
   Bước 3 phải chạy bằng code có bản sửa `nap_mo_hinh` (commit của code bước 4): code cũ ghi commit Hugging Face của
   Qwen3.5 là rỗng, và lượt chính thức của bước 3 giờ dừng ngay (mã 1) khi không có commit.
2. Trên Hugging Face, bằng tài khoản có token: mở `google/gemma-4-12B-it` và đồng ý giấy phép (Gemma cần đồng ý
   trước khi tải). `Qwen/Qwen3.5-9B` là Apache 2.0, không cần đồng ý.
3. (Nên làm, không bắt buộc; cần môi trường Python có `requirements.txt` và token HF.) Chạy bước 4a cho cả hai mô
   hình ngay trên máy cá nhân: chỉ tải tokenizer và config (vài MB), không cần GPU. Mã 2 thì biết trước khi trả tiền
   máy; mã 0 thì `kiem_truoc.json` có luôn `commit_hf`.
   ```bash
   cd /mnt/d/thesis/thi_nghiem_2 && source .venv/bin/activate      # venv của kho trên máy cá nhân
   read -rsp "HF token: " HF_TOKEN; echo; export HF_TOKEN
   python scripts/buoc3_kiem_truoc.py --mo_hinh_dac_ta qwen35_9b; echo "MÃ THOÁT: $?"
   python scripts/buoc3_kiem_truoc.py --mo_hinh_dac_ta gemma4_12b; echo "MÃ THOÁT: $?"
   unset HF_TOKEN
   ```
4. WSL cần `rsync` và `ssh` (một lần):
   ```bash
   sudo apt-get install -y rsync openssh-client
   ```
   Các lệnh trên máy cá nhân
   cần Python (`buoc3_kiem_truoc`, `gop4`) chạy trong venv của kho: `source .venv/bin/activate` trước (hoặc đặt
   `PYTHON=<đường dẫn python của venv>` cho `gop4`).
5. Tên `may-thue` dùng trong mọi lệnh `rsync` / `ssh` của runbook là một mục trong `~/.ssh/config` của WSL. Sau khi
   thuê máy, điền địa chỉ và cổng SSH mà Vast.ai cho (mỗi lần đổi máy thì sửa lại mục này):
   ```
   Host may-thue
       HostName <địa chỉ IP>
       Port <cổng>
       User root
   ```

## 1. Chọn máy (Vast.ai)

| Tiêu chí | Giá trị |
|---|---|
| GPU | 1 GPU **A100 80 GB** (SXM4 hay PCIe đều được). Lượt float32 ở 4d của Gemma 4 12B đo đỉnh 48,6 GiB (28/10/2026), nên cần 80 GB. Lượt chỉ có bf16 (bước 4, `kiem_nap`) có thể dùng A100 40 GB, nhưng chỉ khi bộ nhớ đỉnh đo trong `toc_do.json` mới (lô 40, đúng kiểu attention) còn dư rõ |
| Đĩa | ≥ 120 GB (hai mô hình khoảng 45 GB, môi trường, kết quả khoảng 1–2 GB) trên **ổ cục bộ** của máy (`/workspace` không phải ổ mạng; kiểm ở mục 3) |
| Ảnh | Ubuntu có CUDA ≥ 12.8, Python 3.12, có `nvcc` (để thử cài kernel ở mục 2) |
| Mạng | Direct SSH |

Mọi lượt bước 3 và bước 4 dùng GPU **họ A100** (Ân quyết 29/10/2026: SXM4 hay PCIe, 40 hay 80 GB đều cùng chip GA100).
Cổng mã 8 của bước 4 so họ GPU, không so tên đầy đủ; tên đầy đủ vẫn được ghi lại.

## 2. Đưa code lên, dựng môi trường

Có hai nhánh. **Máy đầu tiên**: máy chạy bước 3 lần đầu. **Máy thứ hai trở đi**: đổi máy giữa bước 3 và bước 4, hoặc
giữa chừng bước 4 (`RUNBOOK_BUOC_4.md`, mục 1, cách 2); khi đó cài và tải **đúng** những gì máy đầu đã dùng, không cài
hay tải bản mới nhất.

Trên máy cá nhân. Lệnh `ssh` đầu tiên cài `rsync`, `tmux`, `python3-venv` lên máy thuê (ảnh có sẵn thì không đổi gì);
phải làm trước `rsync`, vì `rsync` cần có mặt ở cả hai đầu:
```bash
ssh may-thue 'apt-get update -qq && apt-get install -y -qq rsync tmux python3-venv >/dev/null && echo "cài xong"'
rsync -rlt --exclude results --exclude __pycache__ /mnt/d/thesis/thi_nghiem_2/ may-thue:/workspace/thi_nghiem_2/
```
(Máy thứ hai trở đi: chép code và `results/` theo đúng lệnh ở `RUNBOOK_BUOC_4.md`, mục 1, cách 2, bước 3, thay cho
lệnh `rsync` trên.)

Trên máy thuê, trong tmux (`tmux new -s b3`). Tạo `env_b3.sh` **trước tiên**, để mọi cửa sổ, kể cả sau khi mất kết
nối giữa chừng, dùng cùng `HF_HOME` và cùng môi trường:
```bash
printf 'export HF_HOME=/workspace/hf\nexport USE_HUB_KERNELS=NO\nexport OMP_NUM_THREADS=16\nexport MKL_NUM_THREADS=16\n[ -f /workspace/venv/bin/activate ] && . /workspace/venv/bin/activate\ncd /workspace/thi_nghiem_2 && . scripts/ham_b.sh\n' > /workspace/env_b3.sh
python3 -m venv /workspace/venv && . /workspace/env_b3.sh
pip install -U pip
```
`OMP_NUM_THREADS=16`: lượt 28/10 cho thấy torch mở 64 luồng trên máy 32 vCPU, pytest chậm hơn nhiều lần. Không nâng
torch ở nhánh nào, kể cả khi pip gợi ý: mọi lượt phải cùng torch 2.11.0 với lượt kiểm hook bước 3.

**Máy đầu tiên:**
```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128     # ⏱ 3–8 phút
pip install -r requirements.txt
```
**Máy thứ hai trở đi:** in các lệnh cài đúng phiên bản từ lượt kiểm hook bước 3 mà bước 4 sẽ dùng, **cho từng mô
hình**:
```bash
may_moi qwen35_9b; may_moi gemma4_12b
```
Lượt bước 4 nằm ở thư mục khác mặc định (ví dụ `hieu_chinh_lo20`) thì thêm `--thu_muc <thư mục>` cho mô hình đó,
để `may_moi` đọc đúng lượt kiểm hook mà lượt đó dùng. Hai khối lệnh in ra phải giống nhau ở các dòng `pip install` (cùng một máy phục vụ cả hai mô hình); khác nhau thì dừng,
báo Ân. Chạy các dòng `pip install torch…`, `pip install -r requirements.txt`, `pip install transformers…` in ra. Hai
dòng `flash-linear-attention`, `causal-conv1d` chạy **sau** pytest (dưới đây), thay cho khối cài kernel của máy đầu.
Dòng `hf download` để đến mục 3.

Cả hai nhánh, rồi chạy test:
```bash
python -c "import torch,transformers;print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
mkdir -p results/nhat_ky
CUDA_VISIBLE_DEVICES="" python -m pytest -q -p no:cacheprovider tests/ 2>&1 | tee -a results/nhat_ky/pytest_$(hostname)_$(date +%Y%m%d_%H%M%S).txt; echo "MÃ THOÁT: ${PIPESTATUS[0]}"     # ⏱ 10–15 phút; phải: 166 passed
```
Có test hỏng thì dừng, chép `results/nhat_ky/` về, không chạy tiếp. `CUDA_VISIBLE_DEVICES=""` chỉ áp cho tiến trình
test: test chạy đúng như đã kiểm trên CPU, còn các lệnh ở mục 4 vẫn thấy GPU.

**Chạy test TRƯỚC khi cài kernel.** Test chạy mô hình nhỏ trên CPU; khi đã cài `flash-linear-attention`, transformers
dùng kernel Triton cho lớp DeltaNet bất kể thiết bị, nên test trên CPU sẽ lỗi. Sau đó mới cài kernel (Ghi chú kỹ
thuật hai mô hình chính, mục 1). Backend thực dùng và phiên bản gói được ghi vào `meta.json` của mọi lượt
(`moi_truong`, `backend_lop_tuyen_tinh*`). Các lượt về sau (hiệu chỉnh, pilot, chạy chính) phải dùng cùng backend và
cùng phiên bản gói với lượt kiểm hook này: bước 4 tự so và dừng mã 4 nếu khác.

**Máy đầu tiên** (cài lỗi thì bỏ qua: transformers tự dùng bản PyTorch thuần, cùng phép toán, chậm và tốn bộ nhớ hơn;
bản `flash-linear-attention` ghim ở 0.6.0, bản đã đọc mã khi rà soát, 27/10/2026):
```bash
pip install flash-linear-attention==0.6.0 2>&1 | tail -2
MAX_JOBS=16 pip install causal-conv1d --no-build-isolation 2>&1 | tail -2           # ⏱ có thể 10–20 phút biên dịch
```
**Máy thứ hai trở đi:** **không** chạy khối trên; chạy đúng hai dòng `flash-linear-attention`, `causal-conv1d` mà
`may_moi` in ra (dòng ghi "KHÔNG cài" thì không cài gói đó; lượt 28/10 không có `causal-conv1d`). Gói `kernels` không
cài ở nhánh nào.

Cả hai nhánh, ghi lại gói đã cài:
```bash
pip freeze > results/nhat_ky/pip_freeze_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
pip list 2>/dev/null | grep -i -E "flash-linear|causal-conv|^kernels " | tee -a results/nhat_ky/kernel_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
```

Kernel Triton của `flash-linear-attention` dùng TF32 bất kể cờ của PyTorch. Vì vậy lượt float32 ở mục 4d của
Qwen3.5-9B tự chuyển lớp DeltaNet / conv về bản PyTorch tham chiếu của transformers (không gỡ kernel; bf16 và các
lượt sau vẫn dùng kernel). Backend thực dùng ghi trong `meta.json` (`backend_lop_tuyen_tinh*`) và
`ket_qua_fp32.json` (`backend_lop_tuyen_tinh`). Lượt float32 của Qwen vì thế chậm hơn; chưa đo được bao nhiêu.

## 3. Tải hai mô hình, kiểm khoá file

Token Hugging Face cần cho Gemma (mô hình có giấy phép). `HF_HUB_OFFLINE=1` chỉ được bật **sau** khi đã tải xong:
bật trước thì `hf download` báo lỗi chế độ offline.

**Máy đầu tiên:**
```bash
. /workspace/env_b3.sh
read -rsp "HF token: " HF_TOKEN; echo; export HF_TOKEN
hf download Qwen/Qwen3.5-9B              # ⏱ khoảng 19 GB
hf download google/gemma-4-12B-it        # ⏱ khoảng 24 GB
unset HF_TOKEN
```
**Máy thứ hai trở đi:** tải **đúng commit** đã dùng ở máy đầu, không tải bản mới nhất (Hugging Face có thể đã có commit
mới; bước 4 nạp offline đúng commit ghi trong `huong.pt`, thiếu thì lỗi). Chạy hai dòng `HF_HUB_OFFLINE=0 hf download
… --revision …` mà `may_moi` ở mục 2 đã in (mỗi mô hình một dòng, commit lấy từ đúng lượt bước 3 mà bước 4 dùng):
```bash
. /workspace/env_b3.sh
read -rsp "HF token: " HF_TOKEN; echo; export HF_TOKEN
may_moi qwen35_9b | grep "^HF_HUB_OFFLINE=0 hf download"      # in lại dòng tải của Qwen; chép và chạy dòng đó
may_moi gemma4_12b | grep "^HF_HUB_OFFLINE=0 hf download"     # tương tự cho Gemma
unset HF_TOKEN
```
**Cả hai nhánh**, sau khi tải xong: từ đây chỉ dùng bản đã tải (không tải lại, không đổi commit):
```bash
printf 'export HF_HUB_OFFLINE=1\n' >> /workspace/env_b3.sh && . /workspace/env_b3.sh
```
Commit của bản đã tải được ghi trong `kiem_truoc.json` (`cau_truc_config.commit_hf`), trong `meta.json` của mọi lượt
và trong `huong.pt`. `buoc3_kiem_hook`, `buoc3_do_toc_do` và bước 4 tự nạp đúng commit ghi trong `huong.pt`.

Kiểm luôn ổ `/workspace` có hỗ trợ khoá file không (bước 4 cần, để không có hai tiến trình cùng sinh một lượt):

```bash
mkdir -p results && python3 -c "import fcntl,os; fd=os.open('results/.thu.khoa', os.O_RDWR|os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB); print('flock ok')"
```

Phải in `flock ok`. Nếu báo lỗi (ổ mạng), báo Ân trước khi chạy tiếp: kết quả bước 4 phải ghi ra một ổ cục bộ của máy.

## 4. Chạy cho từng mô hình

**Bắt đầu kéo kết quả về định kỳ ngay từ bây giờ** (trên máy cá nhân, một cửa sổ riêng, để chạy suốt bước 3 và bước 4).
Mất máy thì chỉ mất phần sinh sau lần `rsync` **thành công** cuối cùng:
```bash
while true; do rsync -rlt may-thue:/workspace/thi_nghiem_2/results/ /mnt/d/thesis/thi_nghiem_2/results/; echo "$(date '+%F %T') rsync: $?"; sleep 300; done
```
Mã 24 của `rsync` trong vòng này là bình thường (một file tạm biến mất giữa lúc chép). Bản kéo giữa chừng có thể tạm
lệch nhau; chỉ bản kéo cuối cùng có kiểm SHA (mục 5) mới là bản chính thức.

Hàm `b3`, `b4`, `gop4`, `may_moi` nằm trong `scripts/ham_b.sh` (đã có trong kho, `env_b3.sh` tự nạp; không dán tay). Mỗi cửa sổ
tmux mới, kể cả sau khi máy khởi động lại:
```bash
. /workspace/env_b3.sh
```
Rồi chạy lần lượt (xong mô hình này mới chạy mô hình kia):
```bash
b3 qwen35_9b
```
```bash
b3 gemma4_12b
```
Chỉ kiểm nạp trên máy mới (4a, 4b, kể cả quét attention), khi đã có lượt dựng hướng và kiểm hook đạt:
```bash
b3 qwen35_9b kiem_nap
```
Xem tiến độ ở một cửa sổ khác:
```bash
python3 scripts/tien_do_b3.py
```
`b3` không chạy nếu GPU đang bị chiếm (`nvidia-smi` báo đã dùng từ 2 000 MiB trở lên, ví dụ mô hình kia còn nạp ở cửa
sổ khác). Nếu `nvidia-smi` không có tiến trình nào của mình mà bộ nhớ đã dùng vẫn từ 2 000 MiB, hỏi Ân; được đồng ý thì
gọi:
```bash
KHONG_KIEM_GPU=1 b3 <mô hình>
```
Log ghi vào `results/nhat_ky/log_b3_<máy>.txt`.

Kiểu attention: Gemma nạp bằng `eager`, Qwen bằng `sdpa`. Lý do: ở Gemma, kernel EFFICIENT_ATTENTION của `sdpa`
(PyTorch 2.11) cho logit sai khi số vị trí truy vấn m ≡ 1 (mod 32), m ≥ 33 (PyTorch issue #191937; báo cáo kết quả
bước 3). `nap_mo_hinh` dừng nếu transformers không dùng đúng kiểu của đặc tả.

`b3` dừng ngay ở bước đầu tiên có mã thoát khác 0. Riêng mã 4 ("cần xem") ở 4c và 4d thì chạy tiếp: phần 2 kiểm cơ
chế hook (đúng với hướng nào cũng được), đo tốc độ không phụ thuộc hướng. Mã 4 vẫn **không phải đạt** (mục 6): bước
4 không chạy với hướng hay kiểm hook "cần xem", trừ khi Ân, sau khi hai reviewer xem, quyết đi tiếp bằng một file
quyết định (`RUNBOOK_BUOC_4.md`, mục 0). Khi hàm dừng: kéo `results/` về, không chạy tiếp mô hình đó, không sửa code
trên máy thuê.

| Bước trong `b3` | Việc | ⏱ ước |
|---|---|---|
| 4a | Kiểm trước (không nạp trọng số): khối 9 token, vị trí lấy −5, token kết thúc, `layer_types`, greedy | vài giây |
| 4b | Nạp và kiểm: một GPU, cấu trúc, bộ đếm vị trí, kiểu attention, quét attention bf16 (m = 2…320, lô đệm, bốn lô thật của bước 4, KV cache), sinh thử 32 câu, rò thinking, token kết thúc | 4–8 phút |
| 4c | Dựng hướng và s_ℓ: chẩn đoán position_ids / đệm, 512 câu dựng, mọi lớp | 3–8 phút |
| 4d | Bản thu nhỏ phần 2, bf16 rồi float32; quét attention float32 (m = 2…320 và lô đệm 8 câu ở độ dài ≡ 1 mod 32) | 35–60 phút (đo 28/10) |
| 4e | Đo tốc độ và bộ nhớ: lô 16, 32, 40, 64, 128; nền và can thiệp α = 0,8; ước giờ GPU, kể cả riêng cho bước 4 | 7–15 phút (đo 28/10) |

**Máy chết hoặc mất kết nối giữa `b3`:** trước hết xem tmux và GPU:
```bash
tmux ls
```
```bash
nvidia-smi
```
Nếu tiến trình cũ vẫn chạy thì cứ để nó chạy, nối lại bằng:
```bash
tmux attach -t b3
```
Nếu đã chết:

| Chết trong | Chạy lại | Lý do |
|---|---|---|
| 4a, 4b, 4c | `b3 <mô hình>` | Các bước này nhanh. Thư mục `dung_huong_*` dở không có `huong.pt` (file ghi cuối cùng) nên bị bỏ qua |
| 4d | `b3 <mô hình> tu_4d` | Dùng lượt dựng hướng đã xong mới nhất, không dựng lại hướng; chạy 4d rồi cả 4e. Thư mục `kiem_hook_*` dở không có `ket_luan.json` nên bước 4 bỏ qua |
| 4e | `b3 <mô hình> tu_4e` | Chỉ đo lại tốc độ |

Không xoá hay sửa thư mục dở; chúng chỉ bị bỏ qua.

Mã thoát:

| Script | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| buoc3_kiem_truoc | đạt | cấu trúc config hoặc cấu hình sinh lệch | **khối sau câu hỏi / template / token kết thúc lệch dự kiến: DỪNG** | — | — |
| buoc3_nap_kiem | đạt (kể cả quét attention `can_kiem`: log ghi `CHÚ Ý`; đọc theo luật ở `RUNBOOK_LUOT_THUE_TIEP.md`, mục 4, 5) | có phép kiểm không đạt (xem `nap_kiem.json`), kể cả kiểu attention sai đặc tả hoặc quét attention `khong_dat` | — | không nằm trọn một GPU | — |
| buoc3_dung_huong | xong | lỗi, hoặc lượt chính thức không xác minh được commit mô hình | — | không nằm trọn một GPU | **cần xem**: cảnh báo đệm bật, hướng chưa dùng cho bước 4 |
| buoc3_kiem_hook | đạt | không đạt (kể cả quét attention float32 `khong_dat`), hoặc commit mô hình không khớp file hướng, hoặc file hướng không dựng bằng kiểu attention của đặc tả | khối lệch | không nằm trọn một GPU | **cần xem** (không phải đạt) |
| buoc3_do_toc_do | xong | lỗi | — | không nằm trọn một GPU | — |

Nếu `buoc3_kiem_truoc` trả mã 2: **không sửa code trên máy thuê**. Kéo `results/` về. Trước khi kết luận "khối
lệch", xem `khoi_sau_cau_hoi` trong `kiem_truoc.json`: nếu `K_theo_so_cau` vẫn là 9 và vị trí lấy vẫn −5, chỉ chuỗi
giải mã của token khác chuỗi dự kiến (ví dụ có dấu cách đầu) thì đó là lệch ở đặc tả, không phải ở khối. Dù trường hợp
nào cũng quyết lại cùng Ân và hai reviewer trước khi chạy tiếp (Nhap: "cách tách token cần xác nhận khi nạp
tokenizer").

## 5. Kéo kết quả về, kiểm SHA, lưu trữ

Làm sau khi `b3` của cả hai mô hình xong, **và bắt buộc trước lệnh `b4` đầu tiên** nếu chạy bước 4 trên cùng máy: bước 4
gắn với SHA của `huong.pt`, nên mất kết quả bước 3 là không chạy tiếp được bước 4.

Trên máy thuê:
```bash
. /workspace/env_b3.sh
pip freeze > results/nhat_ky/pip_freeze_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
SHA=results/nhat_ky/sha256_buoc3_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
find results/buoc3 -type f -print0 | sort -z | xargs -0 sha256sum > $SHA; echo $SHA
```
Trên máy cá nhân (`<file SHA>` là đường dẫn vừa in ra):
```bash
cd /mnt/d/thesis/thi_nghiem_2
rsync -rlt may-thue:/workspace/thi_nghiem_2/results/ results/; echo "rsync: $?"
sha256sum -c --quiet <file SHA> && echo "SHA ĐỦ"
D=/mnt/d/thesis/luu_tru/results_$(date +%Y%m%d_%H%M%S); mkdir -p $D && rsync -rlt results/ $D/ && echo "Lưu trữ: $D"
```
Phải in `rsync: 0` và `SHA ĐỦ`. Mỗi lần lưu trữ là một thư mục mới có ngày giờ, nên không bao giờ ghi đè bản lưu trước.
File `kich_hoat_vi_tri_lay.pt` khoảng 134 MB (Qwen) và 189 MB (Gemma); cần cho bootstrap của C3 ở bước phân tích.

Chỉ huỷ máy khi **không** chạy bước 4 ngay trên máy này. Nếu chạy (cách 1 của `RUNBOOK_BUOC_4.md`), giữ máy và để vòng
`rsync` ở mục 4 chạy tiếp.

## 6. Đọc kết quả (`results/buoc3/<mô hình>/`)

| Lượt | File chính | Đọc gì |
|---|---|---|
| `kiem_truoc_*` | `kiem_truoc.json` | `khoi_sau_cau_hoi` (9 token trên 512 câu, vị trí −5), `token_ket_thuc`, `cau_truc_config`, `cau_hinh_sinh.la_greedy_dung`, `chat_template` đã dùng |
| `nap_kiem_*` | `nap_kiem.json`, `sinh_thu.jsonl` | `dat_thanh_phan`; `kieu_attention` (đang dùng, đặc tả); `quet_attention`: `trang_thai` (bf16: > 5 logit `khong_dat`, 2–5 `can_kiem`, ≤ 2 `dat`), `lech_lon_nhat`, `lech_lon_nhat_m_xau` / `lech_lon_nhat_m_khac` (m ≡ 1 mod 32 hay không), `so_o_khong_huu_han`, `do_dai_sai`, bảng `don`, `lo`, `lo_thuc`, `cache`; `phan_bo_thiet_bi`, `cau_truc` (đường dẫn lớp, full attention, `layer_scalar` của Gemma), `goi_embedding`, `sinh_thu.ro_thinking`, `sinh_thu.so_cau_theo_eos` (đạt khi có câu dừng ở id chính; 0 câu → xem lại eos); đọc vài câu trong `sinh_thu.jsonl` bằng mắt |
| `dung_huong_*` | `tom_tat.json`, `do_position_ids.json`, `huong.pt` | **`trang_thai`** / **`canh_bao_dem`** (True → mã 4: không dùng hướng cho bước 4 trước khi Ân xem); `do_position_ids.json` chỉ chẩn đoán: lệch tương đối B–C theo lớp và theo câu (cờ tắt không chứng nhận không có lỗi đệm); ‖r‖, ‖r‖/s, s theo lớp; cosine chia đôi; cosine hai vị trí (chỉ chẩn đoán, Nhap điểm 7, null ở lớp có hướng bằng 0); cosine en–vi (chỉ mô tả). Thư mục không có `huong.pt` là lượt dở |
| `kiem_hook_*` | `ket_luan.json`, `ket_qua_fp32.json`, `quet_attention_fp32.json` | `ket_luan.quet_attention_fp32` (float32: ≤ 0,5 logit `dat`, không đạt thì cả lượt `khong_dat`); như bước 1: (ii) ở hai kiểu số; (i), (iii), kiểm âm ở float32; `trang_thai` `dat` / `can_xem` / `khong_dat`, luật ba mức (docstring `tn2.kiem_hook`). Chẩn đoán: `tong_ket.khong_hook_lap_lai_khop` (False: đường nền không tất định, nên khi (ii) trượt chưa thể quy riêng độ lệch cho hook; cũng chưa loại trừ lỗi hook: dừng, xem cùng Ân); `backend_lop_tuyen_tinh` của lượt float32 phải toàn `la_ban_tham_chieu: true` với Qwen. Thư mục không có `ket_luan.json` là lượt dở |
| `toc_do_*` | `toc_do.json` | câu/giờ, bộ nhớ đỉnh theo cỡ lô; ước giờ GPU theo số câu của Nhap cho riêng mô hình này (chạy chính theo số lớp); **`uoc_gio_gpu_buoc4`**: giờ GPU cho bước 4 theo đúng số câu của kế hoạch (19 520 câu mỗi mô hình), ở tốc độ đo tại lô 40 |
| `nhat_ky/` | `log_b3_<máy>.txt`, `pytest_<máy>_<giờ>.txt`, `pip_freeze_<máy>_<giờ>.txt`, `kernel_<máy>_<giờ>.txt`, `sha256_buoc3_<máy>_<giờ>.txt` | log đầy đủ (kể cả traceback), kết quả pytest, phiên bản gói, danh sách SHA |

Lưu ý khi đọc tốc độ: số đo trên máy 80 GB không áp sang loại GPU khác. Trước khi thuê máy cho lượt hiệu chỉnh và
chạy chính, đo lại vài phút trên đúng loại máy đó, cùng môi trường và backend (báo cáo kết quả bước 1, mục 7).
