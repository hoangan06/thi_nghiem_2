# Runbook lượt thuê tiếp theo: làm lại bước 3 của Gemma, kiểm nạp Qwen, rồi bước 4 cho cả hai (A100 80 GB)

Theo các đề xuất Q-E1 đến Q-E6 mà Ân đã duyệt ngày 29/10/2026 (`reviewer/Phan_tich_ra_soat_ket_qua_buoc3_29_10_2026.md`).
Ân quyết thêm: "cùng loại GPU" nghĩa là cùng họ A100.

Lý do làm lại: ở Gemma 4 12B, attention `sdpa` (kernel EFFICIENT_ATTENTION của PyTorch 2.11) cho logit sai khi số vị
trí truy vấn m ≡ 1 (mod 32), m ≥ 33. Từ commit này, Gemma luôn nạp bằng `eager`, còn Qwen giữ `sdpa`. Hai bước 4b
và 4d giờ có thêm phép quét attention theo độ dài.

**Cách dùng các khối lệnh.** Mỗi khối dán một lần rồi Enter; các dòng trong khối chạy lần lượt. Khối nào phải đọc kết
quả trước khi chạy khối sau thì runbook ghi rõ điều cần đọc ngay dưới khối. Ba loại lệnh luôn để **riêng một khối**,
vì dán chung với lệnh khác sẽ hỏng: `ssh may-thue` và `tmux new` (lệnh sau sẽ không vào đúng phiên), `read -rsp`
(dòng dán phía sau bị đọc làm token). Ghi "máy cá nhân" là chạy trong WSL; ghi "máy thuê" là chạy trong tmux trên máy
thuê.

Lần thuê này chỉ cần runbook này. `RUNBOOK_BUOC_3.md` và `RUNBOOK_BUOC_4.md` chỉ mở khi bảng "Khi có sự cố" ở cuối
trỏ tới.

## Thứ tự và thời gian (ước, chưa đo)

| # | Việc | ⏱ ước |
|---|---|---|
| 1 | Chuẩn bị trên máy cá nhân (trước khi thuê) | 5 phút |
| 2 | Thuê máy, chép code và `results/`, dựng môi trường, pytest, cài kernel | 30–45 phút |
| 3 | Tải hai mô hình đúng commit | 10–20 phút |
| 4 | `b3 qwen35_9b kiem_nap`: 4a, 4b của Qwen, có quét attention | 5–10 phút |
| 5 | `b3 gemma4_12b`: 4a đến 4e của Gemma bằng eager (dựng hướng mới) | 60–100 phút |
| 6 | Kéo kết quả về, kiểm SHA | 10 phút |
| 7 | `b4 qwen35_9b; b4 gemma4_12b` | 4–5 giờ; số sát hơn ở `uoc_gio_gpu_buoc4` trong `toc_do.json` mới của Gemma |
| 8 | Kéo kết quả, kiểm SHA, lưu trữ, huỷ máy | 10 phút |

Tổng khoảng 6–8 giờ máy. Bước 7 không cần ngồi canh. Vòng `rsync` 5 phút một lần trên máy cá nhân giữ an toàn kết
quả.

## 1. Trước khi thuê (máy cá nhân)

Sau khi đã commit và push (Ân tự chạy git):

```bash
cd /mnt/d/thesis/thi_nghiem_2
git status --porcelain
git rev-parse HEAD
ls results/buoc3/qwen35_9b/ results/buoc3/gemma4_12b/
```

Đọc:
- `git status --porcelain` không in gì;
- ghi lại commit mà `git rev-parse HEAD` in ra;
- có `dung_huong_20261028_080952` và `kiem_hook_20261028_081023` (Qwen), `dung_huong_20261028_085421` (Gemma). Bước 4
  của Qwen dùng lại hướng và kiểm hook cũ.

## 2. Thuê máy, dựng môi trường

**Chọn máy (Vast.ai):**

| Tiêu chí | Giá trị |
|---|---|
| GPU | 1 GPU **A100 80 GB**, SXM4 hay PCIe đều được (Ân quyết 29/10/2026: cùng họ A100). Cần 80 GB vì lượt float32 ở 4d của Gemma đo đỉnh 48,6 GiB |
| Đĩa | ≥ 120 GB, ổ cục bộ |
| Ảnh | Ubuntu, CUDA ≥ 12.8, Python 3.12 |
| Mạng | Direct SSH |

Sau khi thuê, sửa mục `may-thue` trong `~/.ssh/config` của WSL theo địa chỉ và cổng SSH mà Vast.ai cho:

```
Host may-thue
    HostName <địa chỉ IP>
    Port <cổng>
    User root
```

**Máy cá nhân:** cài công cụ lên máy thuê, rồi chép code và `results/`. `results/` cần cho `may_moi` và `b4`:

```bash
ssh may-thue 'apt-get update -qq && apt-get install -y -qq rsync tmux python3-venv >/dev/null && echo "cài xong"'
rsync -rlt --exclude results --exclude __pycache__ /mnt/d/thesis/thi_nghiem_2/ may-thue:/workspace/thi_nghiem_2/; echo "rsync code: $?"
rsync -rlt /mnt/d/thesis/thi_nghiem_2/results/ may-thue:/workspace/thi_nghiem_2/results/; echo "rsync results: $?"
```

Phải in `cài xong`, `rsync code: 0` và `rsync results: 0`.

**Máy thuê:** vào máy và mở tmux, mỗi lệnh một khối:

```bash
ssh may-thue
```

```bash
tmux new -s b3
```

Trong tmux: kiểm đĩa, tạo `env_b3.sh`, tạo venv, rồi in lệnh cài đúng phiên bản của lượt bước 3. `env_b3.sh` đặt
`HF_HOME`, tắt kernel từ Hub, giới hạn 16 luồng CPU (lượt 28/10 cho thấy torch mở 64 luồng trên máy 32 vCPU, làm
pytest rất chậm), bật venv và nạp các hàm `b3`, `b4`, `may_moi`:

```bash
df -h /workspace
printf 'export HF_HOME=/workspace/hf\nexport USE_HUB_KERNELS=NO\nexport OMP_NUM_THREADS=16\nexport MKL_NUM_THREADS=16\n[ -f /workspace/venv/bin/activate ] && . /workspace/venv/bin/activate\ncd /workspace/thi_nghiem_2 && . scripts/ham_b.sh\n' > /workspace/env_b3.sh && \
python3 -m venv /workspace/venv && . /workspace/env_b3.sh && \
[ "$VIRTUAL_ENV" = /workspace/venv ] && echo "venv: $VIRTUAL_ENV" && \
pip install -U pip && \
may_moi qwen35_9b && \
may_moi gemma4_12b
```

Đọc (chuỗi `&&` dừng ngay ở lệnh đầu tiên lỗi; khi đó không chạy tiếp, báo Ân):
- cột `Avail` của `/workspace` từ 100 GB trở lên;
- dòng `venv: /workspace/venv` (venv đã bật; venv chưa bật thì chuỗi dừng trước `pip`, không cài gì);
- hai khối `may_moi` in ra giống nhau ở mọi dòng `pip install`, và trùng ba dòng `pip install` ở khối dưới; khác thì
  dừng, báo Ân.

Cài torch, gói, rồi kiểm phiên bản (⏱ 5–10 phút; `&&` để dừng ngay nếu một lệnh lỗi):

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && \
pip install -r requirements.txt && \
pip install transformers==5.17.0 tokenizers==0.23.2 && \
python -c "import torch,transformers;print(torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))"
```

Phải in `2.11.0+cu128 5.17.0 NVIDIA A100…`. Ba điều **không** làm:
- không nâng torch, kể cả khi pip gợi ý (lỗi sdpa đã sửa ở bản nightly, nhưng bước 3 và bước 4 phải cùng torch 2.11);
- không cài `causal-conv1d` (`may_moi` in "KHÔNG cài": lượt 28/10 không có gói này);
- không cài gói `kernels`.

Chạy test **trước** khi cài `flash-linear-attention` (⏱ 10–15 phút):

```bash
mkdir -p results/nhat_ky
CUDA_VISIBLE_DEVICES="" python -m pytest -q -p no:cacheprovider tests/ 2>&1 | tee -a results/nhat_ky/pytest_$(hostname)_$(date +%Y%m%d_%H%M%S).txt; echo "MÃ THOÁT: ${PIPESTATUS[0]}"
```

Phải in `166 passed` và `MÃ THOÁT: 0`. Có test hỏng thì dừng, chép `results/nhat_ky/` về, báo Ân.

Cài kernel (dòng `flash-linear-attention` mà `may_moi` đã in) và ghi lại gói đã cài:

```bash
pip install flash-linear-attention==0.6.0 2>&1 | tail -2
pip freeze > results/nhat_ky/pip_freeze_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
pip list 2>/dev/null | grep -i -E "flash-linear|causal-conv|^kernels " | tee -a results/nhat_ky/kernel_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
```

Dòng cuối phải chỉ có `flash-linear-attention 0.6.0`.

## 3. Tải hai mô hình đúng commit

Nhập token Hugging Face (khối riêng; gõ token rồi Enter, token không hiện ra):

```bash
read -rsp "HF token: " HF_TOKEN; echo; export HF_TOKEN
```

In dòng tải mà `may_moi` lấy từ lượt bước 3:

```bash
may_moi qwen35_9b | grep "^HF_HUB_OFFLINE=0 hf download"
may_moi gemma4_12b | grep "^HF_HUB_OFFLINE=0 hf download"
```

Hai dòng in ra phải trùng hai dòng `hf download` ở khối dưới (commit của lượt 28/10); khác thì dừng, báo Ân.

Tải, xoá token khỏi phiên, chuyển sang chỉ dùng bản đã tải, rồi kiểm khoá file:

```bash
HF_HUB_OFFLINE=0 hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a && \
HF_HUB_OFFLINE=0 hf download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 && \
printf 'export HF_HUB_OFFLINE=1\n' >> /workspace/env_b3.sh && echo "TẢI XONG"
unset HF_TOKEN
. /workspace/env_b3.sh
python3 -c "import fcntl,os; fd=os.open('results/.thu.khoa', os.O_RDWR|os.O_CREAT); fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB); print('flock ok')"
```

Phải in `TẢI XONG` và `flock ok`. Không thấy `TẢI XONG` thì dừng, báo Ân (chưa bật chế độ offline).

**Máy cá nhân, một cửa sổ WSL riêng:** bật vòng kéo kết quả về, để chạy suốt lượt thuê:

```bash
while true; do rsync -rlt may-thue:/workspace/thi_nghiem_2/results/ /mnt/d/thesis/thi_nghiem_2/results/; echo "$(date '+%F %T') rsync: $?"; sleep 300; done
```

## 4. Kiểm nạp Qwen trên máy mới (`kiem_nap`)

Qwen không dựng lại hướng: bước 4 dùng hướng và kiểm hook ngày 28/10. Lượt kiểm hook đó chưa ghi kiểu attention, nên
bước 4 nhận nó theo một ngoại lệ hẹp đã duyệt (Q-E2): khớp commit kho e61aa007, commit mô hình c2022362…, SHA hướng
3bfaf7e6…, và mô hình đang nạp dùng `sdpa`. Bước này chỉ chạy 4a và 4b để kiểm máy mới, kể cả phép quét attention.
Khối dưới chạy `kiem_nap`, rồi in các dòng cần đọc:

```bash
b3 qwen35_9b kiem_nap
grep -E "Commit HF|Quét attention|ĐẠT:|MÃ THOÁT" results/nhat_ky/log_b3_$(hostname).txt | tail -6
```

Bình thường kết thúc bằng `xong kiem_nap` và hai dòng `MÃ THOÁT: 0`. Đầu log có dòng `Commit HF cho 4a–4c:
c202236235762e1c871ad0ccb60c8ee5ba337b9a` (đúng commit đã tải; `b3` lấy từ lượt dựng hướng mới nhất, vì bản tải theo
commit không có nhãn `main` để nạp offline).

Dòng `Quét attention:` ghi trạng thái ở cuối phần đầu (`→ dat`, `→ can_kiem`, `→ khong_dat`), rồi ghi "dạng một
chuỗi: lệch lớn nhất ở m ≡ 1 (mod 32) **X**, ở m khác **Y**; các dạng lô đệm và cache **Z**; ô không hữu hạn **N**".
Lỗi kernel đã biết làm X lớn hơn hẳn Y (15–27 logit). Các lô đệm đều có độ dài ≡ 1 (mod 32), nên Z lớn hơn hẳn X và Y
cũng là dấu hiệu lỗi dồn ở các độ dài này. Nhiễu làm tròn thì không dồn. Dòng ghi `LỖI` (phép quét không chạy xong)
thì không có X, Y, Z, N.

**Luật Ân duyệt trước ngày 29/10/2026 (phương án 2)** — trạng thái các lượt giữ nguyên như máy ghi; quyết định đi tiếp
được ghi vào báo cáo kết quả:

| Kết quả của `kiem_nap` | Làm gì |
|---|---|
| `→ dat` | Sang mục 5 |
| `→ can_kiem` (4b mã 0), X ≤ Y, Z ≤ max(X, Y) và N = 0 | Sang mục 5. Qwen dùng tiếp kiểm hook 28/10: 4d float32 hôm đó sạch, kể cả lô 48 dài 33 có đệm |
| `→ can_kiem` (4b mã 0), N = 0, nhưng X > Y hoặc Z > max(X, Y) | Chạy lại kiểm hook của Qwen để lượt float32 (một chuỗi và lô đệm, ngưỡng 0,5) phân xử, khoảng 45 phút (chạy cả 4e). Khối ngay dưới bảng |
| 4b mã 1, dòng `ĐẠT:` cho thấy **chỉ** "quét attention khong_dat" (mọi thành phần khác `True`), N = 0 | Như dòng trên: khối ngay dưới bảng |
| Mọi trường hợp khác: 4a hay 4b có mã khác 0 và 1, 4b mã 1 vì thành phần khác, N > 0, hoặc dòng `Quét attention:` có `LỖI` | Dừng, kéo `results/` về, báo Ân |

Chỉ chạy khối này khi bảng trên chỉ tới nó:

```bash
b3 qwen35_9b tu_4d
grep -E "Quét attention float32|MÃ THOÁT" results/nhat_ky/log_b3_$(hostname).txt | tail -4
```

Đi tiếp khi 4d `MÃ THOÁT: 0` và dòng `Quét attention float32:` ghi `→ dat`; nếu không thì dừng, báo Ân. `b4` sẽ tự dùng
lượt kiểm hook mới này (mới nhất, cùng file hướng 28/10). Máy chết giữa chừng: chạy lại đúng khối trên, và chưa chạy
`b4 qwen35_9b` khi 4d của nó chưa `MÃ THOÁT: 0`. Nếu không, `b4` sẽ lặng lẽ dùng lại kiểm hook 28/10, vì thư mục kiểm
hook dở không có `ket_luan.json`.

## 5. Bước 3 của Gemma bằng eager

Khối dưới chạy 4a–4e của Gemma (60–100 phút), rồi in các dòng cần đọc:

```bash
b3 gemma4_12b
grep -E "Commit HF|MÃ THOÁT|Quét attention|CHÚ Ý|ĐẠT|trạng thái" results/nhat_ky/log_b3_$(hostname).txt | tail -20
```

Dòng `Commit HF cho 4a–4c:` phải là `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`.

Xem tiến độ trong lúc chạy: mở một cửa sổ tmux khác (bấm `Ctrl-b` rồi `c`), rồi:

```bash
. /workspace/env_b3.sh && python3 scripts/tien_do_b3.py
```

Cách đọc dòng `Quét attention:` (X, Y, N): như mục 4. Eager bf16 làm tròn khác tham chiếu MATH (MATH giữ phần tính
trung gian ở float32), nên phép quét bf16 của Gemma có thể ra `can_kiem` hay `khong_dat` chỉ vì nhiễu làm tròn; eager
không đi qua kernel có lỗi. Lượt quét float32 ở 4d là căn cứ quyết.

**Luật Ân duyệt trước ngày 29/10/2026 (phương án 2)** — trạng thái các lượt giữ nguyên như máy ghi; quyết định đi tiếp
được ghi vào báo cáo kết quả:

| Kết quả 4b của Gemma | Làm gì |
|---|---|
| `→ dat`, hoặc `→ can_kiem` (4b mã 0) | `b3` tự chạy tiếp 4c–4e. Đọc điều kiện đi tiếp ở dưới |
| 4b mã 1, dòng `ĐẠT:` cho thấy **chỉ** "quét attention khong_dat" (mọi thành phần khác `True`), lệch lớn nhất ≤ 12 logit (dưới mức lỗi đã biết 15–27), X ≤ Y và N = 0 | Chạy tay 4c, kiểm mã thoát, rồi `b3 gemma4_12b tu_4d` (hai khối ngay dưới bảng) |
| Mọi trường hợp khác: 4b mã 1 mà không thoả dòng trên (lệch > 12, X > Y, N > 0, thành phần khác hỏng), 4a hay 4b có mã khác, hoặc dòng `Quét attention:` có `LỖI` | Dừng, kéo `results/` về, báo Ân. Có thể là lỗi thật |

Chỉ khi bảng trên chỉ tới nhánh chạy tay. Khối 1, chạy 4c:

```bash
python scripts/buoc3_dung_huong.py --mo_hinh_dac_ta gemma4_12b --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 2>&1 | tee -a results/nhat_ky/log_b3_$(hostname).txt; echo "MÃ THOÁT 4c: ${PIPESTATUS[0]}"
```

Khối 2, chỉ khi dòng cuối của khối 1 là `MÃ THOÁT 4c: 0` (mã 4 là "cần xem", mã khác là lỗi: dừng, báo Ân):

```bash
b3 gemma4_12b tu_4d
grep -E "MÃ THOÁT|Quét attention float32|trạng thái" results/nhat_ky/log_b3_$(hostname).txt | tail -8
```

**Điều kiện để chạy bước 4 của Gemma** (cả hai nhánh trên):
- 4a, 4c, 4d đều `MÃ THOÁT: 0` (mã 4 ở 4c hoặc 4d là "cần xem", không phải đạt: dừng, báo Ân);
- 4b: `→ dat`; hoặc `→ can_kiem` với X ≤ Y và N = 0; hoặc nhánh mã 1 ở bảng trên;
- dòng `Quét attention float32:` của 4d ghi `→ dat`;
- 4e chạy xong và `toc_do.json` không có `het_bo_nho` ở lô 40.

Thiếu điều kiện nào thì dừng, kéo `results/` về, báo Ân.

## 6. Kéo kết quả, kiểm SHA

**Máy thuê:** lập bảng SHA của `results/buoc3` (in ra đường dẫn file SHA):

```bash
SHA=results/nhat_ky/sha256_buoc3_$(hostname)_$(date +%Y%m%d_%H%M%S).txt
find results/buoc3 -type f -print0 | sort -z | xargs -0 sha256sum > $SHA; echo $SHA
```

**Máy cá nhân:** kéo về, lấy bảng SHA mới nhất vừa kéo về, rồi kiểm:

```bash
cd /mnt/d/thesis/thi_nghiem_2
rsync -rlt may-thue:/workspace/thi_nghiem_2/results/ results/; echo "rsync: $?"
SHA=$(ls -t results/nhat_ky/sha256_buoc3_*.txt | head -1); echo "Bảng SHA: $SHA"
sha256sum -c --quiet "$SHA" && echo "SHA ĐỦ"
```

Phải in `rsync: 0`, dòng `Bảng SHA:` trùng đường dẫn mà máy thuê vừa in, và `SHA ĐỦ`. Xong thì sang mục 7 ngay.

## 7. Bước 4 cho cả hai mô hình

**Máy thuê:**

```bash
b4 qwen35_9b; b4 gemma4_12b
```

`b4` tự chọn lượt dựng hướng và kiểm hook đã xong mới nhất: Qwen dùng hướng 28/10, Gemma dùng lượt eager vừa chạy.
Đầu log của mỗi mô hình ghi thư mục `--huong`, `--kiem_hook`. Nếu Qwen vẫn dùng kiểm hook 28/10, log có thêm một dòng
`CHÚ Ý` nói kiểu attention của lượt kiểm hook là suy ra theo ngoại lệ (bình thường, đã duyệt). Hướng không dựng bằng
đúng kiểu attention của đặc tả (ví dụ hướng Gemma sdpa ngày 28/10) thì `b4` dừng mã 4.

Theo dõi ở một cửa sổ tmux khác (`Ctrl-b` rồi `c`; thoát `tail` bằng `Ctrl-c`, không ảnh hưởng `b4`):

```bash
. /workspace/env_b3.sh && tail -f results/nhat_ky/log_b4_$(hostname).txt
```

**Trong lúc bước 4 chạy: so hướng mới và cũ của Gemma.** Ở một cửa sổ tmux khác nữa. Script chỉ dùng CPU, vài giây,
không đụng lượt bước 4. Đây chỉ là mô tả, không chặn gì. Dự đoán viết trước: cosine ≥ 0,999 ở năm lớp hiệu chỉnh;
≥ 0,99 ở lớp có ‖r‖/s < 0,03; chênh s_ℓ < 1 %.

```bash
. /workspace/env_b3.sh && CUDA_VISIBLE_DEVICES="" python scripts/so_sanh_huong_gemma.py 2>&1 | tee -a results/nhat_ky/so_sanh_huong_$(hostname).txt
```

Gửi phần in ra cho Claude để ghi vào báo cáo. File kết quả nằm ở `results/buoc3/gemma4_12b/so_sanh_huong_<giờ>/`.

Mã thoát và cách xử lý: `RUNBOOK_BUOC_4.md`, mục 2. Mã 4 có thêm ba nguyên nhân mới:
- kiểu attention của mô hình đang nạp khác lượt kiểm hook;
- lượt kiểm hook không ghi kiểu attention và không thuộc ngoại lệ;
- file hướng không dựng bằng kiểu attention của đặc tả.

Nếu transformers không nạp được đúng kiểu attention của đặc tả, `nap_mo_hinh` báo lỗi ngay khi nạp: mã 1, log có
traceback `ValueError`.

Mã 8 giờ so **họ** GPU: mọi A100 là cùng một họ.

## 8. Kéo kết quả, kiểm SHA, lưu trữ, huỷ máy

**Máy thuê:** đọc lại cả hai dòng `=== … MÃ THOÁT` của `b4` (với `b4 qwen35_9b; b4 gemma4_12b`, Gemma vẫn chạy khi Qwen
dừng, nên phải đọc cả hai):

```bash
grep "MÃ THOÁT" results/nhat_ky/log_b4_$(hostname).txt
```

**Máy cá nhân:** dừng vòng `rsync` (bấm `Ctrl-c` ở cửa sổ đó), rồi:

```bash
cd /mnt/d/thesis/thi_nghiem_2
rsync -rlt may-thue:/workspace/thi_nghiem_2/results/ results/ && echo "rsync: 0" && \
(cd results/buoc4/qwen35_9b/hieu_chinh && sha256sum -c --quiet sha256.txt) && echo "qwen35_9b SHA ĐỦ" && \
(cd results/buoc4/gemma4_12b/hieu_chinh && sha256sum -c --quiet sha256.txt) && echo "gemma4_12b SHA ĐỦ" && \
D=/mnt/d/thesis/luu_tru/results_$(date +%Y%m%d_%H%M%S) && mkdir -p $D && rsync -rlt results/ $D/ && echo "Lưu trữ: $D"
```

Phải thấy `rsync: 0`, hai dòng `SHA ĐỦ` và dòng `Lưu trữ:` (chỉ lưu trữ khi `rsync` thành công và cả hai mô hình đủ
SHA). Sau đó mới huỷ máy
(Vast.ai: Destroy). Thiếu dòng nào thì chưa huỷ máy, báo Ân.

## Khi có sự cố

| Tình huống | Làm gì |
|---|---|
| Mất kết nối SSH | `ssh may-thue`, rồi `tmux attach -t b3` (hai khối riêng); tiến trình vẫn chạy |
| Máy chết trong `b3 gemma4_12b` | `RUNBOOK_BUOC_3.md`, mục 4, bảng "Máy chết" (`tu_4d`, `tu_4e`) |
| Máy chết trong `b4` | Chạy lại đúng `b4 <mô hình>`; đổi máy thì `RUNBOOK_BUOC_4.md`, mục 1, cách 2 (máy mới cùng họ A100; chạy `kiem_nap` cho từng mô hình trước `b4`, phải `→ dat`, khác thì dừng, báo Ân; 40 GB chỉ khi `toc_do.json` mới cho thấy lô 40 còn dư rõ) |
| Hết bộ nhớ ở `b4` | `RUNBOOK_BUOC_4.md`, mục 2, "Hết bộ nhớ GPU" |
| Phép quét attention ra `can_kiem` hay `khong_dat` | Theo luật ở mục 4 (Qwen) và mục 5 (Gemma). Gemma: với `can_kiem` ở 4b, `b3` tự chạy tiếp tới 4e; quyết theo 4d. Qwen (`kiem_nap` dừng sau 4b): theo bảng ở mục 4 |
| Máy chết trong `b3 qwen35_9b tu_4d` | Chạy lại đúng lệnh đó. Chưa chạy `b4 qwen35_9b` khi 4d chưa `MÃ THOÁT: 0` (mục 4) |
| `can_xem` (mã 4) ở 4c hoặc 4d, quét float32 `khong_dat`, hay bất kỳ `khong_dat` nào khác | Dừng mô hình đó, kéo `results/` về, báo Ân. Không sửa code trên máy thuê |
