# Runbook bước 4: lượt hiệu chỉnh (Qwen3.5-9B, Gemma 4 12B; GPU thuê)

## Mục tiêu

Theo Nhap (Trình tự chạy, mục 4), bước này sinh toàn bộ câu trả lời trên tập hiệu chỉnh cho cả hai mô hình. Bước 5 sẽ
dựng tập nhãn tham chiếu và chọn trọng tài từ các câu trả lời này. Bước 6 chấm chúng, rồi chọn α và chọn giữa phương
án 1 và 3.

Mỗi mô hình có 488 đơn vị, mỗi đơn vị 40 câu (một lô), mỗi câu trả lời tối đa 256 token:

| Phần | Số đơn vị | Gồm |
|---|---|---|
| Lượt nền | 4 | α = 0 (hook gắn với vectơ 0), gồm D_plain_test và D_harm_test, mỗi tập en và vi |
| Lưới chính | 320 | 2 kiểu × 4 mức α (0,1; 0,2; 0,4; 0,8) × 5 lớp đại diện × 2 phương án × 4 chiều |
| Lượt nền không hook | 4 | Như lượt nền nhưng không gắn hook nào; chỉ để chẩn đoán: so từng câu với lượt nền có hook (Ân duyệt Q-B ngày 28/10/2026) |
| Mức mở rộng | 160 | α = 0,05 và 1,6, sinh sẵn và để riêng; chỉ chấm khi luật mở rộng lưới của Nhap 3.3 kích hoạt |

Tổng cộng 19 520 câu trả lời mỗi mô hình. Phần dùng cho bước 5, 6 (nền và lưới chính) là 12 960 câu mỗi mô hình,
25 920 câu cho cả hai, khớp con số khoảng 25 900 của Nhap.

## Thời gian ước (chưa đo)

Mỗi đơn vị gồm khoảng 256 bước sinh ở lô 40, tức 10–15 giây.

| Mô hình | Thời gian ước |
|---|---|
| Qwen3.5-9B | khoảng 1 giờ 30 đến 2 giờ |
| Gemma 4 12B | khoảng 2 đến 3 giờ |
| Cả hai | khoảng 3 giờ 30 đến 5 giờ, chưa tính phần dựng máy |

Con số sát hơn: `uoc_gio_gpu_buoc4` trong `toc_do.json` của bước 4e (đúng số câu của kế hoạch, tốc độ đo ở lô 40).
Đo ngày 28/10/2026 trên A100 80 GB: Qwen khoảng 1,5–1,8 giờ; Gemma khoảng 2,5 giờ bằng `sdpa`. Gemma giờ chạy bằng
`eager`, nên dùng số của lượt 4e mới.

**Cập nhật 29/10/2026 (Ân duyệt Q-E1 đến Q-E6).** Lượt thuê tiếp theo làm theo `RUNBOOK_LUOT_THUE_TIEP.md`. Thay đổi
ở bước 4:
- cổng mã 4 so thêm kiểu attention: mô hình đang nạp phải đúng đặc tả (Gemma `eager`, Qwen `sdpa`) và trùng lượt kiểm
  hook bước 3. Lượt kiểm hook thiếu trường này chỉ được nhận theo ngoại lệ hẹp cho lượt Qwen ngày 28/10
  (`NGOAI_LE_ATTN_CU` trong `scripts/buoc4_hieu_chinh.py`); log ghi `CHÚ Ý … suy ra`;
- cổng mã 8 so **họ** GPU (mọi A100 là một họ) và kiểu attention, thay cho tên GPU đầy đủ;
- bước 4 chỉ chạy bf16, nên có thể dùng A100 40 GB, nhưng chỉ khi bộ nhớ đỉnh ở lô 40 trong `toc_do.json` mới còn dư
  rõ. Lô 40 của Gemma đo 30,8 GiB reserved với `sdpa`, trên lô benchmark ngắn hơn lô bước 4 (bước 4 đệm tới 81 token);
  chưa đo với `eager`. Không chắc thì thuê 80 GB: hết bộ nhớ giữa lượt thì phải sang thư mục mới (mã 6).

## 0. Điều kiện trước khi chạy

1. **Bước 3 đã xong cho mô hình cần chạy:**
   - `buoc3_dung_huong` có mã 0, tức `huong.pt` ở trạng thái `dat`;
   - `buoc3_kiem_hook` có mã 0, tức `ket_luan.json` ở trạng thái `dat`.

   Nếu một trong hai là mã 4 ("cần xem"), script dừng mã 4. Chỉ đi tiếp được khi Ân, sau khi hai reviewer xem, quyết
   đi tiếp (Nhap, mục Kiểm pipeline): Ân viết một file quyết định (ví dụ `results/buoc3/<mô hình>/quyet_dinh_can_xem.md`,
   ghi quyết định và câu hạn chế), chép **riêng file đó** lên máy (không đẩy cả `results/`, vì sẽ đè các file máy thuê
   đang ghi), rồi chạy `b4 <mô hình> --quyet_dinh_can_xem <file>`:

   ```bash
   scp /mnt/d/thesis/thi_nghiem_2/results/buoc3/<M>/quyet_dinh_can_xem.md may-thue:/workspace/thi_nghiem_2/results/buoc3/<M>/
   ```

   Dùng đường dẫn tương đối (`results/buoc3/<M>/quyet_dinh_can_xem.md`). File được chép vào thư mục lượt và SHA của nó
   nằm trong dấu vân tay. Khi lượt đã có đơn vị, các lần chạy tiếp `b4` tự đọc lại đường dẫn. **Trước khi lượt có đơn
   vị đầu tiên** (ví dụ lần đầu dừng ở bước kiểm), mỗi lần gọi lại `b4` phải ghi lại `--quyet_dinh_can_xem <file>`;
   quên thì script chỉ dừng mã 4, không hỏng gì.
2. **Code đã commit, có cả code bước 4.** Đây là `git status --porcelain` rỗng, như ở bước 3.
3. **Kết quả bước 3 đã kéo về và kiểm SHA** (`RUNBOOK_BUOC_3.md`, mục 5), và vòng `rsync` định kỳ (`RUNBOOK_BUOC_3.md`,
   mục 4) đang chạy. Bước 4 gắn với SHA của `huong.pt`: mất kết quả bước 3 là không chạy tiếp được bước 4.
4. **Bộ nhớ GPU:** xem `toc_do.json` của bước 4e: lô 40 phải chạy được (không `het_bo_nho`).
5. **Khoá file:** dòng `flock ok` ở mục 3 của `RUNBOOK_BUOC_3.md` (máy mới thì chạy lại lệnh đó). Không khoá được
   thì script dừng mã 9 trước khi nạp mô hình.
6. Script tự kiểm file chia tập (`data/chia_tap/chia_tap.json`) trước khi làm gì khác: tính lại từ dữ liệu hiện có,
   khác hoặc không đạt thì dừng mã 4. Có thể xem kế hoạch trên máy cá nhân, không cần GPU:

   ```bash
   python scripts/buoc4_hieu_chinh.py --mo_hinh_dac_ta qwen35_9b --chi_ke_hoach
   ```

## 1. Máy

**Cách 1 (nên dùng): chạy ngay trên máy của bước 3**, sau khi `b3` của cả hai mô hình đã xong và đã làm mục 0. Mô hình
đã tải, kernel đã cài, nên không phải dựng lại. Nếu code bước 4 được commit sau khi máy đã nhận code bước 3, chép lại
code lên máy:

```bash
rsync -rlt --exclude results --exclude __pycache__ /mnt/d/thesis/thi_nghiem_2/ may-thue:/workspace/thi_nghiem_2/
```

Chỉ chép code **trước khi bắt đầu** lượt. Khi lượt còn dở (chưa có `HOAN_TAT.json`) thì không chép code mới lên máy:
dấu vân tay mã nguồn tính trên mọi file `.py` trong `src/` và `scripts/`, nên code đổi thì script dừng với mã 6.

**Cách 2: máy mới** (bước 4 trên máy khác bước 3, hoặc đổi máy giữa chừng bước 4):

1. **Huỷ hẳn máy cũ trước** (Vast.ai: Destroy, không chỉ Stop). Một máy chỉ mất kết nối có thể sống lại với tmux vẫn
   chạy; khoá không có tác dụng giữa hai máy, nên hai máy sẽ cùng sinh một lượt. Nếu không huỷ được (máy không trả
   lời), coi như nó có thể sống lại: **không bao giờ** kéo kết quả của nó về chung thư mục `results/`; nếu cần thì kéo
   vào một thư mục riêng.
2. Sửa mục `may-thue` trong `~/.ssh/config` sang máy mới (`RUNBOOK_BUOC_3.md`, mục 0.5).
3. Chép code **đúng commit** của lượt và **toàn bộ** `results/` lên máy mới, **trước** khi dựng môi trường (bước dựng
   môi trường cần `results/buoc3` để đọc phiên bản gói và commit mô hình). Commit của lượt là trường `commit_kho` trong
   `phien.jsonl` (lượt bước 4 chưa có thì là commit vừa dùng cho bước 3). Nếu HEAD của kho trên máy cá nhân đã khác
   commit đó, chép từ một bản sạch của commit đó (`git worktree add /mnt/d/thesis/tn2_<commit> <commit>`, Ân tự chạy
   git) và đặt `SRC` là thư mục đó. `--delete` bỏ các file code không thuộc commit đó (không đụng `results/`):

   ```bash
   SRC=/mnt/d/thesis/thi_nghiem_2        # hoặc /mnt/d/thesis/tn2_<commit>
   ssh may-thue 'apt-get update -qq && apt-get install -y -qq rsync tmux python3-venv >/dev/null && echo "cài xong"'
   rsync -rlt --delete --exclude results --exclude __pycache__ $SRC/ may-thue:/workspace/thi_nghiem_2/
   rsync -rlt /mnt/d/thesis/thi_nghiem_2/results/ may-thue:/workspace/thi_nghiem_2/results/; echo "rsync: $?"
   ```

   (Chép từ một git worktree thì `commit_kho` của các phiên trên máy mới ghi rỗng; dấu vân tay mã nguồn vẫn được ghi.)
4. Thuê GPU **họ A100** như phiên đầu của lượt (`ho_gpu` trong `moi_truong_goc.json` phải trùng, nếu không script dừng
   mã 8; SXM4 hay PCIe, 40 hay 80 GB đều được). Dựng môi trường và tải mô hình theo mục 2 và mục 3 của `RUNBOOK_BUOC_3.md`, nhánh **"máy thứ hai trở
   đi"**: hàm `may_moi <mô hình>` (thêm `--thu_muc <thư mục>` nếu lượt không ở thư mục mặc định) in đúng lệnh cài từng gói (theo phiên bản ghi ở lượt kiểm hook bước 3 mà `b4` sẽ dùng)
   và đúng dòng `hf download … --revision <commit>`; `HF_HUB_OFFLINE=1` chỉ bật sau khi tải xong. Backend kernel và
   phiên bản gói khác lượt kiểm hook thì script dừng mã 4 và in khác ở
   đâu; khi đó, nếu không dựng lại được đúng môi trường, chạy lại kiểm hook trên máy mới (`b3 <mô hình> tu_4d`; nó
   chạy cả 4e). Lượt bước 4 mới: `b4` tự lấy kiểm hook mới nhất. Lượt bước 4 đã có đơn vị: truyền rõ
   `b4 <mô hình> --kiem_hook <thư mục kiểm hook mới>` (thư mục được ghi vào `phien.jsonl`); vì môi trường đã khác
   phiên đầu, script sẽ dừng mã 8: hỏi Ân trước khi dùng `--cho_phep_doi_moi_truong`.
5. Chạy `kiem_nap` cho từng mô hình trên máy mới (5–10 phút mỗi mô hình; phép quét attention chạy trên đúng GPU sẽ
   dùng). Phải mã 0 và dòng `Quét attention:` ghi `→ dat` cho cả hai mô hình; khác thì dừng, báo Ân (luật đi tiếp ở
   `RUNBOOK_LUOT_THUE_TIEP.md` chỉ dành cho lần thuê làm lại bước 3, không áp khi đổi máy giữa bước 4):

   ```bash
   b3 qwen35_9b kiem_nap
   ```

   ```bash
   b3 gemma4_12b kiem_nap
   ```
6. Bật lại vòng `rsync` định kỳ (`RUNBOOK_BUOC_3.md`, mục 4) với máy mới, rồi chạy `b4` như mục 2.

Mỗi cửa sổ tmux mới, kể cả sau khi máy khởi động lại: `. /workspace/env_b3.sh` (nạp môi trường, `HF_HUB_OFFLINE=1`, và
các hàm `b3`, `b4`, `gop4`, `may_moi` từ `scripts/ham_b.sh`). Script tự nạp đúng commit Hugging Face ghi trong `huong.pt` và dừng
mã 4 nếu không xác minh được commit của mô hình nạp vào.

## 2. Chạy

`b4` không chạy nếu GPU đang bị chiếm (`nvidia-smi` báo đã dùng từ 2 000 MiB trở lên), ví dụ `b3` hoặc `b4` của mô hình kia ở cửa sổ tmux khác.
Nếu `nvidia-smi` không có tiến trình nào của mình mà bộ nhớ đã dùng vẫn từ 2 000 MiB (máy giữ sẵn), hỏi Ân; được đồng
ý thì gọi `KHONG_KIEM_GPU=1 b4 <mô hình>` (log ghi `CHÚ Ý`). Khoá của lượt vẫn chặn hai tiến trình cùng sinh một lượt.

```bash
. /workspace/env_b3.sh
```
```bash
b4 qwen35_9b; b4 gemma4_12b
```

Lượt mới: `b4` dùng lượt dựng hướng và lượt kiểm hook **đã xong** mới nhất của bước 3 (có `huong.pt`, có
`ket_luan.json`; lượt dở bị bỏ qua), và kiểm hook phải là lượt chạy với đúng file hướng đó. Lượt đã có đơn vị: `b4` đọc
lại đúng `--huong`, `--quyet_dinh_can_xem`, `--co_lo` (nếu khác 40) và `--khong_mo_rong` (nếu lượt dùng) từ
`ke_hoach.json`, và `--kiem_hook` từ phiên cuối trong `phien.jsonl`, nên "chạy lại `b4`" luôn là cùng một lệnh (với
`--thu_muc` nếu lượt không ở thư mục mặc định). Tham số thêm viết dạng `--thu_muc <thư mục>` (có dấu cách, không dùng `=`).

**Theo dõi:** `tail -f results/nhat_ky/log_b4_$(hostname).txt`. Mỗi đơn vị in một dòng, gồm thời gian, số câu bị cắt và
số phút ước còn lại. Riêng các đơn vị mức mở rộng không ghi nhật ký từng đơn vị (số câu bị cắt, thời gian, kể cả dấu
giờ, đều lộ độ dài câu trả lời; Nhap 3.3 yêu cầu để riêng, không đọc trước). Xem tiến độ phần này bằng
`ls results/buoc4/<M>/hieu_chinh/don_vi/*.xong.json | wc -l` (xong hết: 488). Sau lượt nền không hook, log in số câu giống hệt lượt nền có hook; có câu khác thì in `CHÚ Ý` (lượt
vẫn chạy tiếp; báo Ân).

**Bị ngắt giữa chừng** (máy lỗi, mất kết nối, tmux bị đóng, hết giờ thuê): trước hết `tmux ls` và `nvidia-smi`; tiến
trình cũ còn chạy thì cứ để nó chạy (nối lại bằng `tmux attach`). Đã chết thì chạy lại đúng `b4 <mô hình>`. Script làm
theo các nguyên tắc sau:
- Giữ nguyên các đơn vị đã xong: mỗi đơn vị có file `.jsonl` và `.xong.json`, kiểm bằng SHA.
- Chỉ sinh phần còn thiếu.
- Đơn vị viết dở hoặc hỏng thì chuyển sang `hong/` rồi sinh lại, không xoá file nào.
- Nếu thư mục đã có một lượt với cấu hình khác thì dừng (mã 6), không ghi đè. Cấu hình ở đây gồm file hướng, commit mô
  hình, cách chia tập, dữ liệu, tham số, file quyết định (nếu có) và mã nguồn.
- Nếu phiên này khác phiên đầu về môi trường số học thì dừng (mã 8). Môi trường số học gồm họ GPU, torch, CUDA,
  transformers, tokenizers, gói kernel, backend lớp DeltaNet, kiểu attention và kiểu số. Lý do: lượt nền và lượt can thiệp được so theo
  cặp trên từng câu, nên phải sinh trong cùng môi trường. Chỉ dùng `--cho_phep_doi_moi_truong` khi Ân đồng ý; việc này
  được ghi lại.
- Chỉ một tiến trình được chạy một lượt, có khoá. Tiến trình cũ còn chạy thì lệnh mới dừng mã 9.
- Phần nền và lưới chính được gộp ngay khi xong (`cau_tra_loi.jsonl`, `HOAN_TAT_CHINH.json`). Vì vậy, nếu hết thời gian
  trong lúc sinh lượt nền không hook hay mức mở rộng thì phần dùng cho bước 5 và bước 6 đã đầy đủ.
- Mọi đơn vị đã xong nhưng chưa gộp (máy dừng đúng lúc gộp): chạy lại thì script gộp ngay, không nạp mô hình.
- Lượt đã hoàn tất mà chạy lại thì script kiểm SHA của mọi file rồi không làm gì (mã 0). Nếu `HOAN_TAT.json`,
  `sha256.txt` hỏng hoặc có file đổi, thêm, mất so với `sha256.txt`, script ghi `CHÚ Ý` liệt kê từng file, gộp lại từ
  các đơn vị và chuyển bản cũ sang `hong/` (không nạp mô hình). Gặp `CHÚ Ý` này thì báo Ân.

**Gộp trên máy cá nhân** (máy thuê đã mất sau khi sinh xong mọi đơn vị nhưng trước khi gộp): sau khi đã kéo `results/`
về, trong một môi trường Python có `requirements.txt` (torch, transformers; bật venv trước bằng
`source .venv/bin/activate`, hoặc đặt `PYTHON=<đường dẫn python của venv>`), dùng `gop4`. Nó chỉ gộp, **không bao giờ nạp
mô hình**; thiếu đơn vị thì báo thiếu đơn vị nào và dừng mã 1 (khi đó phải sinh tiếp trên một máy GPU, cách 2). Mã nguồn
khác lúc sinh (ví dụ đã có thêm code bước 5) thì vẫn gộp, có ghi lại. In `CHÚ Ý: hệ file không hỗ trợ khoá` là bình
thường trên `/mnt/d` của WSL; chỉ cần không chạy hai lệnh cùng lúc.

```bash
cd /mnt/d/thesis/thi_nghiem_2 && . scripts/ham_b.sh
gop4 qwen35_9b; gop4 gemma4_12b
```

**Hết bộ nhớ GPU** (mã 1, log có `CUDA out of memory`): dừng, hỏi Ân. Nếu Ân đồng ý giảm lô:
- chưa có đơn vị nào xong (thường gặp ngay đơn vị đầu): chạy `b4 <mô hình> --co_lo 20` trong cùng thư mục; ghi
  `--co_lo 20` ở mọi lần gọi cho đến khi có đơn vị đầu tiên, sau đó `b4 <mô hình>` tự đọc lại cỡ lô từ lượt;
- đã có đơn vị xong: script dừng mã 6 (cỡ lô nằm trong dấu vân tay, để mọi đơn vị của một lượt cùng cách chia lô),
  nên phải dùng thư mục mới: `b4 <mô hình> --co_lo 20 --thu_muc results/buoc4/<mô hình>/hieu_chinh_lo20` (chạy lại:
  `b4 <mô hình> --thu_muc results/buoc4/<mô hình>/hieu_chinh_lo20`, cỡ lô tự đọc lại; gộp: `gop4` cùng `--thu_muc`). Khi đó ở
  mục 3 và mục 4, thay `hieu_chinh` bằng `hieu_chinh_lo20` cho mô hình đó, và ghi lại việc đổi cỡ lô.

**Phòng mất máy hẳn:** vòng `rsync` định kỳ của `RUNBOOK_BUOC_3.md`, mục 4 (5 phút một lần, kéo toàn bộ `results/`,
kể cả log). Mất máy hẳn thì chỉ mất phần sinh sau lần `rsync` thành công cuối cùng; sinh tiếp trên máy mới theo cách 2.

**Mã thoát:**

| Mã | Nghĩa | Làm gì |
|---|---|---|
| 0 | Hoàn tất (hoặc đã hoàn tất từ trước) | Sang mục 3 |
| 1 | Lỗi (kể cả transformers không nạp đúng kiểu attention của đặc tả: `ValueError` khi nạp); hoặc `--chi_gop` mà còn đơn vị chưa xong / chưa có lượt | Kéo `results/` về (log có traceback), dừng |
| 2 | Khối sau câu hỏi lệch, hoặc cấu hình sinh không phải greedy | Dừng, không sửa code trên máy thuê |
| 3 | Mô hình không nằm trọn một GPU | Dừng, kiểm lại máy |
| 4 | Đầu vào không dùng được: file chia tập khác hoặc không đạt; file hướng / kiểm hook của bước 3 không đạt, hoặc "cần xem" mà chưa có file quyết định; mô hình nạp không xác minh được commit hoặc khác commit của file hướng; backend kernel, phiên bản gói khác lượt kiểm hook bước 3; kiểu attention khác lượt kiểm hook, hoặc lượt kiểm hook không ghi kiểu attention (ngoài ngoại lệ Qwen 28/10); file hướng không dựng bằng kiểu attention của đặc tả (thiếu trường chỉ nhận hướng Qwen 28/10) | Đọc thông báo DỪNG; trường hợp môi trường khác: mục 1, cách 2; thiếu kiểu attention: chạy lại kiểm hook bằng code hiện tại |
| 6 | Thư mục đã có một lượt khác | Không ghi đè. Đọc thông báo; đừng dùng `--cho_phep_doi_ma_nguon` nếu chưa hỏi Ân |
| 7 | Dừng theo `--so_don_vi_toi_da` | Chạy lại để tiếp |
| 8 | Môi trường số học khác phiên đầu (kể cả họ GPU, kiểu attention) | Dừng. Hỏi Ân; xem `moi_truong_goc.json` và dòng cuối `phien.jsonl` |
| 9 | Một tiến trình khác đang chạy cùng lượt; hoặc còn đơn vị phải sinh mà hệ file không hỗ trợ khoá | Kiểm `tmux ls` và `nvidia-smi`, không chạy song song; trường hợp sau: chạy trên ổ cục bộ |

## 3. Kéo kết quả về, kiểm, lưu trữ, rồi huỷ máy

Đọc lại **cả hai** dòng `=== … MÃ THOÁT` trong `results/nhat_ky/log_b4_<máy>.txt`: với `b4 qwen35_9b; b4 gemma4_12b`,
Gemma vẫn chạy khi Qwen dừng, nên mã thoát của Qwen nằm xa phía trên. Rồi trên máy cá nhân:

```bash
cd /mnt/d/thesis/thi_nghiem_2
rsync -rlt may-thue:/workspace/thi_nghiem_2/results/ results/; echo "rsync: $?"
for M in qwen35_9b gemma4_12b; do
  (cd results/buoc4/$M/hieu_chinh && sha256sum -c --quiet sha256.txt && echo "$M SHA ĐỦ")
done
D=/mnt/d/thesis/luu_tru/results_$(date +%Y%m%d_%H%M%S); mkdir -p $D && rsync -rlt results/ $D/ && echo "Lưu trữ: $D"
```

Phải thấy `rsync: 0`, và mỗi mô hình in một dòng `SHA ĐỦ`. Sau đó mới huỷ máy. Mỗi lần lưu trữ là một thư mục mới có
ngày giờ, nên không ghi đè bản lưu trước.

## 4. Đọc kết quả (`results/buoc4/<mô hình>/hieu_chinh/`)

| File | Nội dung |
|---|---|
| `HOAN_TAT.json` | Số đơn vị, số câu theo từng phần, SHA |
| `HOAN_TAT_CHINH.json` | Phần nền và lưới chính đã gộp xong, kèm SHA |
| `cau_tra_loi.jsonl` | Lượt nền và lưới chính, 12 960 câu. Đây là đầu vào của bước 5 và bước 6 |
| `tom_tat.json` | Riêng phần chính, theo đơn vị và theo nhóm: số câu bị cắt ở 256 token, số câu rò thinking (đếm thô), phân bố token kết thúc |
| `cau_tra_loi_nen_khong_hook.jsonl`, `tom_tat_nen_khong_hook.json`, `HOAN_TAT_NEN_KHONG_HOOK.json` | Lượt nền không hook, 160 câu, chỉ để chẩn đoán. `tom_tat_nen_khong_hook.json` → `so_khop_voi_nen_co_hook`: số câu có dãy token giống hệt lượt nền có hook; `tat_ca_giong_het` phải là `true`. Không dùng cho bước 5, 6 |
| `cau_tra_loi_mo_rong.jsonl`, `tom_tat_mo_rong.json`, `HOAN_TAT_MO_RONG.json` | Mức 0,05 và 1,6, 6 400 câu. **Không mở**, không chấm trước khi luật mở rộng lưới kích hoạt (Nhap 3.3) |
| `ke_hoach.json`, `moi_truong_goc.json`, `phien.jsonl`, `nhat_ky.txt` | Kế hoạch, dấu vân tay, môi trường phiên đầu, môi trường từng phiên (kể cả kết quả đối chiếu với lượt kiểm hook bước 3), nhật ký |
| `quyet_dinh_can_xem.*` | Chỉ có khi đi tiếp với trạng thái "cần xem": bản chép file quyết định của Ân |
| `hong/` | File của đơn vị viết dở hoặc hỏng đã được thay. Bình thường thư mục này rỗng hoặc không có |

Mỗi dòng trong `cau_tra_loi*.jsonl` có:
- `ma_don_vi`, `id`, `mo_hinh`;
- `loai` (`nen`, `can_thiep`, `nen_khong_hook`), `nhom`, `tap`, `phan`, `kieu`;
- `ngon_ngu_cau_hoi`, `huong`, `chieu`, `lop`, `alpha`, `phuong_an`, `lop_hook`, `phuong_an_hook`;
- `cau_hoi`, `van_ban` (giải mã tới token kết thúc, không gồm chính token kết thúc, giữ các token đặc biệt khác),
  `token`;
- `so_token`, `cham_eos`, `bi_cat`, `token_ket_thuc`;
- `nhan_doi_chieu`: làm nặng hoặc làm mềm với D_harm_test, rỗng với D_plain_test.

Quy ước cho phần đọc kết quả về sau:
- Khoá của một câu trả lời là bộ ba (`mo_hinh`, `ma_don_vi`, `id`), vì `ma_don_vi` không chứa tên mô hình.
- Lượt nền chọn theo `loai == "nen"`. Dòng của lượt nền ghi `lop`, `phuong_an`, `huong` là rỗng (Ân duyệt Q-C ngày
  28/10/2026), nên phép nhóm theo lớp không lẫn dòng nền vào một ô; lớp và phương án của hook α = 0 nằm ở `lop_hook`,
  `phuong_an_hook`.
- `token_ket_thuc` cho biết câu trả lời dừng ở token nào. Gemma 4 có ba token kết thúc (1, 106, 50); id 50 không nằm
  trong `van_ban`, nên phép kiểm vỡ tự động ở bước 5 (Nhap 4.4: "tự sinh token điều khiển") phải đọc `token_ket_thuc`.

Bước 5 cũng sẽ tự đánh hai cờ là ngôn ngữ và rò thinking (Nhap 4.1), nên những cờ này không cần sinh ở đây.
