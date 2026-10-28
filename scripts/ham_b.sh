# Hàm chạy bước 3 (b3) và bước 4 (b4, gop4) — xem docs/RUNBOOK_BUOC_3.md và docs/RUNBOOK_BUOC_4.md.
# Nạp bằng `source scripts/ham_b.sh` trong thư mục kho (trên máy thuê: `source /workspace/env_b3.sh` đã làm việc này).
# Để trong kho (không dán tay) nên mỗi máy, mỗi cửa sổ tmux đều dùng đúng một bản.
#
# Log của từng lần chạy: results/nhat_ky/log_b3_<máy>.txt, log_b4_<máy>.txt. Nằm trong results/ để vòng rsync định kỳ
# kéo về; tên theo máy để các lần thuê khác nhau không ghi đè lên nhau.

_nk() { mkdir -p results/nhat_ky; echo "results/nhat_ky/$1_$(hostname).txt"; }

# Trình thông dịch Python cho các script: mặc định `python` của venv đang bật (source env_b3.sh trên máy thuê;
# trên máy cá nhân/WSL phải `source .venv/bin/activate` trước). Đặt PYTHON=... để chỉ định khác.
_py() { "${PYTHON:-python}" "$@"; }

# GPU rảnh: bộ nhớ đã dùng < 2000 MiB (vd. b3 / b4 của mô hình kia ở cửa sổ tmux khác giữ hàng chục GB). Không dựa
# vào danh sách tiến trình: trong container (Vast.ai) nvidia-smi thường không thấy tiến trình nào.
# Bỏ qua phép kiểm này: KHONG_KIEM_GPU=1 b3 ... / b4 ... — CHỈ khi đã xem nvidia-smi, chắc không có tiến trình nào
# của mình đang chạy (vd. driver giữ sẵn ≥ 2000 MiB), và Ân đồng ý.
_gpu_ranh() {
  [ "${KHONG_KIEM_GPU:-0}" = 1 ] && { echo "CHÚ Ý: KHONG_KIEM_GPU=1, bỏ qua kiểm GPU rảnh." >&2; return 0; }
  command -v nvidia-smi >/dev/null || return 0
  local m
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  case "$m" in ''|*[!0-9]*) return 0 ;; esac
  [ "$m" -lt 2000 ]
}

# Lượt dựng hướng chính thức mới nhất mà ĐÃ XONG (có huong.pt: huong.pt được ghi cuối cùng, nguyên tử). Sắp theo
# tên (dung_huong_<ngày>_<giờ>), không theo mtime.
_huong_moi_nhat() {
  local d
  for d in $(ls -d results/buoc3/$1/dung_huong_2*/ 2>/dev/null | sort -r); do
    [ -f "${d}huong.pt" ] && { echo "${d}huong.pt"; return 0; }
  done
  return 1
}

# Lượt kiểm hook chính thức mới nhất mà ĐÃ XONG (có ket_luan.json, file ghi cuối của buoc3_kiem_hook) và chạy với
# ĐÚNG file hướng $2 (meta.json: tham_so.huong_sha256 = SHA của $2).
_kiem_hook_moi_nhat() {
  local d sha
  sha=$(sha256sum "$2" | cut -d' ' -f1)
  for d in $(ls -d results/buoc3/$1/kiem_hook_2*/ 2>/dev/null | sort -r); do
    [ -f "${d}ket_luan.json" ] || continue
    python3 -c "import json,sys;m=json.load(open(sys.argv[1]));sys.exit(0 if (m.get('tham_so') or {}).get('huong_sha256')==sys.argv[2] else 1)" "${d}meta.json" "$sha" 2>/dev/null \
      && { echo "${d%/}"; return 0; }
  done
  return 1
}

# b3 <mô hình> [tu_4a | tu_4d | tu_4e]
#   tu_4a (mặc định): chạy cả 4a → 4e.
#   tu_4d: máy chết trong 4d (kiểm hook): dùng lượt dựng hướng đã xong mới nhất, chạy 4d rồi 4e.
#   tu_4e: máy chết trong 4e (đo tốc độ): chỉ chạy 4e.
#   Máy chết trong 4a, 4b, 4c: chạy lại `b3 <mô hình>` (thư mục dở của 4c không có huong.pt nên bị bỏ qua).
b3() {
  local M=$1 TU=${2:-tu_4a} L rc H r4c=- r4d=-
  case "$TU" in tu_4a|tu_4d|tu_4e) ;; *) echo "Cách gọi: b3 <mô hình> [tu_4a|tu_4d|tu_4e]"; return 1 ;; esac
  L=$(_nk log_b3)
  _gpu_ranh || { echo "DỪNG: GPU đang bị chiếm (nvidia-smi: đã dùng ≥ 2000 MiB). Không chạy song song." | tee -a $L; return 1; }
  _b() { local ten=$1; shift; echo "=== $M $ten: $*" | tee -a $L; _py "$@" 2>&1 | tee -a $L; rc=${PIPESTATUS[0]}; echo "=== $M $ten MÃ THOÁT: $rc" | tee -a $L; return $rc; }
  echo "=== $M b3 bắt đầu ($TU) trên $(hostname), $(date '+%F %T')" | tee -a $L
  if [ "$TU" = tu_4a ]; then
    # 4a. Kiểm trước (vài giây, không nạp trọng số): khối 9 token, vị trí lấy −5, token kết thúc, layer_types, greedy.
    _b 4a scripts/buoc3_kiem_truoc.py --mo_hinh_dac_ta $M || return
    # 4b. Nạp và kiểm (⏱ 3–6 phút): một GPU, cấu trúc, bộ đếm vị trí, sinh thử 32 câu, rò thinking, token kết thúc.
    _b 4b scripts/buoc3_nap_kiem.py --mo_hinh_dac_ta $M || return
    # 4c. Dựng hướng và s_ℓ (⏱ 3–8 phút): chẩn đoán position_ids / đệm, 512 câu dựng, mọi lớp.
    _b 4c scripts/buoc3_dung_huong.py --mo_hinh_dac_ta $M; r4c=$rc; [ $rc -eq 0 ] || [ $rc -eq 4 ] || return $rc
  fi
  H=$(_huong_moi_nhat $M) || { echo "Không thấy lượt dựng hướng đã xong (có huong.pt) của $M" | tee -a $L; return 1; }
  echo "Hướng: $H" | tee -a $L
  if [ "$TU" != tu_4e ]; then
    # 4d. Bản thu nhỏ phần 2, bf16 rồi float32 (⏱ ước 40–120 phút; Qwen có thể lâu hơn vì float32 dùng bản tham chiếu).
    _b 4d scripts/buoc3_kiem_hook.py --mo_hinh_dac_ta $M --huong $H; r4d=$rc; [ $rc -eq 0 ] || [ $rc -eq 4 ] || return $rc
  fi
  # 4e. Đo tốc độ và bộ nhớ (⏱ 10–30 phút): lô 16, 32, 40, 64, 128; nền và can thiệp α = 0,8; ước giờ cho bước 4.
  _b 4e scripts/buoc3_do_toc_do.py --mo_hinh_dac_ta $M --huong $H || return
  echo "=== $M: xong bước 3 ($TU); mã 4c = $r4c, mã 4d = $r4d (0 đạt, 4 cần xem: KHÔNG phải đạt; - là không chạy lần này)" | tee -a $L
}

# Đọc một trường của lượt bước 4 đã có: huong, quyet_dinh_can_xem từ ke_hoach.json; kiem_hook từ dòng cuối của
# phien.jsonl có trường này (có thể đã đổi bằng --kiem_hook ở một phiên sau), không có thì từ ke_hoach.json;
# co_lo, khong_mo_rong từ ke_hoach.json/tham_so (in tham số dòng lệnh cần thêm, rỗng nếu là mặc định).
_ke_hoach() {
  python3 - "$1" "$2" <<'PY'
import json, os, sys
tm, k = sys.argv[1], sys.argv[2]
v = None
if k == "kiem_hook" and os.path.exists(os.path.join(tm, "phien.jsonl")):
    for dong in open(os.path.join(tm, "phien.jsonl"), encoding="utf-8", errors="replace"):
        try:
            x = json.loads(dong)
        except ValueError:
            continue
        if isinstance(x, dict) and x.get("kiem_hook"):
            v = x["kiem_hook"]
if v is None:
    d = json.load(open(os.path.join(tm, "ke_hoach.json"), encoding="utf-8"))
    ts = d.get("tham_so") or {}
    if k == "co_lo":
        v = "" if ts.get("co_lo", 40) == 40 else f"--co_lo {int(ts['co_lo'])}"
    elif k == "khong_mo_rong":
        v = "--khong_mo_rong" if ts.get("alpha_mo_rong", None) == [] else ""
    else:
        v = d.get(k)
        if isinstance(v, dict):
            v = v.get("file_goc")
print("" if v is None else v)
PY
}

# Thư mục lượt bước 4 (mặc định hoặc theo --thu_muc trong các tham số thêm).
_thu_muc_b4() {
  local M=$1; shift; local TM=results/buoc4/$M/hieu_chinh
  while [ $# -gt 0 ]; do [ "$1" = --thu_muc ] && TM=$2; shift; done
  echo "$TM"
}

# Các tham số --huong, --kiem_hook, --quyet_dinh_can_xem (và --co_lo, --khong_mo_rong nếu lượt dùng) cho một lượt
# bước 4. Lượt ĐÃ CÓ ĐƠN VỊ: đọc lại từ lượt (đúng lệnh lần trước; đối chiếu lần cuối, R2 F3: lượt chạy --co_lo 20 thì
# lần sau tự thêm --co_lo 20, không dừng mã 6). Lượt chưa có đơn vị (mới, hoặc lần trước dừng ở bước kiểm): lấy lượt
# bước 3 đã xong mới nhất, kiểm hook phải chạy với đúng file hướng đó; file quyết định (nếu cần) phải truyền rõ ở MỌI
# lần gọi cho đến khi lượt có đơn vị đầu tiên.
_tham_so_b4() {
  local M=$1 TM=$2 H K Q CL KM
  if ls "$TM"/don_vi/*.xong.json >/dev/null 2>&1; then
    H=$(_ke_hoach "$TM" huong); K=$(_ke_hoach "$TM" kiem_hook); Q=$(_ke_hoach "$TM" quyet_dinh_can_xem)
    CL=$(_ke_hoach "$TM" co_lo); KM=$(_ke_hoach "$TM" khong_mo_rong)
    if [ -n "$Q" ] && [ ! -f "$Q" ]; then          # vd. gộp trên máy cá nhân: dùng bản chép trong thư mục lượt
      Q=$(ls "$TM"/quyet_dinh_can_xem.* 2>/dev/null | head -1)
    fi
  else
    H=$(_huong_moi_nhat $M) || return 1
    K=$(_kiem_hook_moi_nhat $M "$H") || return 1
  fi
  [ -n "$H" ] && [ -n "$K" ] || return 1
  echo "--huong $H --kiem_hook $K${Q:+ --quyet_dinh_can_xem $Q}${CL:+ $CL}${KM:+ $KM}"
}

# b4 <mô hình> [tham số thêm, vd. --quyet_dinh_can_xem <file> | --co_lo 20 --thu_muc <thư mục>]
b4() {
  local M=$1 L rc TM TS
  L=$(_nk log_b4)
  _gpu_ranh || { echo "DỪNG: GPU đang bị chiếm (nvidia-smi: đã dùng ≥ 2000 MiB). Không chạy song song." | tee -a $L; return 1; }
  TM=$(_thu_muc_b4 "$@")
  TS=$(_tham_so_b4 $M "$TM") || { echo "Không thấy lượt bước 3 đã xong (huong.pt / ket_luan.json) của $M" | tee -a $L; return 1; }
  echo "=== $M b4 trên $(hostname), $(date '+%F %T'): $TS ${*:2}" | tee -a $L
  _py scripts/buoc4_hieu_chinh.py --mo_hinh_dac_ta $M $TS "${@:2}" 2>&1 | tee -a $L
  rc=${PIPESTATUS[0]}; echo "=== $M b4 MÃ THOÁT: $rc" | tee -a $L; return $rc
}

# gop4 <mô hình> [--thu_muc <thư mục>]: chỉ gộp lượt bước 4 đã sinh xong (vd. trên máy cá nhân), không nạp mô hình.
# Trên máy cá nhân: bật venv của kho trước (source .venv/bin/activate) hoặc đặt PYTHON=<đường dẫn python của venv>.
gop4() {
  local M=$1 TM TS
  TM=$(_thu_muc_b4 "$@")
  [ -f "$TM/ke_hoach.json" ] || { echo "Không có $TM/ke_hoach.json"; return 1; }
  TS=$(_tham_so_b4 $M "$TM") || return 1
  _py scripts/buoc4_hieu_chinh.py --mo_hinh_dac_ta $M $TS "${@:2}" --chi_gop
}

# may_moi <mô hình> [--thu_muc <thư mục bước 4>]: MÁY THỨ HAI TRỞ ĐI (đổi máy giữa bước 3 và bước 4, hoặc giữa chừng
# bước 4), chạy SAU khi đã chép results/ lên máy mới. In commit Hugging Face cần tải và các lệnh cài đúng phiên bản gói,
# lấy từ meta.json của ĐÚNG lượt kiểm hook bước 3 mà b4 sẽ dùng cho mô hình này (lượt bước 4 đã có đơn vị: lượt ghi
# trong lượt đó; chưa có: lượt đã xong mới nhất chạy với hướng đã xong mới nhất). Chỉ đọc file và in, không cài, không
# tải; chỉ cần python3 của hệ thống (chạy được trước khi có venv). Đối chiếu lần cuối, R2 F1, 28/10/2026.
may_moi() {
  local M=$1 TM H K ID
  case "$M" in
    qwen35_9b) ID=Qwen/Qwen3.5-9B ;;
    gemma4_12b) ID=google/gemma-4-12B-it ;;
    *) echo "Cách gọi: may_moi qwen35_9b|gemma4_12b [--thu_muc <thư mục bước 4>]"; return 1 ;;
  esac
  TM=$(_thu_muc_b4 "$@")
  if ls "$TM"/don_vi/*.xong.json >/dev/null 2>&1; then
    H=$(_ke_hoach "$TM" huong); K=$(_ke_hoach "$TM" kiem_hook)
  else
    H=$(_huong_moi_nhat $M) && K=$(_kiem_hook_moi_nhat $M "$H") \
      || { echo "Không thấy lượt bước 3 đã xong (huong.pt / ket_luan.json) của $M trong results/ (đã chép results/ lên chưa?)"; return 1; }
  fi
  [ -f "$K/meta.json" ] || { echo "Không có $K/meta.json"; return 1; }
  python3 - "$M" "$ID" "$H" "$K" <<'PY'
import json, sys
M, ID, H, K = sys.argv[1:5]
m = json.load(open(K + "/meta.json", encoding="utf-8"))
mt = m.get("moi_truong") or {}
c = (m.get("mo_hinh") or {}).get("commit_hf")
print(f"# {M}: hướng {H}")
print(f"# {M}: kiểm hook {K}")
print(f"# GPU lúc kiểm hook: {mt.get('gpu')}; CUDA của torch: {mt.get('cuda')}; USE_HUB_KERNELS: {mt.get('bien_USE_HUB_KERNELS')}")
if not c:
    print("# DỪNG: meta.json không có commit mô hình; báo Ân."); sys.exit(1)
tv = str(mt.get("torch") or "")
base, _, local = tv.partition("+")
if not base:
    print("# DỪNG: meta.json không có phiên bản torch; báo Ân."); sys.exit(1)
print("# --- các lệnh cho máy mới (chạy trong venv đã bật, theo thứ tự) ---")
if local.startswith("cu"):
    print(f"pip install torch=={base} --index-url https://download.pytorch.org/whl/{local}")
else:
    print(f"# torch {tv}: không rõ bản CUDA từ phiên bản; hỏi Ân trước khi cài torch")
print("pip install -r requirements.txt")
pin = [f"{g}=={mt[k]}" for g, k in (("transformers", "transformers"), ("tokenizers", "tokenizers")) if mt.get(k)]
if pin:
    print("pip install " + " ".join(pin))           # trùng requirements.txt thì không đổi gì
print("# (chạy pytest như mục 2 TRƯỚC khi cài hai gói dưới)")
for g, k, them in (("flash-linear-attention", "goi_flash-linear-attention", ""),
                   ("causal-conv1d", "goi_causal-conv1d", " --no-build-isolation")):
    v = mt.get(k)
    print(f"pip install {g}=={v}{them}" if v else f"# {g}: KHÔNG có ở lượt kiểm hook → KHÔNG cài")
if mt.get("goi_kernels"):
    print(f"# CHÚ Ý: lượt kiểm hook có gói kernels=={mt['goi_kernels']} (runbook không cài gói này): báo Ân")
print(f"HF_HUB_OFFLINE=0 hf download {ID} --revision {c}")
PY
}
