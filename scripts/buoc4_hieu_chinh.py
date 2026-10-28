"""Bước 4, lượt hiệu chỉnh: sinh toàn bộ câu trả lời trên tập hiệu chỉnh cho một mô hình chính (Nhap, Trình tự chạy
mục 4; 3.2, 3.3, 3.4, 3.5, 3.6; mục Dữ liệu).

Chạy (sau bước 3, với file hướng đã đạt):
    python scripts/buoc4_hieu_chinh.py --mo_hinh_dac_ta qwen35_9b --huong results/buoc3/qwen35_9b/<lượt>/huong.pt \
        --kiem_hook results/buoc3/qwen35_9b/<lượt kiem_hook>
Bị ngắt (máy lỗi, mất mạng, hết giờ): chạy lại ĐÚNG lệnh đó, script tiếp từ đơn vị chưa xong. Xem kế hoạch mà
không nạp mô hình: thêm --chi_ke_hoach.

Kế hoạch mỗi mô hình (tn2.sinh_luot.ke_hoach_hieu_chinh): 40 cặp hiệu chỉnh của D_plain_test (kiểu a) và
D_harm_test (kiểu b) (data/chia_tap/chia_tap.json), 4 chiều, 5 lớp đại diện, 2 phương án (1 và 3), lưới α 0,1 /
0,2 / 0,4 / 0,8 (320 đơn vị, 12 800 câu) + lượt nền α = 0 (4 đơn vị, 160 câu) + lượt nền KHÔNG gắn hook (4 đơn vị,
160 câu; chẩn đoán: so từng câu với lượt nền có hook, Ân duyệt Q-B ngày 28/10/2026) + hai mức mở rộng 0,05 và 1,6
sinh sẵn (160 đơn vị, 6 400 câu; chỉ đem chấm khi luật mở rộng lưới của Nhap 3.3 kích hoạt). Mỗi đơn vị = 40 câu,
một lô.

Kiểm trước khi sinh (dừng nếu trượt): file chia tập trùng kết quả tính lại từ dữ liệu hiện có và đạt mọi ràng
buộc; bản thu nhỏ phần 2 của bước 3 (--kiem_hook) đạt với đúng file hướng này ("cần xem" chỉ qua được khi có
--quyet_dinh_can_xem: file quyết định của Ân, được chép vào thư mục lượt và vào dấu vân tay);
file hướng của đúng mô hình, trạng thái "dat", lượt chính thức, dựng theo position_ids của mặt nạ; commit HF khớp
mô hình đang nạp (mặc định nạp đúng commit ghi trong file hướng); mô hình nằm trọn một GPU; khối 9 token và vị trí
lấy −5 trên mọi câu sẽ dùng; cấu hình sinh hiệu lực là greedy; backend kernel và phiên bản gói trùng lượt kiểm hook
của bước 3; môi trường số học giống phiên đầu (nếu là phiên tiếp).

Ghi ra results/buoc4/<mô hình>/hieu_chinh/ (cố định, để chạy lại là tiếp; xem tn2.sinh_luot):
  ke_hoach.json, moi_truong_goc.json, phien.jsonl, nhat_ky.txt, don_vi/<mã>.jsonl + <mã>.xong.json;
  xong nền + lưới chính: cau_tra_loi.jsonl (12 960 câu), tom_tat.json, HOAN_TAT_CHINH.json;
  xong lượt nền không hook: cau_tra_loi_nen_khong_hook.jsonl, tom_tat_nen_khong_hook.json (so khớp từng câu với
  lượt nền có hook), HOAN_TAT_NEN_KHONG_HOOK.json;
  xong mức mở rộng: cau_tra_loi_mo_rong.jsonl (6 400 câu; để riêng, chỉ đọc / chấm khi luật mở rộng lưới kích
  hoạt), tom_tat_mo_rong.json, HOAN_TAT_MO_RONG.json; xong cả hai: sha256.txt, HOAN_TAT.json.
Mọi đơn vị đã xong mà chưa gộp (ví dụ máy dừng đúng lúc gộp): chạy lại sẽ gộp mà KHÔNG nạp mô hình. Gộp trên máy cá
nhân: thêm --chi_gop (không bao giờ nạp mô hình; thiếu đơn vị thì báo và dừng mã 1).
Mã thoát: 0 hoàn tất (hoặc đã hoàn tất từ trước); 1 lỗi; 2 khối sau câu hỏi lệch hoặc cấu hình sinh không phải
greedy; 3 mô hình không nằm trọn một GPU; 4 file hướng / kiểm hook của bước 3 không dùng được; 6 thư mục đã có
một lượt khác (không ghi đè); 7 dừng giữa chừng theo --so_don_vi_toi_da (chạy lại để tiếp); 8 môi trường số học
khác phiên đầu; 9 một tiến trình khác đang chạy cùng lượt. Mã 4 cũng dùng khi mô hình nạp không xác minh được
commit hoặc khác commit của file hướng, và khi backend kernel / phiên bản gói khác lượt kiểm hook bước 3.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from _chung import GOC, NhatKy, chon_dac_ta, device_map_mot_gpu

import torch

from tn2 import nguon_goc as NG
from tn2.buoc3.nap_kiem import kiem_khoi_tokenizer
from tn2.mo_hinh import backend_lop_tuyen_tinh, eos_tu_mo_hinh, nap_mo_hinh, phan_bo_thiet_bi
from tn2.sinh_luot import (ALPHA_MO_RONG, LUOI_ALPHA_HIEU_CHINH, MAX_NEW_TOKENS, DangChay, KhoKetQua, chay_ke_hoach,
                           bam_moi_truong, da_hoan_tat, dau_van_tay_ma_nguon, doi_sau_hoan_tat, ke_hoach_hieu_chinh,
                           kiem_ke_hoach, phan_cua, ghi_nguyen_tu, sha256_chuoi, tong_hop_cuoi, tong_hop_phan)
from tn2.tap_test import FILE_CHIA, THU_MUC_TEST, doc_chia_tap, tap_con

CO_LO = 40          # một đơn vị = 40 câu = một lô (cùng cách đệm cho lượt nền và lượt can thiệp)
SO_CAU = 40
# Môi trường phải trùng với lượt kiểm hook của bước 3 (rà soát bước 4, R2 B2; Ân duyệt Q2 ngày 28/10/2026).
# Tên GPU khác thì chỉ ghi lại: ghép cặp nền–can thiệp trong bước 4 đã có cổng mã 8.
KHOA_MOI_TRUONG_BUOC3 = ("torch", "transformers", "cuda", "tokenizers", "goi_flash-linear-attention",
                         "goi_causal-conv1d", "goi_kernels", "bien_USE_HUB_KERNELS")


# Năm file dữ liệu test (tên cố định; không dùng glob để file lạ như ~$….csv của Excel không làm đổi dấu vân tay).
TEP_DU_LIEU_TEST = ("D_harm_test.jsonl", "D_harm_test_vi.jsonl", "D_plain_test.jsonl", "D_plain_test_vi.jsonl",
                    "nhan_doi_chieu_dich_chot.csv")
_DA_KIEM_CHIA_TAP: dict[tuple, list[str]] = {}


def _kiem_chia_tap() -> list[str]:
    """File chia tập phải trùng kết quả tính lại từ dữ liệu hiện có và đạt mọi ràng buộc, kiểm trùng (Nhap: kiểm
    trùng chạy trên bộ file cuối cùng ở bước lượt hiệu chỉnh). Vài giây trên CPU; nhớ kết quả theo SHA đầu vào."""
    from tn2.tap_test import chia_tap
    dau_vao = [FILE_CHIA] + [THU_MUC_TEST / n for n in TEP_DU_LIEU_TEST] + \
        sorted((GOC / "data" / "dung_huong").glob("*.jsonl"))
    try:
        khoa = tuple(NG.sha256_file(f) for f in dau_vao)
    except OSError as e:
        return [f"thiếu file dữ liệu: {e}"]
    if khoa in _DA_KIEM_CHIA_TAP:
        return _DA_KIEM_CHIA_TAP[khoa]
    loi = []
    try:
        cu = json.loads(FILE_CHIA.read_text(encoding="utf-8"))
        moi = json.loads(json.dumps(chia_tap(), ensure_ascii=False))
    except Exception as e:                                              # noqa: BLE001  (dữ liệu hỏng: dừng mã 4)
        return [f"không đọc / tính lại được chia tập: {type(e).__name__}: {e}"]
    if cu != moi:
        loi.append("data/chia_tap/chia_tap.json KHÁC kết quả tính lại từ dữ liệu hiện có (chạy "
                   "scripts/buoc4_chia_tap.py --kiem để xem)")
    if not (isinstance(cu, dict) and cu.get("kiem_tra", {}).get("dat")
            and cu.get("kiem_trung", {}).get("dat_sau_ngoai_le")):
        loi.append("chia tập không đạt ràng buộc hoặc kiểm trùng (kiem_tra.dat / kiem_trung.dat_sau_ngoai_le)")
    _DA_KIEM_CHIA_TAP[khoa] = loi
    return loi


def _cau_theo_tap(gioi_han: int | None) -> dict[str, list[dict]]:
    chia = doc_chia_tap()
    ra = {t: tap_con(t, "hieu_chinh", chia) for t in ("D_plain_test", "D_harm_test")}
    if gioi_han is not None:
        ra = {t: v[:gioi_han] for t, v in ra.items()}
    return ra


def _kiem_huong(h: dict, dac_ta, thu: bool, co_quyet_dinh: bool = False) -> list[str]:
    loi = []
    if h.get("dac_ta") != dac_ta.ten:
        loi.append(f"file hướng của {h.get('dac_ta')}, không phải {dac_ta.ten}")
    if not thu:
        tt = h.get("trang_thai")
        if tt == "can_xem" and not co_quyet_dinh:
            loi.append("trạng thái hướng là 'can_xem' (cảnh báo đệm): chỉ đi tiếp khi Ân đã xem và quyết, với "
                       "--quyet_dinh_can_xem <file quyết định>")
        elif tt not in ("dat", "can_xem"):
            loi.append(f"trạng thái hướng là {tt!r}, cần 'dat'")
        if h.get("chinh_thuc") is not True:
            loi.append("file hướng từ lượt dựng hướng thử, không chính thức")
        if h.get("position_ids_theo_mat_na") is not True:
            loi.append("file hướng không dựng theo position_ids của mặt nạ (Q1)")
        if not h.get("commit_hf"):
            loi.append("file hướng không ghi commit Hugging Face của mô hình: không xác minh được mô hình sẽ nạp "
                       "(rà soát bước 4, R2 B1)")
    for k in ("r_hat_en", "r_hat_vi", "s"):
        if k not in h:
            loi.append(f"thiếu khoá {k}")
    return loi


def _kiem_kiem_hook(thu_muc: str | None, dac_ta, huong_sha: str, thu: bool,
                    co_quyet_dinh: bool = False) -> tuple[list[str], dict | None]:
    """Bản thu nhỏ phần 2 của bước 3 (buoc3_kiem_hook) phải ĐẠT, cho đúng mô hình và đúng file hướng này.
    Trả về (lỗi, meta.json của lượt kiểm hook) — meta dùng để đối chiếu môi trường sau khi nạp mô hình."""
    if thu_muc is None:
        return ([] if thu else ["thiếu --kiem_hook (thư mục kết quả buoc3_kiem_hook của mô hình này)"]), None
    d = Path(thu_muc)
    try:
        kl = json.loads((d / "ket_luan.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"không đọc được ket_luan.json / meta.json trong {d}: {e}"], None
    if not isinstance(kl, dict) or not isinstance(meta, dict):
        return [f"ket_luan.json / meta.json trong {d} không đúng dạng"], None
    loi = []
    tham_so = meta.get("tham_so") if isinstance(meta.get("tham_so"), dict) else {}
    if meta.get("dac_ta") != dac_ta.ten:
        loi.append(f"kiểm hook của {meta.get('dac_ta')}, không phải {dac_ta.ten}")
    if tham_so.get("huong_sha256") != huong_sha:
        loi.append("kiểm hook chạy với file hướng KHÁC file --huong")
    if not thu:
        tt = kl.get("trang_thai")
        if tt == "can_xem" and not co_quyet_dinh:
            loi.append("kiểm hook có trạng thái 'can_xem': chỉ đi tiếp khi Ân, sau khi hai reviewer xem, quyết đi tiếp "
                       "(Nhap, mục Kiểm pipeline), với --quyet_dinh_can_xem <file quyết định>")
        elif tt not in ("dat", "can_xem"):
            loi.append(f"kiểm hook có trạng thái {tt!r}, cần 'dat'")
        if meta.get("chinh_thuc") is not True:
            loi.append("kiểm hook là lượt thử, không chính thức")
    return loi, meta


def _chuan_backend(d) -> dict | None:
    """{tên hàm: (cài đặt, có phải bản PyTorch tham chiếu)}; None nếu không đúng dạng. Bước 3 phải chạy bằng cùng
    code (cùng cách đặt tên cài đặt) với bước 4."""
    if not isinstance(d, dict) or not all(isinstance(v, dict) for v in d.values()):
        return None
    return {k: (str(v.get("cai_dat")), v.get("la_ban_tham_chieu")) for k, v in d.items()}


def doi_chieu_buoc3(meta: dict | None, tt_moi_truong: dict, backend: dict, commit_hf: str | None,
                    thu: bool) -> dict:
    """So môi trường của phiên bước 4 này với lượt kiểm hook bước 3 đã đạt (meta.json của nó): backend kernel bf16
    của lớp attention tuyến tính, phiên bản các gói số học, commit mô hình. Khác → "khac" không rỗng → script
    dừng mã 4 (không có cờ vượt: chạy lại kiểm hook bước 3 trên máy này). Tên GPU khác → chỉ ghi "gpu_khac".
    Lượt thử (--thu) không có meta hoặc meta thiếu trường thì bỏ qua phần đó."""
    if meta is None:
        return {"bo_qua": True, "khac": [], "gpu_khac": None}
    if not isinstance(meta, dict):
        return {"bo_qua": False, "khac": ["meta kiểm hook không đúng dạng"], "gpu_khac": None}
    khac: list[str] = []
    mt3 = meta.get("moi_truong")
    if not isinstance(mt3, dict):
        mt3 = None
    if mt3 is None:
        if not thu:
            khac.append("meta kiểm hook không có 'moi_truong'")
    else:
        for k in KHOA_MOI_TRUONG_BUOC3:
            if mt3.get(k) != tt_moi_truong.get(k):
                khac.append(f"{k}: bước 3 {mt3.get(k)!r}, bây giờ {tt_moi_truong.get(k)!r}")
    if "backend_lop_tuyen_tinh_bf16" not in meta:
        if not thu:
            khac.append("meta kiểm hook không có 'backend_lop_tuyen_tinh_bf16'")
    else:
        b3, b4 = _chuan_backend(meta["backend_lop_tuyen_tinh_bf16"]), _chuan_backend(backend)
        if b3 is None or b3 != b4:
            khac.append(f"backend lớp attention tuyến tính: bước 3 {b3}, bây giờ {b4}")
    mh3 = meta.get("mo_hinh")
    c3 = mh3.get("commit_hf") if isinstance(mh3, dict) else None
    if (c3 is not None or not thu) and c3 != commit_hf:
        khac.append(f"commit mô hình: bước 3 {c3!r}, bây giờ {commit_hf!r}")
    g3, g4 = (mt3 or {}).get("gpu"), tt_moi_truong.get("gpu")
    return {"bo_qua": False, "khac": khac, "gpu_khac": None if g3 == g4 else {"buoc3": g3, "bay_gio": g4}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh_dac_ta", required=True)
    p.add_argument("--huong", default=None, help="huong.pt của bước 3 (bắt buộc, trừ khi --chi_ke_hoach)")
    p.add_argument("--kiem_hook", default=None,
                   help="thư mục kết quả buoc3_kiem_hook của mô hình này (bắt buộc): phải đạt, cùng file hướng")
    p.add_argument("--mo_hinh", default=None,
                   help="thư mục mô hình cục bộ (mặc định id trên Hugging Face). Lượt chính thức với thư mục cục bộ "
                        "không có commit sẽ dừng mã 4 (không xác minh được mô hình)")
    p.add_argument("--revision", default=None, help="mặc định: commit HF ghi trong file hướng")
    p.add_argument("--thu_muc", default=None, help="mặc định results/buoc4/<mô hình>/hieu_chinh")
    p.add_argument("--khong_mo_rong", action="store_true", help="không sinh sẵn hai mức mở rộng 0,05 và 1,6")
    p.add_argument("--chi_ke_hoach", action="store_true", help="in kế hoạch rồi thoát, không nạp mô hình")
    p.add_argument("--so_don_vi_toi_da", type=int, default=None, help="dừng sau từng ấy đơn vị trong phiên này")
    p.add_argument("--cho_phep_doi_moi_truong", action="store_true",
                   help="cho chạy tiếp dù môi trường số học (GPU, torch, backend) khác phiên đầu (ghi lại; hỏi Ân trước)")
    p.add_argument("--cho_phep_doi_ma_nguon", action="store_true",
                   help="cho chạy tiếp dù mã nguồn đã đổi so với lúc bắt đầu lượt (ghi lại; chỉ dùng khi chắc chắn)")
    p.add_argument("--quyet_dinh_can_xem", default=None,
                   help="file quyết định của Ân cho trường hợp bước 3 ra 'cần xem' (Nhap, mục Kiểm pipeline): được "
                        "chép vào thư mục lượt, SHA vào dấu vân tay. Chỉ dùng khi hướng hoặc kiểm hook là 'can_xem'")
    p.add_argument("--chi_gop", action="store_true",
                   help="chỉ gộp lượt đã sinh xong (vd. trên máy cá nhân): không bao giờ nạp mô hình; cho qua khi mã "
                        "nguồn khác lúc sinh (ghi lại); thiếu đơn vị thì báo và dừng mã 1")
    p.add_argument("--co_lo", type=int, default=CO_LO,
                   help=f"cỡ lô (mặc định {CO_LO} = cả đơn vị một lô). CHỈ đổi khi đã hết bộ nhớ GPU và Ân đồng ý; "
                        "nằm trong dấu vân tay nên lượt đã có đơn vị sẽ dừng mã 6 (dùng thư mục mới)")
    # chỉ cho test: lượt thử không chính thức
    p.add_argument("--thu", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gioi_han_cau", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    dac_ta = chon_dac_ta(a.mo_hinh_dac_ta)
    if not 1 <= a.co_lo <= SO_CAU:
        print(f"--co_lo phải trong khoảng 1–{SO_CAU}.")
        return 1
    if not a.thu and (a.gioi_han_cau is not None or a.max_new_tokens != MAX_NEW_TOKENS):
        print("--gioi_han_cau / --max_new_tokens chỉ dùng với --thu.")
        return 1
    ten_luot = "hieu_chinh_thu" if a.thu else "hieu_chinh"
    thu_muc = a.thu_muc or str(GOC / "results" / "buoc4" / dac_ta.ten / ten_luot)
    ke = ke_hoach_hieu_chinh(dac_ta, LUOI_ALPHA_HIEU_CHINH, () if a.khong_mo_rong else ALPHA_MO_RONG)
    if a.chi_ke_hoach:
        try:
            cau = _cau_theo_tap(a.gioi_han_cau)
        except Exception as e:                                          # noqa: BLE001
            print(f"DỪNG: không đọc được chia tập / dữ liệu test: {type(e).__name__}: {e}")
            return 4
        so_cau = {t: len(v) for t, v in cau.items()}
        nhom: dict[str, int] = {}
        for d in ke:
            nhom[d.nhom] = nhom.get(d.nhom, 0) + 1
        print(f"{dac_ta.ten}: {len(ke)} đơn vị {nhom}; số câu mỗi tập {so_cau}; "
              f"tổng {sum(so_cau[d.tap] for d in ke)} câu trả lời; thư mục {thu_muc}")
        for d in ke[:6] + ke[-2:]:
            print("  ", d.ma)
        return 0
    if not a.huong:
        print("Cần --huong (file huong.pt của bước 3).")
        return 1

    # ── dữ liệu chia tập, file hướng và kết quả kiểm hook của bước 3 ──
    # Kiểm chia tập TRƯỚC khi đọc câu theo chia tập: dữ liệu lệch hay hỏng thì dừng mã 4, không traceback (đối chiếu
    # lần cuối, R2, 28/10/2026).
    loi = _kiem_chia_tap()
    if loi:
        print("DỪNG: " + "; ".join(loi))
        return 4
    try:
        cau = _cau_theo_tap(a.gioi_han_cau)
    except Exception as e:                                              # noqa: BLE001
        print(f"DỪNG: không đọc được chia tập / dữ liệu test: {type(e).__name__}: {e}")
        return 4
    so_cau = {t: len(v) for t, v in cau.items()}
    qd = None
    if a.quyet_dinh_can_xem:
        try:
            noi_qd = open(a.quyet_dinh_can_xem, "rb").read()
        except OSError as e:
            print(f"DỪNG: không đọc được file quyết định {a.quyet_dinh_can_xem}: {e}")
            return 4
        if not noi_qd.strip():
            print(f"DỪNG: file quyết định {a.quyet_dinh_can_xem} rỗng.")
            return 4
        qd = {"file_goc": str(a.quyet_dinh_can_xem), "sha256": NG.sha256_file(a.quyet_dinh_can_xem),
              "noi_dung": noi_qd}
    h = torch.load(a.huong, map_location="cpu")
    loi = _kiem_huong(h, dac_ta, a.thu, co_quyet_dinh=qd is not None)
    if loi:
        print("DỪNG: " + "; ".join(loi))
        return 4
    huong_sha = NG.sha256_file(a.huong)
    loi, meta_kh = _kiem_kiem_hook(a.kiem_hook, dac_ta, huong_sha, a.thu, co_quyet_dinh=qd is not None)
    if loi:
        print("DỪNG: " + "; ".join(loi))
        return 4
    if qd is not None:
        tt_kh = None
        if a.kiem_hook:
            try:
                tt_kh = json.loads((Path(a.kiem_hook) / "ket_luan.json").read_text(encoding="utf-8")).get("trang_thai")
            except (OSError, ValueError, AttributeError):
                tt_kh = None
        qd["ap_cho"] = [x for x, tt in (("huong", h.get("trang_thai")), ("kiem_hook", tt_kh)) if tt == "can_xem"]
        if not qd["ap_cho"]:
            print("DỪNG: có --quyet_dinh_can_xem nhưng hướng và kiểm hook đều không ở trạng thái 'can_xem'.")
            return 4
    revision = a.revision or h.get("commit_hf")

    # ── kế hoạch và thư mục (trước khi nạp mô hình: lệch thì dừng sớm, không tốn GPU) ──
    ma_nguon = dau_van_tay_ma_nguon(GOC)
    ke_hoach = {
        "phien_ban_dinh_dang": 1, "buoc": "4", "viec": "lượt hiệu chỉnh", "chinh_thuc": not a.thu,
        "dac_ta": dac_ta.ten, "huong": str(a.huong), "kiem_hook": a.kiem_hook,
        "quyet_dinh_can_xem": None if qd is None else {k: v for k, v in qd.items() if k != "noi_dung"},
        "tham_so": {"max_new_tokens": a.max_new_tokens, "co_lo": a.co_lo, "giai_ma": "den_eos", "dtype": "bfloat16",
                    "lop_hieu_chinh": list(dac_ta.lop_hieu_chinh), "luoi_alpha": list(LUOI_ALPHA_HIEU_CHINH),
                    "alpha_mo_rong": [] if a.khong_mo_rong else list(ALPHA_MO_RONG), "so_cau": so_cau,
                    "revision": revision},
        "id_cau": {t: [c["id"] for c in v] for t, v in cau.items()},
        "don_vi": [{"ma": d.ma, **d.__dict__} for d in ke],
    }
    ke_hoach["dau_van_tay"] = {
        "dac_ta": dac_ta.ten, "commit_hf": h.get("commit_hf"), "huong_sha256": huong_sha,
        "chia_tap_sha256": NG.sha256_file(FILE_CHIA),
        "du_lieu_test_sha256": {n: NG.sha256_file(THU_MUC_TEST / n) for n in TEP_DU_LIEU_TEST},
        "du_lieu_dung_sha256": {f.name: NG.sha256_file(f) for f in sorted((GOC / "data" / "dung_huong").glob("*.jsonl"))},
        "quyet_dinh_can_xem_sha256": None if qd is None else qd["sha256"],
        "tham_so": sha256_chuoi(json.dumps(ke_hoach["tham_so"], sort_keys=True)),
        "ke_hoach_don_vi": sha256_chuoi(json.dumps([d["ma"] for d in ke_hoach["don_vi"]])),
        "id_cau": sha256_chuoi(json.dumps(ke_hoach["id_cau"], sort_keys=True)),
        "ma_nguon": ma_nguon["sha256"],
    }
    kho = KhoKetQua(thu_muc)
    try:
        kho.khoa()
    except DangChay as e:
        print(f"DỪNG: {e}")
        return 9
    if kho.canh_bao_khoa:
        print(f"CHÚ Ý: {kho.canh_bao_khoa}")
    try:
        return _chay(a, dac_ta, ke, cau, h, revision, ke_hoach, ma_nguon, kho, meta_kh, qd)
    finally:
        kho.nha_khoa()


def _bao_gop(log, phan: str, r: dict) -> None:
    if phan == "chinh":
        log(f"Đã gộp nền + lưới chính: {r['so_cau_tra_loi']} câu → {r['file']}; theo nhóm {r['tom_tat']['theo_nhom']}")
    elif phan == "chan_doan":
        sk = r["tom_tat"]["so_khop_voi_nen_co_hook"]
        log(f"Đã gộp lượt nền không hook: {r['so_cau_tra_loi']} câu → {r['file']}; giống hệt lượt nền có hook "
            f"{sk['giong_het']}/{sk['tong_cau']} câu"
            + ("." if sk["tat_ca_giong_het"] else ". CHÚ Ý: có câu KHÁC (sinh không tất định ở cỡ lô này, hoặc hook "
               "α = 0 không trung tính); xem tom_tat_nen_khong_hook.json và báo Ân. Lượt vẫn chạy tiếp."))
    else:
        log(f"Đã gộp mức mở rộng: {r['so_cau_tra_loi']} câu → {r['file']} (để riêng, chưa đọc).")


def _moi_truong_so_hoc(model, tt_moi_truong: dict) -> dict:
    """Những gì có thể làm đầu ra số học khác đi giữa hai phiên (so với phiên đầu)."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu,
            "transformers": tt_moi_truong.get("transformers"), "tokenizers": tt_moi_truong.get("tokenizers"),
            "goi_flash-linear-attention": tt_moi_truong.get("goi_flash-linear-attention"),
            "goi_causal-conv1d": tt_moi_truong.get("goi_causal-conv1d"),
            "backend_lop_tuyen_tinh": {k: v["cai_dat"] for k, v in backend_lop_tuyen_tinh(model).items()},
            "dtype": str(next(model.parameters()).dtype)}


def _chay(a, dac_ta, ke, cau, h, revision, ke_hoach, ma_nguon, kho, meta_kh=None, qd=None) -> int:
    ma_nguon_khac = False
    if a.chi_gop:
        if not kho.file_ke_hoach.exists() or not kho.co_don_vi_nao():
            print(f"DỪNG: --chi_gop nhưng {kho.goc} chưa có lượt nào đã sinh đơn vị để gộp.")
            return 1
        try:
            cu = json.loads(kho.file_ke_hoach.read_text(encoding="utf-8"))
            ma_nguon_khac = cu["dau_van_tay"]["ma_nguon"] != ke_hoach["dau_van_tay"]["ma_nguon"]
        except (OSError, ValueError, KeyError, TypeError):
            ma_nguon_khac = False
    try:
        trang_thai = kho.mo(ke_hoach, cho_phep_doi_ma_nguon=a.cho_phep_doi_ma_nguon or a.chi_gop)
    except ValueError as e:
        print(f"DỪNG: {e}")
        return 6
    # Bản chép file quyết định trong thư mục lượt luôn trùng file đang dùng: bản cũ khác nội dung (hoặc không còn cần)
    # thì chuyển sang hong/ (rà độc lập 28/10/2026). Đã có đơn vị thì dấu vân tay đã chặn mọi thay đổi ở đây.
    p_qd = None if qd is None else kho.goc / ("quyet_dinh_can_xem" + (Path(qd["file_goc"]).suffix or ".txt"))
    for cu in sorted(kho.goc.glob("quyet_dinh_can_xem.*")):
        if cu != p_qd or NG.sha256_file(cu) != qd["sha256"]:
            kho.chuyen_vao_hong(cu, "quyet_dinh_cu")
    if p_qd is not None and not p_qd.exists():
        ghi_nguyen_tu(p_qd, qd["noi_dung"])
    if da_hoan_tat(kho, ke, cau):
        print(f"Lượt đã hoàn tất từ trước và còn nguyên ({kho.file_hoan_tat}); không làm gì thêm.")
        return 0
    log = NhatKy(kho.goc / "nhat_ky.txt")
    if kho.file_hoan_tat.exists():
        # Lượt từng hoàn tất mà nay không còn nguyên (đối chiếu lần cuối, R2 F5): báo rõ file nào đổi trước khi gộp lại;
        # bản tổng hợp cũ sang hong/, không ghi đè. File không dựng lại được từ đơn vị thì Ân xem trước khi dùng.
        doi = doi_sau_hoan_tat(kho)
        log(f"CHÚ Ý: lượt đã có HOAN_TAT.json nhưng không còn nguyên ({len(doi)} thay đổi so với sha256.txt cũ: "
            + "; ".join(doi[:20]) + ("; …" if len(doi) > 20 else "") + "). Gộp lại từ các đơn vị; bản cũ sang hong/.")
    log({"moi": "Bắt đầu lượt mới", "tiep": "Chạy TIẾP lượt cũ",
         "lam_lai_ke_hoach": "Lập lại kế hoạch (lượt cũ chưa sinh đơn vị nào)"}[trang_thai]
        + f" tại {kho.goc}; {len(ke)} đơn vị; chính thức: {not a.thu}")
    if kho.canh_bao_khoa:
        log(f"CHÚ Ý: {kho.canh_bao_khoa}")
    if qd is not None:
        log(f"Đi tiếp với trạng thái 'cần xem' ({qd['ap_cho']}) theo quyết định của Ân: {qd['file_goc']} "
            f"(SHA {qd['sha256'][:12]}…), đã chép vào thư mục lượt.")
    if ma_nguon_khac:
        log("CHÚ Ý (--chi_gop): mã nguồn khác lúc sinh; chỉ gộp, không sinh gì nên vẫn cho qua (ghi lại).")

    def gop_va_ket_thuc() -> int:
        for phan in ("chinh", "chan_doan", "mo_rong"):
            r = tong_hop_phan(kho, ke, cau, phan, dac_ta.dau_hieu_thinking)
            if r:
                _bao_gop(log, phan, r)
        ht = tong_hop_cuoi(kho, ke)
        log(f"HOÀN TẤT: {ht['so_don_vi']} đơn vị, {ht['so_cau_tra_loi']} câu trả lời.")
        return 0

    chua = kiem_ke_hoach(kho, ke, cau)[1]
    if not chua:
        log("Mọi đơn vị đã xong; gộp mà không nạp mô hình.")
        return gop_va_ket_thuc()
    if a.chi_gop:
        log(f"DỪNG (--chi_gop): còn {len(chua)} đơn vị chưa xong, không gộp được và không nạp mô hình. Ví dụ: "
            + "; ".join(f"{dv.ma} ({ly_do})" for dv, ly_do in chua[:5]))
        return 1
    if kho.canh_bao_khoa:
        log("DỪNG: còn đơn vị phải sinh mà không khoá được thư mục lượt (hệ file không hỗ trợ khoá), nên không chặn "
            "được hai tiến trình cùng sinh. Chạy trên ổ cục bộ của máy (ext4), không trên ổ mạng hay /mnt của WSL.")
        return 9

    # ── nạp và kiểm ──
    bat_dau = time.time()
    dm = device_map_mot_gpu()
    model, tok = nap_mo_hinh(dac_ta, a.mo_hinh, revision=revision, device_map=dm)
    pb = phan_bo_thiet_bi(model, can_gpu=dm is not None)
    log(f"Đã nạp sau {(time.time() - bat_dau) / 60:.1f} phút; phân bố thiết bị {pb['so_tensor_theo_thiet_bi']}")
    if not pb["mot_thiet_bi"]:
        log("DỪNG: mô hình không nằm trọn trên một GPU.")
        return 3
    tt_mo_hinh = NG.thong_tin_mo_hinh(model, tok, dac_ta)
    # Lượt chính thức: commit của mô hình đang nạp phải xác minh được và trùng commit ghi trong file hướng (không
    # có commit, vd. nạp từ thư mục cục bộ, thì dừng; rà soát bước 4, R2 B1, Ân duyệt Q1 ngày 28/10/2026).
    if (not a.thu and tt_mo_hinh["commit_hf"] != h.get("commit_hf")) or (
            a.thu and h.get("commit_hf") and tt_mo_hinh["commit_hf"] not in (None, h["commit_hf"])):
        log(f"DỪNG: file hướng dựng từ commit HF {h.get('commit_hf')!r}, mô hình đang nạp có commit "
            f"{tt_mo_hinh['commit_hf']!r} (None = không xác minh được). Nạp đúng id Hugging Face như runbook.")
        return 4
    tat_ca_cau = sorted({c[ng] for v in cau.values() for c in v for ng in ("en", "vi")})
    khoi = kiem_khoi_tokenizer(tok, dac_ta, tat_ca_cau)
    eos = eos_tu_mo_hinh(model, dac_ta)
    from buoc1_kiem_truoc import cau_hinh_hieu_luc
    gc = cau_hinh_hieu_luc(a.mo_hinh or dac_ta.hf_id, revision, tuple(eos), tok.pad_token_id)
    tt_moi_truong = NG.thong_tin_moi_truong()
    mt = _moi_truong_so_hoc(model, tt_moi_truong)
    dc3 = doi_chieu_buoc3(meta_kh, tt_moi_truong, backend_lop_tuyen_tinh(model), tt_mo_hinh["commit_hf"], a.thu)
    mt_ok, mt_khac = kho.kiem_moi_truong(mt, cho_phep_doi=a.cho_phep_doi_moi_truong)
    so_phien = kho.ghi_phien({
        "luc": time.strftime("%Y-%m-%d %H:%M:%S"), "trang_thai_mo": trang_thai, "moi_truong": tt_moi_truong,
        "moi_truong_so_hoc": mt, "moi_truong_khac_phien_dau": mt_khac, "mo_hinh": tt_mo_hinh,
        "bien_moi_truong": NG.bien_moi_truong_quan_trong(), "phan_bo_thiet_bi": pb, "ma_nguon": ma_nguon,
        "commit_kho": NG.commit_kho(), "cho_phep_doi_ma_nguon": a.cho_phep_doi_ma_nguon,
        "cho_phep_doi_moi_truong": a.cho_phep_doi_moi_truong, "khoi_sau_cau_hoi_dat": khoi["dat"],
        "cau_hinh_sinh": gc, "eos": list(eos), "huong_trang_thai": h.get("trang_thai"),
        "huong_chinh_thuc": h.get("chinh_thuc"), "kiem_hook": a.kiem_hook,
        "quyet_dinh_can_xem_sha256": None if qd is None else qd["sha256"], "doi_chieu_buoc3": dc3})
    if not khoi["dat"]:
        log(f"DỪNG: khối sau câu hỏi lệch đặc tả: {khoi['khoi_theo_so_cau']}")
        return 2
    if not gc["la_greedy_dung"]:
        log(f"DỪNG: cấu hình sinh hiệu lực không phải greedy: {gc['hieu_luc']}")
        return 2
    if dc3["khac"]:
        log("DỪNG: môi trường khác lượt kiểm hook bước 3 đã đạt (" + "; ".join(dc3["khac"]) + "). Kiểm hook chỉ "
            "chứng nhận đúng backend và phiên bản đã dùng: dựng lại môi trường như bước 3, hoặc chạy lại "
            "buoc3_kiem_hook trên máy này rồi dùng thư mục kết quả mới cho --kiem_hook.")
        return 4
    if dc3["gpu_khac"]:
        log(f"CHÚ Ý: GPU khác lượt kiểm hook bước 3 ({dc3['gpu_khac']}); chỉ ghi lại (backend và gói đã trùng).")
    if not mt_ok:
        log(f"DỪNG: môi trường số học khác phiên đầu ở {mt_khac} (xem moi_truong_goc.json). Lượt nền và lượt can "
            f"thiệp được so theo cặp nên phải cùng môi trường; hỏi Ân trước khi dùng --cho_phep_doi_moi_truong.")
        return 8
    if mt_khac:
        log(f"CHÚ Ý: môi trường khác phiên đầu ở {mt_khac}; đã cho phép bằng --cho_phep_doi_moi_truong (ghi lại).")

    # ── sinh: phần chính trước, gộp ngay; rồi phần mở rộng ──
    huong = {"en": h["r_hat_en"], "vi": h["r_hat_vi"]}
    meta_phien = {"phien": so_phien, "moi_truong_so_hoc_sha256": bam_moi_truong(mt)}
    con_han = a.so_don_vi_toi_da
    for phan in ("chinh", "chan_doan", "mo_rong"):
        ds = [d for d in ke if phan_cua(d) == phan]
        if not ds:
            continue
        kq = chay_ke_hoach(model, tok, dac_ta, ds, cau, huong, h["s"], eos, kho, a.max_new_tokens, a.co_lo, log,
                           so_don_vi_toi_da=con_han, meta_phien=meta_phien)
        log(f"Phần {phan}, phiên này: {kq}")
        if con_han is not None:
            con_han -= kq["sinh_trong_phien"]
        if kq["con_lai"] > 0:
            log(f"Còn {kq['con_lai']} đơn vị phần {phan}: chạy lại đúng lệnh này để tiếp.")
            return 7
        r = tong_hop_phan(kho, ke, cau, phan, dac_ta.dau_hieu_thinking)
        _bao_gop(log, phan, r)
    ht = tong_hop_cuoi(kho, ke)
    log(f"HOÀN TẤT: {ht['so_don_vi']} đơn vị, {ht['so_cau_tra_loi']} câu trả lời. "
        f"Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
