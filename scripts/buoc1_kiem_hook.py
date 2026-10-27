"""Bước 1, phần 2: kiểm đường can thiệp của ta trên Llama-3-8B-Instruct.

Chạy:
    python scripts/buoc1_kiem_hook.py --mo_hinh meta-llama/Meta-Llama-3-8B-Instruct

Hướng dùng cho phép kiểm: dựng theo giao thức của ta (hiệu trung bình ở ĐẦU RA khối ℓ, tại vị trí lấy
kích hoạt, không lọc) trên 128 + 128 câu dựng của Arditi et al. (tiếng Anh); s_ℓ là độ lớn trung bình
của luồng dư trên 256 câu đó. Hướng này chỉ để kiểm cơ chế hook, không dùng cho phân tích nào.

Câu kiểm: 8 câu đầu của harmless_val (kiểu a) và 8 câu đầu của harmful_val (kiểu b), mỗi kiểu một lô.
Đo nhiễu ở α = 0: 128 câu dựng mỗi kiểu (harmless_train cho kiểu a, harmful_train cho kiểu b), lô 8 câu.
Lớp kiểm mặc định: 25 %, 50 %, 75 % độ sâu (Llama-3: 8, 16, 24; đánh số từ 0).

Chạy hai lượt: bf16 (kiểu số của lượt chạy thật) rồi float32 (giải phóng bản bf16, nạp lại cùng checkpoint ở
float32 — checkpoint lưu bf16 nên đổi lên float32 là chính xác; TF32 tắt); kết luận theo
kiem_hook.ket_luan_hai_kieu_so (luật ba mức, xem docstring của tn2.kiem_hook). Bản float32 nạp với
device_map={"": 0} để ép lên một GPU: thiếu bộ nhớ thì báo lỗi ngay khi nạp.
Mã thoát: 0 đạt; 1 không đạt; 2 khuôn chat lệch cách tác giả mã hoá; 3 mô hình bị chia ra CPU/đĩa hoặc nhiều
thiết bị; 4 "cần xem" (KHÔNG phải đạt: đọc ket_luan.json và chan_doan_vi_tri theo luật viết trước).
"""
from __future__ import annotations

import argparse
import sys
import time

from _chung import GOC, NhatKy, ten_luot_mac_dinh

import torch

from tn2 import nguon_goc as NG
from tn2.buoc1.tai_lap_arditi import DU_LIEU_MAC_DINH, kiem_khuon, nap_du_lieu
from tn2.huong import chuan_hoa, hieu_trung_binh
from tn2.kich_hoat import trung_binh_kich_hoat
from tn2.khuon import ma_hoa
from tn2.kiem_hook import (ALPHA_KIEM, MAX_NEW_TOKENS_KIEM, DauVaoKiemHook, chay_kiem_hook,
                           ket_luan_hai_kieu_so)
from tn2.mo_hinh import LLAMA3_8B, eos_tu_mo_hinh, nap_mo_hinh, phan_bo_thiet_bi, tim_cac_lop


def lop_mac_dinh(so_lop: int) -> list[int]:
    return sorted({round(so_lop * f) for f in (0.25, 0.5, 0.75)})


def dung_huong_kiem(model, tok, co_hai, vo_hai, lop):
    """r̂_ℓ, ‖r_ℓ‖, s_ℓ ở đầu ra khối, vị trí lấy kích hoạt của ta."""
    vt = ma_hoa(tok, [co_hai[0]], LLAMA3_8B).vi_tri_lay
    kh_a = trung_binh_kich_hoat(model, tok, LLAMA3_8B, co_hai, [vt], lop, "ra_khoi")
    kh_b = trung_binh_kich_hoat(model, tok, LLAMA3_8B, vo_hai, [vt], lop, "ra_khoi")
    r = hieu_trung_binh(kh_a.trung_binh[0], kh_b.trung_binh[0])           # [nL, d]
    r_hat, do_lon = chuan_hoa(r)
    s = (kh_a.trung_binh_do_lon[0] * kh_a.so_cau + kh_b.trung_binh_do_lon[0] * kh_b.so_cau) / (kh_a.so_cau + kh_b.so_cau)
    return vt, r_hat, do_lon, s


def thiet_lap_so_hoc() -> dict:
    """Tắt TF32 (để lượt float32 đúng là float32) và ghi lại cấu hình số học."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def _bao(log, ten, t):
    log(f"[{ten}] (ii) α=0 khớp tuyệt đối: {t['ii_dat']}")
    log(f"[{ten}] nhiễu (i) {t['nhieu_i']:.4g}, nhiễu (iii) {t['nhieu_iii']:.4g} → ngưỡng {t['nguong']:.4g}")
    log(f"[{ten}] (i) lệch lớn nhất {t['lech_lon_nhat_i']:.4g} → {t['trang_thai_i']}")
    log(f"[{ten}] (iii) lệch lớn nhất {t['lech_lon_nhat_iii']:.4g} → {t['trang_thai_iii']}")
    log(f"[{ten}] kiểm âm: {[(k, round(v['lech_lon_nhat'], 4), v['dat']) for k, v in t['kiem_am'].items()]}")
    log(f"[{ten}] số lượt α≠0 làm đổi token so với không hook: {t['so_luot_alpha_doi_token']}/{t['so_luot_alpha']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh", default=LLAMA3_8B.hf_id)
    p.add_argument("--revision", default=None)
    p.add_argument("--ra", default=str(GOC / "results" / "buoc1"))
    p.add_argument("--ten_luot", default=None)
    p.add_argument("--so_cau", type=int, default=8)
    p.add_argument("--so_cau_nhieu", type=int, default=128)
    p.add_argument("--lop", default=None, help="danh sách lớp (đánh số từ 0), ví dụ 8,16,24")
    p.add_argument("--gioi_han_dung_huong", type=int, default=None, help="lượt thử nhỏ: số câu dựng mỗi loại")
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS_KIEM)
    p.add_argument("--bo_fp32", action="store_true", help="lượt thử: chỉ chạy bf16 (không cho kết luận)")
    a = p.parse_args(argv)

    chinh_thuc = (a.so_cau == 8 and a.so_cau_nhieu == 128 and a.gioi_han_dung_huong is None
                  and a.max_new_tokens == MAX_NEW_TOKENS_KIEM and a.lop is None and not a.bo_fp32)
    thu_muc = NG.thu_muc_moi(a.ra, a.ten_luot or ten_luot_mac_dinh("kiem_hook" if chinh_thuc else "kiem_hook_thu"))
    log = NhatKy(thu_muc / "nhat_ky.txt")
    log(f"Thư mục kết quả: {thu_muc}; lượt chính thức: {chinh_thuc}")
    bat_dau = time.time()
    so_hoc = thiet_lap_so_hoc()
    du_lieu = nap_du_lieu()
    model, tok = nap_mo_hinh(LLAMA3_8B, a.mo_hinh, revision=a.revision)
    pb = phan_bo_thiet_bi(model)
    log(f"Phân bố thiết bị: {pb}")
    if not pb["mot_thiet_bi"]:
        log("DỪNG: mô hình không nằm trọn trên một GPU.")
        return 3
    kk = kiem_khuon(tok, LLAMA3_8B, sorted(set(du_lieu.harmful_train + du_lieu.harmless_train
                                               + du_lieu.harmful_val + du_lieu.harmless_val)))
    NG.ghi_json(thu_muc / "kiem_khuon.json", kk)
    if not kk["dat"]:
        log(f"DỪNG: khuôn chat lệch cách tác giả mã hoá ở {kk['so_cau_lech']} câu.")
        return 2
    so_lop = len(tim_cac_lop(model))
    lop = [int(x) for x in a.lop.split(",")] if a.lop else lop_mac_dinh(so_lop)
    eos = eos_tu_mo_hinh(model, LLAMA3_8B)

    gh = a.gioi_han_dung_huong
    co_hai = du_lieu.harmful_train[:gh] if gh else du_lieu.harmful_train
    vo_hai = du_lieu.harmless_train[:gh] if gh else du_lieu.harmless_train
    vt, r_hat, do_lon, s = dung_huong_kiem(model, tok, co_hai, vo_hai, lop)
    torch.save({"lop": lop, "r_hat": r_hat, "do_lon_tho": do_lon, "s": s, "vi_tri_lay": vt},
               thu_muc / "huong_kiem.pt")
    log(f"Hướng kiểm ở lớp {lop}: ‖r‖ = {do_lon.tolist()}, s = {s.tolist()}")

    dv = DauVaoKiemHook(cau_hoi_a=du_lieu.harmless_val[: a.so_cau], cau_hoi_b=du_lieu.harmful_val[: a.so_cau],
                        lop=lop, r_hat={l: r_hat[j] for j, l in enumerate(lop)},
                        s={l: float(s[j]) for j, l in enumerate(lop)}, eos_ids=eos, alpha=ALPHA_KIEM,
                        max_new_tokens=a.max_new_tokens,
                        cau_hoi_nhieu_a=du_lieu.harmless_train[: a.so_cau_nhieu],
                        cau_hoi_nhieu_b=du_lieu.harmful_train[: a.so_cau_nhieu], co_lo=a.so_cau)
    NG.ghi_json(thu_muc / "meta.json", {
        "buoc": "1", "phan": "2 — kiểm đường can thiệp", "chinh_thuc": chinh_thuc,
        "moi_truong": NG.thong_tin_moi_truong(), "mo_hinh": NG.thong_tin_mo_hinh(model, tok, LLAMA3_8B),
        "bien_moi_truong": NG.bien_moi_truong_quan_trong(),
        "tham_so": {"lop": lop, "so_cau": a.so_cau, "so_cau_nhieu": a.so_cau_nhieu, "alpha": ALPHA_KIEM,
                    "max_new_tokens": a.max_new_tokens, "eos": list(eos), "vi_tri_lay": vt,
                    "so_cau_dung_huong": [len(co_hai), len(vo_hai)], "bo_fp32": a.bo_fp32},
        "so_hoc": so_hoc, "phan_bo_thiet_bi_bf16": pb,
        "du_lieu_sha256": {f.name: NG.sha256_file(f) for f in sorted(DU_LIEU_MAC_DINH.glob("*.jsonl"))},
    })

    kq16 = chay_kiem_hook(model, tok, LLAMA3_8B, dv, log)
    NG.ghi_json(thu_muc / "ket_qua_bf16.json", kq16)
    _bao(log, "bf16", kq16["tong_ket"])
    if a.bo_fp32:
        log(f"Bỏ lượt float32 (lượt thử). Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
        return 1

    # Nạp lại ở float32 thay vì model.float() tại chỗ: đỉnh bộ nhớ thấp hơn (không giữ hai bản cùng lúc).
    import gc
    del model
    gc.collect()                  # mô hình HF có tham chiếu vòng: phải gom rác thì bản bf16 mới được giải phóng
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        trong, tong = torch.cuda.mem_get_info()
        log(f"Bộ nhớ GPU trước khi nạp float32: trống {trong / 2**30:.1f} / {tong / 2**30:.1f} GiB")
    model, _ = nap_mo_hinh(LLAMA3_8B, a.mo_hinh, dtype=torch.float32, revision=a.revision,
                           device_map=({"": 0} if torch.cuda.is_available() else None))
    pb32 = phan_bo_thiet_bi(model)
    log(f"Phân bố thiết bị (float32): {pb32}")
    if not pb32["mot_thiet_bi"]:
        log("DỪNG: bản float32 không nằm trọn trên một GPU (cần GPU ≥ 40 GB trống).")
        return 3
    kq32 = chay_kiem_hook(model, tok, LLAMA3_8B, dv, log)
    NG.ghi_json(thu_muc / "ket_qua_fp32.json", kq32)
    _bao(log, "fp32", kq32["tong_ket"])

    kl = ket_luan_hai_kieu_so(kq16, kq32)
    NG.ghi_json(thu_muc / "ket_luan.json", kl)
    log(f"Kết luận phần 2: {kl}")
    log(f"Trạng thái: {kl['trang_thai']}. Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
    return {"dat": 0, "can_xem": 4}.get(kl["trang_thai"], 1)


if __name__ == "__main__":
    sys.exit(main())
