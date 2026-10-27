"""Bước 3, đo tốc độ sinh và bộ nhớ đỉnh của một mô hình chính, rồi ước giờ GPU (Nhap, Trình tự chạy bước 3;
"Quy mô và chi phí": tốc độ sinh đo khi nạp từng mô hình chính).

Chạy (sau buoc3_dung_huong.py):
    python scripts/buoc3_do_toc_do.py --mo_hinh_dac_ta qwen35_9b --huong results/buoc3/qwen35_9b/<lượt>/huong.pt

Đo bằng đúng hàm sinh của pipeline chính (greedy, bf16, đệm trái, 256 token) trên 128 câu dựng (32 câu đầu của
mỗi file en/vi), ở các cỡ lô 16, 32, 64, 128, hai chế độ:
- "nen": gắn hook α = 0 ở lớp giữa của lop_kiem_hook (như lượt nền của Nhap 3.6);
- "can_thiep": kiểu (a), phương án 1, α = 0,8 (mức lớn nhất của lưới hiệu chỉnh), cùng lớp. Đây là một cấu hình
  đo, không phải cận trên đã chứng minh (độ dài câu trả lời và điểm dừng thay đổi theo cấu hình).
Ghi toc_do.json (mỗi cấu hình: thời gian, câu/giờ, token/giây, bộ nhớ đỉnh, hết bộ nhớ hay không) và ước giờ GPU
theo số câu của Nhap. Tốc độ đo trên loại GPU này KHÔNG áp sang loại GPU khác (báo cáo kết quả bước 1, mục 7):
trước khi thuê máy cho lượt chính phải đo lại vài phút trên đúng loại máy đó.
"""
from __future__ import annotations

import argparse
import sys
import time

from _chung import GOC, NhatKy, chon_dac_ta, device_map_mot_gpu, ten_luot_mac_dinh

import torch

from tn2 import nguon_goc as NG
from tn2.buoc3.toc_do import do_toc_do, uoc_gio_gpu
from tn2.can_thiep import can_thiep_chinh
from tn2.du_lieu import nap_tap_dung
from tn2.mo_hinh import backend_lop_tuyen_tinh, eos_tu_mo_hinh, nap_mo_hinh, phan_bo_thiet_bi

MAX_NEW_TOKENS = 256
ALPHA_LON_NHAT = 0.8


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh_dac_ta", required=True)
    p.add_argument("--huong", required=True)
    p.add_argument("--mo_hinh", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--ra", default=str(GOC / "results" / "buoc3"))
    p.add_argument("--ten_luot", default=None)
    p.add_argument("--co_lo", default="16,32,64,128")
    p.add_argument("--so_cau_moi_file", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    a = p.parse_args(argv)
    dac_ta = chon_dac_ta(a.mo_hinh_dac_ta)
    thu_muc = NG.thu_muc_moi(f"{a.ra}/{dac_ta.ten}", a.ten_luot or ten_luot_mac_dinh("toc_do"))
    log = NhatKy(thu_muc / "nhat_ky.txt")
    bat_dau = time.time()
    h = torch.load(a.huong, map_location="cpu")
    if h.get("dac_ta") != dac_ta.ten:
        log(f"DỪNG: file hướng của mô hình {h.get('dac_ta')}, không phải {dac_ta.ten}.")
        return 1
    tap = nap_tap_dung()
    n = a.so_cau_moi_file
    cau = tap.harm_en[:n] + tap.harm_vi[:n] + tap.harmless_en[:n] + tap.harmless_vi[:n]
    dm = device_map_mot_gpu()
    model, tok = nap_mo_hinh(dac_ta, a.mo_hinh, revision=a.revision, device_map=dm)
    pb = phan_bo_thiet_bi(model, can_gpu=dm is not None)        # chỉ đòi GPU khi chính script xin GPU
    if not pb["mot_thiet_bi"]:
        log("DỪNG: mô hình không nằm trọn trên một GPU.")
        return 3
    if h.get("commit_hf") and NG.thong_tin_mo_hinh(model, tok, dac_ta)["commit_hf"] not in (None, h["commit_hf"]):
        log(f"DỪNG: file hướng dựng từ commit HF {h['commit_hf']}, mô hình đang nạp khác.")
        return 1
    if not h.get("chinh_thuc", False):
        log("CHÚ Ý: file hướng từ lượt dựng hướng thử (không chính thức).")
    if h.get("canh_bao_dem"):
        log("CHÚ Ý: file hướng có cảnh báo đệm ('cần xem'); đo tốc độ không phụ thuộc điều này.")
    eos = eos_tu_mo_hinh(model, dac_ta)
    lop = dac_ta.lop_kiem_hook[len(dac_ta.lop_kiem_hook) // 2]
    r_hat, s = h["r_hat_en"][lop].to(torch.float32), float(h["s"][lop])
    co_lo = [int(x) for x in a.co_lo.split(",")]

    def nen(enc):
        return [can_thiep_chinh(lop, r_hat, s, 0.0, "a", 1, enc.so_token_khoi)]

    def can_thiep(enc):
        return [can_thiep_chinh(lop, r_hat, s, ALPHA_LON_NHAT, "a", 1, enc.so_token_khoi)]

    # Chạy nháp một lô nhỏ để khởi động (nạp kernel), không tính giờ.
    do_toc_do(model, tok, dac_ta, cau[:4], [4], 8, eos, nen, "khoi_dong")
    kq = {"nen": do_toc_do(model, tok, dac_ta, cau, co_lo, a.max_new_tokens, eos, nen, "nen", log),
          "can_thiep": do_toc_do(model, tok, dac_ta, cau, co_lo, a.max_new_tokens, eos, can_thiep, "can_thiep", log)}
    hop_le = [d for d in kq["can_thiep"] if not d.get("het_bo_nho")]
    tot = max(hop_le, key=lambda d: d["cau_moi_gio"]) if hop_le else None
    kq["uoc_gio_gpu_theo_lo_nhanh_nhat_che_do_can_thiep"] = (
        {"co_lo": tot["co_lo"], **uoc_gio_gpu(tot["cau_moi_gio"], dac_ta)} if tot else None)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    tong = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None
    kq["gpu"] = {"ten": gpu, "tong_bo_nho_GiB": tong}
    kq["tham_so"] = {"lop": lop, "alpha_can_thiep": ALPHA_LON_NHAT, "so_cau": len(cau), "co_lo": co_lo,
                     "max_new_tokens": a.max_new_tokens, "eos": list(eos)}
    NG.ghi_json(thu_muc / "toc_do.json", kq)
    NG.ghi_json(thu_muc / "meta.json", {"buoc": "3", "viec": "đo tốc độ", "dac_ta": dac_ta.ten,
                                        "moi_truong": NG.thong_tin_moi_truong(),
                                        "mo_hinh": NG.thong_tin_mo_hinh(model, tok, dac_ta),
                                        "huong": str(a.huong), "huong_sha256": NG.sha256_file(a.huong),
                                        "huong_chinh_thuc": bool(h.get("chinh_thuc", False)),
                                        "backend_lop_tuyen_tinh": backend_lop_tuyen_tinh(model),
                                        "phan_bo_thiet_bi": pb})
    if tot:
        log(f"Lô nhanh nhất (chế độ can thiệp): {tot['co_lo']}, {tot['cau_moi_gio']:.0f} câu/giờ; "
            f"ước giờ GPU: {kq['uoc_gio_gpu_theo_lo_nhanh_nhat_che_do_can_thiep']['gio_gpu']}")
    log(f"Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
