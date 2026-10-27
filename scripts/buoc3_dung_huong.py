"""Bước 3, dựng hướng và s_ℓ cho một mô hình chính (Nhap, mục Dựng hướng điểm 1–8; mục 3.3).

Chạy:
    python scripts/buoc3_dung_huong.py --mo_hinh_dac_ta qwen35_9b

Trình tự:
1. Nạp bf16 lên một GPU, kiểm phân bố thiết bị (mã 3 nếu không nằm trọn một GPU).
2. Phép chẩn đoán position_ids / đệm (tn2.buoc3.nap_kiem.do_position_ids) trên một lô 32 câu đầu của
   D_harm_build tiếng Anh; chỉ ghi lại, không quyết định gì. Cảnh báo đệm bật → hướng ở trạng thái "cần xem".
3. Lấy kích hoạt ở đầu ra mọi khối, tại vị trí lấy và tại token cuối thật, cho 4 × 128 câu dựng (lô 32), LUÔN
   truyền position_ids = cumsum(mask) − 1 như generate (Ân duyệt Q1, 27/10/2026).
4. Dựng r_ℓ, r̂_ℓ, ‖r_ℓ‖ cho từng ngôn ngữ; s_ℓ trên 512 câu gộp; độ lớn tương đối; chia đôi; cosine hai vị trí;
   cosine en–vi (ước lượng điểm, chỉ mô tả).
Ghi ra results/buoc3/<mô hình>/<ten_luot>/:
  huong.pt                 — r̂, ‖r‖, s, độ lớn tương đối, hướng ở token cuối thật, chia đôi, cosine (CPU, float64);
                             kèm dac_ta, commit_hf, position_ids_theo_mat_na, chinh_thuc, canh_bao_dem, trang_thai
  kich_hoat_vi_tri_lay.pt  — kích hoạt từng câu ở vị trí lấy, mọi lớp ([128, L, d] × 4, kiểu số của mô hình)
  tom_tat.json, do_position_ids.json, meta.json, nhat_ky.txt
Mã thoát: 0 xong; 4 xong nhưng "cần xem" (cảnh báo đệm bật: KHÔNG dùng hướng này cho bước 4 trước khi Ân xem
do_position_ids.json); 3 mô hình không nằm trọn một GPU; 1 lỗi khác.
"""
from __future__ import annotations

import argparse
import sys
import time

from _chung import GOC, NhatKy, chon_dac_ta, device_map_mot_gpu, ten_luot_mac_dinh

import torch

from tn2 import nguon_goc as NG
from tn2.buoc3.dung_huong import SEED_CHIA_DOI, dung_huong
from tn2.buoc3.nap_kiem import do_position_ids
from tn2.du_lieu import nap_tap_dung
from tn2.khuon import ma_hoa
from tn2.mo_hinh import backend_lop_tuyen_tinh, nap_mo_hinh, phan_bo_thiet_bi

CO_LO = 32
SO_CAU_DO_POSITION_IDS = 32


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh_dac_ta", required=True)
    p.add_argument("--mo_hinh", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--ra", default=str(GOC / "results" / "buoc3"))
    p.add_argument("--ten_luot", default=None)
    p.add_argument("--co_lo", type=int, default=CO_LO)
    p.add_argument("--gioi_han", type=int, default=None, help="lượt thử: số câu mỗi file (không chính thức)")
    a = p.parse_args(argv)
    dac_ta = chon_dac_ta(a.mo_hinh_dac_ta)
    chinh_thuc = a.gioi_han is None and a.co_lo == CO_LO
    thu_muc = NG.thu_muc_moi(f"{a.ra}/{dac_ta.ten}",
                             a.ten_luot or ten_luot_mac_dinh("dung_huong" if chinh_thuc else "dung_huong_thu"))
    log = NhatKy(thu_muc / "nhat_ky.txt")
    log(f"Thư mục kết quả: {thu_muc}; lượt chính thức: {chinh_thuc}")
    bat_dau = time.time()
    tap = nap_tap_dung(so_cau=None if a.gioi_han else 128)
    if a.gioi_han:
        g = a.gioi_han
        tap.harm_en, tap.harm_vi = tap.harm_en[:g], tap.harm_vi[:g]
        tap.harmless_en, tap.harmless_vi = tap.harmless_en[:g], tap.harmless_vi[:g]
        tap.id_harm, tap.id_harmless = tap.id_harm[:g], tap.id_harmless[:g]
    dm = device_map_mot_gpu()
    model, tok = nap_mo_hinh(dac_ta, a.mo_hinh, revision=a.revision, device_map=dm)
    pb = phan_bo_thiet_bi(model, can_gpu=dm is not None)        # chỉ đòi GPU khi chính script xin GPU
    log(f"Phân bố thiết bị: {pb['so_tensor_theo_thiet_bi']}; một GPU: {pb['mot_thiet_bi']}")
    if not pb["mot_thiet_bi"]:
        log("DỪNG: mô hình không nằm trọn trên một GPU.")
        return 3
    meta = {"buoc": "3", "viec": "dựng hướng và s_ℓ", "chinh_thuc": chinh_thuc, "dac_ta": dac_ta.ten,
            "moi_truong": NG.thong_tin_moi_truong(), "mo_hinh": NG.thong_tin_mo_hinh(model, tok, dac_ta),
            "bien_moi_truong": NG.bien_moi_truong_quan_trong(), "phan_bo_thiet_bi": pb,
            "backend_lop_tuyen_tinh": backend_lop_tuyen_tinh(model),
            "tham_so": {"co_lo": a.co_lo, "gioi_han": a.gioi_han, "seed_chia_doi": SEED_CHIA_DOI,
                        "so_cau_do_position_ids": SO_CAU_DO_POSITION_IDS},
            "du_lieu_sha256": {f.name: NG.sha256_file(f) for f in sorted((GOC / "data" / "dung_huong").glob("*.jsonl"))}}
    NG.ghi_json(thu_muc / "meta.json", meta)

    vt = ma_hoa(tok, [tap.harm_en[0]], dac_ta).vi_tri_lay
    if dac_ta.vi_tri_lay_du_kien is not None and vt != dac_ta.vi_tri_lay_du_kien:
        log(f"LỖI: vị trí lấy đo từ tokenizer {vt} khác đặc tả {dac_ta.vi_tri_lay_du_kien} (chạy buoc3_kiem_truoc trước).")
        return 1
    cau_do = tap.harm_en[:SO_CAU_DO_POSITION_IDS]
    dp = do_position_ids(model, tok, dac_ta, cau_do, vt)
    NG.ghi_json(thu_muc / "do_position_ids.json", dp)
    tb, tc = dp["trung_binh"], dp["theo_cau"]
    log(f"Chẩn đoán position_ids trên {dp['so_cau']} câu (đệm {dp['so_token_dem_moi_cau']}), pipeline dùng cách B: "
        f"lệch tương đối trung bình B–C {tb['lech_B_C_lon_nhat']:.3g} (A–C {tb['lech_A_C_lon_nhat']:.3g}), theo câu "
        f"B–C {tc['lech_B_C_lon_nhat']:.3g} (A–C {tc['lech_A_C_lon_nhat']:.3g}); cosine B–C nhỏ nhất "
        f"{tb['cos_B_C_nho_nhat']:.8f} (theo câu {tc['cos_B_C_nho_nhat']:.6f})")
    if dp["canh_bao_dem"]:
        log(f"CẢNH BÁO ĐỆM: cosine B–C {tb['cos_B_C_nho_nhat']:.6f} < {dp['nguong_cos_dem']}. Hướng ở trạng thái "
            f"'cần xem': KHÔNG dùng cho bước 4 trước khi Ân xem do_position_ids.json (runbook mục 6).")

    kq = dung_huong(model, tok, dac_ta, tap, co_lo=a.co_lo, position_ids_theo_mat_na=True, bao_tien_do=log)
    t = kq["tensor"]
    t["dac_ta"] = dac_ta.ten
    t["commit_hf"] = meta["mo_hinh"]["commit_hf"]
    t["position_ids_theo_mat_na"] = True
    t["chinh_thuc"] = chinh_thuc
    t["canh_bao_dem"] = dp["canh_bao_dem"]
    t["trang_thai"] = "can_xem" if dp["canh_bao_dem"] else "dat"
    torch.save(t, thu_muc / "huong.pt")
    torch.save({"dac_ta": dac_ta.ten, "vi_tri_lay": t["vi_tri_lay"], "id_harm": tap.id_harm,
                "id_harmless": tap.id_harmless, **kq["kich_hoat_vi_tri_lay"]}, thu_muc / "kich_hoat_vi_tri_lay.pt")
    tom = kq["tom_tat"]
    tom["canh_bao_dem"] = dp["canh_bao_dem"]
    tom["trang_thai"] = t["trang_thai"]
    tom["chinh_thuc"] = chinh_thuc
    NG.ghi_json(thu_muc / "tom_tat.json", tom)
    for ng in ("en", "vi"):
        x = tom[ng]
        log(f"[{ng}] ‖r‖ theo lớp: {[round(v, 3) for v in x['do_lon_tho']]}")
        log(f"[{ng}] ‖r‖ / s: {[round(v, 3) for v in x['do_lon_tuong_doi']]}")
        log(f"[{ng}] cosine chia đôi nhỏ nhất {x['cos_chia_doi_nho_nhat']}; "
            f"cosine hai vị trí nhỏ nhất {x['cos_hai_vi_tri_nho_nhat']}"
            + (f"; hướng ở token cuối thật bằng 0 ở lớp {x['lop_huong_cuoi_that_do_lon_0']}"
               if x["lop_huong_cuoi_that_do_lon_0"] else ""))
    log(f"s theo lớp: {[round(v, 2) for v in tom['s']]}")
    log(f"Trạng thái hướng: {t['trang_thai']}. Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
    return 4 if dp["canh_bao_dem"] else 0


if __name__ == "__main__":
    sys.exit(main())
