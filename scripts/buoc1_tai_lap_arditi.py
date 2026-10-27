"""Bước 1, phần 1: tái lập Arditi et al. (2024) trên Llama-3-8B-Instruct.

Chạy (WSL/Linux, trong thư mục gốc của kho):
    python scripts/buoc1_tai_lap_arditi.py --mo_hinh meta-llama/Meta-Llama-3-8B-Instruct

Lượt thử nhỏ (không phải lượt chính thức, chỉ để xem script chạy thông):
    python scripts/buoc1_tai_lap_arditi.py --gioi_han 8 --max_new_tokens 32

Kết quả: results/buoc1/<ten_luot>/ (xem docs/RUNBOOK_BUOC_1.md).
"""
from __future__ import annotations

import argparse
import sys
import time

from _chung import GOC, NhatKy, ten_luot_mac_dinh

import torch

from tn2 import nguon_goc as NG
from tn2.buoc1 import tai_lap_arditi as A
from tn2.mo_hinh import LLAMA3_8B, nap_mo_hinh, phan_bo_thiet_bi, tim_cac_lop


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh", default=LLAMA3_8B.hf_id, help="thư mục cục bộ hoặc id Hugging Face")
    p.add_argument("--revision", default=None)
    p.add_argument("--ra", default=str(GOC / "results" / "buoc1"))
    p.add_argument("--ten_luot", default=None)
    p.add_argument("--gioi_han", type=int, default=None, help="chỉ lấy N câu đầu mỗi tập (lượt thử nhỏ)")
    p.add_argument("--max_new_tokens", type=int, default=A.MAX_NEW_TOKENS)
    a = p.parse_args(argv)

    chinh_thuc = a.gioi_han is None and a.max_new_tokens == A.MAX_NEW_TOKENS
    thu_muc = NG.thu_muc_moi(a.ra, a.ten_luot or ten_luot_mac_dinh("arditi" if chinh_thuc else "arditi_thu"))
    log = NhatKy(thu_muc / "nhat_ky.txt")
    log(f"Thư mục kết quả: {thu_muc}; lượt chính thức: {chinh_thuc}")
    bat_dau = time.time()

    du_lieu = A.nap_du_lieu(gioi_han=a.gioi_han)
    file_du_lieu = sorted(A.DU_LIEU_MAC_DINH.glob("*.jsonl"))
    model, tok = nap_mo_hinh(LLAMA3_8B, a.mo_hinh, revision=a.revision)
    so_lop = len(tim_cac_lop(model))
    NG.ghi_json(thu_muc / "meta.json", {
        "buoc": "1", "phan": "1 — tái lập Arditi et al. (2024)", "chinh_thuc": chinh_thuc,
        "moi_truong": NG.thong_tin_moi_truong(), "mo_hinh": NG.thong_tin_mo_hinh(model, tok, LLAMA3_8B),
        "bien_moi_truong": NG.bien_moi_truong_quan_trong(), "phan_bo_thiet_bi": phan_bo_thiet_bi(model),
        "tham_so": {"gioi_han": a.gioi_han, "max_new_tokens": a.max_new_tokens, "co_lo_diem": A.CO_LO_DIEM,
                    "co_lo_sinh": A.CO_LO_SINH, "kl_threshold": A.KL_THRESHOLD,
                    "induce_threshold": A.INDUCE_THRESHOLD, "prune": A.PRUNE_LAYER_PERCENTAGE,
                    "token_tu_choi": list(A.TOKEN_TU_CHOI_LLAMA3), "eos": [128001]},
        "du_lieu_sha256": {f.name: NG.sha256_file(f) for f in file_du_lieu},
    })

    # 0. Khuôn: id của ta phải trùng id theo cách tác giả mã hoá, cho mọi câu dùng trong lượt này.
    tat_ca = sorted(set(du_lieu.harmful_train + du_lieu.harmless_train + du_lieu.harmful_val + du_lieu.harmless_val
                        + du_lieu.harmless_test + du_lieu.jailbreakbench))
    kk = A.kiem_khuon(tok, LLAMA3_8B, tat_ca)
    NG.ghi_json(thu_muc / "kiem_khuon.json", kk)
    log(f"Kiểm khuôn: {kk['so_cau_lech']}/{kk['so_cau']} câu lệch; khối sau câu hỏi {kk['token_khoi']}")
    if not kk["dat"]:
        log("DỪNG: khuôn chat không trùng cách tác giả mã hoá.")
        return 2

    # 1. Lọc theo điểm từ chối (filter_train, filter_val).
    du_lieu_loc, loc = A.loc_theo_diem(model, tok, LLAMA3_8B, du_lieu)
    NG.ghi_json(thu_muc / "loc.json", loc)
    log("Lọc: " + ", ".join(f"{k} {v['truoc']}→{v['sau']}" for k, v in loc.items()))

    # 2. Hướng ứng viên ở đầu vào mọi khối, 5 vị trí cuối.
    uv = A.huong_ung_vien(model, tok, LLAMA3_8B, du_lieu_loc.harmful_train, du_lieu_loc.harmless_train,
                          kk["so_token_khoi"])
    torch.save(uv, thu_muc / "huong_ung_vien.pt")
    log(f"Hướng ứng viên: {tuple(uv.shape)}")

    # 3. Chọn hướng.
    bang, chon = A.chon_huong(model, tok, LLAMA3_8B, uv, du_lieu_loc.harmful_val, du_lieu_loc.harmless_val, log)
    NG.ghi_json(thu_muc / "bang_chon_huong.json", bang)
    NG.ghi_json(thu_muc / "huong_chon.json", chon)
    log(f"Chọn: vị trí {chon['vi_tri']}, lớp {chon['lop']} (bypass {chon['diem_chieu_bo']:.3f}, "
        f"induce {chon['diem_cong']:.3f}, KL {chon['kl']:.4f})")
    huong = uv[chon["vi_tri"], chon["lop"]]
    torch.save(huong, thu_muc / "huong_chon.pt")

    # 4. Đánh giá bằng sinh văn bản.
    sinh = A.danh_gia(model, tok, LLAMA3_8B, huong, chon["lop"], du_lieu_loc, a.max_new_tokens)
    for k, v in sinh.items():
        NG.ghi_jsonl(thu_muc / "sinh" / f"{k}.jsonl", v)
    tl = A.ti_le(sinh)
    tc = A.tieu_chi(chon["lop"], so_lop, tl)
    ce = A.so_cau_cham_eos(sinh)
    te = A.ti_le_truoc_eot(sinh, tok)
    NG.ghi_json(thu_muc / "ket_qua.json", {"ti_le_tu_choi": tl, "tieu_chi": tc, "chon": chon,
                                           "so_cau_cham_eos_128001": ce,
                                           "chan_doan_truoc_eot_id": te, "chinh_thuc": chinh_thuc})
    log(f"Tỉ lệ từ chối: {tl}")
    log(f"Số câu dừng ở <|end_of_text|> trước {a.max_new_tokens} token: {ce}")
    log("Chẩn đoán, tỉ lệ từ chối chỉ trên phần trước <|eot_id|> đầu tiên: "
        + ", ".join(f"{k} {v['ti_le_tu_choi']:.2f} ({v['so_cau_co_eot']}/{v['so_cau']} câu có eot)" for k, v in te.items()))
    log(f"Đạt cả bốn tiêu chí: {tc['dat_ca_bon']}")

    # 5. Chẩn đoán so với sản phẩm của tác giả (không có ngưỡng).
    cd = A.chan_doan(uv, bang, chon, A.DU_LIEU_MAC_DINH / "tac_gia_llama3")
    NG.ghi_json(thu_muc / "chan_doan.json", cd)
    log(f"Chẩn đoán: trùng ứng viên của tác giả = {cd.get('trung_ung_vien_tac_gia')}; "
        f"cosine tại ứng viên chọn = {cd.get('cosine_voi_mean_diffs_tac_gia', {}).get('tai_ung_vien_chon')}")
    log(f"Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
    return 0 if tc["dat_ca_bon"] else 1


if __name__ == "__main__":
    sys.exit(main())
