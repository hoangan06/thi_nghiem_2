"""Bước 3, nạp và kiểm một mô hình chính (cần GPU; chạy sau buoc3_kiem_truoc.py).

Chạy:
    python scripts/buoc3_nap_kiem.py --mo_hinh_dac_ta qwen35_9b

Kiểm (Nhap, Trình tự chạy bước 3; Ghi chú kỹ thuật hai mô hình chính, mục 5, việc 1, 2, 3, 7):
1. Phân bố thiết bị theo từng tensor (tham số và buffer): mô hình nằm trọn trên một GPU.
2. Cấu trúc: lớp mô hình, đường dẫn tới các lớp decoder, kiểu khối trong từng lớp so với danh sách full attention
   của đặc tả và với layer_types của config; Gemma 4: 48 giá trị layer_scalar; embedding đầu vào.
3. Khối sau câu hỏi trên tokenizer đã nạp (như kiem_truoc, chạy lại cho chắc).
4. Bộ đếm vị trí của hook: mỗi lượt forward đi qua embedding đầu vào đúng một lần (tn2.buoc3.nap_kiem).
5. Sinh thử không gắn hook trên 8 câu đầu của mỗi tập dựng (32 câu, cả hai ngôn ngữ), 256 token: đếm câu còn
   rò khối thinking (Nhap Dựng hướng điểm 5) và câu dừng ở token kết thúc. Câu trả lời lưu ra sinh_thu.jsonl;
   không tính tỉ lệ từ chối (không dùng để phân tích).
Mã thoát: 0 đạt; 1 có phép kiểm không đạt (xem nap_kiem.json); 3 mô hình không nằm trọn trên một GPU.
Rò thinking vượt 5 % số câu sinh thử (cùng ngưỡng điều kiện 3 của Nhap 7.7) tính là không đạt; không câu sinh
thử nào dừng ở token kết thúc chính (Qwen3.5 <|im_end|>, Gemma 4 <turn|>) cũng tính là không đạt (cần xem lại).
"""
from __future__ import annotations

import argparse
import sys
import time

from _chung import GOC, NhatKy, chon_dac_ta, device_map_mot_gpu, ten_luot_mac_dinh

import torch

from tn2 import nguon_goc as NG
from tn2.buoc3.nap_kiem import dem_ro_thinking, kiem_cau_truc, kiem_goi_embedding, kiem_khoi_tokenizer
from tn2.du_lieu import nap_tap_dung
from tn2.mo_hinh import backend_lop_tuyen_tinh, eos_tu_mo_hinh, nap_mo_hinh, phan_bo_thiet_bi
from tn2.sinh import sinh_theo_lo

NGUONG_RO_THINKING = 0.05
SO_CAU_SINH_THU = 8
MAX_NEW_TOKENS = 256      # Nhap 3.5


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh_dac_ta", required=True)
    p.add_argument("--mo_hinh", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--ra", default=str(GOC / "results" / "buoc3"))
    p.add_argument("--ten_luot", default=None)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--so_cau_sinh_thu", type=int, default=SO_CAU_SINH_THU)
    a = p.parse_args(argv)
    dac_ta = chon_dac_ta(a.mo_hinh_dac_ta)
    chinh_thuc = a.max_new_tokens == MAX_NEW_TOKENS and a.so_cau_sinh_thu == SO_CAU_SINH_THU
    thu_muc = NG.thu_muc_moi(f"{a.ra}/{dac_ta.ten}", a.ten_luot or ten_luot_mac_dinh("nap_kiem"))
    log = NhatKy(thu_muc / "nhat_ky.txt")
    bat_dau = time.time()
    dm = device_map_mot_gpu()
    model, tok = nap_mo_hinh(dac_ta, a.mo_hinh, revision=a.revision, device_map=dm)
    log(f"Đã nạp {type(model).__name__} sau {(time.time() - bat_dau) / 60:.1f} phút")
    pb = phan_bo_thiet_bi(model, can_gpu=dm is not None)        # chỉ đòi GPU khi chính script xin GPU
    log(f"Phân bố thiết bị: {pb['so_tensor_theo_thiet_bi']}; một GPU: {pb['mot_thiet_bi']}")
    ket: dict = {"phan_bo_thiet_bi": pb}
    meta = {"buoc": "3", "viec": "nạp và kiểm", "chinh_thuc": chinh_thuc, "dac_ta": dac_ta.ten,
            "moi_truong": NG.thong_tin_moi_truong(), "mo_hinh": NG.thong_tin_mo_hinh(model, tok, dac_ta),
            "bien_moi_truong": NG.bien_moi_truong_quan_trong(),
            "backend_lop_tuyen_tinh": backend_lop_tuyen_tinh(model),
            "du_lieu_sha256": {f.name: NG.sha256_file(f) for f in sorted((GOC / "data" / "dung_huong").glob("*.jsonl"))}}
    NG.ghi_json(thu_muc / "meta.json", meta)
    if not pb["mot_thiet_bi"]:
        NG.ghi_json(thu_muc / "nap_kiem.json", ket)
        log("DỪNG: mô hình không nằm trọn trên một GPU.")
        return 3
    ct = kiem_cau_truc(model, dac_ta)
    ket["cau_truc"] = ct
    log(f"Cấu trúc: {ct['lop_mo_hinh']}, lớp {ct['lop_decoder']} tại {ct['duong_dan_lop']}, {ct['so_lop']} lớp; "
        f"full attention theo config {ct['full_attention_config']}, theo khối {ct['full_attention_theo_khoi']}, "
        f"đặc tả {ct['full_attention_dac_ta']} → đạt {ct['dat']}")
    if ct["layer_scalar"] is not None:
        log(f"layer_scalar: {[round(x, 4) for x in ct['layer_scalar']]}")
    tap = nap_tap_dung()
    khoi = kiem_khoi_tokenizer(tok, dac_ta, tap.harm_en + tap.harm_vi + tap.harmless_en + tap.harmless_vi)
    ket["khoi_sau_cau_hoi"] = khoi
    log(f"Khối sau câu hỏi: {khoi['khoi_theo_so_cau']} → đạt {khoi['dat']}")
    eos = eos_tu_mo_hinh(model, dac_ta)
    ge = kiem_goi_embedding(model, tok, dac_ta, tap.harmless_en[:4], 8, eos)
    ket["goi_embedding"] = ge
    log(f"Bộ đếm vị trí: {ge} → đạt {ge['dat']}")

    n = a.so_cau_sinh_thu
    cau, nhan = [], []
    for ten in ("harm_en", "harm_vi", "harmless_en", "harmless_vi"):
        cau += getattr(tap, ten)[:n]
        nhan += [ten] * n
    t0 = time.time()
    sinh = sinh_theo_lo(model, tok, dac_ta, cau, len(cau), a.max_new_tokens, eos)
    dt = time.time() - t0
    ro = dem_ro_thinking([r["van_ban"] for r in sinh], dac_ta)
    theo_eos: dict[str, int] = {}
    for r in sinh:
        if r["cham_eos"]:
            k = str(r["token"][-1])
            theo_eos[k] = theo_eos.get(k, 0) + 1
    eos_chinh = dac_ta.eos_du_kien[0][0] if dac_ta.eos_du_kien else eos[0]
    ket["sinh_thu"] = {"so_cau": len(sinh), "max_new_tokens": a.max_new_tokens, "thoi_gian_giay": dt,
                       "so_cau_cham_eos": sum(r["cham_eos"] for r in sinh), "so_cau_theo_eos": theo_eos,
                       "eos_chinh": eos_chinh, "ro_thinking": ro, "eos": list(eos)}
    NG.ghi_jsonl(thu_muc / "sinh_thu.jsonl", [{"tap": t, **r} for t, r in zip(nhan, sinh)])
    log(f"Sinh thử {len(sinh)} câu × {a.max_new_tokens} token trong {dt:.0f} giây: "
        f"{ket['sinh_thu']['so_cau_cham_eos']} câu dừng ở token kết thúc; rò thinking {ro['so_cau_ro']} câu")
    for t in ("harm_en", "harm_vi", "harmless_en", "harmless_vi"):
        r = sinh[nhan.index(t)]
        log(f"  [{t}] {r['cau_hoi'][:80]!r} → {r['van_ban'][:200]!r}")
    dat_ro = (ro["ti_le"] or 0.0) <= NGUONG_RO_THINKING
    # Token kết thúc (phép kiểm nhanh; Ân duyệt cách kết hợp ý hai reviewer, 27/10/2026): ít nhất một câu sinh thử
    # dừng ở token kết thúc CHÍNH (Qwen3.5 <|im_end|>, Gemma 4 <turn|>) trong 256 token; ghi phân bố theo từng id.
    # Không đòi mọi câu (câu dài có thể chưa xong). Không câu nào dừng ở id chính → xem lại danh sách eos và vài câu
    # trong sinh_thu.jsonl; đạt cũng chưa chứng minh mọi id trong danh sách đều được nhận.
    dat_eos = theo_eos.get(str(eos_chinh), 0) > 0
    ket["dat_thanh_phan"] = {"cau_truc": ct["dat"], "khoi_sau_cau_hoi": khoi["dat"], "goi_embedding": ge["dat"],
                             "ro_thinking_toi_da_5_phan_tram": dat_ro, "co_cau_cham_eos_chinh": dat_eos}
    ket["dat"] = bool(ct["dat"] and khoi["dat"] and ge["dat"] and dat_ro and dat_eos)
    NG.ghi_json(thu_muc / "nap_kiem.json", ket)
    log(f"ĐẠT: {ket['dat']} (cấu trúc {ct['dat']}, khối {khoi['dat']}, bộ đếm vị trí {ge['dat']}, "
        f"rò thinking ≤ 5 % {dat_ro}, có câu dừng ở token kết thúc chính {dat_eos}; theo id {theo_eos}). "
        f"Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
    return 0 if ket["dat"] else 1


if __name__ == "__main__":
    sys.exit(main())
