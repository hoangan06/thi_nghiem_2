"""Bước 3, bản thu nhỏ của phần 2 kiểm pipeline trên một mô hình chính (Nhap, mục Kiểm pipeline, đoạn cuối).

Chạy (sau buoc3_dung_huong.py):
    python scripts/buoc3_kiem_hook.py --mo_hinh_dac_ta qwen35_9b --huong results/buoc3/qwen35_9b/<lượt>/huong.pt

Theo Nhap (chốt 27/10/2026):
- chạy ở bf16 và float32, tắt TF32 ở lượt float32; (ii) phải khớp tuyệt đối ở cả hai; (i), (iii) và kiểm âm lấy
  kết luận ở float32 theo cùng luật ba mức như bước 1 (tn2.kiem_hook); số bf16 báo kèm để tham khảo;
- quy mô như bước 1: 8 câu mỗi kiểu, 64 token sinh, α = 0,4, nhiễu đo trên 128 câu mỗi kiểu (lô 8), kiểm âm như
  bước 1 (lệch vùng một token; bỏ cộng khi sinh; phương án 1 và 3; kiểu (a); lớp giữa);
- câu lấy từ tập dựng tiếng Anh: D_harmless_build cho kiểu (a), D_harm_build cho kiểu (b); 8 câu đầu để kiểm, cả
  128 câu để đo nhiễu; hướng là hướng tiếng Anh (r̂_en, s_ℓ) dựng ở bước 3 (file huong.pt);
- ba lớp có cả hai loại khối: Qwen3.5-9B 8, 15, 24; Gemma 4 12B 12, 23, 36 (lớp giữa là full attention).
Bản float32 nạp lại cùng checkpoint với device_map={"": 0}; thiếu bộ nhớ thì báo lỗi ngay khi nạp. Lượt float32 chạy
lớp attention tuyến tính của Qwen3.5 (Gated DeltaNet, conv1d) bằng bản PyTorch tham chiếu của transformers thay
cho kernel flash-linear-attention / causal-conv1d, vì kernel Triton dùng TF32 bất kể cờ của PyTorch (Ân duyệt Q4,
27/10/2026); bf16 giữ backend tối ưu. Backend thực dùng ghi trong meta.json (bf16) và ket_qua_fp32.json (float32). Hạn chế: lượt float32
kiểm cơ chế hook bằng bản tham chiếu, không chứng nhận kernel tối ưu dùng ở bf16.
Chẩn đoán thêm (không vào tiêu chí): mỗi kiểu chạy lượt không hook hai lần và so với nhau (khong_hook_lap_lai).
Trong lượt float32, trước phép kiểm: quét attention theo độ dài (tn2.kiem_attention, m = 2…320 và lô đệm, ngưỡng
0,5 logit; sửa sau kết quả bước 3, Ân duyệt 29/10/2026), ghi quet_attention_fp32.json; không đạt thì cả lượt không đạt.
Mã thoát: 0 đạt; 1 không đạt (kể cả file hướng không dựng bằng kiểu attention của đặc tả); 2 khối sau câu hỏi
lệch đặc tả; 3 mô hình không nằm trọn một GPU;
4 "cần xem" (KHÔNG phải đạt: đọc ket_luan.json và chan_doan_vi_tri theo luật viết trước).
"""
from __future__ import annotations

import argparse
import sys
import time

from _chung import GOC, NhatKy, chon_dac_ta, device_map_mot_gpu, ten_luot_mac_dinh, thiet_lap_so_hoc

import torch

from tn2 import nguon_goc as NG
from tn2.buoc3.nap_kiem import kiem_khoi_tokenizer
from tn2.du_lieu import nap_tap_dung
from tn2 import kiem_attention as KA
from tn2.kiem_hook import ALPHA_KIEM, MAX_NEW_TOKENS_KIEM, DauVaoKiemHook, chay_kiem_hook, ket_luan_hai_kieu_so
from tn2.mo_hinh import (backend_lop_tuyen_tinh, ban_tham_chieu_pytorch, eos_tu_mo_hinh, nap_mo_hinh,
                         phan_bo_thiet_bi)

SO_CAU = 8
SO_CAU_NHIEU = 128


def _bao(log, ten, t):
    log(f"[{ten}] (ii) α=0 khớp tuyệt đối: {t['ii_dat']}")
    log(f"[{ten}] nhiễu (i) {t['nhieu_i']:.4g}, nhiễu (iii) {t['nhieu_iii']:.4g} → ngưỡng {t['nguong']:.4g}")
    log(f"[{ten}] (i) lệch lớn nhất {t['lech_lon_nhat_i']:.4g} → {t['trang_thai_i']}")
    log(f"[{ten}] (iii) lệch lớn nhất {t['lech_lon_nhat_iii']:.4g} → {t['trang_thai_iii']}")
    log(f"[{ten}] kiểm âm (ngưỡng đạt = 5 × ngưỡng): "
        f"{[(k, round(v['lech_lon_nhat'], 4), v['dat']) for k, v in t['kiem_am'].items()]}")
    log(f"[{ten}] số lượt α≠0 làm đổi token so với không hook: {t['so_luot_alpha_doi_token']}/{t['so_luot_alpha']}")
    log(f"[{ten}] chẩn đoán: hai lượt không hook khớp tuyệt đối: {t.get('khong_hook_lap_lai_khop')}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh_dac_ta", required=True)
    p.add_argument("--huong", required=True, help="huong.pt do buoc3_dung_huong.py ghi")
    p.add_argument("--mo_hinh", default=None)
    p.add_argument("--revision", default=None, help="mặc định: commit HF ghi trong file hướng")
    p.add_argument("--ra", default=str(GOC / "results" / "buoc3"))
    p.add_argument("--ten_luot", default=None)
    p.add_argument("--so_cau", type=int, default=SO_CAU)
    p.add_argument("--so_cau_nhieu", type=int, default=SO_CAU_NHIEU)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS_KIEM)
    p.add_argument("--bo_fp32", action="store_true", help="lượt thử: chỉ chạy bf16 (không cho kết luận)")
    p.add_argument("--quet_den", type=int, default=KA.M_TOI_DA, help="độ dài lớn nhất của phép quét attention fp32")
    a = p.parse_args(argv)
    dac_ta = chon_dac_ta(a.mo_hinh_dac_ta)
    chinh_thuc = (a.so_cau == SO_CAU and a.so_cau_nhieu == SO_CAU_NHIEU and a.quet_den == KA.M_TOI_DA
                  and a.max_new_tokens == MAX_NEW_TOKENS_KIEM
                  and not a.bo_fp32)
    # Đọc file hướng TRƯỚC khi đặt tên thư mục: hướng thử → thư mục kiem_hook_thu_… (rà lại pipeline 28/10/2026).
    h = torch.load(a.huong, map_location="cpu")
    if h.get("dac_ta") != dac_ta.ten:
        print(f"DỪNG: file hướng của mô hình {h.get('dac_ta')}, không phải {dac_ta.ten}.")
        return 1
    # Hướng phải dựng bằng đúng kiểu attention của đặc tả (rà soát 29/10/2026, R1 B.3, R2 B2): dừng trước khi nạp
    # mô hình, để không chạy 4d với hướng cũ (vd. hướng Gemma dựng bằng sdpa ngày 28/10).
    loi_attn = KA.loi_kieu_attention_huong(h, NG.sha256_file(a.huong), dac_ta)
    if loi_attn:
        print(f"DỪNG: {loi_attn}.")
        return 1
    huong_thu = not h.get("chinh_thuc", False)
    if huong_thu:
        chinh_thuc = False
    thu_muc = NG.thu_muc_moi(f"{a.ra}/{dac_ta.ten}",
                             a.ten_luot or ten_luot_mac_dinh("kiem_hook" if chinh_thuc else "kiem_hook_thu"))
    log = NhatKy(thu_muc / "nhat_ky.txt")
    log(f"Thư mục kết quả: {thu_muc}; lượt chính thức: {chinh_thuc}")
    bat_dau = time.time()
    so_hoc = thiet_lap_so_hoc()
    revision = a.revision or h.get("commit_hf")
    if huong_thu:
        log("CHÚ Ý: file hướng từ lượt dựng hướng thử (không chính thức) → lượt này không chính thức.")
    if h.get("canh_bao_dem"):
        log("CHÚ Ý: file hướng có cảnh báo đệm (trạng thái 'cần xem'). Phần 2 vẫn kiểm được cơ chế hook; hướng chưa "
            "được dùng cho bước 4 trước khi Ân xem.")
    lop = list(dac_ta.lop_kiem_hook)
    if not lop:
        log("DỪNG: đặc tả không có lop_kiem_hook.")
        return 1
    tap = nap_tap_dung()
    cau_a, cau_b = tap.harmless_en[: a.so_cau], tap.harm_en[: a.so_cau]
    nhieu_a, nhieu_b = tap.harmless_en[: a.so_cau_nhieu], tap.harm_en[: a.so_cau_nhieu]

    dm = device_map_mot_gpu()
    model, tok = nap_mo_hinh(dac_ta, a.mo_hinh, revision=revision, device_map=dm)
    pb = phan_bo_thiet_bi(model, can_gpu=dm is not None)        # chỉ đòi GPU khi chính script xin GPU
    log(f"Phân bố thiết bị (bf16): {pb['so_tensor_theo_thiet_bi']}; một GPU: {pb['mot_thiet_bi']}")
    if not pb["mot_thiet_bi"]:
        log("DỪNG: mô hình không nằm trọn trên một GPU.")
        return 3
    c_nap = NG.thong_tin_mo_hinh(model, tok, dac_ta)["commit_hf"]
    if (chinh_thuc and (not c_nap or c_nap != h.get("commit_hf"))) or (
            h.get("commit_hf") and c_nap not in (None, h["commit_hf"])):
        log(f"DỪNG: file hướng dựng từ commit HF {h.get('commit_hf')!r}, mô hình đang nạp có commit {c_nap!r} "
            f"(None = không xác minh được). Lượt chính thức cần đúng commit của file hướng.")
        return 1
    khoi = kiem_khoi_tokenizer(tok, dac_ta, sorted(set(cau_a + cau_b + nhieu_a + nhieu_b)))
    if not khoi["dat"]:
        NG.ghi_json(thu_muc / "khoi_sau_cau_hoi.json", khoi)
        log("DỪNG: khối sau câu hỏi lệch đặc tả.")
        return 2
    eos = eos_tu_mo_hinh(model, dac_ta)
    dv = DauVaoKiemHook(cau_hoi_a=cau_a, cau_hoi_b=cau_b, lop=lop,
                        r_hat={l: h["r_hat_en"][l].to(torch.float32) for l in lop},
                        s={l: float(h["s"][l]) for l in lop}, eos_ids=eos, alpha=ALPHA_KIEM,
                        max_new_tokens=a.max_new_tokens, cau_hoi_nhieu_a=nhieu_a, cau_hoi_nhieu_b=nhieu_b,
                        co_lo=a.so_cau, lap_lai_khong_hook=True)
    NG.ghi_json(thu_muc / "meta.json", {
        "buoc": "3", "viec": "bản thu nhỏ phần 2 — kiểm đường can thiệp", "chinh_thuc": chinh_thuc,
        "dac_ta": dac_ta.ten, "moi_truong": NG.thong_tin_moi_truong(),
        "mo_hinh": NG.thong_tin_mo_hinh(model, tok, dac_ta), "bien_moi_truong": NG.bien_moi_truong_quan_trong(),
        "huong_chinh_thuc": bool(h.get("chinh_thuc", False)),
        "huong_canh_bao_dem": bool(h.get("canh_bao_dem", False)),
        "backend_lop_tuyen_tinh_bf16": backend_lop_tuyen_tinh(model),
        "tham_so": {"lop": lop, "lop_full_attention": [l for l in lop if l in dac_ta.lop_full_attention],
                    "so_cau": a.so_cau, "so_cau_nhieu": a.so_cau_nhieu, "alpha": ALPHA_KIEM,
                    "max_new_tokens": a.max_new_tokens, "eos": list(eos), "bo_fp32": a.bo_fp32,
                    "huong": str(a.huong), "huong_sha256": NG.sha256_file(a.huong),
                    "s_tai_lop": {l: float(h["s"][l]) for l in lop}},
        "so_hoc": so_hoc, "phan_bo_thiet_bi_bf16": pb,
        "du_lieu_sha256": {f.name: NG.sha256_file(f) for f in sorted((GOC / "data" / "dung_huong").glob("*.jsonl"))},
    })

    kq16 = chay_kiem_hook(model, tok, dac_ta, dv, log)
    NG.ghi_json(thu_muc / "ket_qua_bf16.json", kq16)
    _bao(log, "bf16", kq16["tong_ket"])
    if a.bo_fp32:
        log(f"Bỏ lượt float32 (lượt thử). Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
        return 1

    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trong, tong = torch.cuda.mem_get_info()
        log(f"Bộ nhớ GPU trước khi nạp float32: trống {trong / 2**30:.1f} / {tong / 2**30:.1f} GiB")
    model, _ = nap_mo_hinh(dac_ta, a.mo_hinh, dtype=torch.float32, revision=revision, device_map=dm)
    pb32 = phan_bo_thiet_bi(model, can_gpu=dm is not None)
    log(f"Phân bố thiết bị (float32): {pb32['so_tensor_theo_thiet_bi']}; một GPU: {pb32['mot_thiet_bi']}")
    if not pb32["mot_thiet_bi"]:
        log("DỪNG: bản float32 không nằm trọn trên một GPU.")
        return 3
    with ban_tham_chieu_pytorch(model) as bk:
        log(f"Backend lớp attention tuyến tính, float32: trước {bk.truoc} → trong lượt {bk.sau}")
        # Quét attention theo độ dài ở float32 (sửa sau kết quả bước 3; Ân duyệt Q-E3 ngày 29/10/2026): không đạt
        # thì bản thu nhỏ phần 2 không đạt. Không thay luật ba mức.
        try:
            X = KA.chuoi_quet(tok, dac_ta, tap.harmless_en + tap.harm_en, a.quet_den + 64)
            qa32 = KA.quet_attention(model, X, tok.pad_token_id, fp32=True, m_toi_da=a.quet_den)
        except Exception as e:                                      # noqa: BLE001
            qa32 = {"trang_thai": "khong_dat", "loi": f"{type(e).__name__}: {e}"}
        NG.ghi_json(thu_muc / "quet_attention_fp32.json", qa32)
        log("Quét attention float32: " + (KA.tom_tat(qa32) if "loi" not in qa32 else f"LỖI {qa32['loi']}"))
        kq32 = chay_kiem_hook(model, tok, dac_ta, dv, log)
    kq32["backend_lop_tuyen_tinh"] = {"truoc_khi_chuyen": bk.truoc, "dung_trong_luot": bk.sau}
    NG.ghi_json(thu_muc / "ket_qua_fp32.json", kq32)
    _bao(log, "fp32", kq32["tong_ket"])
    if torch.cuda.is_available():
        log(f"Bộ nhớ GPU đỉnh (float32): {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    kl = ket_luan_hai_kieu_so(kq16, kq32)
    kl["quet_attention_fp32"] = qa32["trang_thai"]
    if qa32["trang_thai"] != "dat":
        kl["dat"] = False
        kl["trang_thai"] = "khong_dat"
    NG.ghi_json(thu_muc / "ket_luan.json", kl)
    log(f"Kết luận bản thu nhỏ phần 2: {kl}")
    log(f"Trạng thái: {kl['trang_thai']}. Xong sau {(time.time() - bat_dau) / 60:.1f} phút.")
    return {"dat": 0, "can_xem": 4}.get(kl["trang_thai"], 1)


if __name__ == "__main__":
    sys.exit(main())
