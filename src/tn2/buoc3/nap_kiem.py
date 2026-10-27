"""Các phép kiểm khi nạp một mô hình chính (bước 3; Ghi chú kỹ thuật hai mô hình chính, mục 5).

- kiem_khoi_tokenizer: khối sau câu hỏi và vị trí lấy kích hoạt đo trên tokenizer thật, so với đặc tả.
- kiem_cau_truc: lớp mô hình, đường dẫn tới các lớp, kiểu lớp so với danh sách full attention của đặc tả,
  layer_scalar của Gemma 4, embedding đầu vào.
- kiem_goi_embedding: mỗi lượt forward đi qua embedding đầu vào đúng MỘT lần (điều kiện của bộ đếm vị trí
  trong tn2.can_thiep), và bộ đếm vị trí khớp số token thật.
- dem_ro_thinking: đếm câu trả lời còn chứa dấu hiệu khối thinking (Nhap Dựng hướng điểm 5).
- do_position_ids: phép chẩn đoán position_ids / đệm (pipeline chính luôn truyền position_ids theo mặt nạ).
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from ..can_thiep import can_thiep_chinh, gan_can_thiep
from ..khuon import ma_hoa, phan_tich_mot_cau
from ..kich_hoat import kich_hoat_tung_cau
from ..mo_hinh import DacTaMoHinh, thiet_bi_cua, tim_cac_lop
from ..sinh import sinh_tham_lam

# Pipeline chính LUÔN truyền position_ids = cumsum(mask) − 1 khi lấy kích hoạt (đúng cách generate đánh vị trí).
# Luật "đo rồi tự chọn" cũ (hệ số 10 × nhiễu) bị bỏ sau rà soát code bước 3 (R1 B1–B2, R2 B2; Ân duyệt Q1 ngày
# 27/10/2026): ở bf16 ngưỡng quá rộng để chứng minh bỏ position_ids là an toàn, còn truyền thì không thể sai.
# do_position_ids chỉ còn là phép chẩn đoán.
# Cảnh báo đệm: cosine giữa trung bình theo lô (cách B, cách được dùng) và trung bình từng câu không đệm (C) ở mức
# làm tròn phải rất gần 1. Dưới 0,999 ở một lớp nào đó: token đệm có thể đã làm đổi kích hoạt (mặt nạ, lớp DeltaNet /
# conv) → hướng ở trạng thái "cần xem" (Ân duyệt Q2). Cờ TẮT không chứng nhận không có lỗi đệm: cosine không đổi khi
# nhân hệ số (ví dụ B = 1,01 × C không bật cờ), nên đọc kèm lệch tương đối theo lớp và theo câu.
NGUONG_COS_DEM = 0.999


def token_cua_khoi(tok, ids: Sequence[int], k: int) -> list[str]:
    """Giải mã từng token của K token cuối (giữ token đặc biệt)."""
    return [tok.decode([int(i)], skip_special_tokens=False) for i in list(ids)[-k:]]


def kiem_khoi_tokenizer(tok, dac_ta: DacTaMoHinh, cau_hoi: Sequence[str]) -> dict[str, Any]:
    """Đo K và vị trí lấy trên TỪNG câu (không dừng ở câu lỗi đầu tiên); so với đặc tả.

    Đạt khi mọi câu cho cùng K = so_token_khoi_du_kien, cùng vị trí lấy = vi_tri_lay_du_kien, và các token của
    khối (giải mã từng token) đúng bằng khoi_du_kien (nếu đặc tả có).
    """
    ks, vts, khoi, loi = {}, {}, {}, []
    vi_du = None
    for i, q in enumerate(cau_hoi):
        try:
            vb, ids, k, vt = phan_tich_mot_cau(tok, q, dac_ta)
        except ValueError as e:
            loi.append({"cau": i, "loi": str(e)[:300]})
            continue
        ks[k] = ks.get(k, 0) + 1
        vts[vt] = vts.get(vt, 0) + 1
        t = tuple(token_cua_khoi(tok, ids, k))
        khoi[t] = khoi.get(t, 0) + 1
        if vi_du is None:
            vi_du = {"van_ban": vb, "ids_cuoi": list(ids[-(k + 4):]),
                     "token_cuoi": token_cua_khoi(tok, ids, min(len(ids), k + 4))}
    dat = (not loi and len(ks) == 1 and len(vts) == 1 and len(khoi) == 1)
    if dac_ta.so_token_khoi_du_kien is not None:
        dat = dat and set(ks) == {dac_ta.so_token_khoi_du_kien}
    if dac_ta.vi_tri_lay_du_kien is not None:
        dat = dat and set(vts) == {dac_ta.vi_tri_lay_du_kien}
    if dac_ta.khoi_du_kien:
        dat = dat and set(khoi) == {tuple(dac_ta.khoi_du_kien)}
    return {"so_cau": len(cau_hoi), "so_cau_loi": len(loi), "loi": loi[:20],
            "K_theo_so_cau": {str(k): v for k, v in ks.items()},
            "vi_tri_lay_theo_so_cau": {str(k): v for k, v in vts.items()},
            "khoi_theo_so_cau": [{"token": list(t), "so_cau": v} for t, v in khoi.items()],
            "khoi_du_kien": list(dac_ta.khoi_du_kien), "K_du_kien": dac_ta.so_token_khoi_du_kien,
            "vi_tri_lay_du_kien": dac_ta.vi_tri_lay_du_kien, "vi_du": vi_du, "dat": bool(dat)}


def kiem_eos_tokenizer(tok, dac_ta: DacTaMoHinh) -> dict[str, Any]:
    giai_ma = {int(i): tok.convert_ids_to_tokens(int(i)) for i in dac_ta.eos_chinh}
    sai = [{"id": i, "du_kien": s, "thuc_te": giai_ma.get(i, tok.convert_ids_to_tokens(i))}
           for i, s in dac_ta.eos_du_kien if tok.convert_ids_to_tokens(int(i)) != s]
    return {"eos_chinh": list(dac_ta.eos_chinh), "token_cua_eos": {str(k): v for k, v in giai_ma.items()},
            "eos_du_kien": [list(x) for x in dac_ta.eos_du_kien], "sai": sai,
            "pad": [tok.pad_token, tok.pad_token_id], "bos": [tok.bos_token, tok.bos_token_id],
            "pad_khong_phai_eos_chinh": tok.pad_token_id not in dac_ta.eos_chinh or dac_ta.pad_bang_eos,
            "dat": not sai and tok.pad_token_id is not None}


def kieu_lop_theo_config(model) -> list[str] | None:
    cfg = model.config
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    lt = getattr(cfg, "layer_types", None)
    return list(lt) if lt is not None else None


def kiem_cau_truc(model, dac_ta: DacTaMoHinh) -> dict[str, Any]:
    cac_lop = tim_cac_lop(model)
    lt = kieu_lop_theo_config(model)
    full_cfg = tuple(i for i, x in enumerate(lt) if x == "full_attention") if lt is not None else None
    # Kiểu khối thật trong từng lớp (Qwen3.5: linear_attn cho DeltaNet, self_attn cho full attention).
    khoi_that = []
    for l in cac_lop:
        if hasattr(l, "linear_attn"):
            khoi_that.append("linear_attn")
        elif hasattr(l, "self_attn"):
            sa = l.self_attn
            truot = getattr(sa, "is_sliding", None)
            if truot is None:
                truot = getattr(sa, "sliding_window", None) is not None
            khoi_that.append("self_attn_truot" if truot else "self_attn")
        else:
            khoi_that.append(type(l).__name__)
    full_that = tuple(i for i, x in enumerate(khoi_that) if x == "self_attn")
    emb = model.get_input_embeddings()
    ten_emb = [n for n, m in model.named_modules() if m is emb]
    scalar = None
    if all(hasattr(l, "layer_scalar") for l in cac_lop):
        scalar = [float(l.layer_scalar.detach().float().reshape(-1)[0]) for l in cac_lop]
    ra = {"lop_mo_hinh": type(model).__name__, "lop_decoder": sorted({type(l).__name__ for l in cac_lop}),
          "so_lop": len(cac_lop), "so_lop_dac_ta": dac_ta.so_lop,
          "duong_dan_lop": [n for n, m in model.named_modules() if m is cac_lop],
          "layer_types_config": lt, "full_attention_config": list(full_cfg) if full_cfg is not None else None,
          "full_attention_dac_ta": list(dac_ta.lop_full_attention), "khoi_trong_tung_lop": khoi_that,
          "full_attention_theo_khoi": list(full_that),
          "embedding_dau_vao": {"lop": type(emb).__name__, "ten": ten_emb,
                                "kich_thuoc": list(emb.weight.shape)},
          "layer_scalar": scalar, "dtype": str(next(model.parameters()).dtype),
          "so_tham_so": int(sum(p.numel() for p in model.parameters()))}
    ra["dat"] = bool(len(cac_lop) == dac_ta.so_lop and full_cfg == tuple(dac_ta.lop_full_attention)
                     and full_that == tuple(dac_ta.lop_full_attention) and len(ten_emb) == 1)
    return ra


@torch.no_grad()
def kiem_goi_embedding(model, tok, dac_ta: DacTaMoHinh, cau_hoi: Sequence[str], max_new_tokens: int,
                       eos_ids: Sequence[int]) -> dict[str, Any]:
    """Mỗi lượt forward trong generate phải đi qua embedding đầu vào đúng một lần và đi qua lớp 0 đúng một lần;
    lượt đầu mang cả lời nhắc (L token), các lượt sau mỗi lượt một token; bộ đếm vị trí của gan_can_thiep cộng
    đúng L + (số lượt − 1). Đây là điều kiện để hook đánh đúng vị trí token sinh (tn2.can_thiep)."""
    enc = ma_hoa(tok, list(cau_hoi), dac_ta)
    cac_lop = tim_cac_lop(model)
    emb_dai, lop0_dai = [], []
    h1 = model.get_input_embeddings().register_forward_hook(lambda m, a, o: emb_dai.append(int(o.shape[1])))

    def pre(m, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        lop0_dai.append(int(h.shape[1]))
    h2 = cac_lop[0].register_forward_pre_hook(pre, with_kwargs=True)
    vec = torch.zeros(model.get_input_embeddings().weight.shape[1])
    ct = can_thiep_chinh(0, vec, 1.0, 0.0, "a", 1, enc.so_token_khoi)
    try:
        L = enc.do_dai_dem
        dem_cuoi = {}
        thiet_bi = thiet_bi_cua(model)
        enc.den(thiet_bi)
        from ..sinh import cau_hinh_greedy
        gc = cau_hinh_greedy(max_new_tokens, eos_ids, tok.pad_token_id)
        try:
            with gan_can_thiep(model, [ct], do_dai_loi_nhac=L) as dem:
                out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, generation_config=gc)
                dem_cuoi["da_qua"] = dem.da_qua
        except RuntimeError as e:       # chốt chặn trong can_thiep: bộ đếm vị trí không khớp độ dài chuỗi
            return {"L": L, "loi": str(e)[:300], "so_lan_goi_embedding": len(emb_dai),
                    "so_lan_qua_lop_0": len(lop0_dai), "dat": False}
    finally:
        h1.remove()
        h2.remove()
    so_luot = len(lop0_dai)
    so_sinh = int(out.shape[1]) - L
    dung = (emb_dai == lop0_dai and so_luot >= 1 and emb_dai[0] == L and all(x == 1 for x in emb_dai[1:])
            and dem_cuoi["da_qua"] == L + so_luot - 1 and so_luot == so_sinh)
    return {"L": L, "so_luot_forward": so_luot, "so_token_sinh": so_sinh,
            "do_dai_qua_embedding": emb_dai[:5] + (["…"] if len(emb_dai) > 5 else []),
            "so_lan_goi_embedding": len(emb_dai), "so_lan_qua_lop_0": len(lop0_dai),
            "bo_dem_vi_tri_cuoi": dem_cuoi["da_qua"], "dat": bool(dung)}


def dem_ro_thinking(van_ban: Sequence[str], dac_ta: DacTaMoHinh) -> dict[str, Any]:
    co = [i for i, v in enumerate(van_ban) if any(d in v for d in dac_ta.dau_hieu_thinking)]
    return {"dau_hieu": list(dac_ta.dau_hieu_thinking), "so_cau": len(van_ban), "so_cau_ro": len(co),
            "ti_le": (len(co) / len(van_ban)) if van_ban else None, "chi_so": co[:50]}


def _lech_tuong_doi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """‖a − b‖ / ‖b‖ theo từng lớp; a, b [nL, d] float64."""
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)


@torch.no_grad()
def do_position_ids(model, tok, dac_ta: DacTaMoHinh, cau_hoi: Sequence[str], vi_tri: int,
                    co_lo: int | None = None) -> dict[str, Any]:
    """Phép chẩn đoán position_ids và đệm (không quyết định gì; pipeline chính luôn dùng cách B).

    Trên MỘT lô của tập dựng (đệm trái, độ dài khác nhau), lấy kích hoạt ở vị trí lấy, mọi lớp, theo ba cách:
    (A) lô, không truyền position_ids; (B) lô, position_ids = cumsum(mask) − 1 như generate (cách pipeline dùng);
    (C) từng câu một, không đệm (chuẩn: không có token đệm, vị trí hiển nhiên đúng).
    Báo: lệch tương đối của trung bình theo lớp (A–C, B–C, A–B); lệch tương đối lớn nhất theo TỪNG câu (lỗi do
    đệm chỉ nằm ở câu đệm nhiều, trung bình làm loãng nó); cosine nhỏ nhất của trung bình (A–C, B–C) và theo câu.
    canh_bao_dem = cosine B–C nhỏ nhất (trung bình, trên các lớp) < NGUONG_COS_DEM.
    Phép đo chạy ở kiểu số của mô hình đang nạp (bf16 ở bước 3).
    """
    cau_hoi = list(cau_hoi)
    co_lo = co_lo or len(cau_hoi)
    if co_lo < len(cau_hoi):
        raise ValueError("Phép đo dùng đúng một lô.")
    a = kich_hoat_tung_cau(model, tok, dac_ta, cau_hoi, [vi_tri], co_lo=co_lo, position_ids_theo_mat_na=False)
    b = kich_hoat_tung_cau(model, tok, dac_ta, cau_hoi, [vi_tri], co_lo=co_lo, position_ids_theo_mat_na=True)
    c = torch.cat([kich_hoat_tung_cau(model, tok, dac_ta, [q], [vi_tri], co_lo=1) for q in cau_hoi], dim=0)
    tb = {k: v[:, 0].to(torch.float64).mean(dim=0) for k, v in (("A", a), ("B", b), ("C", c))}   # [nL, d]
    tc = {k: v[:, 0].to(torch.float64) for k, v in (("A", a), ("B", b), ("C", c))}                 # [N, nL, d]
    cos = torch.nn.functional.cosine_similarity

    def theo_cau(x: str) -> tuple[float, float]:
        lech = (tc[x] - tc["C"]).norm(dim=-1) / tc["C"].norm(dim=-1).clamp_min(1e-30)              # [N, nL]
        return float(lech.max()), float(cos(tc[x], tc["C"], dim=-1).min())

    lech_ac, lech_bc, lech_ab = (_lech_tuong_doi(tb["A"], tb["C"]), _lech_tuong_doi(tb["B"], tb["C"]),
                                 _lech_tuong_doi(tb["A"], tb["B"]))
    cos_ac, cos_bc = cos(tb["A"], tb["C"], dim=-1), cos(tb["B"], tb["C"], dim=-1)
    (lc_ac, cc_ac), (lc_bc, cc_bc) = theo_cau("A"), theo_cau("B")
    canh_bao_dem = float(cos_bc.min()) < NGUONG_COS_DEM
    enc = ma_hoa(tok, cau_hoi, dac_ta)
    so_dem = (enc.attention_mask == 0).sum(dim=1)
    return {"so_cau": len(cau_hoi), "vi_tri": vi_tri, "do_dai_dem": enc.do_dai_dem,
            "so_token_dem_moi_cau": {"nho_nhat": int(so_dem.min()), "lon_nhat": int(so_dem.max())},
            "kieu_so": str(next(model.parameters()).dtype),
            "cach_pipeline_dung": "B_truyen_position_ids_theo_mat_na",
            "trung_binh": {"lech_A_C_theo_lop": lech_ac.tolist(), "lech_B_C_theo_lop": lech_bc.tolist(),
                           "lech_A_B_theo_lop": lech_ab.tolist(),
                           "lech_A_C_lon_nhat": float(lech_ac.max()), "lech_B_C_lon_nhat": float(lech_bc.max()),
                           "cos_A_C_nho_nhat": float(cos_ac.min()), "cos_B_C_nho_nhat": float(cos_bc.min())},
            "theo_cau": {"lech_A_C_lon_nhat": lc_ac, "lech_B_C_lon_nhat": lc_bc,
                         "cos_A_C_nho_nhat": cc_ac, "cos_B_C_nho_nhat": cc_bc},
            "nguong_cos_dem": NGUONG_COS_DEM, "canh_bao_dem": bool(canh_bao_dem),
            "ghi_chu": "Chỉ chẩn đoán. canh_bao_dem = False KHÔNG chứng nhận không có lỗi đệm (cosine không đổi "
                       "khi nhân hệ số); đọc kèm lệch tương đối theo lớp và theo câu."}
