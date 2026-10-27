"""Kiểm đường can thiệp của ta (Nhap, mục "Kiểm pipeline", phần 2). Dùng cho Llama-3-8B-Instruct ở bước 1
và cho bản thu nhỏ trên Qwen3.5-9B, Gemma 4 12B ở bước 3.

Ba phép kiểm, cho phương án 1 và 3, kiểu (a) và (b):
  (i)   hook của ta ở α ≠ 0 (KV cache, đệm trái, theo lô) so với cách tính tham chiếu (một câu, không đệm,
        không cache: tính lại toàn bộ chuỗi, cộng vector đúng vào các vị trí định trước);
  (ii)  hook ở α = 0 so với không gắn hook: phải khớp tuyệt đối (token và logit);
  (iii) hook chạy theo lô so với hook chạy từng câu (một câu, không đệm, có cache).

Tiêu chí khớp cho (i), (iii): so logit theo cùng một chuỗi — lấy chuỗi do lượt theo lô của ta sinh ra, đưa
nguyên chuỗi đó vào cách tính kia (teacher forcing), so logit ở từng vị trí. Ngưỡng = 2 × mức nhiễu số học
đo được ở α = 0 (Nhap). Đạt khi ở mọi câu, mọi lớp, mọi vị trí, độ lệch không vượt ngưỡng.

Các điểm cài đặt (Claude đề xuất trong báo cáo bước 1; hai reviewer đồng ý; Ân chốt 27/10/2026):
1. Mức nhiễu đo ở α = 0 trên một tập câu LỚN HƠN tập kiểm (script bước 1: 128 câu dựng mỗi kiểu, tức
   256 hàng so, nhiều hơn 96 hàng so ở α ≠ 0; chạy theo lô cùng cỡ với lô kiểm). Ngưỡng dùng CHUNG cho (i) và
   (iii), lấy giá trị lớn hơn của hai mức nhiễu. Hai phép có chung phần lô có đệm so với một câu, nhưng (i)
   còn so KV cache với tính lại cả chuỗi — reviewer 1 đo được đây là nguồn nhiễu chính, có cả khi không đệm.
   Lấy giá trị lớn hơn là cách bảo thủ, chấp nhận được vì có kiểm âm đi kèm.
   Lý do đo trên tập lớn: độ lệch lớn nhất là một đại lượng cực trị; đo nhiễu trên ít câu hơn số lần so ở
   α ≠ 0 làm ngưỡng bị thấp một cách hệ thống (mô hình nhỏ, float32: trượt giả với tỉ số 2,9–3,2 lần khi
   nhiễu chỉ đo trên chính các câu kiểm; hết trượt khi đo trên tập lớn hơn).
2. Điều chỉnh so với chữ cũ của Nhap ("2 lần mức nhiễu bf16"), đã ghi lại vào Nhap: hàm này không tự đổi
   dtype; script bước 1 chạy nó HAI lần, bf16 và float32 (TF32 tắt). Phép (ii) phải đạt ở cả hai. Phép (i),
   (iii) và kiểm âm lấy kết quả float32 làm kết luận: ở bf16 nhiễu có thể lớn ngang tác động của lỗi cài vào,
   nên kiểm âm không đạt, đúng trường hợp Nhap nói "phép kiểm không có giá trị". Cơ chế hook không phụ thuộc
   kiểu số (vector được ép về kiểu số của luồng dư). Số bf16 vẫn được báo đủ. Hạn chế: phép kiểm xác nhận cơ
   chế hook, không xác nhận sai số làm tròn của lượt chạy bf16.
3. Kiểm âm: cố ý cài lỗi vào hook (tính vùng với L − 1 thay cho L, tức lệch vùng một token; bỏ cộng khi sinh)
   rồi chạy lại phép (i), ở kiểu (a), lớp giữa của danh sách, cho CẢ phương án 1 và 3 (bốn lượt); "vượt rõ
   ngưỡng" là > HE_SO_KIEM_AM (5) × ngưỡng.
4. Sàn ngưỡng: nếu nhiễu đo được bằng 0 tuyệt đối thì dùng NGUONG_SAN (1e−6 logit), ghi "dung_san" = True.

Luật ba mức (viết trước khi chạy; phương án (a), Ân chốt 27/10/2026), áp cho (i) và (iii) ở float32:
- "dat": lệch lớn nhất ≤ ngưỡng (2 × nhiễu);
- "can_xem": lệch > ngưỡng nhưng ≤ HE_SO_CAN_XEM (10) × nhiễu, VÀ cả bốn kiểm âm > HE_SO_KIEM_AM_CAN_XEM (100)
  × nhiễu. "can_xem" KHÔNG phải đạt và không được trình bày như đã đạt tiêu chí 2×;
- "khong_dat": các trường hợp còn lại.
Khi "can_xem", làm đúng một phép chẩn đoán đã viết trước (tong_ket["chan_doan_vi_tri"], không tốn GPU):
với mỗi vị trí vượt ngưỡng, so độ lệch với đường bao nhiễu α = 0 tại CÙNG vị trí j; vị trí không có câu đo
nhiễu nào thì đánh dấu "thiếu dữ liệu nhiễu", không ngoại suy (sửa sau đối chiếu của reviewer 2, 27/10).
- Có vị trí lệch > HE_SO_NGHI_LOI (10) × bao nhiễu tại đó → dấu hiệu nghi lỗi cơ chế (cần kiểm, chưa phải bằng
  chứng về nguyên nhân): dừng, tìm lỗi trong hook, sửa và chạy lại riêng phần 2.
- Không có vị trí nào như vậy → chỉ có nghĩa là chẩn đoán chưa thấy dấu hiệu lỗi; CHƯA xác nhận đạt về cơ chế,
  kết quả vẫn là "can_xem". Ân, sau khi hai reviewer xem, quyết một trong hai: đi tiếp sang bước 2 và ghi riêng
  quyết định cùng câu hạn chế (ngưỡng 2× không đạt; phép kiểm chưa loại trừ sai lệch nhỏ ở mức này), hoặc tìm
  lỗi như trên.

Cách tính tham chiếu được viết độc lập với hook chính: nó không dùng bộ đếm vị trí mà tính sẵn danh sách
chỉ số trên một chuỗi không đệm, rồi cộng vector bằng phép gán theo chỉ số.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .can_thiep import CanThiep, can_thiep_chinh, gan_can_thiep
from .khuon import ma_hoa
from .mo_hinh import DacTaMoHinh, thiet_bi_cua, tim_cac_lop
from .sinh import KetQuaSinh, sinh_tham_lam

ALPHA_KIEM = 0.4            # Nhap: mức α cho phép kiểm
HE_SO_NGUONG = 2.0          # Nhap: ngưỡng = 2 × nhiễu đo ở α = 0
HE_SO_KIEM_AM = 5.0         # kiểm âm đạt khi độ lệch > 5 × ngưỡng (đề xuất, xem docstring)
MAX_NEW_TOKENS_KIEM = 64    # đề xuất: đủ để kiểm cơ chế cộng ở token sinh; ghi trong báo cáo
NGUONG_SAN = 1e-6           # sàn ngưỡng khi nhiễu đo được bằng 0 (đề xuất, xem docstring)
HE_SO_CAN_XEM = 10.0        # luật ba mức: (i)/(iii) trượt nhưng ≤ 10 × nhiễu …
HE_SO_KIEM_AM_CAN_XEM = 100.0   # … và mọi kiểm âm > 100 × nhiễu → "can_xem" (không phải đạt)
HE_SO_NGHI_LOI = 10.0       # chẩn đoán: lệch > 10 × bao nhiễu tại cùng vị trí → nghi lỗi cơ chế


@torch.no_grad()
def logit_tham_chieu(model, ids_loi_nhac: Sequence[int], ids_tiep: Sequence[int], lop: int,
                     vector: torch.Tensor | None, so_token_khoi: int, so_token_sinh: int | None) -> torch.Tensor:
    """Cách tính tham chiếu: một forward trên [lời nhắc + ids_tiep[:-1]], không đệm, không cache.

    Cộng `vector` vào đầu ra khối `lop` tại: K vị trí cuối của lời nhắc, và vị trí của token sinh thứ j
    (j = 0 là token sinh đầu tiên) nếu so_token_sinh là None hoặc j < so_token_sinh.
    Trả về logit [g, V] float32 dự đoán ids_tiep[0..g−1].
    """
    n, g = len(ids_loi_nhac), len(ids_tiep)
    if g == 0:
        raise ValueError("Chuỗi tiếp rỗng.")
    chuoi = list(ids_loi_nhac) + list(ids_tiep[:-1])
    chi_so = list(range(n - so_token_khoi, n))
    chi_so += [n + j for j in range(g - 1) if so_token_sinh is None or j < so_token_sinh]
    handle = None
    if vector is not None:
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            moi = h.clone()
            idx = torch.tensor(chi_so, device=h.device, dtype=torch.long)
            moi[0, idx, :] = moi[0, idx, :] + vector.to(device=h.device, dtype=h.dtype)
            return (moi, *output[1:]) if isinstance(output, tuple) else moi
        handle = tim_cac_lop(model)[lop].register_forward_hook(hook)
    try:
        x = torch.tensor([chuoi], dtype=torch.long, device=thiet_bi_cua(model))
        out = model(input_ids=x, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return out.logits[0, n - 1: n - 1 + g, :].to(torch.float32)


@torch.no_grad()
def logit_tung_cau_co_cache(model, ids_loi_nhac: Sequence[int], ids_tiep: Sequence[int],
                            can_thiep: Sequence[CanThiep] | None) -> torch.Tensor:
    """Hook của ta trên MỘT câu không đệm, có KV cache, teacher forcing theo ids_tiep. [g, V] float32."""
    n, g = len(ids_loi_nhac), len(ids_tiep)
    tb = thiet_bi_cua(model)
    ra = []
    with gan_can_thiep(model, can_thiep, do_dai_loi_nhac=n):
        out = model(input_ids=torch.tensor([list(ids_loi_nhac)], device=tb), use_cache=True)
        ra.append(out.logits[0, -1, :].to(torch.float32))
        past = out.past_key_values
        for t in ids_tiep[:-1]:
            out = model(input_ids=torch.tensor([[int(t)]], device=tb), past_key_values=past, use_cache=True)
            ra.append(out.logits[0, -1, :].to(torch.float32))
            past = out.past_key_values
    return torch.stack(ra, dim=0)


def _lech_theo_vi_tri(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """Độ lệch logit lớn nhất (trên từ vựng) ở từng vị trí sinh j = 0…g−1."""
    return (a - b).abs().amax(dim=-1).cpu().tolist()


def _thang_logit(kq: KetQuaSinh) -> float:
    """|logit| lớn nhất trên các vị trí hợp lệ của một lượt sinh (để đọc độ lệch tuyệt đối theo thang)."""
    return max(float(kq.logits[i, : kq.so_hop_le[i]].abs().max()) for i in range(len(kq.loi_nhac)))


def _so_lo_vs_tham_chieu(model, kq: KetQuaSinh, lop: int, ct: CanThiep | None) -> list[list[float]]:
    """(i): với mỗi câu, độ lệch logit theo vị trí giữa lượt theo lô và cách tính tham chiếu."""
    lech = []
    for i in range(len(kq.loi_nhac)):
        g = kq.so_hop_le[i]
        ids_tiep = kq.token_day_du[i, :g].tolist()
        vec = None if ct is None else ct.vector
        so_sinh = None if ct is None else ct.vung.so_token_sinh
        k = kq.loi_nhac.so_token_khoi
        ref = logit_tham_chieu(model, kq.loi_nhac.ids_rieng[i], ids_tiep, lop, vec, k, so_sinh)
        lech.append(_lech_theo_vi_tri(kq.logits[i, :g, :], ref))
    return lech


def _so_lo_vs_tung_cau(model, kq: KetQuaSinh, ct: CanThiep | None) -> list[list[float]]:
    """(iii): với mỗi câu, độ lệch logit theo vị trí giữa lượt theo lô và lượt từng câu (cùng hook)."""
    lech = []
    for i in range(len(kq.loi_nhac)):
        g = kq.so_hop_le[i]
        ids_tiep = kq.token_day_du[i, :g].tolist()
        don = logit_tung_cau_co_cache(model, kq.loi_nhac.ids_rieng[i], ids_tiep, None if ct is None else [ct])
        lech.append(_lech_theo_vi_tri(kq.logits[i, :g, :], don))
    return lech


@dataclass
class DauVaoKiemHook:
    cau_hoi_a: list[str]          # câu vô hại cho kiểu (a) — một lô
    cau_hoi_b: list[str]          # câu có hại cho kiểu (b) — một lô
    lop: list[int]
    r_hat: dict[int, torch.Tensor]
    s: dict[int, float]
    eos_ids: tuple[int, ...]
    alpha: float = ALPHA_KIEM
    max_new_tokens: int = MAX_NEW_TOKENS_KIEM
    phuong_an: tuple[int, ...] = (1, 3)
    cau_hoi_nhieu_a: list[str] | None = None   # tập đo nhiễu ở α = 0; None → dùng lại cau_hoi_a
    cau_hoi_nhieu_b: list[str] | None = None
    co_lo: int | None = None                    # cỡ lô khi đo nhiễu; None → len(cau_hoi_a) (giống lô kiểm)
    # Chẩn đoán thêm ở bước 3 (rà soát code bước 3, R1 nhận xét 2; Ân duyệt Q5): chạy lại lượt không hook lần hai
    # và so với lần một. Nếu hai lượt không hook khác nhau thì đường nền không tất định: khi (ii) trượt, chưa thể quy
    # riêng độ lệch cho hook (cũng chưa loại trừ lỗi hook). Không vào tiêu chí. Mặc định tắt (bước 1 không đổi).
    lap_lai_khong_hook: bool = False


def _do_nhieu(model, tok, dac_ta, dv: DauVaoKiemHook, kieu: str, cau: list[str], co_lo: int) -> list[dict]:
    """Nhiễu số học ở α = 0 cho một kiểu: vector 0 nên không phụ thuộc lớp; dùng lớp đầu của danh sách.

    Chạy theo lô cỡ `co_lo` (giống lô kiểm), mỗi câu cho một độ lệch cho (i) và một cho (iii).
    """
    lop = dv.lop[0]
    ra = []
    for d in range(0, len(cau), co_lo):
        enc = ma_hoa(tok, cau[d: d + co_lo], dac_ta)
        ct0 = can_thiep_chinh(lop, dv.r_hat[lop], dv.s[lop], 0.0, kieu, 1, enc.so_token_khoi)
        g0 = sinh_tham_lam(model, tok, enc, dv.max_new_tokens, dv.eos_ids, [ct0], lay_logit=True)
        vi, viii = _so_lo_vs_tham_chieu(model, g0, lop, ct0), _so_lo_vs_tung_cau(model, g0, ct0)
        ra.append({"kieu": kieu, "lo_bat_dau": d, "i": [max(x) for x in vi], "iii": [max(x) for x in viii],
                   "i_theo_vi_tri": vi, "iii_theo_vi_tri": viii, "so_token_sinh": g0.so_hop_le,
                   "thang_logit": _thang_logit(g0)})
    return ra


def chay_kiem_hook(model, tok, dac_ta: DacTaMoHinh, dv: DauVaoKiemHook, bao_tien_do=None) -> dict[str, Any]:
    """Chạy (ii), đo nhiễu, (i), (iii) và kiểm âm ở kiểu số hiện tại của `model`. Trả về dict có "tong_ket"."""
    nhieu_a = dv.cau_hoi_nhieu_a if dv.cau_hoi_nhieu_a is not None else dv.cau_hoi_a
    nhieu_b = dv.cau_hoi_nhieu_b if dv.cau_hoi_nhieu_b is not None else dv.cau_hoi_b
    co_lo = dv.co_lo or len(dv.cau_hoi_a)
    ket: dict[str, Any] = {"cau_hinh": {"alpha": dv.alpha, "max_new_tokens": dv.max_new_tokens, "lop": dv.lop,
                                        "phuong_an": list(dv.phuong_an), "so_cau_a": len(dv.cau_hoi_a),
                                        "so_cau_b": len(dv.cau_hoi_b), "so_cau_nhieu_a": len(nhieu_a),
                                        "so_cau_nhieu_b": len(nhieu_b), "co_lo_nhieu": co_lo,
                                        "he_so_nguong": HE_SO_NGUONG, "he_so_kiem_am": HE_SO_KIEM_AM,
                                        "eos_ids": list(dv.eos_ids),
                                        "dtype": str(next(model.parameters()).dtype)},
                           "ii": [], "nhieu": [], "i": [], "iii": [], "khong_hook_lap_lai": []}
    for kieu, cau, cau_nhieu in (("a", dv.cau_hoi_a, nhieu_a), ("b", dv.cau_hoi_b, nhieu_b)):
        ket["nhieu"] += _do_nhieu(model, tok, dac_ta, dv, kieu, cau_nhieu, co_lo)
        if bao_tien_do:
            bao_tien_do(f"kiểm hook: xong đo nhiễu kiểu ({kieu}) trên {len(cau_nhieu)} câu")
        enc = ma_hoa(tok, cau, dac_ta)
        k = enc.so_token_khoi
        g_khong = sinh_tham_lam(model, tok, enc, dv.max_new_tokens, dv.eos_ids, None, lay_logit=True)
        if dv.lap_lai_khong_hook:
            g_lai = sinh_tham_lam(model, tok, enc, dv.max_new_tokens, dv.eos_ids, None, lay_logit=True)
            ket["khong_hook_lap_lai"].append({
                "kieu": kieu, "cung_token": bool(torch.equal(g_khong.token_day_du, g_lai.token_day_du)),
                "cung_logit": bool(g_khong.logits.shape == g_lai.logits.shape
                                   and torch.equal(g_khong.logits, g_lai.logits))})
        for lop in dv.lop:
            # (ii) α = 0 so với không hook: khớp tuyệt đối (token và logit), ở từng lớp.
            ct0 = can_thiep_chinh(lop, dv.r_hat[lop], dv.s[lop], 0.0, kieu, 1, k)
            g_0 = sinh_tham_lam(model, tok, enc, dv.max_new_tokens, dv.eos_ids, [ct0], lay_logit=True)
            cung_token = torch.equal(g_khong.token_day_du, g_0.token_day_du)
            cung_logit = (g_khong.logits.shape == g_0.logits.shape) and torch.equal(g_khong.logits, g_0.logits)
            ket["ii"].append({"kieu": kieu, "lop": lop, "cung_token": bool(cung_token), "cung_logit": bool(cung_logit)})
            for pa in dv.phuong_an:
                ct = can_thiep_chinh(lop, dv.r_hat[lop], dv.s[lop], dv.alpha, kieu, pa, k)
                g = sinh_tham_lam(model, tok, enc, dv.max_new_tokens, dv.eos_ids, [ct], lay_logit=True)
                vi, viii = _so_lo_vs_tham_chieu(model, g, lop, ct), _so_lo_vs_tung_cau(model, g, ct)
                ket["i"].append({"kieu": kieu, "lop": lop, "phuong_an": pa,
                                 "lech_theo_cau": [max(x) for x in vi], "lech_theo_vi_tri": vi,
                                 "so_token_sinh": g.so_hop_le, "thang_logit": _thang_logit(g),
                                 "khac_khong_hook": not torch.equal(g.token_day_du, g_khong.token_day_du)})
                ket["iii"].append({"kieu": kieu, "lop": lop, "phuong_an": pa,
                                   "lech_theo_cau": [max(x) for x in viii], "lech_theo_vi_tri": viii})
            if bao_tien_do:
                bao_tien_do(f"kiểm hook: xong kiểu ({kieu}), lớp {lop}")

    nhieu_i = max(max(r["i"]) for r in ket["nhieu"])
    nhieu_iii = max(max(r["iii"]) for r in ket["nhieu"])
    nhieu = max(nhieu_i, nhieu_iii)                 # gộp, lấy giá trị lớn hơn (bảo thủ; xem docstring)
    dung_san = nhieu == 0.0
    nguong = NGUONG_SAN if dung_san else HE_SO_NGUONG * nhieu
    max_i = max(max(r["lech_theo_cau"]) for r in ket["i"])
    max_iii = max(max(r["lech_theo_cau"]) for r in ket["iii"])

    # Kiểm âm: kiểu (a), lớp giữa của danh sách, từng phương án × từng lỗi cài vào.
    lop_am = dv.lop[len(dv.lop) // 2]
    enc_a = ma_hoa(tok, dv.cau_hoi_a, dac_ta)
    kiem_am = {}
    for pa in dv.phuong_an:
        ct_am = can_thiep_chinh(lop_am, dv.r_hat[lop_am], dv.s[lop_am], dv.alpha, "a", pa, enc_a.so_token_khoi)
        for loi in ("lech_vung", "bo_cong_khi_sinh"):
            g = sinh_tham_lam(model, tok, enc_a, dv.max_new_tokens, dv.eos_ids, [ct_am], lay_logit=True,
                              loi_gia_lap=loi)
            lech = [max(x) for x in _so_lo_vs_tham_chieu(model, g, lop_am, ct_am)]
            kiem_am[f"pa{pa}_{loi}"] = {"lop": lop_am, "phuong_an": pa, "loi": loi, "lech_lon_nhat": max(lech),
                                        "lech_theo_cau": lech, "dat": max(lech) > HE_SO_KIEM_AM * nguong}

    # Cảnh báo (không phải tiêu chí): can thiệp ở α ≠ 0 không đổi token nào so với không hook → phép (i)/(iii)
    # khi đó chỉ kiểm được logit, không kiểm được nhánh cộng vào token sinh trên chuỗi khác.
    tong = {
        "ii_dat": all(r["cung_token"] and r["cung_logit"] for r in ket["ii"]),
        "nhieu_i": nhieu_i, "nhieu_iii": nhieu_iii, "nhieu_gop": nhieu, "nguong": nguong, "dung_san": dung_san,
        "lech_lon_nhat_i": max_i, "i_dat": max_i <= nguong,
        "lech_lon_nhat_iii": max_iii, "iii_dat": max_iii <= nguong,
        "kiem_am": kiem_am, "kiem_am_dat": all(v["dat"] for v in kiem_am.values()),
        "so_luot_alpha_doi_token": sum(r["khac_khong_hook"] for r in ket["i"]), "so_luot_alpha": len(ket["i"]),
        # chẩn đoán (không phải tiêu chí): lệch lớn nhất chia cho nhiễu; kiểm âm chia cho nhiễu
        "ti_so_i_tren_nhieu": max_i / nhieu if nhieu > 0 else None,
        "ti_so_iii_tren_nhieu": max_iii / nhieu if nhieu > 0 else None,
        "ti_so_kiem_am_tren_nhieu": {k: (v["lech_lon_nhat"] / nhieu if nhieu > 0 else None) for k, v in kiem_am.items()},
        "thang_logit_nhieu": max(r["thang_logit"] for r in ket["nhieu"]),
        "thang_logit_alpha": max(r["thang_logit"] for r in ket["i"]),
        # chẩn đoán Q5: None khi không chạy; True khi hai lượt không hook khớp tuyệt đối ở mọi kiểu
        "khong_hook_lap_lai_khop": (all(r["cung_token"] and r["cung_logit"] for r in ket["khong_hook_lap_lai"])
                                    if ket["khong_hook_lap_lai"] else None),
    }
    tong["dat_tat_ca"] = bool(tong["ii_dat"] and tong["i_dat"] and tong["iii_dat"] and tong["kiem_am_dat"])
    # Luật ba mức cho (i), (iii) (chốt 27/10, phương án (a) của mục C1): "dat" | "can_xem" | "khong_dat".
    kiem_am_rat_ro = nhieu > 0 and all(v["lech_lon_nhat"] > HE_SO_KIEM_AM_CAN_XEM * nhieu for v in kiem_am.values())
    for ten, mx in (("i", max_i), ("iii", max_iii)):
        if mx <= nguong:
            tt = "dat"
        elif nhieu > 0 and mx <= HE_SO_CAN_XEM * nhieu and kiem_am_rat_ro:
            tt = "can_xem"
        else:
            tt = "khong_dat"
        tong[f"trang_thai_{ten}"] = tt
    tong["chan_doan_vi_tri"] = {ten: _chan_doan_vi_tri(ket[ten], ket["nhieu"], ten, nguong)
                                for ten in ("i", "iii")}
    ket["tong_ket"] = tong
    return ket


def _bao_nhieu_theo_vi_tri(nhieu: list[dict], ten: str) -> tuple[list[float], list[int]]:
    """Đường bao nhiễu ở α = 0: độ lệch lớn nhất tại từng vị trí j trên mọi câu đo nhiễu còn sinh tới j,
    kèm số câu đo nhiễu có mặt tại j (để người đọc thấy vị trí nào thưa mẫu)."""
    bao: list[float] = []
    dem: list[int] = []
    for r in nhieu:
        for hang in r[f"{ten}_theo_vi_tri"]:
            for j, x in enumerate(hang):
                if j < len(bao):
                    bao[j] = max(bao[j], x)
                    dem[j] += 1
                else:
                    bao.append(x)
                    dem.append(1)
    return bao, dem


def _chan_doan_vi_tri(cac_luot: list[dict], nhieu: list[dict], ten: str, nguong: float) -> dict[str, Any]:
    """Phép chẩn đoán viết trước cho trạng thái "can_xem" (xem docstring, mục Luật ba mức).

    Với mỗi vị trí vượt ngưỡng: so độ lệch với đường bao nhiễu α = 0 tại CÙNG vị trí j. Nếu không câu đo nhiễu
    nào sinh tới j thì KHÔNG ngoại suy: đánh dấu "thiếu dữ liệu nhiễu" cho vị trí đó (không có tỉ số).
    Cờ nghi lỗi bật nếu có vị trí lệch > HE_SO_NGHI_LOI × bao tại đó (hoặc bao tại đó bằng 0). Cờ là dấu hiệu
    cần kiểm, không phải bằng chứng về nguyên nhân; cờ tắt cũng không chứng minh phần lệch còn lại là nhiễu.
    """
    bao, dem = _bao_nhieu_theo_vi_tri(nhieu, ten)
    vuot = []
    for r in cac_luot:
        for c, hang in enumerate(r["lech_theo_vi_tri"]):
            for j, x in enumerate(hang):
                if x > nguong:
                    co = j < len(bao)
                    b = bao[j] if co else None
                    vuot.append({"kieu": r["kieu"], "lop": r["lop"], "phuong_an": r["phuong_an"], "cau": c,
                                 "vi_tri": j, "lech": x, "thieu_du_lieu_nhieu": not co,
                                 "bao_nhieu_tai_vi_tri": b, "so_cau_nhieu_tai_vi_tri": dem[j] if co else 0,
                                 "ti_so": (x / b) if (co and b > 0) else None})
    co_du_lieu = [v for v in vuot if not v["thieu_du_lieu_nhieu"]]
    nghi = any(v["ti_so"] is None or v["ti_so"] > HE_SO_NGHI_LOI for v in co_du_lieu)
    return {"so_vi_tri_vuot_nguong": len(vuot), "vi_tri_vuot": vuot[:200], "nghi_loi_co_che": bool(nghi),
            "so_vi_tri_thieu_du_lieu_nhieu": len(vuot) - len(co_du_lieu),
            "bao_nhieu_theo_vi_tri": bao, "so_cau_nhieu_theo_vi_tri": dem}


def ket_luan_hai_kieu_so(kq_bf16: dict[str, Any], kq_fp32: dict[str, Any]) -> dict[str, Any]:
    """Kết luận phần 2 từ hai lượt: (ii) phải đạt ở cả hai; (i), (iii), kiểm âm lấy theo float32.

    trang_thai: "dat" | "can_xem" | "khong_dat". "can_xem" KHÔNG phải đạt: (ii) và kiểm âm đạt, (i)/(iii) float32
    trượt ngưỡng 2× nhưng ≤ 10 × nhiễu, và mọi kiểm âm > 100 × nhiễu. Khi đó Ân và hai reviewer đọc
    tong_ket["chan_doan_vi_tri"] theo luật viết trước (docstring của module, mục Luật ba mức).
    """
    t16, t32 = kq_bf16["tong_ket"], kq_fp32["tong_ket"]
    kl = {"ii_bf16": t16["ii_dat"], "ii_fp32": t32["ii_dat"], "i_fp32": t32["i_dat"], "iii_fp32": t32["iii_dat"],
          "kiem_am_fp32": t32["kiem_am_dat"],
          "trang_thai_i_fp32": t32.get("trang_thai_i"), "trang_thai_iii_fp32": t32.get("trang_thai_iii"),
          "tham_khao_bf16": {"i": t16["i_dat"], "iii": t16["iii_dat"], "kiem_am": t16["kiem_am_dat"],
                             "nguong": t16["nguong"]}}
    kl["dat"] = bool(kl["ii_bf16"] and kl["ii_fp32"] and kl["i_fp32"] and kl["iii_fp32"] and kl["kiem_am_fp32"])
    nen = kl["ii_bf16"] and kl["ii_fp32"] and kl["kiem_am_fp32"]
    tt = {kl["trang_thai_i_fp32"], kl["trang_thai_iii_fp32"]}
    if kl["dat"]:
        kl["trang_thai"] = "dat"
    elif nen and tt <= {"dat", "can_xem"} and "can_xem" in tt:
        kl["trang_thai"] = "can_xem"
        kl["nghi_loi_co_che"] = bool(any(t32["chan_doan_vi_tri"][k]["nghi_loi_co_che"] for k in ("i", "iii")))
    else:
        kl["trang_thai"] = "khong_dat"
    return kl
