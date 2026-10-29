"""Quét attention theo độ dài: kiểu attention đang dùng so với một cách tính độc lập (sdpa, backend MATH).

Sửa sau kết quả bước 3 (Ân duyệt Q-E3 ngày 29/10/2026; ghép ý hai reviewer). Lý do: ở Gemma 4 12B, kernel
EFFICIENT_ATTENTION của sdpa (PyTorch 2.11) cho logit sai ở hàng truy vấn cuối khi số vị trí truy vấn m ≡ 1 (mod 32),
m ≥ 33 (k/v stride 0 của lớp full attention 1 KV head; PyTorch issue #191937). Bản thu nhỏ phần 2 (4d) bắt được lỗi
này nhưng muộn; phép quét ở đây bắt ngay khi nạp (4b) và kiểm lại ở float32 (4d).

Các dạng so (cùng chuỗi token, logit float32):
- "don": một chuỗi không đệm X[:m], m = 2…m_toi_da, logit vị trí cuối;
- "lo": lô 8 câu đệm trái, độ dài đệm ép về các độ dài ≡ 1 (mod 32) ≤ m_toi_da, logit vị trí cuối của mọi hàng;
- "lo_thuc": các lô thật do người gọi đưa vào (ví dụ bốn lô 40 câu của bước 4), logit vị trí cuối của mọi hàng;
- "cache": lô đệm trái độ dài 33 và 61, nạp lời nhắc có KV cache rồi teacher forcing 8 token, logit từng bước (lô 61:
  các bước sinh đi qua độ dài KV 65 ≡ 1 mod 32).
float32 chạy "don" và "lo" (rà soát độc lập 29/10/2026: dạng "don" không có mặt nạ nên ở Qwen sdpa chỉ có MATH, so MATH
với MATH; dạng "lo" có mặt nạ đi qua kernel mà bước 4 dùng).

Luật (viết trước, Ân duyệt): bf16 — "khong_dat" khi có ô lệch > 5 logit, "can_kiem" khi lệch lớn nhất trong (2, 5],
"dat" khi ≤ 2; float32 — "dat" khi ≤ 0,5 logit, ngược lại "khong_dat". Lỗi đã biết lệch 15–27 logit.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Sequence

import torch

from .mo_hinh import kieu_attention, thiet_bi_cua

NGUONG_BF16_KHONG_DAT = 5.0
NGUONG_BF16_CAN_KIEM = 2.0
NGUONG_FP32 = 0.5
M_TOI_DA = 320
SO_HANG_LO = 8
SO_BUOC_CACHE = 8
DO_DAI_CACHE = (33, 61)


# Hướng dựng trước khi huong.pt ghi kiểu attention, được coi là dựng bằng sdpa: chỉ hướng Qwen3.5-9B ngày 28/10/2026
# (code e61aa007 không đặt attn_implementation → transformers mặc định sdpa; cùng căn cứ với ngoại lệ ở bước 4).
HUONG_SDPA_CU = {"qwen35_9b": "3bfaf7e6dd8d6c548eaa57fe1bb508f891df1c4fce34e4c157865f5a11f4a2ab"}


def kieu_attention_huong(h: dict, sha256: str | None) -> str | None:
    """Kiểu attention đã dùng khi dựng file hướng h (SHA-256 của file: sha256). Thiếu trường: chỉ "sdpa" với hướng cũ
    trong HUONG_SDPA_CU, còn lại None (không xác định). R1 B.3, R2 B2 (rà soát 29/10/2026)."""
    a = h.get("attn_implementation")
    if a:
        return a
    return "sdpa" if sha256 and HUONG_SDPA_CU.get(h.get("dac_ta")) == sha256 else None


def loi_kieu_attention_huong(h: dict, sha256: str | None, dac_ta) -> str | None:
    """Thông báo lỗi nếu hướng không dựng bằng đúng kiểu attention của đặc tả (Q-E1: bước 4 dùng hướng dựng lại
    bằng kiểu đúng), ngược lại None."""
    if not dac_ta.attn_implementation:
        return None
    a = kieu_attention_huong(h, sha256)
    if a == dac_ta.attn_implementation:
        return None
    return (f"file hướng dựng bằng attention {a!r} (None = không ghi), đặc tả {dac_ta.ten} cần "
            f"{dac_ta.attn_implementation!r}: dựng lại hướng bằng code hiện tại")


@contextmanager
def tham_chieu_math(model):
    """Trong khối này mô hình chạy sdpa với backend MATH (cách tính tham chiếu); ra khỏi khối trả lại kiểu cũ."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cu = kieu_attention(model)
    model.set_attn_implementation("sdpa")
    try:
        with sdpa_kernel([SDPBackend.MATH]):
            yield
    finally:
        if cu:
            model.set_attn_implementation(cu)


def _vi_tri(mask: torch.Tensor) -> torch.Tensor:
    """position_ids như generate của transformers 5.17 (generation/utils.py): cumsum(mask) − 1, vị trí đệm gán 0 —
    dùng đúng hàm của pipeline (kich_hoat.vi_tri_theo_mat_na; sửa theo R2, rà soát 29/10/2026)."""
    from .kich_hoat import vi_tri_theo_mat_na

    return vi_tri_theo_mat_na(mask)


def _dem_trai(hang: Sequence[Sequence[int]], pad: int, thiet_bi) -> tuple[torch.Tensor, torch.Tensor]:
    dai = max(len(h) for h in hang)
    ids = torch.full((len(hang), dai), int(pad), dtype=torch.long)
    mask = torch.zeros((len(hang), dai), dtype=torch.long)
    for i, h in enumerate(hang):
        ids[i, dai - len(h):] = torch.tensor(list(h), dtype=torch.long)
        mask[i, dai - len(h):] = 1
    return ids.to(thiet_bi), mask.to(thiet_bi)


@torch.no_grad()
def _logit_cuoi(model, ids: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Logit float32 ở vị trí cuối của mọi hàng: [B, V]."""
    pos = None if mask is None else _vi_tri(mask)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
    return out.logits[:, -1, :].float()


@torch.no_grad()
def _logit_cache(model, ids: torch.Tensor, mask: torch.Tensor, tiep: Sequence[int]) -> torch.Tensor:
    """Nạp lời nhắc có KV cache rồi đưa lần lượt các token `tiep` (cùng cho mọi hàng): logit [B, 1 + len(tiep), V]."""
    pos = _vi_tri(mask)
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    ra = [out.logits[:, -1, :].float()]
    past, m, p = out.past_key_values, mask, pos[:, -1:]
    for t in tiep:
        m = torch.cat([m, torch.ones_like(m[:, :1])], dim=1)
        p = p + 1
        x = torch.full((ids.shape[0], 1), int(t), dtype=torch.long, device=ids.device)
        out = model(input_ids=x, attention_mask=m, position_ids=p, past_key_values=past, use_cache=True)
        ra.append(out.logits[:, -1, :].float())
        past = out.past_key_values
    return torch.stack(ra, dim=1)


def _lech(a: torch.Tensor, b: torch.Tensor) -> float:
    """Lệch tuyệt đối lớn nhất giữa a và b. Có ô không hữu hạn (NaN, ±Inf) ở a hoặc b → +inf, để ô đó luôn vượt
    ngưỡng: NaN so với ngưỡng luôn sai, và max() của Python bỏ qua NaN (R2 B1, rà soát 29/10/2026)."""
    if not (bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())):
        return math.inf
    return float((a - b).abs().max())


def _so(a: torch.Tensor, b: torch.Tensor) -> tuple[float, int]:
    """(lệch tuyệt đối lớn nhất, số ô mà argmax khác nhau); a, b có cùng hình dạng [..., V]."""
    return _lech(a, b), int((a.argmax(-1) != b.argmax(-1)).sum())


def _hang_lo(X: Sequence[int], dai: int, so_hang: int = SO_HANG_LO) -> list[list[int]]:
    """so_hang chuỗi con của X, dài nhất đúng `dai` (nên độ dài đã đệm là `dai`), các hàng còn lại ngắn dần."""
    ra = []
    for r in range(so_hang):
        n = max(2, dai - 3 * r)
        ra.append(list(X[r: r + n]))
    return ra


def quet_attention(model, X: Sequence[int], pad_id: int, fp32: bool, m_toi_da: int = M_TOI_DA,
                   lo_thuc: dict[str, Sequence[Sequence[int]]] | None = None, day_du: bool = True) -> dict[str, Any]:
    """So kiểu attention đang dùng với sdpa backend MATH trên cùng đầu vào. X: một chuỗi token dài ≥ m_toi_da + 3*8.

    fp32=True: dạng "don" (m = 2…m_toi_da) và "lo", ngưỡng 0,5 logit. fp32=False: "don", "lo", "lo_thuc", "cache"
    (ba dạng sau chỉ khi day_du), luật ba mức bf16. Trả về dict có "trang_thai" ∈ {"dat", "can_kiem", "khong_dat"},
    "do_dai_sai" (m của dạng "don" vượt ngưỡng không đạt), "do_dai_dem_lo_sai" (độ dài đệm của dạng "lo") và bảng chi
    tiết.
    """
    if len(X) < m_toi_da + 3 * SO_HANG_LO + SO_BUOC_CACHE + 1:
        raise ValueError(f"Chuỗi quét quá ngắn ({len(X)} token).")
    tb = thiet_bi_cua(model)
    ra: dict[str, Any] = {"kieu_attention": kieu_attention(model), "tham_chieu": "sdpa (MATH)",
                          "dtype": str(next(model.parameters()).dtype), "fp32": fp32, "m_toi_da": m_toi_da}
    f = getattr(torch.backends.cuda, "fp16_bf16_reduction_math_sdp_allowed", None)
    ra["math_cho_rut_gon_bf16"] = f() if f else None         # False: MATH giữ phần tính trung gian ở float32
    do_dai = list(range(2, m_toi_da + 1))
    xau = [m for m in range(33, m_toi_da + 1, 32)]

    def chay(che_do: str) -> dict[str, Any]:
        kq: dict[str, Any] = {"don": [_logit_cuoi(model, torch.tensor([list(X[:m])], device=tb), None)[0].cpu()
                                      for m in do_dai]}
        if day_du:
            kq["lo"] = []
            for L in xau:
                ids, mask = _dem_trai(_hang_lo(X, L), pad_id, tb)
                kq["lo"].append(_logit_cuoi(model, ids, mask).cpu())
        if not fp32 and day_du:
            kq["lo_thuc"] = {}
            for ten, hang in (lo_thuc or {}).items():
                ids, mask = _dem_trai(hang, pad_id, tb)
                kq["lo_thuc"][ten] = (int(ids.shape[1]), _logit_cuoi(model, ids, mask).cpu())
            kq["cache"] = []
            for L in DO_DAI_CACHE:
                if L > m_toi_da:
                    continue
                ids, mask = _dem_trai(_hang_lo(X, L), pad_id, tb)
                tiep = list(X[L: L + SO_BUOC_CACHE])
                kq["cache"].append((L, _logit_cache(model, ids, mask, tiep).cpu()))
        return kq

    dang = chay("dang_dung")
    with tham_chieu_math(model):
        tc = chay("tham_chieu")

    bang_don = []
    for m, a, b in zip(do_dai, dang["don"], tc["don"]):
        lech, doi = _so(a, b)
        bang_don.append({"m": m, "lech": round(lech, 6), "argmax_doi": doi})
    ra["don"] = bang_don
    lech_max = max(r["lech"] for r in bang_don)
    so_doi = sum(r["argmax_doi"] for r in bang_don)
    if day_du:
        ra["lo"] = []
        for L, a, b in zip(xau, dang["lo"], tc["lo"]):
            lech, doi = _so(a, b)
            ra["lo"].append({"do_dai_dem": L, "lech": round(lech, 6), "argmax_doi": doi})
            lech_max = max(lech_max, lech)
            so_doi += doi
    if not fp32 and day_du:
        ra["lo_thuc"] = []
        for ten, (L, a) in dang["lo_thuc"].items():
            lech, doi = _so(a, tc["lo_thuc"][ten][1])
            ra["lo_thuc"].append({"lo": ten, "do_dai_dem": L, "lech": round(lech, 6), "argmax_doi": doi})
        ra["cache"] = []
        for (L, a), (_, b) in zip(dang["cache"], tc["cache"]):
            lech_buoc = [round(_lech(a[:, j], b[:, j]), 6) for j in range(a.shape[1])]
            ra["cache"].append({"do_dai_dem": L, "lech_theo_buoc": lech_buoc,
                                "argmax_doi": int((a.argmax(-1) != b.argmax(-1)).sum())})
        for r in ra["lo_thuc"]:
            lech_max = max(lech_max, r["lech"])
            so_doi += r["argmax_doi"]
        for r in ra["cache"]:
            lech_max = max(lech_max, max(r["lech_theo_buoc"]))
            so_doi += r["argmax_doi"]
    trung_vi = sorted(r["lech"] for r in bang_don)[len(bang_don) // 2]
    # Để đọc theo luật Ân duyệt 29/10/2026 (phương án 2): lỗi kernel đã biết dồn ở m ≡ 1 (mod 32); nhiễu làm tròn thì không.
    ra["lech_lon_nhat_m_xau"] = max((r["lech"] for r in bang_don if r["m"] >= 33 and r["m"] % 32 == 1), default=None)
    ra["lech_lon_nhat_m_khac"] = max((r["lech"] for r in bang_don if not (r["m"] >= 33 and r["m"] % 32 == 1)),
                                     default=None)
    # Các dạng có đệm (lo, lo_thuc, cache): mọi độ dài của "lo" đều ≡ 1 (mod 32), nên lệch ở đây lớn hơn hẳn dạng một
    # chuỗi cũng là dấu hiệu "dồn" (rà soát độc lập 29/10/2026).
    o_lo = [r["lech"] for r in ra.get("lo", []) + ra.get("lo_thuc", [])]
    o_lo += [x for r in ra.get("cache", []) for x in r["lech_theo_buoc"]]
    ra["lech_lon_nhat_lo_cache"] = max(o_lo, default=None)
    o = [r["lech"] for r in bang_don] + o_lo
    ra["so_o_khong_huu_han"] = sum(1 for x in o if not math.isfinite(x))
    ra["lech_lon_nhat"] = round(lech_max, 6)
    ra["trung_vi_don"] = trung_vi
    ra["so_o_argmax_doi"] = so_doi
    if fp32:
        ra["nguong"] = {"dat_toi_da": NGUONG_FP32}
        ra["trang_thai"] = "dat" if lech_max <= NGUONG_FP32 else "khong_dat"
        nguong_sai = NGUONG_FP32
    else:
        ra["nguong"] = {"dat_toi_da": NGUONG_BF16_CAN_KIEM, "khong_dat_tren": NGUONG_BF16_KHONG_DAT}
        ra["trang_thai"] = ("khong_dat" if lech_max > NGUONG_BF16_KHONG_DAT
                            else "can_kiem" if lech_max > NGUONG_BF16_CAN_KIEM else "dat")
        nguong_sai = NGUONG_BF16_KHONG_DAT
    ra["do_dai_sai"] = [r["m"] for r in bang_don if r["lech"] > nguong_sai]
    ra["do_dai_dem_lo_sai"] = [r["do_dai_dem"] for r in ra.get("lo", []) if r["lech"] > nguong_sai]
    return ra


def chuoi_quet(tok, dac_ta, cau_hoi: Sequence[str], can: int) -> list[int]:
    """Ghép id (đã áp template) của các câu hỏi cho tới khi đủ `can` token."""
    from .khuon import ma_hoa

    X: list[int] = []
    for q in cau_hoi:
        X += ma_hoa(tok, [q], dac_ta).ids_rieng[0]
        if len(X) >= can:
            return X
    raise ValueError(f"Không đủ câu để ghép chuỗi quét {can} token (mới có {len(X)}).")


def tom_tat(kq: dict[str, Any]) -> str:
    """Một dòng cho nhật ký."""
    s = (f"kiểu {kq['kieu_attention']} so với {kq['tham_chieu']}, {kq['dtype']}: lệch lớn nhất "
         f"{kq['lech_lon_nhat']:.4g} logit (trung vị dạng một chuỗi {kq['trung_vi_don']:.3g}); "
         f"{kq['so_o_argmax_doi']} ô đổi argmax; độ dài sai {kq['do_dai_sai']}, lô sai "
         f"{kq['do_dai_dem_lo_sai']} → {kq['trang_thai']}")

    def _g(x):
        return "—" if x is None else f"{x:.4g}"
    s += (f"; dạng một chuỗi: lệch lớn nhất ở m ≡ 1 (mod 32) {_g(kq.get('lech_lon_nhat_m_xau'))}, ở m khác "
          f"{_g(kq.get('lech_lon_nhat_m_khac'))}; các dạng lô đệm và cache {_g(kq.get('lech_lon_nhat_lo_cache'))}; "
          f"ô không hữu hạn {kq.get('so_o_khong_huu_han', 0)}")
    if kq.get("lo_thuc"):
        s += "; lô thật: " + ", ".join(f"{r['lo']} (dài {r['do_dai_dem']}) {r['lech']:.3g}" for r in kq["lo_thuc"])
    return s
