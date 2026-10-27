"""Lấy trung bình kích hoạt ở các lớp và các vị trí (đếm từ cuối lời nhắc), cộng dồn ở float64.

Dùng cho:
- pipeline chính: điểm "ra_khoi", vị trí vi_tri_lay (Nhap Dựng hướng điểm 2–3), và độ lớn trung bình s_ℓ
  (Nhap mục 3.3: độ lớn trung bình của luồng dư tại vị trí lấy kích hoạt);
- tái lập Arditi et al.: điểm "vao_khoi" (resid_pre), 5 vị trí cuối (−5…−1), như generate_directions.py.

Vì lời nhắc được đệm trái, chỉ số âm luôn trỏ vào token thật của mọi câu trong lô.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .khuon import chia_lo, ma_hoa
from .mo_hinh import DacTaMoHinh, thiet_bi_cua, tim_cac_lop


@dataclass
class KetQuaKichHoat:
    vi_tri: list[int]                 # các chỉ số âm
    lop: list[int]
    diem: str
    trung_binh: torch.Tensor          # [P, nL, d] float64
    trung_binh_do_lon: torch.Tensor   # [P, nL] float64: trung bình ‖h‖ theo câu
    so_cau: int


@torch.no_grad()
def trung_binh_kich_hoat(model, tok, dac_ta: DacTaMoHinh, cau_hoi: Sequence[str], vi_tri: Sequence[int],
                         lop: Sequence[int] | None = None, diem: str = "ra_khoi",
                         co_lo: int = 32) -> KetQuaKichHoat:
    if diem not in ("ra_khoi", "vao_khoi"):
        raise ValueError("diem phải là 'ra_khoi' hoặc 'vao_khoi'.")
    if len(cau_hoi) == 0:
        raise ValueError("Không có câu nào.")
    if any(v >= 0 for v in vi_tri):
        raise ValueError("Vị trí phải là chỉ số âm (đếm từ cuối lời nhắc).")
    cac_lop = tim_cac_lop(model)
    lop = list(range(len(cac_lop))) if lop is None else [int(x) for x in lop]
    vi_tri = [int(v) for v in vi_tri]
    thiet_bi = thiet_bi_cua(model)
    d = model.get_input_embeddings().weight.shape[1]
    tong = torch.zeros((len(vi_tri), len(lop), d), dtype=torch.float64, device=thiet_bi)
    tong_do_lon = torch.zeros((len(vi_tri), len(lop)), dtype=torch.float64, device=thiet_bi)
    n = len(cau_hoi)

    def ghi(j: int, h: torch.Tensor):
        x = h[:, vi_tri, :].to(device=thiet_bi, dtype=torch.float64)     # [B, P, d]
        tong[:, j] += x.sum(dim=0)
        tong_do_lon[:, j] += x.norm(dim=-1).sum(dim=0)

    def tao_hook(j: int):
        if diem == "ra_khoi":
            def hook(module, args, output):
                ghi(j, output[0] if isinstance(output, tuple) else output)
            return hook

        def hook_vao(module, args, kwargs):
            ghi(j, args[0] if len(args) > 0 else kwargs["hidden_states"])
        return hook_vao

    handles = []
    try:
        for j, l in enumerate(lop):
            if diem == "ra_khoi":
                handles.append(cac_lop[l].register_forward_hook(tao_hook(j)))
            else:
                handles.append(cac_lop[l].register_forward_pre_hook(tao_hook(j), with_kwargs=True))
        for a, b in chia_lo(n, co_lo):
            enc = ma_hoa(tok, cau_hoi[a:b], dac_ta).den(thiet_bi)
            if -min(vi_tri) > enc.do_dai_dem:
                raise ValueError("Vị trí âm vượt quá độ dài lời nhắc.")
            model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return KetQuaKichHoat(vi_tri, lop, diem, (tong / n).cpu(), (tong_do_lon / n).cpu(), n)
