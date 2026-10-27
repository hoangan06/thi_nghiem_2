"""Hướng từ chối: hiệu trung bình, chuẩn hoá, cosine, chia đôi (Nhap, Dựng hướng điểm 1–8).

- r_ℓ = trung bình kích hoạt câu có hại − trung bình kích hoạt câu vô hại (không lọc theo hành vi).
- r̂_ℓ = r_ℓ / ‖r_ℓ‖; lưu cả độ lớn thô ‖r_ℓ‖ và độ lớn tương đối ‖r_ℓ‖ / s_ℓ.
- s_ℓ = độ lớn trung bình của luồng dư ở lớp ℓ, tại vị trí lấy kích hoạt, trên tập dựng gộp.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch


def hieu_trung_binh(tb_co_hai: torch.Tensor, tb_vo_hai: torch.Tensor) -> torch.Tensor:
    if tb_co_hai.shape != tb_vo_hai.shape:
        raise ValueError("Hai trung bình khác kích thước.")
    return (tb_co_hai.to(torch.float64) - tb_vo_hai.to(torch.float64))


def chuan_hoa(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trả về (v̂, ‖v‖) theo chiều cuối. Báo lỗi nếu có vector bằng 0."""
    n = v.norm(dim=-1, keepdim=True)
    if bool((n == 0).any()):
        raise ValueError("Có hướng có độ lớn bằng 0.")
    return v / n, n.squeeze(-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def chia_doi(n: int, seed: int) -> tuple[list[int], list[int]]:
    """Chia ngẫu nhiên (có seed) các chỉ số 0..n−1 thành hai nửa rời nhau; nửa đầu nhận phần lẻ."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    k = (n + 1) // 2
    return sorted(idx[:k]), sorted(idx[k:])


def lay_con(ds: Sequence, chi_so: Sequence[int]) -> list:
    return [ds[i] for i in chi_so]
