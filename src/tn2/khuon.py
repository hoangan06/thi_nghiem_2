"""Áp chat template, mã hoá, xác định khối template sau câu hỏi và vị trí lấy kích hoạt.

Quy ước (Nhap, mục 3.2 và Dựng hướng điểm 2):
- "Khối template sau câu hỏi" = mọi token sau nội dung câu hỏi (Llama-3: 5 token; Qwen3.5, Gemma 4: dự kiến 9).
- Vị trí lấy kích hoạt = token cuối của phần mở đầu lượt trả lời, KHÔNG tính khối thinking rỗng
  mà template tự chèn (Llama-3 không có khối này nên là token cuối, tức −1).

Cả hai đại lượng được ĐO từ tokenizer thật, không gõ tay: tách văn bản đã áp template ở cuối câu hỏi
(và ở đầu khối thinking), mã hoá phần đầu riêng rồi so với mã hoá của cả chuỗi. Nếu phần đầu không
khớp đúng tiền tố (token vắt qua ranh giới), hàm báo lỗi thay vì đoán.

Lời nhắc được đệm TRÁI, nên với mọi câu trong lô, khối sau câu hỏi luôn là K vị trí cuối của chuỗi đã đệm,
và vị trí lấy kích hoạt luôn là chỉ số âm vi_tri_lay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .mo_hinh import DacTaMoHinh


@dataclass
class LoiNhac:
    cau_hoi: list[str]
    van_ban: list[str]              # văn bản đã áp template
    ids_rieng: list[list[int]]      # id từng câu, không đệm
    input_ids: torch.Tensor         # [B, L], đệm trái
    attention_mask: torch.Tensor    # [B, L]
    so_token_khoi: int              # K: số token của khối template sau câu hỏi
    vi_tri_lay: int                 # chỉ số âm của vị trí lấy kích hoạt (ví dụ −1)

    @property
    def do_dai_dem(self) -> int:
        return int(self.input_ids.shape[1])

    def __len__(self) -> int:
        return len(self.cau_hoi)

    def den(self, thiet_bi: torch.device | str) -> "LoiNhac":
        self.input_ids = self.input_ids.to(thiet_bi)
        self.attention_mask = self.attention_mask.to(thiet_bi)
        return self


def ap_template(tok, cau_hoi: str, dac_ta: DacTaMoHinh) -> str:
    """Văn bản lời nhắc một lượt user, có phần mở đầu lượt trả lời (add_generation_prompt=True)."""
    return tok.apply_chat_template(
        [{"role": "user", "content": cau_hoi}],
        tokenize=False,
        add_generation_prompt=True,
        **dac_ta.template_kwargs,
    )


def _ma(tok, van_ban: str) -> list[int]:
    # Văn bản đã áp template đã chứa BOS (nếu mô hình dùng BOS), nên KHÔNG thêm token đặc biệt.
    return list(tok(van_ban, add_special_tokens=False)["input_ids"])


def _so_token_sau(tok, van_ban: str, ids: list[int], cat: int, ten: str) -> int:
    dau = _ma(tok, van_ban[:cat])
    if ids[: len(dau)] != dau:
        raise ValueError(f"Mã hoá phần trước {ten} không phải tiền tố của mã hoá cả chuỗi "
                         f"(token vắt qua ranh giới). Văn bản: {van_ban!r}")
    return len(ids) - len(dau)


def phan_tich_mot_cau(tok, cau_hoi: str, dac_ta: DacTaMoHinh) -> tuple[str, list[int], int, int]:
    """Trả về (văn bản, ids, K, vi_tri_lay) cho một câu hỏi."""
    van_ban = ap_template(tok, cau_hoi, dac_ta)
    ids = _ma(tok, van_ban)
    # Template có thể cắt khoảng trắng hai đầu nội dung (Llama-3: `| trim`), nên tìm cả hai dạng.
    vt, chuoi = van_ban.rfind(cau_hoi), cau_hoi
    if vt < 0:
        chuoi = cau_hoi.strip()
        vt = van_ban.rfind(chuoi)
    if vt < 0 or not chuoi:
        raise ValueError(f"Không tìm thấy câu hỏi trong văn bản đã áp template: {cau_hoi!r}")
    het_cau = vt + len(chuoi)
    k = _so_token_sau(tok, van_ban, ids, het_cau, "cuối câu hỏi")
    if dac_ta.chuoi_thinking is None:
        vi_tri_lay = -1
    else:
        if not van_ban.endswith(dac_ta.chuoi_thinking):
            raise ValueError(f"{dac_ta.ten}: lời nhắc không kết thúc bằng khối thinking "
                             f"{dac_ta.chuoi_thinking!r}: {van_ban[-80:]!r}")
        dau_thinking = len(van_ban) - len(dac_ta.chuoi_thinking)
        if dau_thinking < het_cau:
            raise ValueError("Khối thinking nằm trước cuối câu hỏi.")
        so_token_thinking = _so_token_sau(tok, van_ban, ids, dau_thinking, "khối thinking")
        if so_token_thinking >= k:
            raise ValueError("Khối thinking chiếm hết khối sau câu hỏi.")
        vi_tri_lay = -(so_token_thinking + 1)
    return van_ban, ids, k, vi_tri_lay


def ma_hoa(tok, cau_hoi: Sequence[str], dac_ta: DacTaMoHinh) -> LoiNhac:
    """Mã hoá một lô câu hỏi, đệm trái bằng tok.pad_token_id.

    Báo lỗi nếu K hoặc vi_tri_lay khác nhau giữa các câu, hoặc K khác so_token_khoi_du_kien của đặc tả.
    """
    if len(cau_hoi) == 0:
        raise ValueError("Lô rỗng.")
    if tok.pad_token_id is None:
        raise ValueError("Tokenizer chưa có pad_token_id.")
    van_ban, ids_rieng, ks, vts = [], [], set(), set()
    for q in cau_hoi:
        vb, ids, k, vt = phan_tich_mot_cau(tok, q, dac_ta)
        van_ban.append(vb)
        ids_rieng.append(ids)
        ks.add(k)
        vts.add(vt)
    if len(ks) != 1 or len(vts) != 1:
        raise ValueError(f"Số token khối sau câu hỏi {sorted(ks)} hoặc vị trí lấy {sorted(vts)} khác nhau giữa các câu.")
    k, vt = ks.pop(), vts.pop()
    if dac_ta.so_token_khoi_du_kien is not None and k != dac_ta.so_token_khoi_du_kien:
        raise ValueError(f"{dac_ta.ten}: khối sau câu hỏi có {k} token, đặc tả ghi {dac_ta.so_token_khoi_du_kien}.")
    dai = max(len(x) for x in ids_rieng)
    input_ids = torch.full((len(ids_rieng), dai), int(tok.pad_token_id), dtype=torch.long)
    mask = torch.zeros((len(ids_rieng), dai), dtype=torch.long)
    for i, x in enumerate(ids_rieng):
        input_ids[i, dai - len(x):] = torch.tensor(x, dtype=torch.long)
        mask[i, dai - len(x):] = 1
    return LoiNhac(list(cau_hoi), van_ban, ids_rieng, input_ids, mask, k, vt)


def chia_lo(n: int, co_lo: int) -> list[tuple[int, int]]:
    """Các khoảng [đầu, cuối) chia n phần tử thành lô cỡ co_lo, giữ thứ tự (như code tác giả)."""
    if co_lo <= 0:
        raise ValueError("co_lo phải dương.")
    return [(i, min(i + co_lo, n)) for i in range(0, n, co_lo)]
