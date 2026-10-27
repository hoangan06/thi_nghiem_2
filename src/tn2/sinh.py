"""Sinh greedy theo lô (đệm trái, bf16, KV cache), có hoặc không có can thiệp — Nhap mục 3.5.

Mọi tham số giải mã đặt tường minh: do_sample=False, num_beams=1, repetition_penalty=1.0,
danh sách token kết thúc truyền vào. Không phụ thuộc generation_config mặc định của mô hình
(Qwen3.5 không có file này; Gemma 4 mặc định lấy mẫu ngẫu nhiên) — có test kiểm điều này.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .can_thiep import CanThiep, gan_can_thiep
from .khuon import LoiNhac, chia_lo, ma_hoa
from .mo_hinh import DacTaMoHinh, thiet_bi_cua


@dataclass
class KetQuaSinh:
    loi_nhac: LoiNhac
    token_day_du: torch.Tensor          # [B, T] token sinh (sau khi xong, phần còn lại là pad)
    so_hop_le: list[int]                # số token hợp lệ mỗi câu (tính cả token kết thúc nếu có)
    cham_eos: list[bool]                # câu có dừng ở token kết thúc trước giới hạn không
    van_ban: list[str]
    logits: torch.Tensor | None         # [B, T, V] float32 trên thiết bị của mô hình (khi lay_logit=True)

    def token_hop_le(self, i: int) -> list[int]:
        return self.token_day_du[i, : self.so_hop_le[i]].tolist()


def cau_hinh_greedy(max_new_tokens: int, eos_ids: Sequence[int], pad_id: int):
    from transformers import GenerationConfig

    return GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=1,
        repetition_penalty=1.0,
        # transformers 5 điền mọi trường để trống (None) từ generation_config của mô hình; đặt rõ hai trường
        # có thể đổi token chọn khi giải mã greedy. scripts/buoc1_kiem_truoc.py in cấu hình hiệu lực cuối cùng.
        no_repeat_ngram_size=0,
        min_new_tokens=0,          # được ưu tiên hơn min_length nên không cần đặt min_length
        # Gemma 4 12B đặt suppress_tokens = [258883, 258882] (<audio|>, <image|>) trong generation_config.json;
        # để None thì transformers điền lại và ép logit hai token đó về −∞, không còn greedy thuần (phát hiện
        # bởi buoc3_kiem_truoc trên tokenizer thật, 27/10/2026; Ân duyệt). [] được giữ nguyên, không chặn token nào.
        # Llama-3 và Qwen3.5 không có trường này nên kết quả của chúng không đổi.
        suppress_tokens=[],
        eos_token_id=[int(x) for x in eos_ids],
        pad_token_id=int(pad_id),
    )


def _so_hop_le(token: torch.Tensor, eos_ids: Sequence[int]) -> tuple[list[int], list[bool]]:
    eos = torch.tensor(list(eos_ids), device=token.device)
    la_eos = torch.isin(token, eos)
    so, cham = [], []
    for i in range(token.shape[0]):
        idx = torch.nonzero(la_eos[i]).flatten()
        if idx.numel() > 0:
            so.append(int(idx[0]) + 1)
            cham.append(True)
        else:
            so.append(int(token.shape[1]))
            cham.append(False)
    return so, cham


@torch.no_grad()
def sinh_tham_lam(model, tok, enc: LoiNhac, max_new_tokens: int, eos_ids: Sequence[int],
                  can_thiep: Sequence[CanThiep] | None = None, lay_logit: bool = False,
                  giai_ma: str = "den_eos", loi_gia_lap: str | None = None) -> KetQuaSinh:
    """Sinh cho một lô đã mã hoá.

    giai_ma:
    - "den_eos": giải mã các token trước token kết thúc đầu tiên, giữ token đặc biệt khác (để phát hiện
      tự sinh lượt hội thoại mới / token điều khiển — Nhap mục 4.1, 4.4).
    - "arditi" : giải mã cả hàng token sinh với skip_special_tokens=True rồi strip(), đúng như
      model_base.generate_completions của Arditi et al.
    """
    if giai_ma not in ("den_eos", "arditi"):
        raise ValueError(f"giai_ma không hợp lệ: {giai_ma}")
    thiet_bi = thiet_bi_cua(model)
    enc.den(thiet_bi)
    L = enc.do_dai_dem
    gc = cau_hinh_greedy(max_new_tokens, eos_ids, tok.pad_token_id)
    gc.return_dict_in_generate = True
    gc.output_logits = bool(lay_logit)      # logit thô (trước bộ xử lý logit), dùng cho phép kiểm (i), (iii)
    if not can_thiep and loi_gia_lap is None:
        # "Không gắn hook" đúng nghĩa: không đăng ký cả bộ đếm vị trí (phép (ii) và kiểm "không gắn hook" ở pilot).
        out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, generation_config=gc)
    else:
        with gan_can_thiep(model, can_thiep, do_dai_loi_nhac=L, loi_gia_lap=loi_gia_lap):
            out = model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, generation_config=gc)
    token = out.sequences[:, L:]
    so, cham = _so_hop_le(token, eos_ids)
    van_ban = []
    for i in range(token.shape[0]):
        if giai_ma == "arditi":
            van_ban.append(tok.decode(token[i], skip_special_tokens=True).strip())
        else:
            dung = so[i] - 1 if cham[i] else so[i]
            van_ban.append(tok.decode(token[i, :dung], skip_special_tokens=False))
    logits = None
    if lay_logit:
        logits = torch.stack([x.to(torch.float32) for x in out.logits], dim=1)
    return KetQuaSinh(enc, token, so, cham, van_ban, logits)


def sinh_theo_lo(model, tok, dac_ta: DacTaMoHinh, cau_hoi: Sequence[str], co_lo: int, max_new_tokens: int,
                 eos_ids: Sequence[int], tao_can_thiep=None, giai_ma: str = "den_eos") -> list[dict]:
    """Sinh cho cả danh sách câu, chia lô giữ thứ tự (đệm theo từng lô như code của Arditi et al.).

    tao_can_thiep: None, hoặc hàm (LoiNhac) -> list[CanThiep] (cần K của lô để dựng vùng).
    Trả về list dict: cau_hoi, van_ban, token, so_token, cham_eos.
    """
    kq = []
    for a, b in chia_lo(len(cau_hoi), co_lo):
        enc = ma_hoa(tok, cau_hoi[a:b], dac_ta)
        ct = tao_can_thiep(enc) if tao_can_thiep is not None else None
        r = sinh_tham_lam(model, tok, enc, max_new_tokens, eos_ids, can_thiep=ct, giai_ma=giai_ma)
        for i in range(len(enc)):
            kq.append({"cau_hoi": enc.cau_hoi[i], "van_ban": r.van_ban[i], "token": r.token_hop_le(i),
                       "so_token": r.so_hop_le[i], "cham_eos": r.cham_eos[i]})
    return kq
