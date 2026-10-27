"""Đặc tả mô hình, nạp mô hình và tìm các lớp decoder.

Nguồn số liệu:
- Llama-3-8B-Instruct: Nhap mục 3.2 (khối 5 token sau câu hỏi), 3.5 (token kết thúc),
  phần "Kiểm pipeline" (dừng ở <|end_of_text|> khi tái lập Arditi et al.).
- Qwen3.5-9B, Gemma 4 12B: Ghi_chu_ky_thuat_hai_mo_hinh_chinh_24_10_2026 mục 1, 2, 4. Các giá trị
  ghi "dự kiến" (số token của khối sau câu hỏi, chuỗi khối thinking) phải xác nhận khi nạp ở bước 3;
  code tự đo các giá trị này từ tokenizer thật (khuon.py) và báo nếu lệch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class DacTaMoHinh:
    ten: str
    hf_id: str
    so_lop: int
    # Token kết thúc dùng cho pipeline chính (giải mã greedy, đặt tường minh — Nhap mục 3.5).
    eos_chinh: tuple[int, ...]
    # Tham số truyền thêm cho apply_chat_template (ví dụ enable_thinking=False).
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    # Chuỗi khối thinking rỗng mà template tự chèn ở cuối lời nhắc (None nếu không có).
    # Vị trí lấy kích hoạt là token cuối TRƯỚC khối này (Nhap, Dựng hướng điểm 2).
    chuoi_thinking: str | None = None
    # Số token dự kiến của khối template sau câu hỏi; None = chưa xác nhận, chỉ đo và báo.
    so_token_khoi_du_kien: int | None = None
    # Có gán pad_token = eos_token không (tokenizer không có pad token sẵn).
    pad_bang_eos: bool = False
    # Lớp full attention (đánh số từ 0), chỉ để vẽ và để kiểm khi nạp.
    lop_full_attention: tuple[int, ...] = ()
    ghi_chu: str = ""


LLAMA3_8B = DacTaMoHinh(
    ten="llama3_8b",
    hf_id="meta-llama/Meta-Llama-3-8B-Instruct",
    so_lop=32,
    # <|eot_id|> (128009) cùng các id trong generation_config.json ([128001, 128009]); khi nạp,
    # code hợp thêm các id thật đọc từ generation_config để chắc không sót (xem eos_tu_mo_hinh).
    eos_chinh=(128009, 128001),
    template_kwargs={},
    chuoi_thinking=None,
    so_token_khoi_du_kien=5,
    pad_bang_eos=True,
    lop_full_attention=tuple(range(32)),
    ghi_chu="Chỉ dùng để kiểm pipeline (bước 1), không đưa vào phân tích.",
)

# Token kết thúc của phần tái lập Arditi et al.: theo Nhap, bảng "Chi tiết phần 1", dòng "Sinh văn bản"
# (mô hình viết tiếp qua <|eot_id|>, dừng ở <|end_of_text|> = 128001).
# Ghi chú của người rà soát code (27/10): bộ kết quả tác giả công bố (transformers 4.41.2) thực ra sinh đủ
# 512 token, không dừng sớm (eos để trống); với transformers 4.44.2 ở commit 9d852fae thì generate lấy eos từ
# generation_config của mô hình. Ta giữ đúng Nhap; script báo số câu chạm 128001 trong ket_qua.json để thấy
# khác biệt với bản không dừng sớm (chỉ ở các câu này, phần sau 128001 không được sinh).
EOS_ARDITI_LLAMA3 = (128001,)

QWEN35_9B = DacTaMoHinh(
    ten="qwen35_9b",
    hf_id="Qwen/Qwen3.5-9B",
    so_lop=32,
    eos_chinh=(248046, 248044),
    template_kwargs={"enable_thinking": False},
    chuoi_thinking="<think>\n\n</think>\n\n",
    so_token_khoi_du_kien=None,   # dự kiến 9; xác nhận ở bước 3
    pad_bang_eos=False,
    lop_full_attention=tuple(range(3, 32, 4)),
    ghi_chu="Mô hình chính. Đường dẫn lớp, cách tách token cần xác nhận khi nạp (bước 3).",
)

GEMMA4_12B = DacTaMoHinh(
    ten="gemma4_12b",
    hf_id="google/gemma-4-12B-it",
    so_lop=48,
    eos_chinh=(1, 106, 50),
    template_kwargs={"enable_thinking": False},
    chuoi_thinking="<|channel>thought\n<channel|>",
    so_token_khoi_du_kien=None,   # dự kiến 9; xác nhận ở bước 3
    pad_bang_eos=False,
    lop_full_attention=tuple(range(5, 48, 6)),
    ghi_chu="Mô hình chính. id 106 cần xác nhận là <turn|> khi nạp (bước 3).",
)

CAC_MO_HINH = {d.ten: d for d in (LLAMA3_8B, QWEN35_9B, GEMMA4_12B)}

# Các đường dẫn thường gặp tới danh sách lớp decoder, thử theo thứ tự.
_DUONG_LOP = (
    "model.layers",                    # LlamaForCausalLM, Qwen3_5ForCausalLM, Gemma4UnifiedForCausalLM
    "model.language_model.layers",     # Qwen3_5ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration
    "language_model.model.layers",
    "model.model.layers",
    "layers",                          # khi truyền thẳng mô hình gốc (không có đầu LM)
)


def _lay_thuoc_tinh(obj: Any, duong: str) -> Any:
    for ten in duong.split("."):
        obj = getattr(obj, ten)
    return obj


def tim_cac_lop(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Trả về ModuleList các lớp decoder. Báo lỗi nếu không tìm thấy đường dẫn nào."""
    for duong in _DUONG_LOP:
        try:
            lop = _lay_thuoc_tinh(model, duong)
        except AttributeError:
            continue
        if isinstance(lop, torch.nn.ModuleList) and len(lop) > 0:
            return lop
    raise AttributeError("Không tìm thấy danh sách lớp decoder trong mô hình "
                         f"{type(model).__name__}; đã thử {_DUONG_LOP}.")


def khoi_attn(lop: torch.nn.Module) -> torch.nn.Module:
    """Khối attention của một lớp (Llama: self_attn). Chỉ dùng cho chiếu bỏ kiểu Arditi."""
    for ten in ("self_attn", "linear_attn", "attention", "attn"):   # linear_attn: lớp GatedDeltaNet của Qwen3.5
        if hasattr(lop, ten):
            return getattr(lop, ten)
    raise AttributeError(f"Lớp {type(lop).__name__} không có khối attention quen thuộc.")


def khoi_mlp(lop: torch.nn.Module) -> torch.nn.Module:
    for ten in ("mlp", "feed_forward"):
        if hasattr(lop, ten):
            return getattr(lop, ten)
    raise AttributeError(f"Lớp {type(lop).__name__} không có khối MLP quen thuộc.")


def eos_tu_mo_hinh(model: torch.nn.Module, dac_ta: DacTaMoHinh) -> tuple[int, ...]:
    """Hợp eos_chinh của đặc tả với eos trong generation_config của mô hình (nếu có).

    Với Llama-3-8B-Instruct, Nhap ghi: <|eot_id|> (128009) cùng các id trong generation_config.json.
    Với Qwen3.5-9B (không có generation_config) và Gemma 4 12B, dùng đúng danh sách của đặc tả.
    """
    ids = list(dac_ta.eos_chinh)
    if dac_ta.ten == LLAMA3_8B.ten:
        gc = getattr(model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc is not None else None
        if e is not None:
            for x in ([e] if isinstance(e, int) else list(e)):
                if x not in ids:
                    ids.append(int(x))
    return tuple(ids)


def nap_mo_hinh(dac_ta: DacTaMoHinh, duong_dan: str | None = None, dtype: torch.dtype = torch.bfloat16,
                device_map: str | dict | None = "auto", revision: str | None = None):
    """Nạp (model, tokenizer) ở chế độ eval, không tính gradient, đệm trái.

    duong_dan: thư mục cục bộ hoặc id trên Hugging Face; mặc định dac_ta.hf_id.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    nguon = duong_dan or dac_ta.hf_id
    tok = AutoTokenizer.from_pretrained(nguon, revision=revision)
    tok.padding_side = "left"
    if dac_ta.pad_bang_eos or tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(nguon, dtype=dtype, device_map=device_map, revision=revision)
    model.eval()
    model.requires_grad_(False)
    so_lop = len(tim_cac_lop(model))
    if so_lop != dac_ta.so_lop:
        raise ValueError(f"{dac_ta.ten}: mô hình có {so_lop} lớp, đặc tả ghi {dac_ta.so_lop}.")
    return model, tok


def phan_bo_thiet_bi(model: torch.nn.Module) -> dict[str, Any]:
    """Mô hình nằm trọn trên một thiết bị không? (device_map='auto' có thể đẩy lớp ra CPU hoặc đĩa)."""
    m = getattr(model, "hf_device_map", None) or {}
    noi = sorted({str(v) for v in m.values()})
    return {"hf_device_map_noi": noi, "mot_thiet_bi": len(noi) <= 1 and not any(x in ("cpu", "disk") for x in noi)}


def thiet_bi_cua(model: torch.nn.Module) -> torch.device:
    """Thiết bị của embedding đầu vào (nơi phải đặt input_ids)."""
    return model.get_input_embeddings().weight.device
