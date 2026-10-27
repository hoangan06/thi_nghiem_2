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
    # ── Chỉ dùng cho hai mô hình chính (bước 3) ──
    # Các token (giải mã từng token) dự kiến của khối sau câu hỏi; script kiểm trước so với tokenizer thật.
    khoi_du_kien: tuple[str, ...] = ()
    # Chỉ số âm dự kiến của vị trí lấy kích hoạt (token cuối phần mở đầu lượt trả lời, không tính khối thinking).
    vi_tri_lay_du_kien: int | None = None
    # Chuỗi đánh dấu khối thinking: câu trả lời chứa một trong các chuỗi này là "rò thinking" (Nhap Dựng hướng
    # điểm 5; mục 7.7 điều kiện 3).
    dau_hieu_thinking: tuple[str, ...] = ()
    # Ba lớp cho bản thu nhỏ phần 2 (Nhap, Kiểm pipeline): có cả hai loại khối, lớp giữa là full attention.
    lop_kiem_hook: tuple[int, ...] = ()
    # Token kết thúc phải giải mã ra đúng chuỗi này (kiểm khi nạp tokenizer).
    eos_du_kien: tuple[tuple[int, str], ...] = ()


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
    # Nhap mục 3.2: dự kiến 9 token. Đặt cứng 9 để mọi lượt chạy dừng ngay nếu tokenizer thật tách khác;
    # scripts/buoc3_kiem_truoc.py xác nhận trên tokenizer thật trước khi nạp trọng số.
    so_token_khoi_du_kien=9,
    pad_bang_eos=False,
    lop_full_attention=tuple(range(3, 32, 4)),
    ghi_chu="Mô hình chính. Khối 9 token và vị trí −5 xác nhận bằng scripts/buoc3_kiem_truoc.py.",
    khoi_du_kien=("<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>", "\n\n", "</think>", "\n\n"),
    vi_tri_lay_du_kien=-5,
    dau_hieu_thinking=("<think>", "</think>"),
    lop_kiem_hook=(8, 15, 24),
    eos_du_kien=((248046, "<|im_end|>"), (248044, "<|endoftext|>")),
)

GEMMA4_12B = DacTaMoHinh(
    ten="gemma4_12b",
    hf_id="google/gemma-4-12B-it",
    so_lop=48,
    eos_chinh=(1, 106, 50),
    template_kwargs={"enable_thinking": False},
    chuoi_thinking="<|channel>thought\n<channel|>",
    so_token_khoi_du_kien=9,      # như Qwen3.5-9B (Nhap mục 3.2)
    pad_bang_eos=False,
    lop_full_attention=tuple(range(5, 48, 6)),
    ghi_chu="Mô hình chính. id 106 phải là <turn|> (Nhap mục 3.5); kiểm bằng scripts/buoc3_kiem_truoc.py.",
    khoi_du_kien=("<turn|>", "\n", "<|turn>", "model", "\n", "<|channel>", "thought", "\n", "<channel|>"),
    vi_tri_lay_du_kien=-5,
    dau_hieu_thinking=("<|channel>", "<channel|>"),
    lop_kiem_hook=(12, 23, 36),
    # Nhap chỉ yêu cầu xác nhận id 106; id 1 và 50 được in ra để đối chiếu, không đặt chuỗi dự kiến.
    eos_du_kien=((106, "<turn|>"),),
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


def phan_bo_thiet_bi(model: torch.nn.Module, can_gpu: bool | None = None) -> dict[str, Any]:
    """Mô hình có nằm trọn trên MỘT thiết bị không, kiểm trên từng tensor thật (tham số VÀ buffer).

    Sửa sau bước 1 (báo cáo kết quả bước 1, mục 6.2): bản cũ chỉ đọc hf_device_map, mà bản đồ này rỗng khi nạp
    với device_map={"": 0} hay không có device_map, nên phép kiểm đạt mà không kiểm gì. Giờ duyệt mọi
    named_parameters() và named_buffers() (hệ số layer_scalar của Gemma 4 là buffer theo mã transformers 5.17.0).
    can_gpu: True → thiết bị duy nhất phải là GPU (cuda); None → tự đặt True nếu máy có CUDA.
    """
    if can_gpu is None:
        can_gpu = torch.cuda.is_available()
    dem: dict[str, int] = {}
    ten_theo_tb: dict[str, list[str]] = {}
    for loai, it in (("tham_so", model.named_parameters()), ("buffer", model.named_buffers())):
        for ten, t in it:
            tb = str(t.device)
            dem[tb] = dem.get(tb, 0) + 1
            ten_theo_tb.setdefault(tb, []).append(f"{loai}:{ten}")
    thiet_bi = sorted(dem)
    chinh = max(dem, key=dem.get) if dem else None
    lech = [x for tb, ds in ten_theo_tb.items() if tb != chinh for x in ds]
    mot = len(thiet_bi) == 1 and thiet_bi[0] not in ("meta", "disk")
    if can_gpu:
        mot = mot and thiet_bi[0].startswith("cuda")
    m = getattr(model, "hf_device_map", None) or {}
    return {"thiet_bi": thiet_bi, "so_tensor_theo_thiet_bi": dem, "thiet_bi_chinh": chinh,
            "tensor_lech_thiet_bi": lech[:50], "so_tensor_lech": len(lech), "can_gpu": bool(can_gpu),
            "hf_device_map_noi": sorted({str(v) for v in m.values()}), "mot_thiet_bi": bool(mot)}


def thiet_bi_cua(model: torch.nn.Module) -> torch.device:
    """Thiết bị của embedding đầu vào (nơi phải đặt input_ids)."""
    return model.get_input_embeddings().weight.device


# ───────────── backend của lớp attention tuyến tính (Qwen3.5: Gated DeltaNet + conv1d nhân quả) ─────────────
# transformers 5.17 (integrations/hub_kernels.use_kernel_func_from_hub_with_fallback) chọn cài đặt lúc import
# module: có gói `fla` / `causal_conv1d` thì dùng kernel của gói, không thì dùng bản PyTorch tham chiếu; không xét
# kiểu số. Kernel Triton của flash-linear-attention 0.6.0 dùng TF32 cho tl.dot (fla/ops/gated_delta_rule/
# chunk_fwd.py: SOLVE_TRIL_DOT_PRECISION = 'tf32'; Triton mặc định TF32 với đầu vào float32), cờ TF32 của PyTorch
# không chi phối. Nhap yêu cầu tắt TF32 ở lượt float32 của phần 2, nên lượt đó chuyển về bản PyTorch tham chiếu
# (rà soát code bước 3, R2 D1; Ân duyệt Q4 ngày 27/10/2026). Hàm gốc lấy qua __wrapped__ (functools.wraps).
TEN_HAM_TUYEN_TINH = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                      "causal_conv1d_fn", "causal_conv1d_update")


def _module_cac_lop(model: torch.nn.Module):
    import importlib

    lop = tim_cac_lop(model)[0]
    return importlib.import_module(type(lop).__module__)


def _cai_dat_thuc(f) -> Any:
    """Hàm thực sự được gọi bên trong wrapper của transformers (biến `implementation` trong closure)."""
    import inspect

    try:
        nl = inspect.getclosurevars(f).nonlocals
    except TypeError:
        nl = {}
    return nl["implementation"] if "implementation" in nl else inspect.unwrap(f)


def backend_lop_tuyen_tinh(model: torch.nn.Module) -> dict[str, Any]:
    """Với từng hàm của lớp attention tuyến tính: cài đặt đang dùng và có phải bản PyTorch tham chiếu không.
    Rỗng với mô hình không có các hàm này (Llama, Gemma 4)."""
    import inspect

    mod = _module_cac_lop(model)
    ra: dict[str, Any] = {}
    for ten in TEN_HAM_TUYEN_TINH:
        f = getattr(mod, ten, None)
        if f is None:
            continue
        thuc = _cai_dat_thuc(f)
        goc = inspect.unwrap(f)
        ra[ten] = {"cai_dat": f"{getattr(thuc, '__module__', '?')}.{getattr(thuc, '__qualname__', repr(thuc))}",
                   "la_ban_tham_chieu": thuc is goc}
    return ra


def _goi_loc_tham_so(fn):
    """Bọc hàm gốc, chỉ truyền các tham số từ khoá mà hàm nhận (như wrapper của transformers)."""
    import functools
    import inspect

    sig = inspect.signature(fn)
    nhan_tat_ca = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    ten = set(sig.parameters)

    @functools.wraps(fn)
    def goi(*args, **kwargs):
        if not nhan_tat_ca:
            kwargs = {k: v for k, v in kwargs.items() if k in ten}
        return fn(*args, **kwargs)
    return goi


class ban_tham_chieu_pytorch:
    """Ngữ cảnh: trong khối `with`, các hàm của lớp attention tuyến tính chạy bằng bản PyTorch tham chiếu
    (không qua kernel fla / causal_conv1d). Ra khỏi khối thì trả lại như cũ. Không có tác dụng với mô hình
    không có các hàm này. `.truoc` / `.sau` ghi backend trước và trong khối."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._cu: dict[str, Any] = {}
        self.truoc: dict[str, Any] = {}
        self.sau: dict[str, Any] = {}

    def __enter__(self):
        import inspect

        self.truoc = backend_lop_tuyen_tinh(self.model)
        self._mod = _module_cac_lop(self.model)
        for ten in self.truoc:
            f = getattr(self._mod, ten)
            self._cu[ten] = f
            setattr(self._mod, ten, _goi_loc_tham_so(inspect.unwrap(f)))
        self.sau = backend_lop_tuyen_tinh(self.model)
        if not all(v["la_ban_tham_chieu"] for v in self.sau.values()):
            self.__exit__(None, None, None)
            raise RuntimeError(f"Không chuyển được về bản PyTorch tham chiếu: {self.sau}")
        return self

    def __exit__(self, *exc):
        for ten, f in self._cu.items():
            setattr(self._mod, ten, f)
        return False
