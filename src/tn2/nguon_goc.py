"""Ghi nguồn gốc của một lượt chạy: phiên bản thư viện, mô hình, GPU, dữ liệu, commit của kho.

Không chạy git (Ân tự chạy git): commit đọc trực tiếp từ thư mục .git nếu có.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

GOC_KHO = Path(__file__).resolve().parents[2]


def sha256_file(duong: str | Path) -> str:
    h = hashlib.sha256()
    with open(duong, "rb") as f:
        for khoi in iter(lambda: f.read(1 << 20), b""):
            h.update(khoi)
    return h.hexdigest()


def dau_van_tay_ma_nguon(goc: Path = GOC_KHO) -> dict[str, Any]:
    """SHA của mọi file .py trong src/ và scripts/ (không cần git): chứng cứ code thật sự đã chạy, kể cả khi cây làm
    việc không sạch. Đổi code giữa hai phiên thì đầu ra có thể đổi."""
    goc = Path(goc)
    tep = sorted(list((goc / "src").rglob("*.py")) + list((goc / "scripts").glob("*.py")))
    tung = {str(p.relative_to(goc)).replace(os.sep, "/"): sha256_file(p) for p in tep if "__pycache__" not in p.parts}
    return {"sha256": hashlib.sha256(json.dumps(tung, sort_keys=True).encode("utf-8")).hexdigest(),
            "so_file": len(tung)}


def commit_kho(goc: Path = GOC_KHO) -> str | None:
    git = goc / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1].strip()
        f = git / ref
        if f.exists():
            return f.read_text().strip()
        packed = git / "packed-refs"
        if packed.exists():
            for dong in packed.read_text().splitlines():
                if dong.endswith(" " + ref):
                    return dong.split(" ", 1)[0]
    except OSError:
        return None
    return None


def thong_tin_moi_truong() -> dict[str, Any]:
    import torch
    import transformers

    tt: dict[str, Any] = {
        "thoi_gian_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "nen_tang": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "commit_kho": commit_kho(),
        "ma_nguon": dau_van_tay_ma_nguon(),
    }
    try:
        import tokenizers
        tt["tokenizers"] = tokenizers.__version__
    except ImportError:
        pass
    # Kernel của lớp Gated DeltaNet (Qwen3.5): có thì transformers dùng, không có thì tự chuyển sang bản PyTorch
    # thuần (chậm hơn, cùng kết quả về mặt toán). Ghi lại để biết backend thực dùng (báo cáo kết quả bước 1, mục 7).
    import importlib.metadata as _md
    # Gói `kernels` (kernel từ HF Hub) không được cài; có nó thì transformers bọc thêm một lớp quanh các hàm của
    # lớp DeltaNet (đối chiếu code bước 3, R1 P2). Runbook đặt USE_HUB_KERNELS=NO; ghi cả hai để kiểm.
    tt["bien_USE_HUB_KERNELS"] = os.environ.get("USE_HUB_KERNELS")
    for goi in ("flash-linear-attention", "causal-conv1d", "flash_attn", "kernels"):
        try:
            tt[f"goi_{goi}"] = _md.version(goi)
        except _md.PackageNotFoundError:
            tt[f"goi_{goi}"] = None
    return tt


def ho_gpu(ten: str) -> str:
    """Họ GPU dùng cho phép so môi trường giữa các phiên: mọi bản A100 (SXM4/PCIe, 40/80 GB; cùng chip GA100, cùng
    kiến trúc sm80) là "NVIDIA A100"; GPU khác giữ nguyên tên. Ân quyết ngày 29/10/2026: "cùng loại GPU" nghĩa là cùng
    A100."""
    t = str(ten)
    # \bA100\b: không khớp "RTX A1000"; lát MIG (số SM khác) không được coi là cùng họ.
    return "NVIDIA A100" if re.search(r"\bA100\b", t.upper()) and "MIG" not in t.upper() else t


def _kieu_attention(model) -> str | None:
    from .mo_hinh import kieu_attention
    return kieu_attention(model)


def thong_tin_mo_hinh(model, tok, dac_ta) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "dac_ta": dac_ta.ten,
        "nguon": getattr(cfg, "_name_or_path", None),
        "commit_hf": getattr(cfg, "_commit_hash", None),
        "dtype": str(getattr(model, "dtype", None)),
        # Kiểu attention thực dùng của phần chữ (sửa sau kết quả bước 3, Ân duyệt Q-E2 ngày 29/10/2026).
        "attn_implementation": _kieu_attention(model),
        "pad_token": tok.pad_token, "pad_token_id": tok.pad_token_id,
        "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
    }


def _bo_nan(obj: Any) -> Any:
    """Thay float NaN/±inf bằng None (JSON chuẩn không có NaN; sửa sau bước 1, báo cáo kết quả mục 6.1)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _bo_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_bo_nan(v) for v in obj]
    return obj


def fsync_thu_muc(thu_muc: str | Path) -> None:
    """fsync thư mục để phép os.replace vừa làm bền qua mất điện (hệ file không hỗ trợ thì bỏ qua)."""
    try:
        fd = os.open(str(thu_muc), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _ghi_tam_roi_thay(duong: Path, ghi) -> None:
    """Ghi qua file tạm ".<tên>.tam-<pid>" cùng thư mục + fsync + os.replace + fsync thư mục: file đích hoặc là bản
    cũ, hoặc là bản mới đầy đủ, không bao giờ dở dang (đối chiếu lần cuối, R2, 28/10/2026)."""
    duong.parent.mkdir(parents=True, exist_ok=True)
    tam = duong.with_name(f".{duong.name}.tam-{os.getpid()}")
    with open(tam, "w", encoding="utf-8", newline="\n") as f:
        ghi(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tam, duong)
    fsync_thu_muc(duong.parent)


def ghi_json(duong: str | Path, obj: Any) -> None:
    """Ghi JSON nguyên tử (NaN/inf → null)."""
    du_lieu = _bo_nan(obj)
    noi = json.dumps(du_lieu, ensure_ascii=False, indent=2, allow_nan=False) + "\n"   # lỗi tuần tự hoá: chưa đụng file
    _ghi_tam_roi_thay(Path(duong), lambda f: f.write(noi))


def ghi_jsonl(duong: str | Path, dong: Iterable[dict]) -> None:
    """Ghi JSONL nguyên tử."""
    noi = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dong)
    _ghi_tam_roi_thay(Path(duong), lambda f: f.write(noi))


def doc_jsonl(duong: str | Path) -> list[dict]:
    with open(duong, encoding="utf-8") as f:
        return [json.loads(d) for d in f if d.strip()]


def thu_muc_moi(goc: str | Path, ten: str) -> Path:
    """Tạo thư mục kết quả mới; báo lỗi nếu đã tồn tại (không ghi đè lượt cũ)."""
    p = Path(goc) / ten
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.mkdir()            # exist_ok=False: hai lượt khởi động cùng một giây không thể dùng chung một thư mục
    except FileExistsError:
        raise FileExistsError(f"Thư mục kết quả đã có: {p} (không ghi vào lượt cũ; chờ một giây rồi chạy lại "
                              f"hoặc đặt --ten_luot khác)") from None
    return p


def bien_moi_truong_quan_trong() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE",
                                           "USE_HUB_KERNELS")}
