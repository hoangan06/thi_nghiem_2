"""Ghi nguồn gốc của một lượt chạy: phiên bản thư viện, mô hình, GPU, dữ liệu, commit của kho.

Không chạy git (Ân tự chạy git): commit đọc trực tiếp từ thư mục .git nếu có.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
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


def thong_tin_mo_hinh(model, tok, dac_ta) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "dac_ta": dac_ta.ten,
        "nguon": getattr(cfg, "_name_or_path", None),
        "commit_hf": getattr(cfg, "_commit_hash", None),
        "dtype": str(getattr(model, "dtype", None)),
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


def ghi_json(duong: str | Path, obj: Any) -> None:
    duong = Path(duong)
    duong.parent.mkdir(parents=True, exist_ok=True)
    with open(duong, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_bo_nan(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def ghi_jsonl(duong: str | Path, dong: Iterable[dict]) -> None:
    duong = Path(duong)
    duong.parent.mkdir(parents=True, exist_ok=True)
    with open(duong, "w", encoding="utf-8", newline="\n") as f:
        for d in dong:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def doc_jsonl(duong: str | Path) -> list[dict]:
    with open(duong, encoding="utf-8") as f:
        return [json.loads(d) for d in f if d.strip()]


def thu_muc_moi(goc: str | Path, ten: str) -> Path:
    """Tạo thư mục kết quả mới; báo lỗi nếu đã tồn tại (không ghi đè lượt cũ)."""
    p = Path(goc) / ten
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f"Thư mục kết quả đã có dữ liệu: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def bien_moi_truong_quan_trong() -> dict[str, str | None]:
    return {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_OFFLINE")}
