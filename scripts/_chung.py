"""Phần dùng chung của các script: thêm src vào sys.path, ghi nhật ký."""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
if str(GOC / "src") not in sys.path:
    sys.path.insert(0, str(GOC / "src"))


class NhatKy:
    def __init__(self, duong: Path):
        self.duong = duong
        duong.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, thong_diep: str) -> None:
        dong = f"[{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {thong_diep}"
        print(dong, flush=True)
        with open(self.duong, "a", encoding="utf-8", newline="\n") as f:
            f.write(dong + "\n")


def ten_luot_mac_dinh(tien_to: str) -> str:
    return f"{tien_to}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"


def chon_dac_ta(ten: str):
    """Đặc tả của một mô hình chính theo tên (qwen35_9b | gemma4_12b)."""
    from tn2.mo_hinh import GEMMA4_12B, QWEN35_9B

    bang = {QWEN35_9B.ten: QWEN35_9B, GEMMA4_12B.ten: GEMMA4_12B}
    if ten not in bang:
        raise SystemExit(f"--mo_hinh_dac_ta phải là một trong {sorted(bang)}")
    return bang[ten]


def thiet_lap_so_hoc() -> dict:
    """Tắt TF32 (để lượt float32 đúng là float32) và ghi lại cấu hình số học."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def device_map_mot_gpu():
    """Ép mô hình lên GPU 0 khi có CUDA (thiếu bộ nhớ thì báo lỗi ngay khi nạp); không có CUDA thì để trên CPU."""
    import torch

    return {"": 0} if torch.cuda.is_available() else None
