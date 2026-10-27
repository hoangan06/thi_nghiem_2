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
