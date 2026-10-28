"""Bước 4, chuẩn bị dữ liệu: chia tập chính / tập hiệu chỉnh / tập con pilot cho D_harm_test và D_plain_test
(Nhap, mục Dữ liệu) và chạy kiểm trùng trên bộ file cuối cùng. Chạy trên CPU, không cần mô hình.

Chạy:
    python scripts/buoc4_chia_tap.py            # tạo data/chia_tap/chia_tap.json (nếu chưa có)
    python scripts/buoc4_chia_tap.py --kiem     # chỉ tính lại và so với file đã có

Cách chia là tất định (seed và thứ tự rút trong tn2.tap_test). File kết quả nằm trong data/ (vào Git), không ghi
đè: nếu đã có thì chỉ tính lại và so; khác nhau thì báo lỗi, không sửa gì.
Mã thoát: 0 đạt (file mới tạo hoặc trùng file cũ, mọi ràng buộc đạt, kiểm trùng đạt trừ ngoại lệ đã biết
CHAP_NHAN_TRUNG trong tn2.tap_test); 1 lỗi hoặc khác file cũ;
2 ràng buộc hoặc kiểm trùng không đạt (vẫn ghi file nếu chưa có, để xem).
"""
from __future__ import annotations

import argparse
import json
import sys

from _chung import GOC  # noqa: F401  (đặt sys.path tới src/)

from tn2 import nguon_goc as NG
from tn2.tap_test import FILE_CHIA, chia_tap


def _tom_tat(d: dict) -> list[str]:
    dong = []
    for tap, p in d["tap"].items():
        dong.append(f"{tap}: chính {len(p['chinh'])}, hiệu chỉnh {len(p['hieu_chinh'])}, pilot {len(p['pilot'])}")
    for k, v in d["kiem_trung"].items():
        if isinstance(v, dict) and "trung_nguyen_van" in v:
            dong.append(f"  kiểm trùng {k}: nguyên văn {v['trung_nguyen_van']}, gần đúng {v['trung_gan_dung']}, "
                        f"chỉ số nguồn {v['trung_chi_so_nguon']}, Jaccard lớn nhất {v['jaccard_lon_nhat']} → {v['dat']}")
    kt = d["kiem_trung"]
    dong.append(f"Ràng buộc: {d['kiem_tra']}; kiểm trùng đạt: {kt['dat']}; ngoại lệ đã biết: "
                f"{list(kt['ngoai_le_chap_nhan'])}; đạt sau ngoại lệ: {kt['dat_sau_ngoai_le']}")
    return dong


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kiem", action="store_true", help="chỉ tính lại và so với file đã có")
    a = p.parse_args(argv)
    d = chia_tap()
    for dong in _tom_tat(d):
        print(dong)
    dat = d["kiem_tra"]["dat"] and d["kiem_trung"]["dat_sau_ngoai_le"]
    if FILE_CHIA.exists():
        cu = json.loads(FILE_CHIA.read_text(encoding="utf-8"))
        if cu != json.loads(json.dumps(d, ensure_ascii=False)):
            print(f"KHÁC file đã có {FILE_CHIA}: không ghi đè. Kiểm lại dữ liệu nguồn và seed.")
            return 1
        print(f"Trùng file đã có {FILE_CHIA} (sha256 {NG.sha256_file(FILE_CHIA)}).")
        return 0 if dat else 2
    if a.kiem:
        print(f"Chưa có {FILE_CHIA}.")
        return 1
    FILE_CHIA.parent.mkdir(parents=True, exist_ok=True)
    NG.ghi_json(FILE_CHIA, d)
    print(f"Đã ghi {FILE_CHIA} (sha256 {NG.sha256_file(FILE_CHIA)}).")
    return 0 if dat else 2


if __name__ == "__main__":
    sys.exit(main())
