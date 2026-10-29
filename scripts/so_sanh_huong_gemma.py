"""So hướng Gemma 4 12B dựng lại bằng eager với hướng cũ dựng bằng sdpa (28/10/2026). Chỉ mô tả, không phải cổng.

Q-E4 (Ân duyệt 29/10/2026): bước 4 dùng hướng mới (eager). Phép so này cho biết hướng cũ có bị lỗi kernel sdpa ảnh
hưởng không. Dự đoán viết trước khi có số (báo cáo kết quả bước 3, mục cập nhật):
  - cosine(r̂ cũ, r̂ mới) ≥ 0,999 ở năm lớp hiệu chỉnh (9, 16, 23, 31, 38), cả en và vi;
  - cosine ≥ 0,99 ở mọi lớp có ‖r‖/s < 0,03 (hướng nhỏ, nhạy với nhiễu bf16);
  - chênh tương đối của s_ℓ < 1e-2 ở mọi lớp.
Lý do dự đoán: lượt dựng hướng 28/10 không có lô nào có độ dài đệm ≡ 1 (mod 32), nên chỉ khác nhau do nhiễu bf16
giữa hai cách tính attention.

Chạy trong thư mục kho, trên máy thuê (chỉ dùng CPU, chạy được trong lúc bước 4 đang sinh) hoặc bất kỳ máy nào có
torch và results/:
    python scripts/so_sanh_huong_gemma.py
Mặc định: hướng cũ là results/buoc3/gemma4_12b/dung_huong_20261028_085421/huong.pt; hướng mới là lượt dựng hướng
mới nhất có huong.pt ghi attn_implementation = "eager". Ghi kết quả vào results/buoc3/gemma4_12b/so_sanh_huong_<giờ>/.
"""
import argparse
import datetime as dt
import glob
import hashlib
import json
from pathlib import Path

import torch

LOP_HIEU_CHINH = (9, 16, 23, 31, 38)          # = GEMMA4_12B.lop_hieu_chinh (test kiểm)
GOC = Path("results/buoc3/gemma4_12b")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cos(a, b):
    a, b = a.double(), b.double()
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cu", default=str(GOC / "dung_huong_20261028_085421" / "huong.pt"))
    p.add_argument("--moi", default=None)
    a = p.parse_args(argv)
    if a.moi is None:
        ung = []
        for f in sorted(glob.glob(str(GOC / "dung_huong_2*" / "huong.pt"))):
            if torch.load(f, map_location="cpu").get("attn_implementation") == "eager":
                ung.append(f)
        if not ung:
            raise SystemExit("Không thấy lượt dựng hướng nào ghi attn_implementation = 'eager'.")
        a.moi = ung[-1]
    cu, moi = torch.load(a.cu, map_location="cpu"), torch.load(a.moi, map_location="cpu")
    print(f"Hướng cũ: {a.cu} (attention {cu.get('attn_implementation')!r}, commit {cu['commit_hf']})")
    print(f"Hướng mới: {a.moi} (attention {moi.get('attn_implementation')!r}, commit {moi['commit_hf']})")
    if cu["commit_hf"] != moi["commit_hf"]:
        print("CHÚ Ý: hai hướng khác commit mô hình.")
    s_cu, s_moi = cu["s"].double(), moi["s"].double()
    ds = ((s_moi - s_cu).abs() / s_cu)
    ra = {"cu": {"file": a.cu, "sha256": sha(a.cu), "attn": cu.get("attn_implementation")},
          "moi": {"file": a.moi, "sha256": sha(a.moi), "attn": moi.get("attn_implementation")},
          "lop_hieu_chinh": list(LOP_HIEU_CHINH), "chenh_s_tuong_doi": ds.tolist(), "theo_ngon_ngu": {}}
    dd = {"cos_hieu_chinh_toi_thieu_0_999": True, "cos_huong_nho_toi_thieu_0_99": True,
          "chenh_s_duoi_1e_2": bool((ds < 1e-2).all())}
    for ng in ("en", "vi"):
        c = cos(cu[f"r_hat_{ng}"], moi[f"r_hat_{ng}"])
        tl = moi[f"do_lon_tuong_doi_{ng}"].double()
        nho = [i for i in range(len(tl)) if tl[i] < 0.03]
        ra["theo_ngon_ngu"][ng] = {"cos": c.tolist(), "do_lon_tuong_doi_moi": tl.tolist(), "lop_huong_nho": nho,
                                   "cos_nho_nhat": float(c.min()), "lop_cos_nho_nhat": int(c.argmin())}
        dd["cos_hieu_chinh_toi_thieu_0_999"] &= bool(all(c[l] >= 0.999 for l in LOP_HIEU_CHINH))
        dd["cos_huong_nho_toi_thieu_0_99"] &= bool(all(c[l] >= 0.99 for l in nho))
        print(f"\n[{ng}] lớp | cos(cũ, mới) | ‖r‖/s mới | chênh s")
        for l in range(len(c)):
            dau = " *" if l in LOP_HIEU_CHINH else ""
            print(f"  {l:2d}{dau:2s} | {float(c[l]):.6f} | {float(tl[l]):.4f} | {float(ds[l]):.2e}")
        print(f"  nhỏ nhất: {float(c.min()):.6f} ở lớp {int(c.argmin())}; lớp hiệu chỉnh: "
              + ", ".join(f"{l}: {float(c[l]):.6f}" for l in LOP_HIEU_CHINH))
    ra["du_doan_dung"] = dd
    print(f"\nDự đoán viết trước (chỉ mô tả): {dd}")
    print(f"Chênh s lớn nhất: {float(ds.max()):.2e} ở lớp {int(ds.argmax())}")
    d = GOC / f"so_sanh_huong_{dt.datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True)
    (d / "ket_qua.json").write_text(json.dumps(ra, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {d / 'ket_qua.json'}")
    return ra


if __name__ == "__main__":
    main()
