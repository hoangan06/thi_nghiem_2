"""Dữ liệu của thí nghiệm chính: đọc tập dựng hướng en–vi và các phép kiểm dữ liệu (Nhap, mục Dữ liệu).

Tập dựng hướng (Nhap): D_harm_build (128 cặp, AdvBench) và D_harmless_build (128 cặp, Alpaca), mỗi tập có
bản tiếng Anh và tiếng Việt ghép cặp. File nằm ở data/dung_huong/, chép nguyên từ kho cũ (xem data/NGUON.md).
Mỗi dòng: {"id", "text", "nhan", "nguon_idx"} (bản tiếng Anh của D_harmless_build có thêm "tac_vu").
Không lọc theo hành vi (Nhap, Dựng hướng điểm 1): mọi câu đều được dùng.

Phép kiểm trùng (Nhap: "không trùng nguyên văn, không trùng gần đúng, không trùng chỉ số nguồn") dùng cùng
cách chuẩn hoá và cùng ngưỡng Jaccard 0,6 trên tập từ như bộ dữ liệu của thí nghiệm cũ đã dùng khi rút các tập
(tom_tat_buoc1.json: nguong_jaccard = 0,6; chuẩn hoá: NFKC, chữ thường, bỏ dấu câu, gộp khoảng trắng).
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

GOC_DU_LIEU = Path(__file__).resolve().parents[2] / "data"
THU_MUC_DUNG_HUONG = GOC_DU_LIEU / "dung_huong"
TAP_DUNG = ("D_harm_build", "D_harmless_build")
NGON_NGU = ("en", "vi")
SO_CAU_TAP_DUNG = 128
NGUONG_JACCARD_GAN_DUNG = 0.6

_DAU_CAU = re.compile(r"[^\w\s]", flags=re.UNICODE)
_KHOANG = re.compile(r"\s+")


def _ten_file(tap: str, ngon_ngu: str) -> str:
    return f"{tap}.jsonl" if ngon_ngu == "en" else f"{tap}_vi.jsonl"


def doc_jsonl(duong: str | Path) -> list[dict]:
    with open(duong, encoding="utf-8") as f:
        return [json.loads(d) for d in f if d.strip()]


@dataclass
class TapDung:
    """Bốn danh sách câu ghép cặp theo chỉ số: harm_en[i] ↔ harm_vi[i], harmless_en[i] ↔ harmless_vi[i]."""
    harm_en: list[str]
    harm_vi: list[str]
    harmless_en: list[str]
    harmless_vi: list[str]
    id_harm: list[str]
    id_harmless: list[str]

    def cau(self, tap: str, ngon_ngu: str) -> list[str]:
        ten = {"D_harm_build": "harm", "D_harmless_build": "harmless"}[tap]
        return getattr(self, f"{ten}_{ngon_ngu}")


def kiem_ghep_cap(en: list[dict], vi: list[dict], ten: str) -> list[str]:
    """Các lỗi ghép cặp giữa bản en và bản vi (rỗng = đạt)."""
    loi = []
    if len(en) != len(vi):
        loi.append(f"{ten}: số dòng en {len(en)} ≠ vi {len(vi)}")
    for i, (a, b) in enumerate(zip(en, vi)):
        if a.get("id") != b.get("id"):
            loi.append(f"{ten}[{i}]: id en {a.get('id')} ≠ vi {b.get('id')}")
        if a.get("nguon_idx") != b.get("nguon_idx"):
            loi.append(f"{ten}[{i}]: nguon_idx en {a.get('nguon_idx')} ≠ vi {b.get('nguon_idx')}")
        for ng, d in (("en", a), ("vi", b)):
            if not str(d.get("text", "")).strip():
                loi.append(f"{ten}[{i}] ({ng}): câu rỗng")
    for ng, ds in (("en", en), ("vi", vi)):
        ids = [d.get("id") for d in ds]
        if len(set(ids)) != len(ids):
            loi.append(f"{ten} ({ng}): id trùng trong tập")
        txt = [d.get("text") for d in ds]
        if len(set(txt)) != len(txt):
            loi.append(f"{ten} ({ng}): câu trùng nguyên văn trong tập")
    return loi


def nap_tap_dung(thu_muc: str | Path = THU_MUC_DUNG_HUONG, so_cau: int | None = SO_CAU_TAP_DUNG) -> TapDung:
    """Đọc bốn file tập dựng; báo lỗi nếu ghép cặp en–vi sai hoặc số câu khác `so_cau` (None = không kiểm)."""
    thu_muc = Path(thu_muc)
    doc = {(t, n): doc_jsonl(thu_muc / _ten_file(t, n)) for t in TAP_DUNG for n in NGON_NGU}
    loi = []
    for t in TAP_DUNG:
        loi += kiem_ghep_cap(doc[(t, "en")], doc[(t, "vi")], t)
        if so_cau is not None and len(doc[(t, "en")]) != so_cau:
            loi.append(f"{t}: {len(doc[(t, 'en')])} câu, Nhap ghi {so_cau}")
    if loi:
        raise ValueError("Tập dựng không hợp lệ:\n" + "\n".join(loi))
    txt = {k: [d["text"] for d in v] for k, v in doc.items()}
    return TapDung(txt[("D_harm_build", "en")], txt[("D_harm_build", "vi")],
                   txt[("D_harmless_build", "en")], txt[("D_harmless_build", "vi")],
                   [d["id"] for d in doc[("D_harm_build", "en")]], [d["id"] for d in doc[("D_harmless_build", "en")]])


# ─────────────────────────────────────────── kiểm trùng ───────────────────────────────────────────
def chuan_hoa(s: str) -> str:
    """NFKC, chữ thường, bỏ dấu câu, gộp khoảng trắng (như bộ dữ liệu của thí nghiệm cũ)."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = _DAU_CAU.sub(" ", s)
    return _KHOANG.sub(" ", s).strip()


def jaccard_tu(a: str, b: str) -> float:
    ta, tb = set(chuan_hoa(a).split()), set(chuan_hoa(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    chung = len(ta & tb)
    return chung / (len(ta) + len(tb) - chung)


def kiem_trung(ten_1: str, ds_1: Sequence[dict], ten_2: str, ds_2: Sequence[dict],
               nguong: float = NGUONG_JACCARD_GAN_DUNG, cung_nguon: bool = True) -> dict[str, Any]:
    """Ba phép kiểm trùng giữa hai tập (mỗi phần tử là dict có "text" và "nguon_idx").

    cung_nguon: hai tập rút từ cùng một bộ nguồn (ví dụ cùng Alpaca) thì kiểm cả trùng chỉ số nguồn;
    khác bộ nguồn (AdvBench với MultiJail) thì chỉ số nguồn không so được, ghi None.
    """
    chuan_2 = {}
    for j, d in enumerate(ds_2):
        chuan_2.setdefault(chuan_hoa(d["text"]), []).append(j)
    nguyen_van = [(i, j) for i, d in enumerate(ds_1) for j in chuan_2.get(chuan_hoa(d["text"]), [])]
    gan = []
    j_lon_nhat = 0.0
    for i, a in enumerate(ds_1):
        for j, b in enumerate(ds_2):
            g = jaccard_tu(a["text"], b["text"])
            j_lon_nhat = max(j_lon_nhat, g)
            if g >= nguong:
                gan.append({"i": i, "j": j, "jaccard": round(g, 4), "cau_1": a["text"], "cau_2": b["text"]})
    nguon = None
    if cung_nguon:
        s2 = {d.get("nguon_idx") for d in ds_2}
        nguon = sorted({d.get("nguon_idx") for d in ds_1} & s2)
    return {"tap_1": ten_1, "tap_2": ten_2, "so_cau": [len(ds_1), len(ds_2)], "nguong_jaccard": nguong,
            "trung_nguyen_van": len(nguyen_van), "trung_gan_dung": len(gan), "vi_du_gan_dung": gan[:20],
            "jaccard_lon_nhat": round(j_lon_nhat, 4),
            "trung_chi_so_nguon": (len(nguon) if nguon is not None else None),
            "chi_so_nguon_trung": (nguon[:50] if nguon is not None else None),
            "dat": bool(not nguyen_van and not gan and not nguon)}
