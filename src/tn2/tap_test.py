"""Tập test và cách chia tập chính / tập hiệu chỉnh / tập con pilot (Nhap, mục Dữ liệu).

Nguồn (data/test/, chép nguyên từ kho cũ, xem data/NGUON.md):
- D_harm_test (MultiJail), D_plain_test (Alpaca): 315 cặp en–vi mỗi tập, ghép theo id.
- nhan_doi_chieu_dich_chot.csv: nhãn đối chiếu bản dịch của D_harm_test (cột doi_chieu: giu 244, lam_nang 22,
  lam_mem 25, lech_nghia 23, khong_thong_nhat 1), làm bằng tay ở thí nghiệm cũ.

Luật chia (Nhap, mục Dữ liệu; chốt 27/10/2026):
- D_harm_test: tập chính = 244 cặp "giữ nghĩa" (toàn bộ, không rút); tập hiệu chỉnh = 20 cặp làm nặng (rút từ 22)
  + 20 cặp làm mềm (rút từ 25); tập con pilot = 50 cặp rút từ tập chính.
- D_plain_test: bỏ cặp D_plain_test/0221 (gần trùng về nghĩa với D_harmless_build/0025), còn 314 cặp; tập chính =
  244 cặp rút ngẫu nhiên; tập hiệu chỉnh = 40 cặp rút từ 70 cặp còn lại; tập con pilot = 50 cặp rút từ tập chính.
- Mọi lần rút dùng MỘT bộ sinh số ngẫu nhiên random.Random(SEED_CHIA_TAP), theo đúng thứ tự THU_TU_RUT, trên
  danh sách ứng viên đã sắp theo id; kết quả mỗi tập sắp lại theo id. Ghi seed và thứ tự vào chia_tap.json.
- Kiểm trùng (Nhap: "phép kiểm trùng được chạy lại trên bộ file cuối cùng ... cho các cặp tập dựng–chính,
  dựng–hiệu chỉnh, hiệu chỉnh–chính"): làm trên cả bản en và bản vi, ghi vào kết quả chia.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from .du_lieu import (GOC_DU_LIEU, NGON_NGU, NGUONG_JACCARD_GAN_DUNG, THU_MUC_DUNG_HUONG, doc_jsonl, jaccard_tu,
                      kiem_ghep_cap, kiem_trung)
from .nguon_goc import sha256_file

THU_MUC_TEST = GOC_DU_LIEU / "test"
THU_MUC_CHIA = GOC_DU_LIEU / "chia_tap"
FILE_CHIA = THU_MUC_CHIA / "chia_tap.json"
FILE_NHAN_DOI_CHIEU = THU_MUC_TEST / "nhan_doi_chieu_dich_chot.csv"
TAP_TEST = ("D_harm_test", "D_plain_test")
PHAN = ("chinh", "hieu_chinh", "pilot")
TAP_DUNG_TUONG_UNG = {"D_harm_test": "D_harm_build", "D_plain_test": "D_harmless_build"}
# Hai tập cùng bộ nguồn thì so được chỉ số nguồn (Alpaca–Alpaca); MultiJail và AdvBench khác bộ nguồn.
CUNG_NGUON_VOI_TAP_DUNG = {"D_harm_test": False, "D_plain_test": True}

SEED_CHIA_TAP = 20261028
LOAI_BO = {"D_plain_test": {"D_plain_test/0221": "gần trùng về nghĩa với D_harmless_build/0025 (Nhap, mục "
                                                  "D_plain_test; Ân duyệt Q3 ngày 27/10/2026)"}}
SO_CAU = {"chinh": 244, "hieu_chinh": 40, "pilot": 50}
SO_HIEU_CHINH_HARM = {"lam_nang": 20, "lam_mem": 20}
# Ngoại lệ đã biết của phép kiểm trùng (Claude đề xuất khi viết code bước 4; Ân duyệt ngày 28/10/2026 và ghi vào
# Nhap, mục D_harm_test, kèm câu hạn chế): MultiJail có các câu theo khuôn, ví dụ "can i punch people in <nước>?". Trong 22 cặp làm nặng có 5 cặp gần
# trùng (Jaccard ≥ 0,6, bản en) với cặp giữ nghĩa, nên rút 20/22 theo Nhap thì không tránh được ít nhất 3; trong 25 cặp
# làm mềm có 2. Với seed đã đặt, tập hiệu chỉnh có 5 câu gần trùng ở bản en (4 làm nặng, 1 làm mềm) và 3 ở bản vi.
# Một số cặp cùng tác vụ (chỉ khác tên nước), một số chỉ cùng khuôn câu (vd. "shoplift" / "insult someone"). Vì α và
# phương án chọn trên tập hiệu chỉnh rồi dùng cho tập chính, sự gần nhau có thể làm lựa chọn lạc quan; mức ảnh
# hưởng không được đo (rà soát bước 4, R2). Ngoại lệ CHỈ miễn phần "gần đúng": trùng nguyên văn hay trùng chỉ số nguồn vẫn là trượt. Danh sách đầy
# đủ các cặp ghi ở kiem_trung[<khoá>]["cap_gan_trung_day_du"].
CHAP_NHAN_TRUNG = {
    "D_harm_test.en.hieu_chinh_chinh": "câu theo khuôn của MultiJail; 20/22 cặp làm nặng không tránh được",
    "D_harm_test.vi.hieu_chinh_chinh": "câu theo khuôn của MultiJail; 20/22 cặp làm nặng không tránh được",
}
THU_TU_RUT = ("D_harm_test.hieu_chinh.lam_nang", "D_harm_test.hieu_chinh.lam_mem", "D_harm_test.pilot",
              "D_plain_test.chinh", "D_plain_test.hieu_chinh", "D_plain_test.pilot")


def _ten_file(tap: str, ng: str) -> str:
    return f"{tap}.jsonl" if ng == "en" else f"{tap}_vi.jsonl"


def doc_tap_test(tap: str, thu_muc: str | Path = THU_MUC_TEST) -> dict[str, list[dict]]:
    """Hai bản en, vi của một tập test (đủ 315 cặp), đã kiểm ghép cặp."""
    thu_muc = Path(thu_muc)
    d = {ng: doc_jsonl(thu_muc / _ten_file(tap, ng)) for ng in NGON_NGU}
    loi = kiem_ghep_cap(d["en"], d["vi"], tap)
    if loi:
        raise ValueError("Tập test không hợp lệ:\n" + "\n".join(loi))
    return d


def doc_nhan_doi_chieu(p: str | Path = FILE_NHAN_DOI_CHIEU) -> dict[str, str]:
    with open(p, encoding="utf-8", newline="") as f:
        return {r["id"]: r["doi_chieu"] for r in csv.DictReader(f)}


def _rut(rng: random.Random, ung_vien: list[str], k: int, ten: str) -> list[str]:
    ung_vien = sorted(ung_vien)
    if len(ung_vien) < k:
        raise ValueError(f"{ten}: cần {k} cặp, chỉ có {len(ung_vien)} ứng viên.")
    return sorted(rng.sample(ung_vien, k))


def chia_tap(thu_muc_test: str | Path = THU_MUC_TEST, file_nhan: str | Path = FILE_NHAN_DOI_CHIEU,
             seed: int = SEED_CHIA_TAP) -> dict[str, Any]:
    """Chia theo luật của Nhap. Tất định: cùng dữ liệu, cùng seed → cùng kết quả."""
    thu_muc_test = Path(thu_muc_test)
    nhan = doc_nhan_doi_chieu(file_nhan)
    harm = doc_tap_test("D_harm_test", thu_muc_test)
    plain = doc_tap_test("D_plain_test", thu_muc_test)
    id_harm = [d["id"] for d in harm["en"]]
    if set(nhan) != set(id_harm):
        raise ValueError("Nhãn đối chiếu không phủ đúng tập id của D_harm_test.")
    rng = random.Random(seed)
    theo_nhan = {k: [i for i in id_harm if nhan[i] == k] for k in ("giu", "lam_nang", "lam_mem")}
    ket: dict[str, Any] = {"D_harm_test": {}, "D_plain_test": {}}
    # thứ tự rút đúng THU_TU_RUT
    nang = _rut(rng, theo_nhan["lam_nang"], SO_HIEU_CHINH_HARM["lam_nang"], THU_TU_RUT[0])
    mem = _rut(rng, theo_nhan["lam_mem"], SO_HIEU_CHINH_HARM["lam_mem"], THU_TU_RUT[1])
    chinh_h = sorted(theo_nhan["giu"])
    if len(chinh_h) != SO_CAU["chinh"]:
        raise ValueError(f"D_harm_test có {len(chinh_h)} cặp giữ nghĩa, Nhap ghi {SO_CAU['chinh']}.")
    pilot_h = _rut(rng, chinh_h, SO_CAU["pilot"], THU_TU_RUT[2])
    ket["D_harm_test"] = {"chinh": chinh_h, "hieu_chinh": sorted(nang + mem), "pilot": pilot_h,
                          "nhan_hieu_chinh": {i: nhan[i] for i in sorted(nang + mem)}}
    loai = set(LOAI_BO["D_plain_test"])
    nguon_p = [d["id"] for d in plain["en"] if d["id"] not in loai]
    chinh_p = _rut(rng, nguon_p, SO_CAU["chinh"], THU_TU_RUT[3])
    con_lai = [i for i in nguon_p if i not in set(chinh_p)]
    hc_p = _rut(rng, con_lai, SO_CAU["hieu_chinh"], THU_TU_RUT[4])
    pilot_p = _rut(rng, chinh_p, SO_CAU["pilot"], THU_TU_RUT[5])
    ket["D_plain_test"] = {"chinh": chinh_p, "hieu_chinh": hc_p, "pilot": pilot_p,
                           "so_cap_nguon_sau_loai_bo": len(nguon_p), "so_cap_con_lai_sau_tap_chinh": len(con_lai)}
    du_lieu = {"seed": seed, "thu_tu_rut": list(THU_TU_RUT), "loai_bo": LOAI_BO,
               "nguon_sha256": {p.name: sha256_file(p) for p in
                                sorted(list(thu_muc_test.glob("*.jsonl")) + [Path(file_nhan)])},
               "tap": ket}
    du_lieu["kiem_trung"] = kiem_trung_chia_tap(du_lieu, thu_muc_test)
    du_lieu["kiem_tra"] = kiem_chia_tap(du_lieu, thu_muc_test, file_nhan)
    return du_lieu


def _theo_id(ds: list[dict]) -> dict[str, dict]:
    return {d["id"]: d for d in ds}


def kiem_trung_chia_tap(du_lieu: dict[str, Any], thu_muc_test: str | Path = THU_MUC_TEST,
                        thu_muc_dung: str | Path = THU_MUC_DUNG_HUONG) -> dict[str, Any]:
    """Ba cặp tập của Nhap (dựng–chính, dựng–hiệu chỉnh, hiệu chỉnh–chính), mỗi tập test, cả en và vi."""
    ra: dict[str, Any] = {}
    for tap in TAP_TEST:
        test = doc_tap_test(tap, thu_muc_test)
        dung_ten = TAP_DUNG_TUONG_UNG[tap]
        for ng in NGON_NGU:
            dung = doc_jsonl(Path(thu_muc_dung) / _ten_file(dung_ten, ng))
            t = _theo_id(test[ng])
            chinh = [t[i] for i in du_lieu["tap"][tap]["chinh"]]
            hc = [t[i] for i in du_lieu["tap"][tap]["hieu_chinh"]]
            cn = CUNG_NGUON_VOI_TAP_DUNG[tap]
            ra[f"{tap}.{ng}.dung_chinh"] = kiem_trung(dung_ten, dung, f"{tap}.chinh", chinh, cung_nguon=cn)
            ra[f"{tap}.{ng}.dung_hieu_chinh"] = kiem_trung(dung_ten, dung, f"{tap}.hieu_chinh", hc, cung_nguon=cn)
            ra[f"{tap}.{ng}.hieu_chinh_chinh"] = kiem_trung(f"{tap}.hieu_chinh", hc, f"{tap}.chinh", chinh,
                                                            cung_nguon=True)
    # Danh sách đầy đủ cặp gần trùng của các khoá ngoại lệ (kiem_trung chỉ giữ 20 ví dụ đầu).
    for k in CHAP_NHAN_TRUNG:
        if k not in ra:
            continue
        tap, ng, _ = k.split(".")
        t = _theo_id(doc_tap_test(tap, thu_muc_test)[ng])
        hc, chinh = du_lieu["tap"][tap]["hieu_chinh"], du_lieu["tap"][tap]["chinh"]
        ra[k]["cap_gan_trung_day_du"] = [
            {"hieu_chinh": i, "chinh": j, "jaccard": round(jaccard_tu(t[i]["text"], t[j]["text"]), 4)}
            for i in hc for j in chinh if jaccard_tu(t[i]["text"], t[j]["text"]) >= NGUONG_JACCARD_GAN_DUNG]
    ra["dat"] = all(v["dat"] for k, v in ra.items() if isinstance(v, dict))
    ra["ngoai_le_chap_nhan"] = {k: CHAP_NHAN_TRUNG[k] for k in CHAP_NHAN_TRUNG if k in ra and not ra[k]["dat"]}

    def dat_k(k: str, v: dict) -> bool:
        if k in CHAP_NHAN_TRUNG:          # chỉ miễn phần gần đúng
            return v["trung_nguyen_van"] == 0 and not v["trung_chi_so_nguon"]
        return v["dat"]
    ra["dat_sau_ngoai_le"] = all(dat_k(k, v) for k, v in ra.items() if isinstance(v, dict) and "dat" in v)
    return ra


def kiem_chia_tap(du_lieu: dict[str, Any], thu_muc_test: str | Path = THU_MUC_TEST,
                  file_nhan: str | Path = FILE_NHAN_DOI_CHIEU) -> dict[str, Any]:
    """Các ràng buộc cấu trúc của cách chia (số cặp, rời nhau, pilot ⊂ chính, nhãn, loại bỏ)."""
    loi: list[str] = []
    nhan = doc_nhan_doi_chieu(file_nhan)
    for tap in TAP_TEST:
        ids = {d["id"] for d in doc_tap_test(tap, thu_muc_test)["en"]}
        p = du_lieu["tap"][tap]
        for phan in PHAN:
            if len(p[phan]) != SO_CAU[phan]:
                loi.append(f"{tap}.{phan}: {len(p[phan])} cặp, cần {SO_CAU[phan]}")
            if len(set(p[phan])) != len(p[phan]):
                loi.append(f"{tap}.{phan}: id lặp")
            if not set(p[phan]) <= ids:
                loi.append(f"{tap}.{phan}: id không có trong tập")
            if p[phan] != sorted(p[phan]):
                loi.append(f"{tap}.{phan}: chưa sắp theo id")
        if set(p["chinh"]) & set(p["hieu_chinh"]):
            loi.append(f"{tap}: tập chính và tập hiệu chỉnh giao nhau")
        if not set(p["pilot"]) <= set(p["chinh"]):
            loi.append(f"{tap}: pilot không nằm trong tập chính")
        for i in LOAI_BO.get(tap, {}):
            if any(i in p[phan] for phan in PHAN):
                loi.append(f"{tap}: {i} phải bị loại")
    h = du_lieu["tap"]["D_harm_test"]
    if any(nhan[i] != "giu" for i in h["chinh"]):
        loi.append("D_harm_test.chinh: có cặp không phải giữ nghĩa")
    dem = {k: sum(nhan[i] == k for i in h["hieu_chinh"]) for k in ("lam_nang", "lam_mem")}
    if dem != SO_HIEU_CHINH_HARM:
        loi.append(f"D_harm_test.hieu_chinh: {dem}, cần {SO_HIEU_CHINH_HARM}")
    return {"loi": loi, "dat": not loi}


def doc_chia_tap(p: str | Path = FILE_CHIA) -> dict[str, Any]:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def tap_con(tap: str, phan: str, du_lieu_chia: dict[str, Any] | None = None,
            thu_muc_test: str | Path = THU_MUC_TEST) -> list[dict[str, Any]]:
    """Các cặp câu của một phần (chinh | hieu_chinh | pilot), theo thứ tự id: {"id", "en", "vi", "nguon_idx",
    "nhan_doi_chieu"} (nhãn đối chiếu bản dịch: giu / lam_nang / lam_mem với D_harm_test; None với D_plain_test)."""
    du_lieu_chia = du_lieu_chia or doc_chia_tap()
    test = doc_tap_test(tap, thu_muc_test)
    en, vi = _theo_id(test["en"]), _theo_id(test["vi"])
    nhan = doc_nhan_doi_chieu() if tap == "D_harm_test" else {}
    return [{"id": i, "en": en[i]["text"], "vi": vi[i]["text"], "nguon_idx": en[i].get("nguon_idx"),
             "nhan_doi_chieu": nhan.get(i)} for i in du_lieu_chia["tap"][tap][phan]]
