"""Bước 4: chia tập test, kế hoạch lượt hiệu chỉnh, sinh theo đơn vị, ghi bền và chạy tiếp sau khi bị ngắt."""
import json
import os

import pytest
import torch

from tn2 import tap_test as TT
from tn2.mo_hinh import GEMMA4_12B, QWEN35_9B, eos_tu_mo_hinh, nap_mo_hinh
from tn2.sinh import sinh_tham_lam
from tn2.khuon import ma_hoa
from tn2 import sinh_luot as SL


# ─────────────────────────────── chia tập ───────────────────────────────
def test_chia_tap_tat_dinh_va_trung_file_da_khoa():
    d = TT.chia_tap()
    assert json.loads(json.dumps(d, ensure_ascii=False)) == TT.doc_chia_tap()   # đúng file trong data/chia_tap
    assert TT.chia_tap() == d                                                   # tất định


def test_chia_tap_dung_luat_cua_nhap():
    d = TT.doc_chia_tap()
    nhan = TT.doc_nhan_doi_chieu()
    h, p = d["tap"]["D_harm_test"], d["tap"]["D_plain_test"]
    assert d["seed"] == TT.SEED_CHIA_TAP and d["kiem_tra"]["dat"]
    # D_harm_test: chính = đủ 244 cặp giữ nghĩa; hiệu chỉnh = 20 làm nặng + 20 làm mềm; pilot ⊂ chính
    assert h["chinh"] == sorted(i for i, v in nhan.items() if v == "giu")
    assert sorted(nhan[i] for i in h["hieu_chinh"]).count("lam_nang") == 20
    assert sorted(nhan[i] for i in h["hieu_chinh"]).count("lam_mem") == 20
    assert set(h["pilot"]) <= set(h["chinh"]) and len(h["pilot"]) == 50
    # D_plain_test: bỏ 0221; 244 chính; 40 hiệu chỉnh từ 70 cặp còn lại; pilot ⊂ chính
    tat_ca = [r["id"] for r in TT.doc_tap_test("D_plain_test")["en"]]
    assert "D_plain_test/0221" not in p["chinh"] + p["hieu_chinh"] + p["pilot"]
    con_lai = set(tat_ca) - {"D_plain_test/0221"} - set(p["chinh"])
    assert len(con_lai) == 70 and set(p["hieu_chinh"]) <= con_lai
    assert set(p["pilot"]) <= set(p["chinh"]) and len(p["chinh"]) == 244
    # kiểm trùng: chỉ hai ngoại lệ đã biết (câu theo khuôn của MultiJail), còn lại đạt
    kt = d["kiem_trung"]
    assert set(kt["ngoai_le_chap_nhan"]) == {"D_harm_test.en.hieu_chinh_chinh", "D_harm_test.vi.hieu_chinh_chinh"}
    assert kt["dat_sau_ngoai_le"]
    assert all(kt[f"D_plain_test.{ng}.{k}"]["dat"] for ng in ("en", "vi")
               for k in ("dung_chinh", "dung_hieu_chinh", "hieu_chinh_chinh"))


def test_kiem_chia_tap_bat_loi():
    d = json.loads(json.dumps(TT.doc_chia_tap()))
    d["tap"]["D_plain_test"]["hieu_chinh"][0] = d["tap"]["D_plain_test"]["chinh"][0]      # giao nhau
    d["tap"]["D_harm_test"]["pilot"][0] = d["tap"]["D_harm_test"]["hieu_chinh"][0]        # pilot ngoài tập chính
    loi = TT.kiem_chia_tap(d)["loi"]
    assert any("giao nhau" in x for x in loi) and any("pilot" in x for x in loi)


def test_tap_con_ghep_cap():
    c = TT.tap_con("D_harm_test", "hieu_chinh")
    assert len(c) == 40 and all(x["en"] and x["vi"] for x in c)
    assert [x["id"] for x in c] == TT.doc_chia_tap()["tap"]["D_harm_test"]["hieu_chinh"]


# ─────────────────────────────── kế hoạch ───────────────────────────────
@pytest.mark.parametrize("dac_ta", [QWEN35_9B, GEMMA4_12B])
def test_ke_hoach_hieu_chinh(dac_ta):
    ke = SL.ke_hoach_hieu_chinh(dac_ta)
    assert len(ke) == 488 and len({d.ma for d in ke}) == 488
    nen = [d for d in ke if d.loai == "nen"]
    assert ke[:4] == nen and all(d.alpha == 0.0 for d in nen)
    assert {(d.tap, d.kieu, d.ngon_ngu_cau_hoi) for d in nen} == {
        ("D_plain_test", "a", "en"), ("D_plain_test", "a", "vi"), ("D_harm_test", "b", "en"), ("D_harm_test", "b", "vi")}
    chinh = [d for d in ke if d.nhom == "luoi_chinh"]
    mo = [d for d in ke if d.nhom == "mo_rong"]
    assert len(chinh) == 320 and len(mo) == 160 and ke[-160:] == mo          # mở rộng chạy sau cùng
    kh = [d for d in ke if d.nhom == "nen_khong_hook"]                       # Q-B: sau lưới chính, trước mở rộng
    assert ke[324:328] == kh and len(kh) == 4 and all(d.lop is None and d.phuong_an is None for d in kh)
    assert {(d.tap, d.kieu, d.ngon_ngu_cau_hoi) for d in kh} == {(d.tap, d.kieu, d.ngon_ngu_cau_hoi) for d in nen}
    assert {d.alpha for d in chinh} == {0.1, 0.2, 0.4, 0.8} and {d.alpha for d in mo} == {0.05, 1.6}
    assert {d.lop for d in chinh} == set(dac_ta.lop_hieu_chinh) and {d.phuong_an for d in chinh} == {1, 3}
    assert {d.chieu for d in chinh} == {"en->en", "en->vi", "vi->en", "vi->vi"}
    assert all(SL.TAP_THEO_KIEU[d.kieu] == d.tap for d in ke)
    # mỗi ô (kiểu, α, lớp, phương án, chiều) đúng một đơn vị
    o = {(d.kieu, d.alpha, d.lop, d.phuong_an, d.chieu) for d in chinh}
    assert len(o) == 2 * 4 * 5 * 2 * 4
    assert sum(40 for _ in SL.ke_hoach_hieu_chinh(dac_ta, mo_rong=(), chan_doan=False)) * 2 == 25_920   # ~25 900 Nhap


# ─────────────────────────────── sinh một đơn vị ───────────────────────────────
@pytest.fixture(scope="module")
def qwen(thu_muc_qwen35):
    return nap_mo_hinh(QWEN35_9B, str(thu_muc_qwen35), dtype=torch.float32, device_map=None)


def _huong_gia(dac_ta, seed=0, s=1.0):
    g = torch.Generator().manual_seed(seed)
    r = {ng: torch.randn(dac_ta.so_lop, 32, generator=g, dtype=torch.float64) for ng in ("en", "vi")}
    r = {k: v / v.norm(dim=-1, keepdim=True) for k, v in r.items()}
    return r, torch.full((dac_ta.so_lop,), s, dtype=torch.float64)


def _cau(n=3):
    return {t: TT.tap_con(t, "hieu_chinh")[:n] for t in ("D_plain_test", "D_harm_test")}


def test_sinh_don_vi_dung_huong_dau_lop_va_phuong_an(qwen, monkeypatch):
    model, tok = qwen
    huong, s = _huong_gia(QWEN35_9B)
    s = s * torch.arange(1, QWEN35_9B.so_lop + 1, dtype=torch.float64)          # s khác nhau theo lớp
    goi = []
    goc = SL.can_thiep_chinh

    def ghi(lop, r_hat, s_l, alpha, kieu, pa, k):
        goi.append((lop, r_hat.clone(), s_l, alpha, kieu, pa, k))
        return goc(lop, r_hat, s_l, alpha, kieu, pa, k)
    monkeypatch.setattr(SL, "can_thiep_chinh", ghi)
    cau = _cau(2)
    eos = eos_tu_mo_hinh(model, QWEN35_9B)
    dv = SL.DonVi("can_thiep", "luoi_chinh", "D_harm_test", "hieu_chinh", "b", "en", "vi", 21, 0.4, 3)
    dong = SL.sinh_don_vi(model, tok, QWEN35_9B, dv, cau["D_harm_test"], huong, s, eos, 3, 40)
    lop, r_hat, s_l, alpha, kieu, pa, k = goi[0]
    assert lop == 21 and torch.equal(r_hat, huong["vi"][21].to(torch.float32))   # chiều vi→en: hướng tiếng Việt
    assert s_l == float(s[21]) and alpha == 0.4 and kieu == "b" and pa == 3 and k == 9
    assert [d["cau_hoi"] for d in dong] == [c["en"] for c in cau["D_harm_test"]]  # câu hỏi tiếng Anh
    assert all(d["chieu"] == "vi->en" and d["ma_don_vi"] == dv.ma for d in dong)
    # dấu: kiểu (b) là trừ
    assert torch.allclose(goc(21, r_hat, s_l, 0.4, "b", 3, 9).vector, -0.4 * s_l * r_hat)


def test_nen_khop_tuyet_doi_khong_hook(qwen):
    model, tok = qwen
    huong, s = _huong_gia(QWEN35_9B)
    cau = _cau(3)["D_plain_test"]
    eos = eos_tu_mo_hinh(model, QWEN35_9B)
    nen = SL.DonVi("nen", "nen", "D_plain_test", "hieu_chinh", "a", "vi", None, 15, 0.0, 1)
    dong = SL.sinh_don_vi(model, tok, QWEN35_9B, nen, cau, huong, s, eos, 5, 40)
    goc = sinh_tham_lam(model, tok, ma_hoa(tok, [c["vi"] for c in cau], QWEN35_9B), 5, eos)
    assert [d["token"] for d in dong] == [goc.token_hop_le(i) for i in range(3)]
    co = SL.DonVi("can_thiep", "luoi_chinh", "D_plain_test", "hieu_chinh", "a", "vi", "en", 15, 0.8, 1)
    dong_co = SL.sinh_don_vi(model, tok, QWEN35_9B, co, cau, huong, s * 50, eos, 5, 40)
    assert [d["token"] for d in dong_co] != [d["token"] for d in dong]          # can thiệp thật sự có tác dụng


# ─────────────────────────────── ghi bền, chạy tiếp ───────────────────────────────
def test_ghi_nguyen_tu_khong_de_file_tam(tmp_path):
    p = tmp_path / "a" / "x.jsonl"
    SL.ghi_nguyen_tu(p, "1\n")
    SL.ghi_nguyen_tu(p, "2\n")
    assert p.read_text() == "2\n" and sorted(os.listdir(p.parent)) == ["x.jsonl"]


def test_kho_mo_tu_choi_luot_khac(tmp_path):
    kho = SL.KhoKetQua(tmp_path / "l")
    kh = {"dau_van_tay": {"a": 1, "ma_nguon": "x"}}
    assert kho.mo(kh) == "moi" and kho.mo(kh) == "tiep"
    dv = SL.ke_hoach_hieu_chinh(QWEN35_9B)[0]
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {})          # đã có đơn vị sinh xong
    with pytest.raises(ValueError):
        kho.mo({"dau_van_tay": {"a": 2, "ma_nguon": "x"}})
    with pytest.raises(ValueError):
        kho.mo({"dau_van_tay": {"a": 1, "ma_nguon": "y"}})
    assert kho.mo({"dau_van_tay": {"a": 1, "ma_nguon": "y"}}, cho_phep_doi_ma_nguon=True) == "tiep"
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "z.txt").write_text("z")
    with pytest.raises(ValueError):                                             # thư mục lạ: không ghi vào
        SL.KhoKetQua(tmp_path / "k").mo(kh)


def test_chay_ke_hoach_bi_ngat_roi_chay_tiep(qwen, tmp_path, monkeypatch):
    """Máy dừng giữa chừng (ngoại lệ ở đơn vị thứ 4): 3 đơn vị đã xong giữ nguyên từng byte; chạy lại thì chỉ sinh
    phần còn thiếu; không có file viết dở."""
    model, tok = qwen
    huong, s = _huong_gia(QWEN35_9B)
    cau = _cau(2)
    eos = eos_tu_mo_hinh(model, QWEN35_9B)
    ke = SL.ke_hoach_hieu_chinh(QWEN35_9B, luoi=(0.4,), mo_rong=())[:8]
    kho = SL.KhoKetQua(tmp_path / "luot")
    kho.mo({"dau_van_tay": {"x": 1}})
    goc = SL.sinh_don_vi
    dem = {"n": 0}

    def hong_lan_4(*a, **k):
        dem["n"] += 1
        if dem["n"] == 4:
            raise RuntimeError("giả lập máy dừng")
        return goc(*a, **k)
    monkeypatch.setattr(SL, "sinh_don_vi", hong_lan_4)
    with pytest.raises(RuntimeError):
        SL.chay_ke_hoach(model, tok, QWEN35_9B, ke, cau, huong, s, eos, kho, 3, 40, log=lambda m: None)
    da = {dv.ma: (kho.jsonl(dv).stat().st_mtime_ns, SL.sha256_file(kho.jsonl(dv))) for dv in ke[:3]}
    assert all(kho.kiem_don_vi(dv, [c["id"] for c in cau[dv.tap]])[0] for dv in ke[:3])
    assert not kho.jsonl(ke[3]).exists() and not any(p.name.startswith(".") for p in kho.don_vi.iterdir())
    monkeypatch.setattr(SL, "sinh_don_vi", goc)
    kq = SL.chay_ke_hoach(model, tok, QWEN35_9B, ke, cau, huong, s, eos, kho, 3, 40, log=lambda m: None)
    assert kq["da_co_tu_truoc"] == 3 and kq["sinh_trong_phien"] == 5 and kq["con_lai"] == 0
    assert all((kho.jsonl(dv).stat().st_mtime_ns, SL.sha256_file(kho.jsonl(dv))) == da[dv.ma] for dv in ke[:3])


def test_don_vi_hong_chuyen_sang_hong_va_sinh_lai(qwen, tmp_path):
    model, tok = qwen
    huong, s = _huong_gia(QWEN35_9B)
    cau = _cau(2)
    eos = eos_tu_mo_hinh(model, QWEN35_9B)
    ke = SL.ke_hoach_hieu_chinh(QWEN35_9B, luoi=(0.4,), mo_rong=())[:3]
    kho = SL.KhoKetQua(tmp_path / "luot")
    kho.mo({"dau_van_tay": {"x": 1}})
    SL.chay_ke_hoach(model, tok, QWEN35_9B, ke, cau, huong, s, eos, kho, 3, 40, log=lambda m: None)
    kho.xong(ke[0]).unlink()                                       # dừng giữa lúc ghi .jsonl và .xong.json
    with open(kho.jsonl(ke[1]), "a", encoding="utf-8") as f:       # file bị sửa sau khi xong
        f.write('{"id": "la"}\n')
    noi_dung_cu = kho.jsonl(ke[1]).read_text(encoding="utf-8")
    kq = SL.chay_ke_hoach(model, tok, QWEN35_9B, ke, cau, huong, s, eos, kho, 3, 40, log=lambda m: None)
    assert kq["sinh_trong_phien"] == 2 and {x["ly_do"] for x in kq["hong"]} == {"thieu_mot_trong_hai_file",
                                                                                "sai_sha256"}
    trong_hong = sorted(os.listdir(kho.hong))
    assert any(n.startswith(ke[0].ma) for n in trong_hong) and any(n.startswith(ke[1].ma) for n in trong_hong)
    giu = [n for n in trong_hong if n.startswith(f"{ke[1].ma}.jsonl")][0]
    assert (kho.hong / giu).read_text(encoding="utf-8") == noi_dung_cu          # bản hỏng được giữ, không mất
    assert all(kho.kiem_don_vi(dv, [c["id"] for c in cau[dv.tap]])[0] for dv in ke)


# ─────────────────────────────── script chạy thông ───────────────────────────────
def _luu_huong(p, dt, **them):
    huong, s = _huong_gia(dt)
    t = {"dac_ta": dt.ten, "commit_hf": None, "r_hat_en": huong["en"], "r_hat_vi": huong["vi"], "s": s,
         "trang_thai": "dat", "chinh_thuc": True, "position_ids_theo_mat_na": True}
    t.update(them)
    torch.save(t, p)
    return p


def _anh_thu_muc(goc):
    return {str(p.relative_to(goc)): (p.stat().st_mtime_ns, SL.sha256_file(p))
            for p in sorted(goc.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("ten,dac_ta", [("qwen35", QWEN35_9B), ("gemma4", GEMMA4_12B)])
def test_script_hieu_chinh_chay_ngat_tiep_hoan_tat(ten, dac_ta, thu_muc_qwen35, thu_muc_gemma4, tmp_path,
                                                    monkeypatch):
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    d = str(thu_muc_qwen35 if ten == "qwen35" else thu_muc_gemma4)
    huong = _luu_huong(tmp_path / "huong.pt", dac_ta)
    tm = tmp_path / "luot"
    chung = ["--mo_hinh_dac_ta", dac_ta.ten, "--mo_hinh", d, "--huong", str(huong), "--thu_muc", str(tm), "--thu",
             "--gioi_han_cau", "2", "--max_new_tokens", "3", "--khong_mo_rong"]
    # phiên 1: dừng sau 6 đơn vị
    assert S.main(chung + ["--so_don_vi_toi_da", "6"]) == 7
    xong_1 = {k: v for k, v in _anh_thu_muc(tm).items() if k.startswith("don_vi/")}
    assert len(xong_1) == 12 and not (tm / "HOAN_TAT.json").exists()
    # phiên 2: chạy tiếp tới hết; 6 đơn vị cũ giữ nguyên từng byte
    assert S.main(chung) == 0
    sau = _anh_thu_muc(tm)
    assert all(sau[k] == v for k, v in xong_1.items())
    ke = json.loads((tm / "ke_hoach.json").read_text(encoding="utf-8"))
    assert len(ke["don_vi"]) == 328                                     # 324 + 4 lượt nền không hook (Q-B)
    dong = [json.loads(x) for x in (tm / "cau_tra_loi.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(dong) == 324 * 2 and len({(r["ma_don_vi"], r["id"]) for r in dong}) == 648
    assert {r["mo_hinh"] for r in dong} == {dac_ta.ten} and all("van_ban" in r and "token" in r for r in dong)
    assert {r["loai"] for r in dong} == {"nen", "can_thiep"}            # nền không hook không lẫn vào phần chính
    nen = [r for r in dong if r["loai"] == "nen"]                       # Q-C: dòng nền không mang lớp / phương án
    assert all(r["lop"] is None and r["phuong_an"] is None and r["lop_hook"] == dac_ta.lop_hieu_chinh[2]
               and r["phuong_an_hook"] == 1 for r in nen)
    assert all(r["lop"] == r["lop_hook"] for r in dong if r["loai"] == "can_thiep")
    ht = json.loads((tm / "HOAN_TAT.json").read_text(encoding="utf-8"))
    assert ht["so_cau_tra_loi"] == 656 and set(ht["phan"]) == {"chinh", "chan_doan"}
    tk = json.loads((tm / "tom_tat_nen_khong_hook.json").read_text(encoding="utf-8"))["so_khop_voi_nen_co_hook"]
    assert tk["tong_cau"] == 8 and tk["tat_ca_giong_het"] is True        # α = 0 có hook giống hệt không hook
    assert ht["phan"]["chinh"]["sha256"] == SL.sha256_file(tm / "cau_tra_loi.jsonl")
    assert not (tm / "cau_tra_loi_mo_rong.jsonl").exists()
    harm = [r for r in dong if r["tap"] == "D_harm_test"]
    assert {r["nhan_doi_chieu"] for r in harm} <= {"lam_nang", "lam_mem"}
    assert all(r["nhan_doi_chieu"] is None for r in dong if r["tap"] == "D_plain_test")
    for dong_sha in (tm / "sha256.txt").read_text().splitlines():
        h, ten_f = dong_sha.split("  ", 1)
        assert SL.sha256_file(tm / ten_f) == h
    assert len((tm / "phien.jsonl").read_text().splitlines()) == 2
    # phiên 3: đã hoàn tất → không làm gì, không file nào đổi
    truoc = _anh_thu_muc(tm)
    assert S.main(chung) == 0
    assert _anh_thu_muc(tm) == truoc
    # file hướng khác (SHA khác) cùng thư mục → từ chối (mã 6), không đổi file nào
    huong2 = _luu_huong(tmp_path / "huong2.pt", dac_ta)
    torch.save({**torch.load(huong2), "r_hat_en": torch.load(huong2)["r_hat_vi"]}, huong2)
    truoc = _anh_thu_muc(tm)
    assert S.main([x if x != str(huong) else str(huong2) for x in chung]) == 6
    assert _anh_thu_muc(tm) == truoc


def _kiem_hook_gia(d, huong, dac_ta="qwen35_9b", trang_thai="dat", chinh_thuc=True):
    d.mkdir(parents=True, exist_ok=True)
    (d / "ket_luan.json").write_text(json.dumps({"trang_thai": trang_thai}))
    (d / "meta.json").write_text(json.dumps({"dac_ta": dac_ta, "chinh_thuc": chinh_thuc,
                                             "tham_so": {"huong_sha256": SL.sha256_file(huong)}}))
    return d


def test_script_tu_choi_huong_khong_dung_duoc(tmp_path):
    import buoc4_hieu_chinh as S
    for k, them in enumerate(({"trang_thai": "can_xem"}, {"chinh_thuc": False}, {"dac_ta": "gemma4_12b"},
                              {"position_ids_theo_mat_na": False}, {"commit_hf": None})):
        h = _luu_huong(tmp_path / f"h{k}.pt", QWEN35_9B, **{"commit_hf": "abc", **them})
        kh = _kiem_hook_gia(tmp_path / f"kh{k}", h)
        assert S.main(["--mo_hinh_dac_ta", "qwen35_9b", "--huong", str(h), "--kiem_hook", str(kh),
                       "--thu_muc", str(tmp_path / "x")]) == 4
    h = _luu_huong(tmp_path / "tot.pt", QWEN35_9B, commit_hf="abc")
    h2 = _luu_huong(tmp_path / "khac.pt", QWEN35_9B, commit_hf="abc")
    for kh in (None, _kiem_hook_gia(tmp_path / "kh_cx", h, trang_thai="can_xem"),
               _kiem_hook_gia(tmp_path / "kh_thu", h, chinh_thuc=False),
               _kiem_hook_gia(tmp_path / "kh_mh", h, dac_ta="gemma4_12b"),
               _kiem_hook_gia(tmp_path / "kh_huong", h2)):                     # kiểm hook với file hướng khác
        lenh = ["--mo_hinh_dac_ta", "qwen35_9b", "--huong", str(h), "--thu_muc", str(tmp_path / "x")]
        assert S.main(lenh + ([] if kh is None else ["--kiem_hook", str(kh)])) == 4
    assert not (tmp_path / "x").exists()                                # dừng trước khi tạo thư mục lượt


def test_script_doi_ma_nguon_giua_hai_phien(thu_muc_qwen35, tmp_path, monkeypatch):
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    chung = ["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--huong", str(h), "--thu_muc",
             str(tmp_path / "l"), "--thu", "--gioi_han_cau", "1", "--max_new_tokens", "2", "--khong_mo_rong"]
    assert S.main(chung + ["--so_don_vi_toi_da", "1"]) == 7
    goc = S.dau_van_tay_ma_nguon
    monkeypatch.setattr(S, "dau_van_tay_ma_nguon", lambda g: {**goc(g), "sha256": "khac"})
    assert S.main(chung + ["--so_don_vi_toi_da", "1"]) == 6                    # mã nguồn đổi → dừng
    assert S.main(chung + ["--so_don_vi_toi_da", "1", "--cho_phep_doi_ma_nguon"]) == 7
    ph = [json.loads(x) for x in (tmp_path / "l" / "phien.jsonl").read_text().splitlines()]
    assert ph[-1]["cho_phep_doi_ma_nguon"] is True


def test_script_chi_ke_hoach_khong_nap_mo_hinh(capsys):
    import buoc4_hieu_chinh as S
    assert S.main(["--mo_hinh_dac_ta", "gemma4_12b", "--chi_ke_hoach"]) == 0
    assert "488 đơn vị" in capsys.readouterr().out


def test_tong_hop_de_rieng_muc_mo_rong(qwen, tmp_path):
    model, tok = qwen
    huong, s = _huong_gia(QWEN35_9B)
    cau = _cau(2)
    eos = eos_tu_mo_hinh(model, QWEN35_9B)
    tat_ca = SL.ke_hoach_hieu_chinh(QWEN35_9B)
    ke = tat_ca[:2] + [d for d in tat_ca if d.nhom == "mo_rong"][:2]
    kho = SL.KhoKetQua(tmp_path / "luot")
    kho.mo({"dau_van_tay": {"x": 1}})
    nhat_ky = []
    SL.chay_ke_hoach(model, tok, QWEN35_9B, ke, cau, huong, s, eos, kho, 3, 40, log=nhat_ky.append)
    assert all("bị cắt" not in m for m in nhat_ky if "mo_rong" in m or "a1p6" in m or "a0p05" in m)
    assert any("bị cắt" in m for m in nhat_ky if m.startswith("[") and "nen__" in m)
    r1 = SL.tong_hop_phan(kho, ke, cau, "chinh", QWEN35_9B.dau_hieu_thinking)
    r2 = SL.tong_hop_phan(kho, ke, cau, "mo_rong", QWEN35_9B.dau_hieu_thinking)
    ht = SL.tong_hop_cuoi(kho, ke)
    chinh = SL.tach_dong_jsonl((kho.goc / "cau_tra_loi.jsonl").read_text())
    mo = SL.tach_dong_jsonl((kho.goc / "cau_tra_loi_mo_rong.jsonl").read_text())
    assert {r["nhom"] for r in chinh} == {"nen"} and {r["nhom"] for r in mo} == {"mo_rong"}
    assert r1["so_cau_tra_loi"] == 4 and r2["so_cau_tra_loi"] == 4 and ht["so_cau_tra_loi"] == 8
    tt = json.loads((kho.goc / "tom_tat.json").read_text())
    assert all("a1p6" not in k and "a0p05" not in k for k in tt["theo_don_vi"])   # thống kê mở rộng để file riêng
    assert (kho.goc / "tom_tat_mo_rong.json").exists() and SL.da_hoan_tat(kho, ke, cau)
    # tổng hợp lại lần nữa: không file nào đổi
    truoc = _anh_thu_muc(kho.goc)
    SL.tong_hop_phan(kho, ke, cau, "chinh", QWEN35_9B.dau_hieu_thinking)
    SL.tong_hop_phan(kho, ke, cau, "mo_rong", QWEN35_9B.dau_hieu_thinking)
    SL.tong_hop_cuoi(kho, ke)
    assert _anh_thu_muc(kho.goc) == truoc


def test_ky_tu_tach_dong_unicode_khong_lam_hong_don_vi(tmp_path):
    """Câu trả lời chứa U+2028 / U+2029 / U+0085 (json.dumps để nguyên) vẫn được tính là xong."""
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    dv = SL.ke_hoach_hieu_chinh(QWEN35_9B)[0]
    dong = [{"ma_don_vi": dv.ma, "id": "a", "van_ban": "x\u2028y\u2029z\u0085w", "bi_cat": False,
             "token_ket_thuc": 1, "nhom": "nen"}, {"ma_don_vi": dv.ma, "id": "b", "van_ban": "ok", "bi_cat": True,
                                                   "token_ket_thuc": None, "nhom": "nen"}]
    kho.ghi_don_vi(dv, dong, {})
    assert kho.kiem_don_vi(dv, ["a", "b"]) == (True, "xong")
    assert SL.tach_dong_jsonl(kho.jsonl(dv).read_text(encoding="utf-8"))[0]["van_ban"] == "x\u2028y\u2029z\u0085w"


def test_khoa_mot_tien_trinh(thu_muc_qwen35, tmp_path, monkeypatch):
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    tm = tmp_path / "l"
    giu = SL.KhoKetQua(tm)
    giu.khoa()                                                     # giả lập tiến trình khác đang chạy
    try:
        with pytest.raises(SL.DangChay):
            SL.KhoKetQua(tm).khoa()
        assert S.main(["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--huong", str(h),
                       "--thu_muc", str(tm), "--thu", "--gioi_han_cau", "1", "--max_new_tokens", "2",
                       "--khong_mo_rong"]) == 9
        assert not tm.exists()
    finally:
        giu.nha_khoa()
    k2 = SL.KhoKetQua(tm)
    k2.khoa()                                                      # nhả rồi thì khoá lại được
    k2.nha_khoa()


def test_moi_truong_khac_phien_dau(tmp_path):
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    assert kho.kiem_moi_truong({"gpu": "A"}) == (True, [])
    assert kho.kiem_moi_truong({"gpu": "B"}) == (True, [])          # chưa có đơn vị: phiên mới thành gốc
    assert json.loads(kho.file_moi_truong.read_text())["gpu"] == "B" and any(kho.hong.iterdir())
    dv = SL.ke_hoach_hieu_chinh(QWEN35_9B)[0]
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {})
    assert kho.kiem_moi_truong({"gpu": "C"}) == (False, ["gpu"])    # đã có đơn vị: dừng
    assert kho.kiem_moi_truong({"gpu": "C"}, cho_phep_doi=True) == (True, ["gpu"])
    assert json.loads(kho.file_moi_truong.read_text())["gpu"] == "B"   # gốc không bị đổi



def test_mat_moi_truong_goc_thi_so_voi_don_vi_cu(tmp_path):
    """Chỉ chép don_vi/ sang máy mới (mất moi_truong_goc.json): so với dấu vân tay môi trường trong .xong.json."""
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    dv = SL.ke_hoach_hieu_chinh(QWEN35_9B)[0]
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {"moi_truong_so_hoc_sha256": SL.bam_moi_truong({"gpu": "A"})})
    assert not kho.file_moi_truong.exists()
    assert kho.kiem_moi_truong({"gpu": "B"}) == (False, ["thieu_moi_truong_goc_va_khac_don_vi_cu"])
    assert not kho.file_moi_truong.exists()
    assert kho.kiem_moi_truong({"gpu": "A"}) == (True, [])          # cùng môi trường với đơn vị cũ: ghi lại gốc
    assert json.loads(kho.file_moi_truong.read_text())["gpu"] == "A"

def test_script_moi_truong_khac_dung_ma_8(thu_muc_qwen35, tmp_path, monkeypatch):
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    chung = ["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--huong", str(h), "--thu_muc",
             str(tmp_path / "l"), "--thu", "--gioi_han_cau", "1", "--max_new_tokens", "2", "--khong_mo_rong"]
    assert S.main(chung + ["--so_don_vi_toi_da", "2"]) == 7
    goc = S._moi_truong_so_hoc
    monkeypatch.setattr(S, "_moi_truong_so_hoc", lambda m, t: {**goc(m, t), "gpu": "may-khac"})
    truoc = _anh_thu_muc(tmp_path / "l" / "don_vi")
    assert S.main(chung + ["--so_don_vi_toi_da", "1"]) == 8
    assert _anh_thu_muc(tmp_path / "l" / "don_vi") == truoc
    assert S.main(chung + ["--so_don_vi_toi_da", "1", "--cho_phep_doi_moi_truong"]) == 7
    xong = sorted((tmp_path / "l" / "don_vi").glob("*.xong.json"))
    phien = {json.loads(p.read_text())["phien"] for p in xong}
    assert phien == {1, 3}                                          # mỗi đơn vị ghi phiên đã sinh ra nó


def test_script_gop_khong_nap_mo_hinh_khi_moi_don_vi_da_xong(thu_muc_qwen35, tmp_path, monkeypatch):
    """Máy dừng sau đơn vị cuối, trước khi gộp: chạy lại thì gộp mà không nạp mô hình (vd. trên máy cá nhân)."""
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    tm = tmp_path / "l"
    chung = ["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--huong", str(h), "--thu_muc",
             str(tm), "--thu", "--gioi_han_cau", "1", "--max_new_tokens", "2", "--khong_mo_rong"]
    goc_gop = SL.tong_hop_phan
    monkeypatch.setattr(S, "tong_hop_phan", lambda k, ke, c, phan, *a, **kw: (_ for _ in ()).throw(
        RuntimeError("máy dừng")) if phan == "chan_doan" else goc_gop(k, ke, c, phan, *a, **kw))
    with pytest.raises(RuntimeError):
        S.main(chung)
    assert (tm / "cau_tra_loi.jsonl").exists() and not (tm / "HOAN_TAT.json").exists()   # phần chính đã gộp trước
    monkeypatch.setattr(S, "tong_hop_phan", goc_gop)
    monkeypatch.setattr(S, "nap_mo_hinh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("không được nạp")))
    assert S.main(chung) == 0
    assert (tm / "HOAN_TAT.json").exists() and len((tm / "cau_tra_loi.jsonl").read_text().splitlines()) == 324


def test_lap_lai_ke_hoach_khi_chua_sinh_don_vi_nao(tmp_path):
    kho = SL.KhoKetQua(tmp_path / "l")
    assert kho.mo({"dau_van_tay": {"x": 1}}) == "moi"
    assert kho.mo({"dau_van_tay": {"x": 2}}) == "lam_lai_ke_hoach"     # lượt trước dừng ở bước kiểm
    assert json.loads(kho.file_ke_hoach.read_text())["dau_van_tay"] == {"x": 2} and any(kho.hong.iterdir())
    dv = SL.ke_hoach_hieu_chinh(QWEN35_9B)[0]
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {})
    with pytest.raises(ValueError):
        kho.mo({"dau_van_tay": {"x": 3}})



# ───────────────────── sửa sau rà soát code bước 4 của hai reviewer (28/10/2026) ─────────────────────
def test_chet_luc_ghi_ke_hoach_dau_tien_chay_lai_duoc(tmp_path, monkeypatch):
    """R2 C1: chết sau khi ghi file tạm của ke_hoach.json, trước os.replace → chạy lại cùng lệnh phải tiếp được."""
    goc_replace = SL.os.replace

    def chet(*a, **k):
        raise KeyboardInterrupt("giả lập máy chết trước os.replace")
    monkeypatch.setattr(SL.os, "replace", chet)
    kho = SL.KhoKetQua(tmp_path / "l")
    with pytest.raises(KeyboardInterrupt):
        kho.mo({"dau_van_tay": {"x": 1}})
    assert [p.name.startswith(".ke_hoach.json.tam-") for p in (tmp_path / "l").iterdir()] == [True]
    monkeypatch.setattr(SL.os, "replace", goc_replace)
    assert SL.KhoKetQua(tmp_path / "l").mo({"dau_van_tay": {"x": 1}}) == "moi"
    assert kho.file_ke_hoach.exists() and len(list(kho.hong.iterdir())) == 1          # file tạm sang hong/, không xoá
    # thư mục có file lạ (không phải file tạm của kế hoạch) thì vẫn từ chối
    la = tmp_path / "la"
    la.mkdir()
    (la / "ghi_chu.txt").write_text("x")
    with pytest.raises(ValueError):
        SL.KhoKetQua(la).mo({"dau_van_tay": {"x": 1}})


def test_file_hong_khong_phai_utf8(tmp_path):
    """R2 C2 và các file đọc lại khác: byte không phải UTF-8 không làm script hỏng lặp lại."""
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    dv = SL.ke_hoach_hieu_chinh(QWEN35_9B)[0]
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {"moi_truong_so_hoc_sha256": SL.bam_moi_truong({"g": 1})})
    assert kho.kiem_don_vi(dv, ["a"]) == (True, "xong")
    kho.xong(dv).write_bytes(b"\xff\xfe{")
    assert kho.kiem_don_vi(dv, ["a"]) == (False, "xong_json_hong")
    assert len(kho.chuyen_hong(dv, "xong_json_hong")) == 2
    # moi_truong_goc.json hỏng, chưa có đơn vị → chuyển sang hong/, phiên này thành gốc
    kho.file_moi_truong.write_bytes(b"\xff")
    assert kho.kiem_moi_truong({"g": 2}) == (True, [])
    assert json.loads(kho.file_moi_truong.read_text())["g"] == 2
    # phien.jsonl có dòng hỏng → vẫn thêm được phiên mới, đánh số tiếp
    (kho.goc / "phien.jsonl").write_bytes(b'{"phien": 1}\n\xff\xfe')
    assert kho.ghi_phien({"x": 1}) == 3
    # ke_hoach.json hỏng: chưa có đơn vị → lập lại (bản hỏng sang hong/); đã có đơn vị → dừng, không ghi đè
    kho.file_ke_hoach.write_bytes(b"\xff")
    assert kho.mo({"dau_van_tay": {"x": 1}}) == "lam_lai_ke_hoach"
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {})
    kho.file_ke_hoach.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="hỏng"):
        kho.mo({"dau_van_tay": {"x": 1}})
    assert kho.file_ke_hoach.read_bytes() == b"\xff"


def test_da_hoan_tat_voi_file_tong_hop_hong(qwen, tmp_path):
    model, tok = qwen
    huong, s = _huong_gia(QWEN35_9B)
    cau = _cau(1)
    ke = SL.ke_hoach_hieu_chinh(QWEN35_9B, luoi=(0.4,), mo_rong=())[:2]
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    SL.chay_ke_hoach(model, tok, QWEN35_9B, ke, cau, huong, s, eos_tu_mo_hinh(model, QWEN35_9B), kho, 2, 40,
                     log=lambda m: None)
    SL.tong_hop_phan(kho, ke, cau, "chinh")
    SL.tong_hop_cuoi(kho, ke)
    assert SL.da_hoan_tat(kho, ke, cau)
    (kho.goc / "HOAN_TAT_CHINH.json").write_bytes(b"\xff")
    assert not SL.da_hoan_tat(kho, ke, cau)                              # không traceback: gộp lại từ đơn vị


def test_da_hoan_tat_kiem_ca_hoan_tat_va_sha256(qwen, tmp_path):
    """Đối chiếu lần cuối, R2 F5: HOAN_TAT.json hỏng, sha256.txt rỗng / thiếu dòng, một file tổng hợp đổi sau khi
    xong, file gộp không còn đúng bằng ghép các đơn vị → chưa hoàn tất. Gộp lại thì bản cũ sang hong/, rồi lại đạt."""
    model, tok = qwen
    huong, s = _huong_gia(QWEN35_9B)
    cau = _cau(1)
    ke = SL.ke_hoach_hieu_chinh(QWEN35_9B, luoi=(0.4,), mo_rong=())[:2]
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    SL.chay_ke_hoach(model, tok, QWEN35_9B, ke, cau, huong, s, eos_tu_mo_hinh(model, QWEN35_9B), kho, 2, 40,
                     log=lambda m: None)

    def gop_lai():
        SL.tong_hop_phan(kho, ke, cau, "chinh")
        SL.tong_hop_cuoi(kho, ke)
        assert SL.da_hoan_tat(kho, ke, cau)

    gop_lai()
    p_ht, p_sha = kho.goc / "HOAN_TAT.json", kho.goc / "sha256.txt"
    for f, noi in ((p_ht, b"\xff"), (p_ht, b"{}\n"), (p_sha, b""),
                   (p_sha, b"".join(p_sha.read_bytes().splitlines(keepends=True)[1:])),
                   (kho.goc / "tom_tat.json", b'{"sua": 1}\n')):
        goc = f.read_bytes()
        f.write_bytes(noi)
        assert not SL.da_hoan_tat(kho, ke, cau), (f.name, noi[:10])
        so_hong = len(list((kho.goc / "hong").iterdir())) if (kho.goc / "hong").exists() else 0
        gop_lai()
        assert len(list((kho.goc / "hong").iterdir())) > so_hong      # bản hỏng không bị ghi đè
        assert f.read_bytes() == goc
    assert SL.doi_sau_hoan_tat(kho) == []
    (kho.goc / "la.txt").write_text("x")
    (kho.goc / "tom_tat.json").write_text("{}")
    assert SL.doi_sau_hoan_tat(kho) == ["thêm: la.txt", "khác: tom_tat.json"]
    gop_lai()
    (kho.goc / "la.txt").rename(kho.goc / "hong" / "la.txt")
    gop_lai()
    # file gộp và HOAN_TAT của phần cùng bị thay cho khớp nhau, nhưng không còn đúng bằng ghép các đơn vị
    p_gop, p_htc = kho.goc / "cau_tra_loi.jsonl", kho.goc / "HOAN_TAT_CHINH.json"
    gop = p_gop.read_text(encoding="utf-8")
    p_gop.write_text(gop.split("\n", 1)[1], encoding="utf-8")
    htc = json.loads(p_htc.read_text(encoding="utf-8"))
    p_htc.write_text(json.dumps({**htc, "sha256": SL.sha256_file(p_gop)}), encoding="utf-8")
    assert not SL.da_hoan_tat(kho, ke, cau)
    gop_lai()
    assert p_gop.read_text(encoding="utf-8") == gop


def test_khoa_tren_he_file_khong_ho_tro(tmp_path, monkeypatch):
    """R1 C1: chỉ BlockingIOError mới là 'tiến trình khác đang chạy'; lỗi khác của flock → chạy không khoá, cảnh báo."""
    import errno
    import fcntl

    def khong_ho_tro(*a, **k):
        raise OSError(errno.ENOLCK, "No locks available")
    monkeypatch.setattr(fcntl, "flock", khong_ho_tro)
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.khoa()
    assert kho.canh_bao_khoa and "không hỗ trợ khoá" in kho.canh_bao_khoa

    def dang_bi_giu(*a, **k):
        raise BlockingIOError(errno.EWOULDBLOCK, "Resource temporarily unavailable")
    monkeypatch.setattr(fcntl, "flock", dang_bi_giu)
    with pytest.raises(SL.DangChay):
        SL.KhoKetQua(tmp_path / "l").khoa()


def test_doi_chieu_buoc3():
    """R2 B2: backend kernel, phiên bản gói và commit phải trùng lượt kiểm hook bước 3; GPU khác chỉ ghi lại."""
    import buoc4_hieu_chinh as S
    mt = {k: f"v_{k}" for k in S.KHOA_MOI_TRUONG_BUOC3} | {"gpu": ["H100"]}
    bk = {"chunk_gated_delta_rule": {"cai_dat": "fla.ops.chunk_gated_delta_rule", "la_ban_tham_chieu": False}}
    meta = {"moi_truong": dict(mt), "backend_lop_tuyen_tinh_bf16": bk, "mo_hinh": {"commit_hf": "c1"}}
    assert S.doi_chieu_buoc3(meta, mt, bk, "c1", thu=False) == {"bo_qua": False, "khac": [], "gpu_khac": None}
    assert S.doi_chieu_buoc3(meta, mt | {"gpu": ["A100"]}, bk, "c1", False)["khac"] == []
    assert S.doi_chieu_buoc3(meta, mt | {"gpu": ["A100"]}, bk, "c1", False)["gpu_khac"] is not None
    assert S.doi_chieu_buoc3(meta, mt | {"torch": "khac"}, bk, "c1", False)["khac"][0].startswith("torch")
    assert S.doi_chieu_buoc3(meta, mt | {"goi_flash-linear-attention": None}, bk, "c1", False)["khac"]
    bk_pt = {"chunk_gated_delta_rule": {"cai_dat": "transformers.x.torch_chunk", "la_ban_tham_chieu": True}}
    assert "backend" in S.doi_chieu_buoc3(meta, mt, bk_pt, "c1", False)["khac"][0]
    assert S.doi_chieu_buoc3(meta, mt, bk, "c2", False)["khac"][0].startswith("commit")
    # meta sai dạng: không đạt, không traceback
    assert S.doi_chieu_buoc3(meta | {"backend_lop_tuyen_tinh_bf16": ["x"]}, mt, bk, "c1", False)["khac"]
    assert S.doi_chieu_buoc3(meta | {"moi_truong": [1], "mo_hinh": "x"}, mt, bk, "c1", False)["khac"]
    assert S.doi_chieu_buoc3([1], mt, bk, "c1", False)["khac"]
    # lượt chính thức: meta thiếu trường thì không đạt; lượt thử thì bỏ qua phần thiếu; không có meta thì bỏ qua
    assert len(S.doi_chieu_buoc3({"mo_hinh": {"commit_hf": "c1"}}, mt, bk, "c1", False)["khac"]) == 2
    assert S.doi_chieu_buoc3({}, mt, bk, None, True)["khac"] == []
    assert S.doi_chieu_buoc3(None, mt, bk, None, False)["bo_qua"] is True


def test_script_moi_truong_khac_buoc3_dung_ma_4(thu_muc_qwen35, tmp_path, monkeypatch):
    """R2 B2 qua script: môi trường khác lượt kiểm hook → mã 4 sau khi nạp, chưa sinh đơn vị nào; trùng → chạy."""
    import buoc4_hieu_chinh as S
    from tn2 import nguon_goc as NG
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    chung = ["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--huong", str(h), "--thu",
             "--gioi_han_cau", "1", "--max_new_tokens", "2", "--khong_mo_rong", "--so_don_vi_toi_da", "1"]
    kh = _kiem_hook_gia(tmp_path / "kh", h)
    meta = json.loads((kh / "meta.json").read_text())
    (kh / "meta.json").write_text(json.dumps(meta | {"moi_truong": NG.thong_tin_moi_truong() | {"torch": "0.0"}}))
    assert S.main(chung + ["--kiem_hook", str(kh), "--thu_muc", str(tmp_path / "l1")]) == 4
    assert not SL.KhoKetQua(tmp_path / "l1").co_don_vi_nao()
    (kh / "meta.json").write_text(json.dumps(meta | {"moi_truong": NG.thong_tin_moi_truong()}))
    assert S.main(chung + ["--kiem_hook", str(kh), "--thu_muc", str(tmp_path / "l2")]) == 7
    phien = json.loads((tmp_path / "l2" / "phien.jsonl").read_text().splitlines()[0])
    assert phien["doi_chieu_buoc3"]["khac"] == [] and phien["doi_chieu_buoc3"]["bo_qua"] is False


def test_script_chinh_thuc_dung_khi_khong_xac_minh_duoc_commit(thu_muc_qwen35, tmp_path, monkeypatch):
    """R2 B1: lượt chính thức, mô hình nạp không có commit (thư mục cục bộ) → mã 4 trước khi sinh."""
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B, commit_hf="abc")
    kh = _kiem_hook_gia(tmp_path / "kh", h)
    tm = tmp_path / "l"
    assert S.main(["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--revision", "main",
                   "--huong", str(h), "--kiem_hook", str(kh), "--thu_muc", str(tm), "--khong_mo_rong"]) == 4
    assert not SL.KhoKetQua(tm).co_don_vi_nao()
    assert "không xác minh được" in (tm / "nhat_ky.txt").read_text(encoding="utf-8")


def test_script_co_lo(thu_muc_qwen35, tmp_path, monkeypatch):
    """Q5: --co_lo nằm trong dấu vân tay; lượt đã có đơn vị mà đổi cỡ lô thì mã 6; ngoài khoảng 1–40 thì mã 1."""
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    chung = ["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--huong", str(h), "--thu_muc",
             str(tmp_path / "l"), "--thu", "--gioi_han_cau", "2", "--max_new_tokens", "2", "--khong_mo_rong"]
    assert S.main(chung + ["--co_lo", "0"]) == 1 and S.main(chung + ["--co_lo", "41"]) == 1
    assert S.main(chung + ["--so_don_vi_toi_da", "1"]) == 7
    assert S.main(chung + ["--so_don_vi_toi_da", "1", "--co_lo", "1"]) == 6
    assert json.loads((tmp_path / "l" / "ke_hoach.json").read_text())["tham_so"]["co_lo"] == 40



def test_json_sai_dang_khong_lam_hong_lap_lai(tmp_path):
    """JSON hợp lệ nhưng sai dạng (vd. sửa tay, chép xung đột): xử lý như file hỏng, không traceback."""
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    kho.file_ke_hoach.write_text("[]")
    assert kho.mo({"dau_van_tay": {"x": 1}}) == "lam_lai_ke_hoach"
    kho.file_moi_truong.write_text("[1]")
    assert kho.kiem_moi_truong({"g": 1}) == (True, [])                    # gốc sai dạng → hong/, phiên này thành gốc
    dv = SL.ke_hoach_hieu_chinh(QWEN35_9B)[0]
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {})
    kho.xong(dv).write_text("null")
    assert kho.kiem_don_vi(dv, ["a"]) == (False, "xong_json_hong")
    kho.chuyen_vao_hong(kho.file_moi_truong, "thu")
    assert kho.kiem_moi_truong({"g": 1}) == (False, ["thieu_moi_truong_goc_va_khac_don_vi_cu"])
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {})
    kho.file_ke_hoach.write_text("{}")
    with pytest.raises(ValueError, match="hỏng"):
        kho.mo({"dau_van_tay": {"x": 1}})
    (kho.goc / "HOAN_TAT.json").write_text("{}")
    (kho.goc / "HOAN_TAT_CHINH.json").write_text("[]")
    assert not SL.da_hoan_tat(kho, [dv], {dv.tap: [{"id": "a"}]})


def test_script_khong_khoa_duoc_chi_cho_gop(thu_muc_qwen35, tmp_path, monkeypatch):
    """Hệ file không hỗ trợ khoá: còn đơn vị phải sinh thì mã 9 (không chặn được hai tiến trình); chỉ gộp thì chạy."""
    import errno
    import fcntl

    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    tm = tmp_path / "l"
    chung = ["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--huong", str(h), "--thu_muc",
             str(tm), "--thu", "--gioi_han_cau", "1", "--max_new_tokens", "2", "--khong_mo_rong"]
    goc_gop, goc_flock = SL.tong_hop_phan, fcntl.flock
    monkeypatch.setattr(S, "tong_hop_phan", lambda k, ke, c, phan, *a, **kw: (_ for _ in ()).throw(
        RuntimeError("máy dừng")) if phan == "chan_doan" else goc_gop(k, ke, c, phan, *a, **kw))
    with pytest.raises(RuntimeError):
        S.main(chung)                                                    # sinh xong mọi đơn vị, chết lúc gộp
    monkeypatch.setattr(S, "tong_hop_phan", goc_gop)

    def khong_ho_tro(*a, **k):
        raise OSError(errno.ENOLCK, "No locks available")
    monkeypatch.setattr(fcntl, "flock", khong_ho_tro)
    assert S.main(chung) == 0                                            # chỉ gộp: chạy không khoá, có cảnh báo
    assert "không hỗ trợ khoá" in (tm / "nhat_ky.txt").read_text(encoding="utf-8")
    tm2 = tmp_path / "l2"
    assert S.main([x if x != str(tm) else str(tm2) for x in chung]) == 9    # còn phải sinh: dừng
    assert not SL.KhoKetQua(tm2).co_don_vi_nao()
    monkeypatch.setattr(fcntl, "flock", goc_flock)


def test_nap_mo_hinh_ghi_commit_cua_checkpoint_ghep(tmp_path):
    """Qwen3.5 là checkpoint ghép: AutoModelForCausalLM nạp config con không có _commit_hash. nap_mo_hinh phải ghi
    đúng commit khi nạp từ cache Hugging Face (id + revision), và None khi nạp từ thư mục cục bộ."""
    import subprocess
    import sys
    from pathlib import Path
    goc = Path(__file__).resolve().parents[1]
    ma = f"""
import os, sys
sys.path[:0] = [{str(goc / "src")!r}, {str(goc / "tests")!r}]
from pathlib import Path
from conftest import tao_qwen35_vlm_nho, tao_tok_qwen35_that, tao_gemma4_vlm_nho, tao_tok_gemma4_that
from tn2.mo_hinh import QWEN35_9B, GEMMA4_12B, nap_mo_hinh
from tn2 import nguon_goc as NG
h = "c2022362" + "0" * 32
for dt, mk, tk in ((QWEN35_9B, tao_qwen35_vlm_nho, tao_tok_qwen35_that), (GEMMA4_12B, tao_gemma4_vlm_nho, tao_tok_gemma4_that)):
    org, ten = dt.hf_id.split("/")
    repo = Path(os.environ["HF_HOME"]) / "hub" / f"models--{{org}}--{{ten}}"
    snap = repo / "snapshots" / h
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(h)
    mk().save_pretrained(snap); tk().save_pretrained(snap)
    m, t = nap_mo_hinh(dt, None, revision=h, device_map=None)
    print("HF", dt.ten, NG.thong_tin_mo_hinh(m, t, dt)["commit_hf"])
    import shutil
    cucbo = Path(os.environ["HF_HOME"]).parent / ("cucbo_" + dt.ten)       # thư mục cục bộ ngoài cache HF
    shutil.copytree(snap, cucbo)
    m, t = nap_mo_hinh(dt, str(cucbo), device_map=None)
    print("CUCBO", dt.ten, NG.thong_tin_mo_hinh(m, t, dt)["commit_hf"])
"""
    env = {**os.environ, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "CUDA_VISIBLE_DEVICES": ""}
    kq = subprocess.run([sys.executable, "-c", ma], env=env, capture_output=True, text=True, timeout=900)
    assert kq.returncode == 0, kq.stderr[-3000:]
    dong = [x for x in kq.stdout.splitlines() if x.startswith(("HF ", "CUCBO "))]
    h = "c2022362" + "0" * 32
    assert dong == [f"HF qwen35_9b {h}", "CUCBO qwen35_9b None", f"HF gemma4_12b {h}", "CUCBO gemma4_12b None"]



# ───────────── sửa sau rà lại toàn bộ pipeline và đối chiếu của hai reviewer (28/10/2026) ─────────────
def test_script_chi_gop(thu_muc_qwen35, tmp_path, monkeypatch):
    """--chi_gop: không bao giờ nạp mô hình; thiếu đơn vị thì mã 1; mã nguồn khác lúc sinh vẫn gộp (ghi lại);
    thư mục chưa có lượt thì mã 1."""
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    tm = tmp_path / "l"
    chung = ["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--huong", str(h), "--thu_muc",
             str(tm), "--thu", "--gioi_han_cau", "1", "--max_new_tokens", "2", "--khong_mo_rong"]
    assert S.main(chung + ["--so_don_vi_toi_da", "3"]) == 7
    monkeypatch.setattr(S, "nap_mo_hinh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("không được nạp")))
    assert S.main(chung + ["--chi_gop"]) == 1                            # thiếu đơn vị: báo, không nạp
    assert "còn 325 đơn vị chưa xong" in (tm / "nhat_ky.txt").read_text(encoding="utf-8")
    assert S.main([x if x != str(tm) else str(tmp_path / "trong") for x in chung] + ["--chi_gop"]) == 1
    assert not (tmp_path / "trong" / "ke_hoach.json").exists()
    monkeypatch.undo()
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    assert S.main(chung) == 0
    (tm / "HOAN_TAT.json").rename(tmp_path / "ht_cu.json")              # bỏ một file tổng hợp: phải gộp lại
    goc = S.dau_van_tay_ma_nguon
    monkeypatch.setattr(S, "dau_van_tay_ma_nguon", lambda g: {**goc(g), "sha256": "code_buoc5_moi"})
    monkeypatch.setattr(S, "nap_mo_hinh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("không được nạp")))
    assert S.main(chung) == 6                                            # không --chi_gop: code khác → 6
    assert S.main(chung + ["--chi_gop"]) == 0                            # --chi_gop: gộp được, ghi lại
    assert (tm / "HOAN_TAT.json").exists()
    (tm / "tom_tat.json").write_text("{}")                               # đổi một file sau khi hoàn tất: báo rõ
    assert S.main(chung + ["--chi_gop"]) == 0
    assert "khác: tom_tat.json" in (tm / "nhat_ky.txt").read_text(encoding="utf-8")
    assert S.main(chung + ["--chi_gop"]) == 0                            # lần sau: còn nguyên, không làm gì
    assert "mã nguồn khác lúc sinh" in (tm / "nhat_ky.txt").read_text(encoding="utf-8")


def test_kiem_chia_tap_truoc_khi_chay(tmp_path, monkeypatch):
    """Bước 4 tính lại cách chia và đòi trùng file, đạt mọi ràng buộc; khác thì mã 4, không tạo thư mục lượt."""
    import buoc4_hieu_chinh as S
    assert S._kiem_chia_tap() == []
    d = json.loads(S.FILE_CHIA.read_text(encoding="utf-8"))
    d["tap"]["D_plain_test"]["hieu_chinh"][0], d["tap"]["D_plain_test"]["hieu_chinh"][1] = (
        d["tap"]["D_plain_test"]["hieu_chinh"][1], d["tap"]["D_plain_test"]["hieu_chinh"][0])
    sua = tmp_path / "chia_tap.json"
    sua.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(S, "FILE_CHIA", sua)
    assert any("KHÁC" in x for x in S._kiem_chia_tap())
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    assert S.main(["--mo_hinh_dac_ta", "qwen35_9b", "--huong", str(h), "--thu", "--thu_muc",
                   str(tmp_path / "l")]) == 4
    assert not (tmp_path / "l").exists()


def test_quyet_dinh_can_xem(thu_muc_qwen35, tmp_path, monkeypatch):
    """Q-A: 'cần xem' chỉ đi tiếp được với file quyết định của Ân; file được chép vào thư mục lượt, SHA vào dấu vân
    tay. Không có file → mã 4 trước khi tạo thư mục; có file mà không có gì 'cần xem' → mã 4."""
    import buoc4_hieu_chinh as S
    monkeypatch.setattr(S, "device_map_mot_gpu", lambda: None)
    qd = tmp_path / "quyet_dinh.md"
    qd.write_text("Ân quyết đi tiếp (Nhap, mục Kiểm pipeline); câu hạn chế: ...", encoding="utf-8")
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B, commit_hf="abc", trang_thai="can_xem")
    kh = _kiem_hook_gia(tmp_path / "kh", h)
    tm = tmp_path / "l"
    chung = ["--mo_hinh_dac_ta", "qwen35_9b", "--mo_hinh", str(thu_muc_qwen35), "--revision", "main",
             "--huong", str(h), "--kiem_hook", str(kh), "--thu_muc", str(tm), "--khong_mo_rong"]
    assert S.main(chung) == 4 and not tm.exists()                        # cần xem, không có quyết định
    assert S.main(chung + ["--quyet_dinh_can_xem", str(tmp_path / "khong_co.md")]) == 4 and not tm.exists()
    # có quyết định: qua cổng 'cần xem'; dừng sau ở cổng commit (mô hình cục bộ không có commit), chưa sinh gì
    assert S.main(chung + ["--quyet_dinh_can_xem", str(qd)]) == 4
    ke = json.loads((tm / "ke_hoach.json").read_text(encoding="utf-8"))
    assert ke["quyet_dinh_can_xem"]["ap_cho"] == ["huong"]
    assert ke["dau_van_tay"]["quyet_dinh_can_xem_sha256"] == SL.sha256_file(qd)
    assert (tm / "quyet_dinh_can_xem.md").read_bytes() == qd.read_bytes()
    assert not SL.KhoKetQua(tm).co_don_vi_nao()
    qd.write_text("Ân sửa lại quyết định", encoding="utf-8")                # bản chép phải theo file đang dùng
    assert S.main(chung + ["--quyet_dinh_can_xem", str(qd)]) == 4
    assert (tm / "quyet_dinh_can_xem.md").read_bytes() == qd.read_bytes()
    assert any(p.name.startswith("quyet_dinh_can_xem.md") for p in (tm / "hong").iterdir())
    # kiểm hook 'cần xem' cũng cần quyết định; hướng và kiểm hook đều 'dat' mà có quyết định → mã 4
    h2 = _luu_huong(tmp_path / "h2.pt", QWEN35_9B, commit_hf="abc")
    kh2 = _kiem_hook_gia(tmp_path / "kh2", h2, trang_thai="can_xem")
    c2 = [x if x not in (str(h), str(kh), str(tm)) else {str(h): str(h2), str(kh): str(kh2), str(tm): str(tmp_path / "l2")}[x]
          for x in chung]
    assert S.main(c2) == 4
    kh3 = _kiem_hook_gia(tmp_path / "kh3", h2)
    c3 = [x if x != str(kh2) else str(kh3) for x in c2]
    assert S.main(c3 + ["--quyet_dinh_can_xem", str(qd)]) == 4 and not (tmp_path / "l2").exists()
    assert S._kiem_huong({"dac_ta": "qwen35_9b", "trang_thai": "khong_dat", "chinh_thuc": True,
                          "position_ids_theo_mat_na": True, "commit_hf": "x", "r_hat_en": 0, "r_hat_vi": 0, "s": 0},
                         QWEN35_9B, thu=False, co_quyet_dinh=True)                   # 'không đạt' không bao giờ qua


def test_so_phien_khong_trung_khi_phien_jsonl_cu(tmp_path):
    """Số phiên lớn hơn mọi số đã dùng, kể cả số trong .xong.json (phien.jsonl cũ hơn các đơn vị)."""
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    assert kho.ghi_phien({}) == 1
    dv = SL.ke_hoach_hieu_chinh(QWEN35_9B)[0]
    kho.ghi_don_vi(dv, [{"ma_don_vi": dv.ma, "id": "a"}], {"phien": 5})
    assert kho.ghi_phien({}) == 6


def test_so_phien_theo_gia_tri_trong_phien_jsonl(tmp_path):
    """Đối chiếu lần cuối, R2 F4: phien.jsonl có dòng mang số phiên lớn hơn số dòng (vd. sau khi ghép bản kéo về) →
    số mới lớn hơn mọi số đã ghi; dòng hỏng không chặn."""
    kho = SL.KhoKetQua(tmp_path / "l")
    kho.mo({"dau_van_tay": {"x": 1}})
    (kho.goc / "phien.jsonl").write_bytes(b'{"phien": 6}\n\xff\n')
    assert kho.ghi_phien({}) == 7
    assert kho.ghi_phien({}) == 8
    assert [json.loads(x)["phien"] for x in (kho.goc / "phien.jsonl").read_bytes().split(b"\n")
            if x.strip() and x != b"\xff"] == [6, 7, 8]


def test_ghi_json_nguyen_tu(tmp_path):
    """Đối chiếu lần cuối, R2: ghi_json lỗi giữa chừng (đối tượng không tuần tự hoá được) không làm hỏng file cũ, không
    để lại file dở; ghi được thì thay trọn."""
    from tn2 import nguon_goc as NG
    f = tmp_path / "a" / "ket_luan.json"
    NG.ghi_json(f, {"trang_thai": "dat", "x": float("nan")})
    cu = f.read_bytes()
    assert json.loads(cu) == {"trang_thai": "dat", "x": None}
    with pytest.raises(TypeError):
        NG.ghi_json(f, {"trang_thai": object()})
    assert f.read_bytes() == cu and sorted(p.name for p in f.parent.iterdir()) == ["ket_luan.json"]
    NG.ghi_jsonl(tmp_path / "a" / "b.jsonl", [{"a": 1}, {"b": "ă"}])
    assert NG.doc_jsonl(tmp_path / "a" / "b.jsonl") == [{"a": 1}, {"b": "ă"}]


def test_du_lieu_chia_tap_hong_ma_4(tmp_path, monkeypatch):
    """Đối chiếu lần cuối, R2: chia tập kiểm TRƯỚC khi đọc câu; dữ liệu đọc lỗi → mã 4, không traceback."""
    import buoc4_hieu_chinh as S
    import tn2.tap_test as TT2
    monkeypatch.setattr(S, "_DA_KIEM_CHIA_TAP", {})
    monkeypatch.setattr(TT2, "chia_tap", lambda: (_ for _ in ()).throw(AssertionError("ràng buộc hỏng")))
    assert any("AssertionError" in x for x in S._kiem_chia_tap())
    monkeypatch.undo()
    goi = []
    monkeypatch.setattr(S, "_kiem_chia_tap", lambda: goi.append("kiem") or [])
    monkeypatch.setattr(S, "_cau_theo_tap", lambda g: goi.append("doc") or (_ for _ in ()).throw(KeyError("tap")))
    h = _luu_huong(tmp_path / "h.pt", QWEN35_9B)
    assert S.main(["--mo_hinh_dac_ta", "qwen35_9b", "--huong", str(h), "--thu", "--thu_muc",
                   str(tmp_path / "l")]) == 4
    assert goi == ["kiem", "doc"] and not (tmp_path / "l").exists()
    assert S.main(["--mo_hinh_dac_ta", "qwen35_9b", "--chi_ke_hoach"]) == 4


def test_kiem_hook_sai_dang_khong_traceback(tmp_path):
    """Đối chiếu bước 4 (R2 B3): ket_luan.json / meta.json sai dạng → lỗi mã 4, không traceback."""
    import buoc4_hieu_chinh as S
    d = tmp_path / "kh"
    d.mkdir()
    (d / "ket_luan.json").write_text("{}")
    (d / "meta.json").write_text("[]")
    loi, meta = S._kiem_kiem_hook(str(d), QWEN35_9B, "x", thu=False)
    assert loi and meta is None
    (d / "meta.json").write_text(json.dumps({"dac_ta": "qwen35_9b", "tham_so": [1]}))
    loi, _ = S._kiem_kiem_hook(str(d), QWEN35_9B, "x", thu=False)
    assert any("KHÁC" in x for x in loi)


def test_ham_b_sh(tmp_path):
    """scripts/ham_b.sh: b3 chạy tiếp từ 4d / 4e với lượt dựng hướng ĐÃ XONG mới nhất (bỏ thư mục dở không có
    huong.pt); b4 lượt chưa có đơn vị lấy lượt bước 3 đã xong mới nhất, kiểm hook phải khớp SHA file hướng; lượt đã có
    đơn vị thì đọc lại đúng tham số (kiểm hook theo phiên cuối); gop4 thêm --chi_gop. Thay `python` bằng một script
    ghi lại lệnh."""
    import hashlib
    import shutil
    import subprocess
    from pathlib import Path
    goc = Path(__file__).resolve().parents[1]
    w = tmp_path / "kho"
    (w / "scripts").mkdir(parents=True)
    shutil.copy(goc / "scripts" / "ham_b.sh", w / "scripts" / "ham_b.sh")
    b3 = w / "results" / "buoc3" / "qwen35_9b"
    for ten in ("dung_huong_20261028_100000", "dung_huong_20261028_090000", "dung_huong_20261028_110000"):
        (b3 / ten).mkdir(parents=True)
    (b3 / "dung_huong_20261028_100000" / "huong.pt").write_text("huong moi")
    (b3 / "dung_huong_20261028_090000" / "huong.pt").write_text("huong cu")
    (b3 / "dung_huong_20261028_110000" / "tom_tat.json").write_text("x")       # dở: chưa có huong.pt
    sha = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in (("moi", "huong moi"), ("cu", "huong cu"))}
    for ten, sh, xong in (("kiem_hook_20261028_100500", sha["moi"], True),
                          ("kiem_hook_20261028_120000", sha["cu"], True),        # mới hơn nhưng với hướng khác
                          ("kiem_hook_20261028_130000", sha["moi"], False)):     # dở: chưa có ket_luan.json
        (b3 / ten).mkdir(parents=True)
        (b3 / ten / "meta.json").write_text(json.dumps({"tham_so": {"huong_sha256": sh}}))
        if xong:
            (b3 / ten / "ket_luan.json").write_text("{}")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python").write_text('#!/bin/sh\necho "$@" >> "$GHI"\nexit 0\n')
    (bin_ / "python").chmod(0o755)
    (bin_ / "nvidia-smi").write_text('#!/bin/sh\necho "$NV"\n')
    (bin_ / "nvidia-smi").chmod(0o755)
    ghi = tmp_path / "ghi.txt"

    def chay(lenh, nv="3"):
        ghi.write_text("")
        env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "GHI": str(ghi), "NV": nv}
        kq = subprocess.run(["bash", "-c", f"source scripts/ham_b.sh && {lenh}"], cwd=w, env=env,
                            capture_output=True, text=True, timeout=60)
        return kq.returncode, ghi.read_text().splitlines()

    assert subprocess.run(["bash", "-n", str(goc / "scripts" / "ham_b.sh")]).returncode == 0
    rc, dong = chay("b3 qwen35_9b tu_4d")
    assert rc == 0 and [d.split()[0] for d in dong] == ["scripts/buoc3_kiem_hook.py", "scripts/buoc3_do_toc_do.py"]
    assert "dung_huong_20261028_100000/huong.pt" in dong[0]                  # bỏ thư mục dở mới hơn
    rc, dong = chay("b3 qwen35_9b tu_4e")
    assert rc == 0 and len(dong) == 1 and dong[0].startswith("scripts/buoc3_do_toc_do.py")
    rc, dong = chay("b3 qwen35_9b sai")
    assert rc == 1 and dong == []
    rc, dong = chay("b3 qwen35_9b tu_4e", nv="41000")                         # GPU đang bận (mô hình khác)
    assert rc == 1 and dong == []
    rc, dong = chay("b4 qwen35_9b")
    assert rc == 0 and "--huong results/buoc3/qwen35_9b/dung_huong_20261028_100000/huong.pt" in dong[0]
    assert "--kiem_hook results/buoc3/qwen35_9b/kiem_hook_20261028_100500" in dong[0]   # khớp SHA hướng
    tm = w / "results" / "buoc4" / "qwen35_9b" / "hieu_chinh"
    tm.mkdir(parents=True)
    (tm / "ke_hoach.json").write_text(json.dumps({"huong": "H_CU.pt", "kiem_hook": "K_CU",
                                                  "quyet_dinh_can_xem": {"file_goc": "qd.md"}}))
    rc, dong = chay("b4 qwen35_9b")                                           # chưa có đơn vị: lấy lượt bước 3 mới
    assert rc == 0 and "dung_huong_20261028_100000" in dong[0] and "--quyet_dinh_can_xem" not in dong[0]
    (tm / "don_vi").mkdir()
    (tm / "don_vi" / "x.xong.json").write_text("{}")
    (tm / "quyet_dinh_can_xem.md").write_text("qd")
    rc, dong = chay("b4 qwen35_9b")                                           # đã có đơn vị: đúng tham số cũ
    assert rc == 0 and dong[0].split()[1:] == ["--mo_hinh_dac_ta", "qwen35_9b", "--huong", "H_CU.pt", "--kiem_hook",
                                               "K_CU", "--quyet_dinh_can_xem",
                                               "results/buoc4/qwen35_9b/hieu_chinh/quyet_dinh_can_xem.md"]
    (tm / "phien.jsonl").write_text(json.dumps({"phien": 1, "kiem_hook": "K_CU"}) + "\n"
                                    + json.dumps({"phien": 2, "kiem_hook": "K_MOI"}) + "\n")
    rc, dong = chay("b4 qwen35_9b")                                           # kiểm hook theo phiên cuối
    assert rc == 0 and "--kiem_hook K_MOI" in dong[0]
    rc, dong = chay("gop4 qwen35_9b")
    assert rc == 0 and dong[0].endswith("--chi_gop") and "H_CU.pt" in dong[0]
    rc, dong = chay("b4 qwen35_9b --co_lo 20 --thu_muc results/buoc4/qwen35_9b/hieu_chinh_lo20")
    assert rc == 0 and "dung_huong_20261028_100000" in dong[0] and dong[0].endswith("hieu_chinh_lo20")
    assert (w / "results" / "nhat_ky").is_dir() and any((w / "results" / "nhat_ky").iterdir())
    assert "--co_lo" not in chay("b4 qwen35_9b")[1][0]                       # lượt mặc định: không thêm gì
    # Đối chiếu lần cuối, R2 F3: lượt đã có đơn vị chạy với --co_lo 20 / --khong_mo_rong → tự thêm lại, kể cả gop4
    tm2 = w / "results" / "buoc4" / "qwen35_9b" / "hieu_chinh_lo20"
    (tm2 / "don_vi").mkdir(parents=True)
    (tm2 / "don_vi" / "x.xong.json").write_text("{}")
    (tm2 / "ke_hoach.json").write_text(json.dumps({"huong": "H2.pt", "kiem_hook": "K2", "quyet_dinh_can_xem": None,
                                                   "tham_so": {"co_lo": 20, "alpha_mo_rong": []}}))
    for lenh in ("b4 qwen35_9b --thu_muc results/buoc4/qwen35_9b/hieu_chinh_lo20",
                 "gop4 qwen35_9b --thu_muc results/buoc4/qwen35_9b/hieu_chinh_lo20"):
        rc, dong = chay(lenh)
        assert rc == 0 and "--huong H2.pt --kiem_hook K2 --co_lo 20 --khong_mo_rong --thu_muc" in dong[0], dong
    # R1 E.2: bỏ kiểm GPU rảnh chỉ khi đặt rõ KHONG_KIEM_GPU=1; R1 E.3: PYTHON chỉ định trình thông dịch
    assert chay("KHONG_KIEM_GPU=1 b3 qwen35_9b tu_4e", nv="41000")[0] == 0
    (bin_ / "py_khac").write_text('#!/bin/sh\necho "KHAC $@" >> "$GHI"\n')
    (bin_ / "py_khac").chmod(0o755)
    rc, dong = chay(f"PYTHON={bin_ / 'py_khac'} gop4 qwen35_9b")
    assert rc == 0 and dong[0].startswith("KHAC scripts/buoc4_hieu_chinh.py")
    # R2 F1: may_moi in lệnh cài đúng phiên bản và đúng commit, từ kiểm hook mà b4 sẽ dùng cho mô hình này
    meta = {"tham_so": {"huong_sha256": sha["moi"]}, "mo_hinh": {"commit_hf": "c0ffee"},
            "moi_truong": {"torch": "2.11.0+cu128", "cuda": "12.8", "transformers": "5.17.0", "tokenizers": "0.23.2",
                           "goi_flash-linear-attention": "0.6.0", "goi_causal-conv1d": None, "goi_kernels": None,
                           "bien_USE_HUB_KERNELS": "NO", "gpu": ["NVIDIA A100 80GB"]}}
    (b3 / "kiem_hook_20261028_100500" / "meta.json").write_text(json.dumps(meta))
    import shutil as _sh
    _sh.rmtree(tm / "don_vi")                                                # lượt mặc định chưa có đơn vị
    kq = subprocess.run(["bash", "-c", "source scripts/ham_b.sh && may_moi qwen35_9b"], cwd=w,
                        capture_output=True, text=True, timeout=60)
    assert kq.returncode == 0, kq.stdout + kq.stderr
    lenh = [x for x in kq.stdout.splitlines() if not x.startswith("#")]
    assert lenh == ["pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128",
                    "pip install -r requirements.txt", "pip install transformers==5.17.0 tokenizers==0.23.2",
                    "pip install flash-linear-attention==0.6.0",
                    "HF_HUB_OFFLINE=0 hf download Qwen/Qwen3.5-9B --revision c0ffee"], kq.stdout
    assert "causal-conv1d: KHÔNG" in kq.stdout and "kiem_hook_20261028_100500" in kq.stdout
    assert subprocess.run(["bash", "-c", "source scripts/ham_b.sh && may_moi gemma4_12b"], cwd=w,
                          capture_output=True, text=True, timeout=60).returncode == 1   # Gemma chưa có bước 3
    # bảng id Hugging Face trong may_moi trùng đặc tả
    noi = (goc / "scripts" / "ham_b.sh").read_text(encoding="utf-8")
    assert f"qwen35_9b) ID={QWEN35_9B.hf_id} ;;" in noi and f"gemma4_12b) ID={GEMMA4_12B.hf_id} ;;" in noi
