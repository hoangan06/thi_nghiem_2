"""Bước 3 trên mô hình nhỏ cùng kiến trúc (Qwen3.5: 32 lớp DeltaNet + full attention; Gemma 4: 48 lớp cửa sổ trượt
+ full attention, có layer_scalar), nạp qua đúng đường của mô hình thật (checkpoint ảnh–chữ → AutoModelForCausalLM),
với tokenizer đồ chơi đặt token đặc biệt ở đúng id thật."""
import json

import pytest
import torch

from conftest import tao_gemma4_vlm_nho
from tn2 import nguon_goc as NG
from tn2.buoc3.dung_huong import dung_huong
from tn2.buoc3.nap_kiem import (dem_ro_thinking, do_position_ids, kiem_cau_truc, kiem_eos_tokenizer,
                                kiem_goi_embedding, kiem_khoi_tokenizer)
from tn2.du_lieu import TapDung, kiem_ghep_cap, kiem_trung, nap_tap_dung
from tn2.huong import cosine
from tn2.kich_hoat import kich_hoat_tung_cau, trung_binh_kich_hoat
from tn2.mo_hinh import GEMMA4_12B, QWEN35_9B, eos_tu_mo_hinh, nap_mo_hinh, phan_bo_thiet_bi, tim_cac_lop

HAI = [("qwen35", QWEN35_9B), ("gemma4", GEMMA4_12B)]


@pytest.fixture(scope="module")
def tap():
    return nap_tap_dung()


@pytest.fixture(scope="module")
def mo_hinh(thu_muc_qwen35, thu_muc_gemma4):
    """(model float32, tok) nạp qua nap_mo_hinh từ checkpoint ảnh–chữ đã lưu."""
    ra = {}
    for ten, dac_ta in HAI:
        d = thu_muc_qwen35 if ten == "qwen35" else thu_muc_gemma4
        ra[ten] = nap_mo_hinh(dac_ta, str(d), dtype=torch.float32, device_map=None)
    return ra


def _tap_nho(tap, n):
    return TapDung(tap.harm_en[:n], tap.harm_vi[:n], tap.harmless_en[:n], tap.harmless_vi[:n],
                   tap.id_harm[:n], tap.id_harmless[:n])


# ─────────────────────────────── dữ liệu ───────────────────────────────
def test_tap_dung_du_128_cap_va_ghep_dung(tap):
    for ten in ("harm_en", "harm_vi", "harmless_en", "harmless_vi"):
        assert len(getattr(tap, ten)) == 128
    assert tap.id_harm[0] == "D_harm_build/0000" and tap.id_harmless[-1] == "D_harmless_build/0127"
    assert tap.cau("D_harm_build", "vi")[0] == tap.harm_vi[0]


def test_kiem_ghep_cap_bat_loi():
    en = [{"id": "a", "text": "x", "nguon_idx": 1}, {"id": "b", "text": "y", "nguon_idx": 2}]
    assert kiem_ghep_cap(en, [dict(d) for d in en], "t") == []
    vi = [dict(en[0]), {"id": "c", "text": " ", "nguon_idx": 3}]
    loi = kiem_ghep_cap(en, vi, "t")
    assert any("id" in x for x in loi) and any("nguon_idx" in x for x in loi) and any("rỗng" in x for x in loi)


def test_kiem_trung_ba_cach():
    a = [{"text": "Write a poem about the sea.", "nguon_idx": 1}, {"text": "Explain photosynthesis", "nguon_idx": 2},
         {"text": "Name three fruits", "nguon_idx": 9}]
    b = [{"text": "write a POEM about the sea", "nguon_idx": 5},                      # trùng nguyên văn sau chuẩn hoá
         {"text": "Please explain photosynthesis briefly", "nguon_idx": 6},         # Jaccard 2/4 < 0,6
         {"text": "Name three red fruits", "nguon_idx": 7},                          # Jaccard 3/4 ≥ 0,6
         {"text": "Unrelated", "nguon_idx": 2}]                                      # trùng chỉ số nguồn
    r = kiem_trung("A", a, "B", b)
    assert r["trung_nguyen_van"] == 1 and r["trung_chi_so_nguon"] == 1 and not r["dat"]
    assert {(v["i"], v["j"]) for v in r["vi_du_gan_dung"]} == {(0, 0), (2, 2)}
    assert kiem_trung("A", a[1:2], "B", b[3:], cung_nguon=False)["dat"]


# ─────────────────────────────── kiểm khi nạp ───────────────────────────────
@pytest.mark.parametrize("ten,dac_ta", HAI)
def test_khoi_9_token_vi_tri_5_tren_512_cau(ten, dac_ta, tap, tok_qwen35, tok_gemma4):
    tok = tok_qwen35 if ten == "qwen35" else tok_gemma4
    k = kiem_khoi_tokenizer(tok, dac_ta, tap.harm_en + tap.harm_vi + tap.harmless_en + tap.harmless_vi)
    assert k["dat"], k
    assert k["K_theo_so_cau"] == {"9": 512} and k["vi_tri_lay_theo_so_cau"] == {"-5": 512}
    assert kiem_eos_tokenizer(tok, dac_ta)["dat"]


def test_khoi_lech_du_kien_bi_bat(tap, tok_qwen35):
    import copy
    tok = copy.deepcopy(tok_qwen35)
    tok.chat_template = tok.chat_template.replace("<think>\n\n</think>\n\n", "<think>\n</think>\n\n")
    k = kiem_khoi_tokenizer(tok, QWEN35_9B, tap.harm_en[:5])
    assert not k["dat"] and k["so_cau_loi"] == 5          # không kết thúc bằng khối thinking của đặc tả


def test_eos_sai_bi_bat(tok_gemma4):
    from dataclasses import replace
    sai = replace(GEMMA4_12B, eos_du_kien=((106, "<eos>"),))
    assert not kiem_eos_tokenizer(tok_gemma4, sai)["dat"]


@pytest.mark.parametrize("ten,dac_ta", HAI)
def test_nap_qua_auto_va_cau_truc(ten, dac_ta, mo_hinh):
    model, tok = mo_hinh[ten]
    ct = kiem_cau_truc(model, dac_ta)
    assert ct["dat"], ct
    if ten == "qwen35":
        assert ct["lop_mo_hinh"] == "Qwen3_5ForCausalLM" and ct["duong_dan_lop"] == ["model.layers"]
        assert ct["layer_scalar"] is None
    else:
        assert ct["lop_mo_hinh"] == "Gemma4UnifiedForConditionalGeneration"
        assert ct["duong_dan_lop"] == ["model.language_model.layers"]
        assert len(ct["layer_scalar"]) == 48 and abs(ct["layer_scalar"][1] - 1.1) < 1e-6


def test_phan_bo_thiet_bi_kiem_ca_buffer():
    m = tao_gemma4_vlm_nho()
    pb = phan_bo_thiet_bi(m, can_gpu=False)
    assert pb["mot_thiet_bi"] and pb["thiet_bi"] == ["cpu"]
    assert not phan_bo_thiet_bi(m, can_gpu=True)["mot_thiet_bi"]            # đòi GPU mà mô hình ở CPU
    m.model.language_model.layers[7].layer_scalar = torch.ones(1, device="meta")   # một BUFFER lệch thiết bị
    pb = phan_bo_thiet_bi(m, can_gpu=False)
    assert not pb["mot_thiet_bi"] and pb["so_tensor_lech"] == 1
    assert pb["tensor_lech_thiet_bi"] == ["buffer:model.language_model.layers.7.layer_scalar"]
    assert pb["hf_device_map_noi"] == []                 # bản đồ thiết bị rỗng không còn được coi là đạt


@pytest.mark.parametrize("ten,dac_ta", HAI)
def test_moi_luot_forward_qua_embedding_mot_lan(ten, dac_ta, mo_hinh, tap):
    model, tok = mo_hinh[ten]
    ge = kiem_goi_embedding(model, tok, dac_ta, tap.harmless_en[:3], 6, eos_tu_mo_hinh(model, dac_ta))
    assert ge["dat"], ge
    assert ge["bo_dem_vi_tri_cuoi"] == ge["L"] + ge["so_luot_forward"] - 1


def test_goi_embedding_hai_lan_bi_bat(mo_hinh, tap):
    model, tok = mo_hinh["qwen35"]
    emb = model.get_input_embeddings()
    h = tim_cac_lop(model)[0].register_forward_pre_hook(lambda m, a: emb(torch.zeros(1, 1, dtype=torch.long)))
    try:
        ge = kiem_goi_embedding(model, tok, QWEN35_9B, tap.harmless_en[:2], 4, eos_tu_mo_hinh(model, QWEN35_9B))
    finally:
        h.remove()
    assert not ge["dat"]


def test_dem_ro_thinking():
    r = dem_ro_thinking(["ok", "<think>x", "a</think>", "b"], QWEN35_9B)
    assert r["so_cau_ro"] == 2 and r["chi_so"] == [1, 2]
    assert dem_ro_thinking(["<|channel>thought"], GEMMA4_12B)["so_cau_ro"] == 1


# ─────────────────────────────── kích hoạt, position_ids ───────────────────────────────
@pytest.mark.parametrize("ten,dac_ta", HAI)
def test_kich_hoat_tung_cau_khop_trung_binh(ten, dac_ta, mo_hinh, tap):
    model, tok = mo_hinh[ten]
    cau = tap.harm_vi[:6]
    tc = kich_hoat_tung_cau(model, tok, dac_ta, cau, [-5, -1], co_lo=4)
    assert tc.shape == (6, 2, dac_ta.so_lop, 32) and tc.dtype == torch.float32
    tb = trung_binh_kich_hoat(model, tok, dac_ta, cau, [-5, -1], co_lo=4)
    assert torch.allclose(tc.to(torch.float64).mean(dim=0), tb.trung_binh, rtol=1e-10, atol=1e-12)
    # câu thứ 3 chạy riêng (không đệm) ≈ chạy trong lô có đệm
    rieng = kich_hoat_tung_cau(model, tok, dac_ta, [cau[2]], [-5], co_lo=1)
    assert torch.allclose(rieng[0, 0], tc[2, 0], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("ten,dac_ta", HAI)
def test_do_position_ids_chi_chan_doan(ten, dac_ta, mo_hinh, tap):
    model, tok = mo_hinh[ten]
    dp = do_position_ids(model, tok, dac_ta, tap.harm_en[:8], -5)
    assert dp["so_token_dem_moi_cau"]["lon_nhat"] > 0             # lô thật sự có đệm
    assert dp["cach_pipeline_dung"] == "B_truyen_position_ids_theo_mat_na"
    assert "quyet_dinh" not in dp and "giu_cach_hien_tai" not in dp  # không còn tự chọn (Q1)
    tb, tc = dp["trung_binh"], dp["theo_cau"]
    assert len(tb["lech_B_C_theo_lop"]) == dac_ta.so_lop
    assert tb["lech_B_C_lon_nhat"] < 1e-3 and tc["lech_B_C_lon_nhat"] < 1e-3           # float32: chỉ làm tròn
    assert tb["cos_B_C_nho_nhat"] > 0.999999 and not dp["canh_bao_dem"]


def _gia_lap(monkeypatch, bien_doi):
    """Thay kich_hoat_tung_cau trong nap_kiem: bien_doi(out, position_ids_theo_mat_na) áp cho các lượt theo lô."""
    import tn2.buoc3.nap_kiem as NK
    import tn2.kich_hoat as K
    goc = K.kich_hoat_tung_cau

    def sai(*args, position_ids_theo_mat_na=False, **kw):
        out = goc(*args, position_ids_theo_mat_na=position_ids_theo_mat_na, **kw)
        return out if kw.get("co_lo") == 1 else bien_doi(out, position_ids_theo_mat_na)
    monkeypatch.setattr(NK, "kich_hoat_tung_cau", sai)
    return NK


def test_position_ids_loi_chi_o_cach_A_duoc_ghi_khong_canh_bao(mo_hinh, tap, monkeypatch):
    """Lỗi chỉ ở đường không truyền position_ids (A, pipeline không dùng): ghi ở lệch A–C, không bật cảnh báo."""
    NK = _gia_lap(monkeypatch, lambda out, pos: out if pos else out * 1.01)
    model, tok = mo_hinh["gemma4"]
    dp = NK.do_position_ids(model, tok, GEMMA4_12B, tap.harm_en[:8], -5)
    assert abs(dp["trung_binh"]["lech_A_C_lon_nhat"] - 0.01) < 1e-3 and dp["trung_binh"]["lech_B_C_lon_nhat"] < 1e-3
    assert not dp["canh_bao_dem"]


def test_position_ids_dem_lam_lech_huong_bi_canh_bao(mo_hinh, tap, monkeypatch):
    """Đệm làm đổi HƯỚNG kích hoạt ở cả hai cách theo lô → bật canh_bao_dem."""
    g = torch.Generator().manual_seed(1)
    NK = _gia_lap(monkeypatch, lambda out, pos: out + 0.5 * out.abs().mean() * torch.randn(out.shape, generator=g))
    model, tok = mo_hinh["qwen35"]
    dp = NK.do_position_ids(model, tok, QWEN35_9B, tap.harm_en[:8], -5)
    assert dp["canh_bao_dem"] and dp["trung_binh"]["cos_B_C_nho_nhat"] < NK.NGUONG_COS_DEM


def test_position_ids_lech_ti_le_khong_bat_canh_bao_nhung_duoc_ghi(mo_hinh, tap, monkeypatch):
    """Giới hạn đã ghi (R2 B2): A = B = 1,01 × C không bật cờ (cosine không đổi khi nhân hệ số), nhưng lệch
    tương đối B–C được ghi lại để người đọc thấy."""
    NK = _gia_lap(monkeypatch, lambda out, pos: out * 1.01)
    model, tok = mo_hinh["qwen35"]
    dp = NK.do_position_ids(model, tok, QWEN35_9B, tap.harm_en[:8], -5)
    assert not dp["canh_bao_dem"]
    assert abs(dp["trung_binh"]["lech_B_C_lon_nhat"] - 0.01) < 1e-4
    assert abs(dp["theo_cau"]["lech_B_C_lon_nhat"] - 0.01) < 1e-4


# ─────────────────────────────── dựng hướng ───────────────────────────────
@pytest.mark.parametrize("ten,dac_ta", HAI)
def test_dung_huong(ten, dac_ta, mo_hinh, tap):
    model, tok = mo_hinh[ten]
    nho = _tap_nho(tap, 10)
    # So khớp tuyệt đối với hàm trung bình của bước 1 (không truyền position_ids) nên tắt position_ids ở đây.
    kq = dung_huong(model, tok, dac_ta, nho, co_lo=4, position_ids_theo_mat_na=False)
    t, tom, kh = kq["tensor"], kq["tom_tat"], kq["kich_hoat_vi_tri_lay"]
    assert t["vi_tri_lay"] == -5 and t["vi_tri_cuoi_that"] == -1
    L = dac_ta.so_lop
    # r = hiệu trung bình ở vị trí lấy, tính lại độc lập bằng trung_binh_kich_hoat
    a = trung_binh_kich_hoat(model, tok, dac_ta, nho.harm_en, [-5], co_lo=4).trung_binh[0]
    b = trung_binh_kich_hoat(model, tok, dac_ta, nho.harmless_en, [-5], co_lo=4).trung_binh[0]
    assert torch.allclose(t["r_en"], a - b, rtol=1e-9, atol=1e-12)
    assert torch.allclose(t["r_hat_en"].norm(dim=-1), torch.ones(L, dtype=torch.float64))
    assert torch.allclose(t["do_lon_tho_en"], (a - b).norm(dim=-1))
    # s: trung bình ‖h‖ trên 4 × 10 câu gộp
    x = torch.cat([kh[k].to(torch.float64).norm(dim=-1) for k in ("harm_en", "harm_vi", "harmless_en", "harmless_vi")])
    assert x.shape == (40, L) and torch.allclose(t["s"], x.mean(dim=0))
    assert torch.allclose(t["do_lon_tuong_doi_vi"], t["do_lon_tho_vi"] / t["s"])
    assert tom["so_cau_tinh_s"] == 40
    # chia đôi: hai nửa rời nhau, phủ đủ; cosine tính lại khớp
    n1, n2 = t["chia_doi"]["nua_1"], t["chia_doi"]["nua_2"]
    assert sorted(n1 + n2) == list(range(10)) and not set(n1) & set(n2)
    r1 = kh["harm_vi"][n1].to(torch.float64).mean(0) - kh["harmless_vi"][n1].to(torch.float64).mean(0)
    r2 = kh["harm_vi"][n2].to(torch.float64).mean(0) - kh["harmless_vi"][n2].to(torch.float64).mean(0)
    assert torch.allclose(t["cos_chia_doi_vi"], cosine(r1, r2))
    # cosine hai vị trí và en–vi có đủ L giá trị, trong [−1, 1]
    for k in ("cos_hai_vi_tri_en", "cos_hai_vi_tri_vi", "cos_en_vi_uoc_luong_diem"):
        assert t[k].shape == (L,) and bool((t[k].abs() <= 1 + 1e-12).all())
    assert not torch.allclose(t["r_hat_en"], t["r_hat_en_cuoi_that"])      # hai vị trí khác nhau thật
    assert kh["harm_en"].shape == (10, L, 32)
    json.dumps(tom)                                                          # tóm tắt ghi JSON được
    # Mặc định (pipeline chính): truyền position_ids; float32 → hướng gần như trùng cách không truyền.
    kq2 = dung_huong(model, tok, dac_ta, nho, co_lo=4)
    assert kq2["tom_tat"]["position_ids_theo_mat_na"] is True
    assert bool((cosine(kq2["tensor"]["r_hat_en"], t["r_hat_en"]) > 0.9999).all())


def test_dung_huong_huong_chan_doan_bang_0_khong_chan_huong_chinh(tap, tok_qwen35, monkeypatch):
    """R2 B4: hướng ở token cuối thật (chỉ chẩn đoán) bằng 0 ở mọi lớp → ghi NaN/null, hướng chính vẫn dựng."""
    import tn2.buoc3.dung_huong as DH
    nho = _tap_nho(tap, 6)
    co_hai = set(nho.harm_en + nho.harm_vi)
    g = torch.Generator().manual_seed(0)

    def gia(model, tok, dac_ta, cau, vi_tri, co_lo=32, position_ids_theo_mat_na=False):
        out = torch.zeros(len(cau), 2, 4, 8)
        out[:, 0] = torch.randn(len(cau), 4, 8, generator=g) + (1.0 if cau[0] in co_hai else 0.0)
        out[:, 1] = 1.0                                  # vị trí −1 giống hệt nhau ở mọi câu → rB = 0
        return out
    monkeypatch.setattr(DH, "kich_hoat_tung_cau", gia)
    kq = DH.dung_huong(None, tok_qwen35, QWEN35_9B, nho, co_lo=6)
    t, tom = kq["tensor"], kq["tom_tat"]
    assert bool(torch.isfinite(t["r_hat_en"]).all()) and bool(torch.isnan(t["r_hat_en_cuoi_that"]).all())
    assert tom["en"]["lop_huong_cuoi_that_do_lon_0"] == [0, 1, 2, 3]
    assert tom["en"]["cos_hai_vi_tri_nho_nhat"] is None
    with pytest.raises(ValueError):                      # hướng CHÍNH bằng 0 thì vẫn dừng
        monkeypatch.setattr(DH, "kich_hoat_tung_cau", lambda *a, **k: torch.ones(6, 2, 4, 8))
        DH.dung_huong(None, tok_qwen35, QWEN35_9B, nho, co_lo=6)


# ─────────────────────────────── script chạy thông ───────────────────────────────
@pytest.mark.parametrize("ten,dac_ta", HAI)
def test_cac_script_buoc3_chay_thong(ten, dac_ta, thu_muc_qwen35, thu_muc_gemma4, tmp_path, monkeypatch):
    import buoc3_do_toc_do as TD
    import buoc3_dung_huong as DH
    import buoc3_kiem_hook as KH
    import buoc3_kiem_truoc as KT
    import buoc3_nap_kiem as NK

    # Luôn chạy trên CPU, kể cả trên máy thuê có GPU (các ngưỡng của test đã kiểm trên CPU). Phép kiểm thiết bị
    # phải nhận can_gpu=False TƯỜNG MINH (không dựa vào torch.cuda.is_available, R2 B1): ghi lại để kiểm.
    from tn2.mo_hinh import phan_bo_thiet_bi as pb_goc
    can_gpu_da_truyen = []

    def pb_ghi(model, can_gpu=None):
        can_gpu_da_truyen.append(can_gpu)
        return pb_goc(model, can_gpu=can_gpu)
    for m in (TD, DH, KH, NK):
        monkeypatch.setattr(m, "device_map_mot_gpu", lambda: None)
        monkeypatch.setattr(m, "phan_bo_thiet_bi", pb_ghi)

    d = str(thu_muc_qwen35 if ten == "qwen35" else thu_muc_gemma4)
    ra = str(tmp_path / "ra")
    chung = ["--mo_hinh_dac_ta", dac_ta.ten, "--mo_hinh", d, "--ra", ra]
    assert KT.main(chung + ["--ten_luot", "kt"]) == 0
    k = json.loads((tmp_path / "ra" / dac_ta.ten / "kt" / "kiem_truoc.json").read_text(encoding="utf-8"))
    assert k["khoi_sau_cau_hoi"]["dat"] and k["cau_hinh_sinh"]["la_greedy_dung"] and k["cau_truc_config"]["dat"]
    # Mô hình nhỏ khởi tạo ngẫu nhiên không bao giờ sinh token kết thúc → mã 1, chỉ do thành phần eos.
    assert NK.main(chung + ["--ten_luot", "nk", "--max_new_tokens", "6", "--so_cau_sinh_thu", "2"]) == 1
    nk = json.loads((tmp_path / "ra" / dac_ta.ten / "nk" / "nap_kiem.json").read_text(encoding="utf-8"))
    tp = nk["dat_thanh_phan"]
    assert tp.pop("co_cau_cham_eos_chinh") is False and all(tp.values()), nk["dat_thanh_phan"]
    assert nk["sinh_thu"]["eos_chinh"] == dac_ta.eos_du_kien[0][0]
    rc_dh = DH.main(chung + ["--ten_luot", "dh", "--gioi_han", "6", "--co_lo", "4"])
    huong = tmp_path / "ra" / dac_ta.ten / "dh" / "huong.pt"
    h = torch.load(huong)
    assert rc_dh == (4 if h["canh_bao_dem"] else 0)              # cảnh báo đệm → mã 4 "cần xem" (Q2)
    assert h["trang_thai"] == ("can_xem" if h["canh_bao_dem"] else "dat")
    assert h["dac_ta"] == dac_ta.ten and h["r_hat_en"].shape == (dac_ta.so_lop, 32)
    assert h["chinh_thuc"] is False and h["position_ids_theo_mat_na"] is True   # --gioi_han → lượt thử; luôn B
    kh = torch.load(tmp_path / "ra" / dac_ta.ten / "dh" / "kich_hoat_vi_tri_lay.pt")
    assert kh["harm_vi"].shape == (6, dac_ta.so_lop, 32) and len(kh["id_harm"]) == 6
    rc = KH.main(chung + ["--ten_luot", "kh", "--huong", str(huong), "--so_cau", "2", "--so_cau_nhieu", "32",
                          "--max_new_tokens", "6"])
    assert rc == 0
    kl = json.loads((tmp_path / "ra" / dac_ta.ten / "kh" / "ket_luan.json").read_text(encoding="utf-8"))
    assert kl["trang_thai"] == "dat"
    meta = json.loads((tmp_path / "ra" / dac_ta.ten / "kh" / "meta.json").read_text(encoding="utf-8"))
    assert meta["chinh_thuc"] is False and meta["huong_chinh_thuc"] is False      # hướng thử → lượt thử
    for kt in ("bf16", "fp32"):                                                   # chẩn đoán Q5 có mặt và khớp
        kq = json.loads((tmp_path / "ra" / dac_ta.ten / "kh" / f"ket_qua_{kt}.json").read_text(encoding="utf-8"))
        assert kq["tong_ket"]["khong_hook_lap_lai_khop"] is True and len(kq["khong_hook_lap_lai"]) == 2
    bk = kq["backend_lop_tuyen_tinh"]["dung_trong_luot"]                          # float32: bản tham chiếu (Q4)
    assert (len(bk) == 4) == (ten == "qwen35") and all(v["la_ban_tham_chieu"] for v in bk.values())
    assert meta["tham_so"]["lop"] == list(dac_ta.lop_kiem_hook)
    assert meta["tham_so"]["lop_full_attention"] == [dac_ta.lop_kiem_hook[1]]       # lớp giữa là full attention
    assert TD.main(chung + ["--ten_luot", "td", "--huong", str(huong), "--co_lo", "2,4", "--so_cau_moi_file", "1",
                            "--max_new_tokens", "4"]) == 0
    sai = KH.main(chung + ["--ten_luot", "kh2", "--huong", str(huong), "--mo_hinh_dac_ta",
                           ("gemma4_12b" if ten == "qwen35" else "qwen35_9b")])
    assert sai == 1                                              # file hướng của mô hình khác → dừng
    assert can_gpu_da_truyen and all(x is False for x in can_gpu_da_truyen), can_gpu_da_truyen


def test_so_cau_chay_chinh_theo_so_lop():
    """R2 B3: chạy chính quét mọi lớp → số câu theo số lớp; tổng khớp 158 100 của Nhap."""
    from tn2.buoc3.toc_do import so_cau_theo_mo_hinh, uoc_gio_gpu
    q, g = so_cau_theo_mo_hinh(QWEN35_9B), so_cau_theo_mo_hinh(GEMMA4_12B)
    assert q["chay_chinh"] == 63_440 and g["chay_chinh"] == 94_672 and q["chay_chinh"] + g["chay_chinh"] == 158_112
    assert "pilot" in q and "kiem_alpha" not in q and "kiem_alpha" in g and "pilot" not in g
    u = uoc_gio_gpu(1000.0, GEMMA4_12B)
    assert abs(u["gio_gpu"]["chay_chinh"] - 94.672) < 1e-9 and abs(u["gio_gpu"]["tong"] - sum(g.values()) / 1000) < 1e-9


def test_ban_tham_chieu_pytorch_thay_kernel_gia(mo_hinh, tap, monkeypatch):
    """Q4: khi transformers đang dùng một kernel ngoài (giả lập như fla), ngữ cảnh ban_tham_chieu_pytorch chuyển
    bốn hàm của lớp DeltaNet / conv về bản PyTorch tham chiếu, cho đúng kết quả gốc, và trả lại khi ra."""
    import functools
    import inspect

    from tn2.mo_hinh import TEN_HAM_TUYEN_TINH, backend_lop_tuyen_tinh, ban_tham_chieu_pytorch
    model, tok = mo_hinh["qwen35"]
    enc_ids = tok(["xin chao the gioi"], return_tensors="pt")["input_ids"]
    goc = model(enc_ids).logits
    sinh_goc = model.generate(enc_ids, max_new_tokens=3, do_sample=False)
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    def kernel_gia(*a, **k):
        raise RuntimeError("kernel giả bị gọi")

    def boc_nhu_transformers(torch_fn):
        implementation = kernel_gia

        @functools.wraps(torch_fn)
        def wrapped(*a, **k):
            return implementation(*a, **k)
        return wrapped
    for ten in TEN_HAM_TUYEN_TINH:
        monkeypatch.setattr(M, ten, boc_nhu_transformers(inspect.unwrap(getattr(M, ten))))
    bk = backend_lop_tuyen_tinh(model)
    assert len(bk) == 4 and not any(v["la_ban_tham_chieu"] for v in bk.values())
    with pytest.raises(RuntimeError, match="kernel giả"):
        model(enc_ids)
    with ban_tham_chieu_pytorch(model) as b:
        assert all(v["la_ban_tham_chieu"] for v in b.sau.values())
        assert torch.equal(model(enc_ids).logits, goc)
        # đường decode có cache (causal_conv1d_update, torch_recurrent_gated_delta_rule) cũng không gọi kernel giả
        sinh = model.generate(enc_ids, max_new_tokens=3, do_sample=False)
    assert torch.equal(sinh, sinh_goc)
    assert not any(v["la_ban_tham_chieu"] for v in backend_lop_tuyen_tinh(model).values())   # đã trả lại
    assert backend_lop_tuyen_tinh(mo_hinh["gemma4"][0]) == {}


# ─────────────────────────────── sửa sau bước 1 ───────────────────────────────
def test_ghi_json_nan_thanh_null(tmp_path):
    NG.ghi_json(tmp_path / "x.json", {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2.5}})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": None, "b": [1.0, None], "c": {"d": 2.5}}


def test_chan_doan_cosine_bo_o_do_lon_0(tmp_path):
    from tn2.buoc1.tai_lap_arditi import chan_doan
    g = torch.Generator().manual_seed(0)
    ta = torch.randn(5, 4, 8, generator=g, dtype=torch.float64)
    tg = ta + 0.01 * torch.randn(5, 4, 8, generator=g, dtype=torch.float64)
    ta[:, 0] = 0.0                                  # lớp 0: hướng bằng 0 như đầu vào khối 0 của Llama-3
    torch.save(tg, tmp_path / "mean_diffs.pt")
    ra = chan_doan(ta, [], {"vi_tri": 1, "lop": 2}, tmp_path)["cosine_voi_mean_diffs_tac_gia"]
    assert ra["so_o_bo_do_lon_0"] == 5 and ra["so_o_hop_le"] == 15
    assert ra["nho_nhat"] > 0.99 and ra["theo_vi_tri_lop"][0][0] is None
    NG.ghi_json(tmp_path / "c.json", ra)            # không còn NaN trong JSON
