"""Kiểu attention theo đặc tả và phép quét attention theo độ dài (sửa sau kết quả bước 3; Ân duyệt Q-E1–Q-E3 ngày
29/10/2026). Mô hình nhỏ chạy trên CPU nên không tái tạo được lỗi kernel EFFICIENT_ATTENTION của GPU; lỗi được giả lập
bằng cách cộng thêm vào logit ở các độ dài ≡ 1 (mod 32), đúng dạng lỗi đã đo."""
import pytest
import torch

import tn2.kiem_attention as KA
import tn2.mo_hinh as MH
from conftest import tao_gemma4_nho, tao_qwen35_nho
from tn2 import nguon_goc as NG
from tn2.mo_hinh import GEMMA4_12B, QWEN35_9B, kieu_attention, nap_mo_hinh

M = 40                                     # quét tới 40: có đúng một độ dài xấu (33)


def _X(n=M + 64, vocab=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, vocab, (n,), generator=g).tolist()


@pytest.fixture(scope="module")
def gemma_eager():
    m = tao_gemma4_nho(64)
    m.set_attn_implementation("eager")
    return m


def test_dac_ta_kieu_attention():
    assert GEMMA4_12B.attn_implementation == "eager" and QWEN35_9B.attn_implementation == "sdpa"


@pytest.mark.parametrize("dac_ta", [QWEN35_9B, GEMMA4_12B])
def test_nap_mo_hinh_dung_kieu_attention(dac_ta, thu_muc_qwen35, thu_muc_gemma4, monkeypatch):
    d = thu_muc_qwen35 if dac_ta is QWEN35_9B else thu_muc_gemma4
    model, tok = nap_mo_hinh(dac_ta, str(d), dtype=torch.float32, device_map=None)
    assert kieu_attention(model) == dac_ta.attn_implementation
    assert NG.thong_tin_mo_hinh(model, tok, dac_ta)["attn_implementation"] == dac_ta.attn_implementation
    # transformers lặng lẽ đổi kiểu (vd. không hỗ trợ) → dừng ngay khi nạp
    monkeypatch.setattr(MH, "kieu_attention", lambda m: "flash_attention_2")
    with pytest.raises(ValueError):
        nap_mo_hinh(dac_ta, str(d), dtype=torch.float32, device_map=None)


def test_tham_chieu_math_tra_lai_kieu_cu(gemma_eager, monkeypatch):
    # Đếm số lần gọi sdpa: ở float32 trên CPU, eager và MATH của Gemma nhỏ cho kết quả trùng khít, nên chỉ so logit thì
    # không biết tham chiếu có thật sự đổi cách tính hay không (rà soát độc lập 29/10/2026).
    goc = torch.nn.functional.scaled_dot_product_attention
    dem = []

    def dem_sdpa(*a, **k):
        dem.append(1)
        return goc(*a, **k)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", dem_sdpa)
    ids = torch.tensor([_X(20)])
    KA._logit_cuoi(gemma_eager, ids, None)
    assert dem == []
    with KA.tham_chieu_math(gemma_eager):
        assert kieu_attention(gemma_eager) == "sdpa"
        KA._logit_cuoi(gemma_eager, ids, None)
    assert len(dem) == gemma_eager.config.num_hidden_layers
    assert kieu_attention(gemma_eager) == "eager"
    with pytest.raises(RuntimeError):
        with KA.tham_chieu_math(gemma_eager):
            raise RuntimeError("x")
    assert kieu_attention(gemma_eager) == "eager"


@pytest.mark.parametrize("tao", [tao_gemma4_nho, tao_qwen35_nho])
def test_quet_attention_float32_khong_loi_thi_dat(tao):
    m = tao(64)
    if tao is tao_gemma4_nho:
        m.set_attn_implementation("eager")
    kq = KA.quet_attention(m, _X(), 0, fp32=True, m_toi_da=M)
    assert kq["trang_thai"] == "dat" and kq["lech_lon_nhat"] <= 1e-3 and kq["do_dai_sai"] == []
    assert [r["m"] for r in kq["don"]] == list(range(2, M + 1)) and [r["do_dai_dem"] for r in kq["lo"]] == [33]
    assert "lo_thuc" not in kq and "cache" not in kq and kq["do_dai_dem_lo_sai"] == []
    # dạng đầy đủ (lô, lô thật, KV cache) trên mô hình float32: các dạng đều khớp
    lo_thuc = {"a": [_X(20, seed=2), _X(33, seed=3)], "b": [_X(7, seed=4)]}
    kq = KA.quet_attention(m, _X(), 0, fp32=False, m_toi_da=M, lo_thuc=lo_thuc)
    assert kq["trang_thai"] == "dat" and kq["lech_lon_nhat"] <= 1e-3
    assert [r["do_dai_dem"] for r in kq["lo"]] == [33] and [r["do_dai_dem"] for r in kq["lo_thuc"]] == [33, 7]
    assert kq["cache"][0]["do_dai_dem"] == 33 and len(kq["cache"][0]["lech_theo_buoc"]) == 1 + KA.SO_BUOC_CACHE
    assert "lô thật" in KA.tom_tat(kq)
    # quét tới 70: thêm lô cache 61 (bước sinh đi qua độ dài KV 65 ≡ 1 mod 32)
    kq = KA.quet_attention(m, _X(70 + 64), 0, fp32=False, m_toi_da=70)
    assert [r["do_dai_dem"] for r in kq["cache"]] == [33, 61] and [r["do_dai_dem"] for r in kq["lo"]] == [33, 65]
    assert kq["trang_thai"] == "dat"


def _gia_loi(monkeypatch, cong: float):
    """Giả lỗi kernel: khi chạy bằng kiểu đang dùng (không phải tham chiếu sdpa), độ dài ≡ 1 (mod 32), ≥ 33 → cộng
    `cong` vào logit vị trí cuối của mọi hàng (cả lượt nạp có cache)."""
    goc_cuoi, goc_cache = KA._logit_cuoi, KA._logit_cache

    def xau(model, n):
        return kieu_attention(model) != "sdpa" and n >= 33 and n % 32 == 1

    def cuoi(model, ids, mask):
        out = goc_cuoi(model, ids, mask)
        if xau(model, ids.shape[1]):
            out = out.clone()
            out[:, 0] += cong
        return out

    def cache(model, ids, mask, tiep):
        out = goc_cache(model, ids, mask, tiep)
        if xau(model, ids.shape[1]):
            out = out.clone()
            out[:, 0, 0] += cong
        return out
    monkeypatch.setattr(KA, "_logit_cuoi", cuoi)
    monkeypatch.setattr(KA, "_logit_cache", cache)


@pytest.mark.parametrize("cong,bf16,fp32", [(20.0, "khong_dat", "khong_dat"), (3.0, "can_kiem", "khong_dat"),
                                            (1.0, "dat", "khong_dat"), (0.2, "dat", "dat")])
def test_quet_attention_bat_loi_gia_lap(gemma_eager, monkeypatch, cong, bf16, fp32):
    """Luật viết trước: bf16 > 5 → khong_dat, (2, 5] → can_kiem, ≤ 2 → dat; float32 > 0,5 → khong_dat."""
    _gia_loi(monkeypatch, cong)
    k32 = KA.quet_attention(gemma_eager, _X(), 0, fp32=True, m_toi_da=M)
    assert k32["trang_thai"] == fp32 and k32["do_dai_sai"] == ([33] if fp32 == "khong_dat" else [])
    assert k32["do_dai_dem_lo_sai"] == k32["do_dai_sai"]
    kb = KA.quet_attention(gemma_eager, _X(), 0, fp32=False, m_toi_da=M, lo_thuc={"t": [_X(33, seed=5)]})
    assert kb["trang_thai"] == bf16 and kb["do_dai_sai"] == ([33] if bf16 == "khong_dat" else [])
    assert kb["lo"][0]["lech"] == pytest.approx(cong, rel=1e-3, abs=1e-3)
    assert kb["lech_lon_nhat_m_xau"] == pytest.approx(cong, rel=1e-3, abs=1e-3)      # dồn ở m ≡ 1 (mod 32)
    assert kb["lech_lon_nhat_m_khac"] < 1e-3 and kb["so_o_khong_huu_han"] == 0
    assert kb["lech_lon_nhat_lo_cache"] == pytest.approx(cong, rel=1e-3, abs=1e-3)
    assert "m ≡ 1 (mod 32)" in KA.tom_tat(kb)
    assert kb["lo_thuc"][0]["lech"] == pytest.approx(cong, rel=1e-3, abs=1e-3)
    assert kb["cache"][0]["lech_theo_buoc"][0] == pytest.approx(cong, rel=1e-3, abs=1e-3)
    assert kieu_attention(gemma_eager) == "eager"


@pytest.mark.parametrize("so_hang,fp32", [(8, True), (8, False), (2, False)])
def test_quet_attention_loi_chi_o_mot_dang_lo(gemma_eager, monkeypatch, so_hang, fp32):
    """Lỗi chỉ ở dạng lô đệm (8 hàng) hoặc chỉ ở lô thật (ở đây 2 hàng), dạng một chuỗi sạch: vẫn không đạt."""
    goc = KA._logit_cuoi

    def cuoi(model, ids, mask):
        out = goc(model, ids, mask)
        if kieu_attention(model) != "sdpa" and ids.shape[0] == so_hang:
            out = out.clone()
            out[:, 0] += 9.0
        return out
    monkeypatch.setattr(KA, "_logit_cuoi", cuoi)
    kq = KA.quet_attention(gemma_eager, _X(), 0, fp32=fp32, m_toi_da=M,
                           lo_thuc=None if fp32 else {"t": [_X(33, seed=5), _X(20, seed=6)]})
    assert kq["trang_thai"] == "khong_dat" and kq["do_dai_sai"] == []
    assert kq["do_dai_dem_lo_sai"] == ([33] if so_hang == 8 else [])
    if so_hang == 2:
        assert kq["lo_thuc"][0]["lech"] == pytest.approx(9.0, rel=1e-3)


def test_quet_attention_chi_loi_o_lo_hoac_cache_van_bi_bat(gemma_eager, monkeypatch):
    """Lỗi chỉ xuất hiện ở lượt nạp có KV cache (dạng một chuỗi sạch) vẫn đẩy trạng thái lên."""
    goc = KA._logit_cache

    def cache(model, ids, mask, tiep):
        out = goc(model, ids, mask, tiep)
        if kieu_attention(model) != "sdpa":
            out = out.clone()
            out[:, -1, 0] += 9.0
        return out
    monkeypatch.setattr(KA, "_logit_cache", cache)
    kb = KA.quet_attention(gemma_eager, _X(), 0, fp32=False, m_toi_da=M)
    assert kb["trang_thai"] == "khong_dat" and kb["do_dai_sai"] == []        # dạng một chuỗi sạch
    assert kb["cache"][0]["lech_theo_buoc"][-1] == pytest.approx(9.0, rel=1e-3)


def test_quet_attention_chuoi_qua_ngan(gemma_eager):
    with pytest.raises(ValueError):
        KA.quet_attention(gemma_eager, _X(M), 0, fp32=True, m_toi_da=M)


def test_chuoi_quet(tok_qwen35):
    from tn2.du_lieu import nap_tap_dung
    tap = nap_tap_dung()
    X = KA.chuoi_quet(tok_qwen35, QWEN35_9B, tap.harmless_en, 384)
    assert len(X) >= 384
    with pytest.raises(ValueError):
        KA.chuoi_quet(tok_qwen35, QWEN35_9B, tap.harmless_en[:1], 10 ** 6)


def test_so_sanh_huong_gemma(tmp_path, monkeypatch, capsys):
    """Script mô tả (Q-E4): tự tìm hướng eager mới nhất, so với hướng cũ; năm lớp hiệu chỉnh trùng đặc tả."""
    import so_sanh_huong_gemma as SS
    assert SS.LOP_HIEU_CHINH == GEMMA4_12B.lop_hieu_chinh
    g = torch.Generator().manual_seed(0)

    def huong(attn, nhieu=0.0):
        t = {"commit_hf": "c", "attn_implementation": attn, "s": torch.rand(48, generator=g) + 1}
        for ng in ("en", "vi"):
            r = torch.randn(48, 16, generator=g)
            t[f"r_hat_{ng}"] = r / r.norm(dim=-1, keepdim=True)
            t[f"do_lon_tuong_doi_{ng}"] = torch.full((48,), 0.1)
        return t
    goc = tmp_path / "results" / "buoc3" / "gemma4_12b"
    cu = huong("sdpa")
    moi = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cu.items()} | {"attn_implementation": "eager"}
    moi["r_hat_en"] = -moi["r_hat_en"]                                   # cố ý khác ở en
    for ten, t in (("dung_huong_20261028_085421", cu), ("dung_huong_20261101_000000", moi),
                   ("dung_huong_20261102_000000", None)):                # lượt dở: không có huong.pt
        (goc / ten).mkdir(parents=True)
        if t is not None:
            torch.save(t, goc / ten / "huong.pt")
    monkeypatch.chdir(tmp_path)
    ra = SS.main([])
    assert ra["moi"]["file"].endswith("dung_huong_20261101_000000/huong.pt") and ra["moi"]["attn"] == "eager"
    assert ra["theo_ngon_ngu"]["vi"]["cos_nho_nhat"] == pytest.approx(1.0)
    assert ra["theo_ngon_ngu"]["en"]["cos_nho_nhat"] == pytest.approx(-1.0)
    assert ra["du_doan_dung"] == {"cos_hieu_chinh_toi_thieu_0_999": False, "cos_huong_nho_toi_thieu_0_99": True,
                                  "chenh_s_duoi_1e_2": True}
    assert len(list(goc.glob("so_sanh_huong_*/ket_qua.json"))) == 1


@pytest.mark.parametrize("gia_tri,dang", [(float("nan"), "don"), (float("inf"), "don"), (float("nan"), "cache"),
                                          (float("nan"), "tham_chieu")])
def test_quet_attention_o_khong_huu_han_la_khong_dat(gemma_eager, monkeypatch, gia_tri, dang):
    """R2 B1 (rà soát 29/10/2026): NaN / Inf ở bất kỳ dạng nào (kể cả ở phía tham chiếu, kể cả đường cache) →
    khong_dat ở cả bf16 và float32. Trước khi sửa, max() của Python bỏ qua NaN và phép quét ra 'dat'."""
    goc_cuoi, goc_cache = KA._logit_cuoi, KA._logit_cache

    def phia(model):
        return (kieu_attention(model) == "sdpa") if dang == "tham_chieu" else (kieu_attention(model) != "sdpa")

    def cuoi(model, ids, mask):
        out = goc_cuoi(model, ids, mask)
        if dang in ("don", "tham_chieu") and phia(model) and ids.shape == (1, 33):
            out = out.clone()
            out[0, 1] = gia_tri
        return out

    def cache(model, ids, mask, tiep):
        out = goc_cache(model, ids, mask, tiep)
        if dang == "cache" and phia(model):
            out = out.clone()
            out[0, 3, 1] = gia_tri
        return out
    monkeypatch.setattr(KA, "_logit_cuoi", cuoi)
    monkeypatch.setattr(KA, "_logit_cache", cache)
    for fp32 in ((False,) if dang == "cache" else (False, True)):
        kq = KA.quet_attention(gemma_eager, _X(), 0, fp32=fp32, m_toi_da=M)
        assert kq["trang_thai"] == "khong_dat" and kq["so_o_khong_huu_han"] == 1, (fp32, kq["so_o_khong_huu_han"])
        assert kq["lech_lon_nhat"] == float("inf")
        if dang != "cache":
            assert kq["do_dai_sai"] == [33] and kq["lech_lon_nhat_m_xau"] == float("inf")
    from tn2 import nguon_goc as NG
    assert NG._bo_nan({"x": kq["lech_lon_nhat"]}) == {"x": None}                 # JSON: null, không phải lỗi


@pytest.mark.parametrize("tao", [tao_gemma4_nho, tao_qwen35_nho])
def test_cac_dang_quet_nhat_quan_voi_nhau(tao):
    """R1 E (rà soát 29/10/2026): kiểm phần đệm, vị trí và KV cache của chính kiem_attention trên mô hình float32 —
    hàng 0 của lô đệm (= X[:L]) và từng hàng đệm khớp chuỗi riêng không đệm; 9 logit của đường cache khớp logit
    không cache ở độ dài L, L+1, …, L+8. Lỗi giả lập ở các test khác cài ở _logit_cuoi / _logit_cache nên không bảo
    vệ được phần này."""
    m = tao(64)
    if tao is tao_gemma4_nho:
        m.set_attn_implementation("eager")
    X = _X(140)
    for L in (33, 61):
        hang = KA._hang_lo(X, L)
        ids, mask = KA._dem_trai(hang, 0, torch.device("cpu"))
        lo = KA._logit_cuoi(m, ids, mask)
        for i, h in enumerate(hang):
            rieng = KA._logit_cuoi(m, torch.tensor([h]), None)[0]
            assert torch.allclose(lo[i], rieng, atol=1e-4, rtol=1e-4), (L, i, float((lo[i] - rieng).abs().max()))
        tiep = X[L: L + KA.SO_BUOC_CACHE]
        c = KA._logit_cache(m, ids, mask, tiep)
        for i, h in enumerate(hang):                                  # mọi hàng, kể cả hàng có đệm trái
            for j in range(1 + KA.SO_BUOC_CACHE):
                khong_cache = KA._logit_cuoi(m, torch.tensor([list(h) + tiep[:j]]), None)[0]
                assert torch.allclose(c[i, j], khong_cache, atol=1e-4, rtol=1e-4), (L, i, j)


def test_vi_tri_dem_bang_0_nhu_generate():
    """R2 (rà soát 29/10/2026): vị trí của token đệm là 0 như generate của transformers 5.17 và kich_hoat."""
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    assert KA._vi_tri(mask).tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]


@pytest.mark.parametrize("dac_ta,them,sha,loi", [
    (GEMMA4_12B, {}, "abc", True),                                   # hướng Gemma 28/10: không ghi trường
    (GEMMA4_12B, {"attn_implementation": "sdpa"}, "abc", True),
    (GEMMA4_12B, {"attn_implementation": "eager"}, "abc", False),
    (QWEN35_9B, {}, KA.HUONG_SDPA_CU["qwen35_9b"], False),           # hướng Qwen 28/10: nhận theo SHA
    (QWEN35_9B, {}, "abc", True),
    (QWEN35_9B, {"attn_implementation": "sdpa"}, "abc", False)])
def test_kieu_attention_cua_huong(dac_ta, them, sha, loi):
    h = {"dac_ta": dac_ta.ten} | them
    assert (KA.loi_kieu_attention_huong(h, sha, dac_ta) is not None) == loi


def test_kiem_hook_dung_truoc_khi_nap_voi_huong_sai_kieu(thu_muc_gemma4, tmp_path, monkeypatch, capsys):
    """Hướng Gemma không ghi kiểu attention (như lượt sdpa ngày 28/10) → buoc3_kiem_hook dừng mã 1 trước khi nạp."""
    import buoc3_kiem_hook as KH
    torch.save({"dac_ta": "gemma4_12b", "chinh_thuc": True, "commit_hf": "c"}, tmp_path / "huong.pt")
    monkeypatch.setattr(KH, "nap_mo_hinh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("không được nạp")))
    assert KH.main(["--mo_hinh_dac_ta", "gemma4_12b", "--mo_hinh", str(thu_muc_gemma4), "--huong",
                    str(tmp_path / "huong.pt"), "--ra", str(tmp_path / "ra")]) == 1
    assert "attention" in capsys.readouterr().out and not (tmp_path / "ra").exists()
