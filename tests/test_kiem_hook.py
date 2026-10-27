"""Phần 2 (kiểm đường can thiệp) trên mô hình nhỏ: đạt ở float32, kiểm âm bắt được lỗi cài vào."""
import json

import torch

from conftest import doc_cau, tao_llama_nho
from tn2.kiem_hook import DauVaoKiemHook, chay_kiem_hook, ket_luan_hai_kieu_so, logit_tham_chieu
from tn2.khuon import ma_hoa
from tn2.mo_hinh import LLAMA3_8B

EOS = (128001, 128009)


def _dv(lop=(8, 16, 24), so=4, so_nhieu=12, tok_moi=10, seed=1):
    g = torch.Generator().manual_seed(seed)
    rh = {l: torch.nn.functional.normalize(torch.randn(32, generator=g), dim=0) for l in lop}
    return DauVaoKiemHook(doc_cau("harmless_val", so), doc_cau("harmful_val", so), list(lop), rh,
                          {l: 3.0 for l in lop}, EOS, max_new_tokens=tok_moi,
                          cau_hoi_nhieu_a=doc_cau("harmless_train", so_nhieu),
                          cau_hoi_nhieu_b=doc_cau("harmful_train", so_nhieu), co_lo=so)


def test_tham_chieu_khong_vector_bang_forward_thuong(tok_llama, llama_f32):
    ids = ma_hoa(tok_llama, doc_cau("harmless_val", 1), LLAMA3_8B).ids_rieng[0]
    tiep = [11, 12, 13]
    ref = logit_tham_chieu(llama_f32, ids, tiep, 5, None, 5, None)
    out = llama_f32(input_ids=torch.tensor([ids + tiep[:-1]])).logits[0, len(ids) - 1:]
    assert torch.equal(ref, out.float())


def test_float32_dat_va_kiem_am_bat_loi(tok_llama, llama_f32):
    kq = chay_kiem_hook(llama_f32, tok_llama, LLAMA3_8B, _dv())
    t = kq["tong_ket"]
    assert t["ii_dat"]
    assert t["i_dat"] and t["iii_dat"], t
    assert t["kiem_am_dat"], t["kiem_am"]
    assert t["so_luot_alpha_doi_token"] > 0          # can thiệp thật sự làm đổi chuỗi sinh
    assert len(kq["nhieu"]) == 2 * 3                  # 12 câu mỗi kiểu, lô 4
    assert t["trang_thai_i"] == "dat" and t["trang_thai_iii"] == "dat"
    assert set(t["chan_doan_vi_tri"]) == {"i", "iii"} and t["chan_doan_vi_tri"]["i"]["so_vi_tri_vuot_nguong"] == 0
    assert all(len(h) == g for r in kq["i"] for h, g in zip(r["lech_theo_vi_tri"], r["so_token_sinh"]))
    assert set(t["kiem_am"]) == {"pa1_lech_vung", "pa1_bo_cong_khi_sinh", "pa3_lech_vung", "pa3_bo_cong_khi_sinh"}
    assert not t["dung_san"]


def test_bat_duoc_hook_sai_that(tok_llama, llama_f32, monkeypatch):
    """Một lỗi thật trong hook chính (bỏ token sinh thứ hai trở đi) phải làm (i) trượt."""
    import tn2.can_thiep as C
    goc = C.mat_na_vung

    def sai(vung, vi_tri, L, loi=None):
        m = goc(vung, vi_tri, L, loi)
        return m & (vi_tri < L + 1)
    monkeypatch.setattr(C, "mat_na_vung", sai)
    kq = chay_kiem_hook(llama_f32, tok_llama, LLAMA3_8B, _dv(lop=(16,), so_nhieu=4))
    assert not kq["tong_ket"]["i_dat"]


def test_script_hai_kieu_so(tok_llama, tmp_path, monkeypatch):
    import buoc1_kiem_hook as S

    monkeypatch.setattr(S, "nap_mo_hinh",
                        lambda *a, dtype=torch.bfloat16, **k: (tao_llama_nho(dtype=dtype), tok_llama))
    # Mô hình nhỏ ở CPU: phép kiểm thiết bị nhận can_gpu=False tường minh, để test không phụ thuộc máy có CUDA
    # (đối chiếu code bước 3, R1 P1). Script bước 1 khi chạy thật vẫn đòi GPU (không đổi).
    from tn2.mo_hinh import phan_bo_thiet_bi as pb_goc
    monkeypatch.setattr(S, "phan_bo_thiet_bi", lambda m, can_gpu=None: pb_goc(m, can_gpu=False))
    rc = S.main(["--ra", str(tmp_path), "--ten_luot", "thu", "--so_cau", "2", "--so_cau_nhieu", "4",
                 "--max_new_tokens", "6", "--gioi_han_dung_huong", "4"])
    t = tmp_path / "thu"
    for f in ("meta.json", "kiem_khuon.json", "huong_kiem.pt", "ket_qua_bf16.json", "ket_qua_fp32.json",
              "ket_luan.json"):
        assert (t / f).exists(), f
    kl = json.loads((t / "ket_luan.json").read_text())
    assert kl["ii_bf16"] and kl["ii_fp32"]
    assert rc == {"dat": 0, "can_xem": 4}.get(kl["trang_thai"], 1)
    assert kl["trang_thai"] in ("dat", "can_xem", "khong_dat")
    assert json.loads((t / "ket_qua_fp32.json").read_text())["cau_hinh"]["dtype"] == "torch.float32"
    assert json.loads((t / "ket_qua_bf16.json").read_text())["cau_hinh"]["dtype"] == "torch.bfloat16"
    hk = torch.load(t / "huong_kiem.pt")
    assert hk["lop"] == [8, 16, 24] and hk["vi_tri_lay"] == -1


def test_ket_luan_hai_kieu_so():
    def tk(ii, i, iii, am, tt_i=None, tt_iii=None, nghi=False):
        cd = {k: {"nghi_loi_co_che": nghi} for k in ("i", "iii")}
        return {"tong_ket": {"ii_dat": ii, "i_dat": i, "iii_dat": iii, "kiem_am_dat": am, "nguong": 1.0,
                             "trang_thai_i": tt_i or ("dat" if i else "khong_dat"),
                             "trang_thai_iii": tt_iii or ("dat" if iii else "khong_dat"), "chan_doan_vi_tri": cd}}
    kl = ket_luan_hai_kieu_so(tk(True, False, False, False), tk(True, True, True, True))
    assert kl["dat"] and kl["trang_thai"] == "dat"                       # bf16 (i)/(iii) trượt không sao (Q1)
    assert ket_luan_hai_kieu_so(tk(False, True, True, True), tk(True, True, True, True))["trang_thai"] == "khong_dat"
    assert ket_luan_hai_kieu_so(tk(True, True, True, True), tk(True, True, True, False))["trang_thai"] == "khong_dat"
    kl = ket_luan_hai_kieu_so(tk(True, True, True, True), tk(True, False, True, True, tt_i="can_xem", nghi=True))
    assert not kl["dat"] and kl["trang_thai"] == "can_xem" and kl["nghi_loi_co_che"]   # cần xem KHÔNG phải đạt
    kl = ket_luan_hai_kieu_so(tk(True, True, True, True), tk(True, False, False, True, "can_xem", "khong_dat"))
    assert kl["trang_thai"] == "khong_dat"
    kl = ket_luan_hai_kieu_so(tk(True, True, True, True), tk(False, False, True, True, tt_i="can_xem"))
    assert kl["trang_thai"] == "khong_dat"                                # (ii) float32 trượt


def test_chan_doan_vi_tri():
    from tn2.kiem_hook import _chan_doan_vi_tri
    nhieu = [{"i_theo_vi_tri": [[1e-5, 2e-5, 4e-5], [1e-5, 3e-5]]}]
    # vượt ngưỡng 5e-5 nhưng ≤ 10 × bao nhiễu tại vị trí đó → không nghi lỗi
    rai = [{"kieu": "a", "lop": 8, "phuong_an": 1, "lech_theo_vi_tri": [[1e-5, 2e-5, 6e-5]]}]
    cd = _chan_doan_vi_tri(rai, nhieu, "i", 5e-5)
    assert cd["so_vi_tri_vuot_nguong"] == 1 and not cd["nghi_loi_co_che"]
    # lệch dồn ở vị trí 0, nơi nhiễu rất nhỏ → nghi lỗi cơ chế
    don = [{"kieu": "a", "lop": 8, "phuong_an": 3, "lech_theo_vi_tri": [[2e-4, 1e-5, 1e-5]]}]
    assert _chan_doan_vi_tri(don, nhieu, "i", 5e-5)["nghi_loi_co_che"]
    # số câu đo nhiễu tại từng vị trí
    assert cd["so_cau_nhieu_theo_vi_tri"] == [2, 2, 1]
    # vị trí 3 không có câu đo nhiễu nào: không ngoại suy, đánh dấu thiếu dữ liệu
    xa = [{"kieu": "b", "lop": 8, "phuong_an": 1, "lech_theo_vi_tri": [[1e-5, 1e-5, 1e-5, 9e-5]]}]
    cx = _chan_doan_vi_tri(xa, nhieu, "i", 5e-5)
    v = cx["vi_tri_vuot"][0]
    assert v["thieu_du_lieu_nhieu"] and v["ti_so"] is None and v["bao_nhieu_tai_vi_tri"] is None
    assert cx["so_vi_tri_thieu_du_lieu_nhieu"] == 1 and not cx["nghi_loi_co_che"]


def test_dung_som_o_token_ket_thuc(tok_llama, llama_f32):
    """Mô hình thật thường dừng sớm ở <|eot_id|>; giả lập bằng cách lấy một token hay gặp làm token kết thúc."""
    import collections

    from tn2.sinh import sinh_tham_lam
    enc = ma_hoa(tok_llama, doc_cau("harmless_val", 4) + doc_cau("harmful_val", 4), LLAMA3_8B)
    r = sinh_tham_lam(llama_f32, tok_llama, enc, 10, (128001,))
    eos_gia = collections.Counter(r.token_day_du[:, 1:6].flatten().tolist()).most_common(1)[0][0]
    dv = _dv()
    dv.eos_ids = (eos_gia,)
    kq = chay_kiem_hook(llama_f32, tok_llama, LLAMA3_8B, dv)
    so = [g for x in kq["i"] for g in x["so_token_sinh"]]
    assert min(so) < 10 and max(so) == 10              # có câu dừng sớm, có câu chạy hết
    t = kq["tong_ket"]
    assert t["ii_dat"] and t["i_dat"] and t["iii_dat"] and t["kiem_am_dat"], t


def test_luat_ba_muc_trong_chay_kiem_hook(tok_llama, llama_f32, monkeypatch):
    """Ép ngưỡng thấp (0,5 × nhiễu) để (i) trượt ngưỡng nhưng vẫn trong 10 × nhiễu → "can_xem", không phải đạt."""
    import tn2.kiem_hook as KH
    monkeypatch.setattr(KH, "HE_SO_NGUONG", 0.5)
    t = chay_kiem_hook(llama_f32, tok_llama, LLAMA3_8B, _dv(lop=(16,), so_nhieu=8))["tong_ket"]
    assert t["lech_lon_nhat_i"] > t["nguong"] and not t["i_dat"]
    assert t["trang_thai_i"] == "can_xem"
    cd = t["chan_doan_vi_tri"]["i"]
    assert cd["so_vi_tri_vuot_nguong"] > 0 and not cd["nghi_loi_co_che"]     # chỉ là nhiễu: không nghi lỗi
    kl = ket_luan_hai_kieu_so({"tong_ket": t}, {"tong_ket": t})
    assert kl["trang_thai"] in ("can_xem", "khong_dat") and not kl["dat"]
