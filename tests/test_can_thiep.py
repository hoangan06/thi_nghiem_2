"""Hook can thiệp: vùng, khớp tuyệt đối ở α = 0, cộng đúng chỗ, chiếu bỏ, gỡ hook."""
import pytest
import torch

from conftest import doc_cau, tao_llama_nho
from tn2.can_thiep import (CanThiep, Vung, can_thiep_chinh, chieu_bo_moi_lop, gan_can_thiep, mat_na_vung,
                           vector_can_thiep, vung_phuong_an)
from tn2.khuon import ma_hoa
from tn2.mo_hinh import LLAMA3_8B, tim_cac_lop
from tn2.sinh import sinh_tham_lam

EOS = (128001, 128009)


def _vec(d=32, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(d, generator=g), dim=0)


def test_mat_na_vung():
    vt = torch.arange(0, 14)          # L = 10, K = 5: khối 5..9, sinh 10..13
    pa1 = mat_na_vung(vung_phuong_an(1, 5), vt, 10)
    assert pa1.tolist() == [False] * 5 + [True] * 9
    pa3 = mat_na_vung(vung_phuong_an(3, 5), vt, 10)
    assert pa3.tolist() == [False] * 5 + [True] * 6 + [False] * 3
    lech = mat_na_vung(vung_phuong_an(1, 5), vt, 10, "lech_vung")
    assert lech.tolist() == [False] * 4 + [True] * 10
    bo = mat_na_vung(vung_phuong_an(1, 5), vt, 10, "bo_cong_khi_sinh")
    assert bo.tolist() == [False] * 5 + [True] * 5 + [False] * 4
    assert mat_na_vung(Vung("moi_vi_tri"), vt, None).all()
    with pytest.raises(ValueError):
        vung_phuong_an(2, 5)


def test_vector_dau_va_do_lon():
    r = _vec()
    va = vector_can_thiep(r, 3.0, 0.4, "a")
    vb = vector_can_thiep(r, 3.0, 0.4, "b")
    assert torch.allclose(va, 1.2 * r) and torch.allclose(vb, -1.2 * r)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pa", [1, 3])
def test_alpha_0_khop_tuyet_doi(tok_llama, dtype, pa):
    m = tao_llama_nho(dtype=dtype)
    enc = ma_hoa(tok_llama, doc_cau("harmless_val", 4), LLAMA3_8B)
    a = sinh_tham_lam(m, tok_llama, enc, 12, EOS, None, lay_logit=True)
    ct = can_thiep_chinh(16, _vec(), 5.0, 0.0, "a", pa, enc.so_token_khoi)
    b = sinh_tham_lam(m, tok_llama, enc, 12, EOS, [ct], lay_logit=True)
    assert torch.equal(a.token_day_du, b.token_day_du) and torch.equal(a.logits, b.logits)


def _bat_dau_vao(model, lop):
    """Bắt đầu vào của khối `lop` (pre-hook) trong lượt forward."""
    giu = {}

    def hook(module, args, kwargs):
        giu["h"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()
    return giu, tim_cac_lop(model)[lop].register_forward_pre_hook(hook, with_kwargs=True)


def test_cong_ra_khoi_dung_vi_tri(tok_llama, llama_f32):
    m = llama_f32
    enc = ma_hoa(tok_llama, doc_cau("harmful_val", 3), LLAMA3_8B)
    l = 10
    giu, h = _bat_dau_vao(m, l + 1)
    m(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    goc = giu["h"]
    v = 2.0 * _vec()
    ct = CanThiep((l,), "ra_khoi", "cong", v, vung_phuong_an(1, enc.so_token_khoi))
    with gan_can_thiep(m, [ct], do_dai_loi_nhac=enc.do_dai_dem):
        m(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    h.remove()
    moi = giu["h"]
    K = enc.so_token_khoi
    assert torch.equal(moi[:, :-K], goc[:, :-K])                 # ngoài vùng: giữ nguyên tuyệt đối
    assert torch.equal(moi[:, -K:], goc[:, -K:] + v)              # trong vùng: cộng đúng v


def test_vao_khoi_tuong_duong_ra_khoi_lop_truoc(tok_llama, llama_f32):
    m = llama_f32
    enc = ma_hoa(tok_llama, doc_cau("harmful_val", 3), LLAMA3_8B)
    v = 3.0 * _vec()
    vung = vung_phuong_an(1, enc.so_token_khoi)
    a = sinh_tham_lam(m, tok_llama, enc, 8, EOS, [CanThiep((7,), "ra_khoi", "cong", v, vung)], lay_logit=True)
    b = sinh_tham_lam(m, tok_llama, enc, 8, EOS, [CanThiep((8,), "vao_khoi", "cong", v, vung)], lay_logit=True)
    assert torch.equal(a.logits, b.logits)


def test_chieu_bo_moi_lop_truc_giao(tok_llama, llama_f32):
    m = llama_f32
    enc = ma_hoa(tok_llama, doc_cau("harmful_val", 3), LLAMA3_8B)
    d = 4.0 * _vec(seed=3)
    dh = d / d.norm()
    cts = chieu_bo_moi_lop(32, d)
    with gan_can_thiep(m, cts) as _:
        thu = []
        hs = []
        for l in (0, 15, 31):
            def hook(module, args, kwargs, l=l):
                thu.append((args[0] if args else kwargs["hidden_states"]).detach())
            hs.append(tim_cac_lop(m)[l].register_forward_pre_hook(hook, with_kwargs=True))
        out = m(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        for h in hs:
            h.remove()
    for h in thu:
        assert float((h @ dh).abs().max()) < 1e-4 * float(h.norm(dim=-1).max())
    # ảnh hưởng thật lên đầu ra
    goc = m(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    assert not torch.equal(goc.logits, out.logits)


def test_go_hook_ke_ca_khi_loi(tok_llama, llama_f32):
    m = llama_f32
    def dem():
        return sum(len(x._forward_hooks) + len(x._forward_pre_hooks) for x in m.modules())
    truoc = dem()
    ct = can_thiep_chinh(5, _vec(), 1.0, 0.4, "a", 1, 5)
    with pytest.raises(RuntimeError):
        with gan_can_thiep(m, chieu_bo_moi_lop(32, _vec()) + [ct], do_dai_loi_nhac=10):
            assert dem() > truoc
            raise RuntimeError("thử")
    assert dem() == truoc


def test_lop_ngoai_khoang(llama_f32):
    ct = can_thiep_chinh(32, _vec(), 1.0, 0.4, "a", 1, 5)
    with pytest.raises(ValueError):
        with gan_can_thiep(llama_f32, [ct], do_dai_loi_nhac=10):
            pass
