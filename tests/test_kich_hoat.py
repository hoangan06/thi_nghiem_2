"""Trung bình kích hoạt theo lô đệm trái = trung bình tính từng câu không đệm."""
import torch

from conftest import doc_cau
from tn2.kich_hoat import trung_binh_kich_hoat
from tn2.khuon import ma_hoa
from tn2.mo_hinh import LLAMA3_8B, tim_cac_lop


def _tham_chieu(m, tok, cau, lop, vi_tri, vao):
    tong = None
    do_lon = None
    for q in cau:
        ids = torch.tensor([ma_hoa(tok, [q], LLAMA3_8B).ids_rieng[0]])
        giu = {}
        if vao:
            h = tim_cac_lop(m)[lop].register_forward_pre_hook(
                lambda mod, a, kw: giu.__setitem__("h", a[0] if a else kw["hidden_states"]), with_kwargs=True)
        else:
            h = tim_cac_lop(m)[lop].register_forward_hook(
                lambda mod, a, o: giu.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        m(input_ids=ids)
        h.remove()
        x = giu["h"][0, vi_tri].to(torch.float64)
        tong = x if tong is None else tong + x
        do_lon = x.norm(dim=-1) if do_lon is None else do_lon + x.norm(dim=-1)
    return tong / len(cau), do_lon / len(cau)


def test_trung_binh_khop_tung_cau(tok_llama, llama_f32):
    cau = doc_cau("harmful_train", 7)
    for diem, vao in (("ra_khoi", False), ("vao_khoi", True)):
        kq = trung_binh_kich_hoat(llama_f32, tok_llama, LLAMA3_8B, cau, [-5, -1], [3, 20], diem, co_lo=3)
        for j, l in enumerate([3, 20]):
            tb, dl = _tham_chieu(llama_f32, tok_llama, cau, l, [-5, -1], vao)
            assert torch.allclose(kq.trung_binh[:, j], tb, atol=1e-4, rtol=1e-4)
            assert torch.allclose(kq.trung_binh_do_lon[:, j], dl, atol=1e-4, rtol=1e-4)
    assert kq.so_cau == 7


def test_vao_khoi_bang_ra_khoi_lop_truoc(tok_llama, llama_f32):
    cau = doc_cau("harmless_train", 5)
    a = trung_binh_kich_hoat(llama_f32, tok_llama, LLAMA3_8B, cau, [-1], [9], "ra_khoi")
    b = trung_binh_kich_hoat(llama_f32, tok_llama, LLAMA3_8B, cau, [-1], [10], "vao_khoi")
    assert torch.equal(a.trung_binh, b.trung_binh)
