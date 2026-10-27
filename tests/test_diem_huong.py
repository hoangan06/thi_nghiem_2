"""Điểm từ chối, KL, so khớp chuỗi; hiệu trung bình, chuẩn hoá, chia đôi."""
import math

import pytest
import torch

from tn2.diem import CUM_TU_CHOI_ARDITI, diem_tu_choi, kl_cuoi, la_tu_choi_chuoi, ti_le_tu_choi_chuoi
from tn2.huong import chia_doi, chuan_hoa, cosine, hieu_trung_binh


def test_diem_tu_choi_cong_thuc():
    lg = torch.tensor([[0.0, math.log(3.0), 0.0, 0.0]], dtype=torch.float64)     # p = [1,3,1,1]/6
    pr = 3 / 6
    d = diem_tu_choi(lg, [1])
    assert d.dtype == torch.float64
    assert abs(float(d) - (math.log(pr + 1e-8) - math.log(1 - pr + 1e-8))) < 1e-12
    d2 = diem_tu_choi(lg, [1, 2])
    assert abs(float(d2) - (math.log(4 / 6 + 1e-8) - math.log(2 / 6 + 1e-8))) < 1e-12


def test_kl():
    a = torch.randn(3, 50)
    assert torch.allclose(kl_cuoi(a, a), torch.zeros(3, dtype=torch.float64), atol=1e-12)
    b = torch.randn(3, 50)
    p, q = a.double().softmax(-1), b.double().softmax(-1)
    assert torch.allclose(kl_cuoi(a, b), (p * ((p + 1e-6).log() - (q + 1e-6).log())).sum(-1))


def test_so_khop_chuoi():
    assert len(CUM_TU_CHOI_ARDITI) == 12
    assert la_tu_choi_chuoi("well, i CANNOT help")
    assert la_tu_choi_chuoi("As an AI, ...")
    assert not la_tu_choi_chuoi("Sure, here is how")
    assert ti_le_tu_choi_chuoi(["I'm sorry", "Sure", "I am unable to", "ok"]) == 0.5
    with pytest.raises(ValueError):
        ti_le_tu_choi_chuoi([])


def test_huong():
    a, b = torch.randn(4, 8), torch.randn(4, 8)
    r = hieu_trung_binh(a, b)
    assert r.dtype == torch.float64 and torch.allclose(r, (a - b).double())
    rh, n = chuan_hoa(r)
    assert torch.allclose(rh.norm(dim=-1), torch.ones(4, dtype=torch.float64))
    assert torch.allclose(n, r.norm(dim=-1))
    assert torch.allclose(cosine(r, 2 * r), torch.ones(4, dtype=torch.float64))
    with pytest.raises(ValueError):
        chuan_hoa(torch.zeros(2, 3))


def test_chia_doi():
    a, b = chia_doi(11, 7)
    assert len(a) == 6 and len(b) == 5 and sorted(a + b) == list(range(11))
    assert (a, b) == chia_doi(11, 7) and (a, b) != chia_doi(11, 8)
