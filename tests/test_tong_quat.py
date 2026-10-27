"""Cơ chế hook trên hai kiến trúc của mô hình chính (Qwen3.5: attention tuyến tính + đầy đủ;
Gemma 4: cửa sổ trượt) — bản nhỏ khởi tạo ngẫu nhiên, dùng tokenizer đồ chơi kiểu Qwen."""
import pytest
import torch

from conftest import doc_cau, tao_gemma4_nho, tao_qwen35_nho
from tn2.can_thiep import can_thiep_chinh, chieu_bo_moi_lop, gan_can_thiep
from tn2.kiem_hook import DauVaoKiemHook, chay_kiem_hook
from tn2.khuon import ma_hoa
from tn2.mo_hinh import QWEN35_9B, tim_cac_lop
from tn2.sinh import sinh_tham_lam


def _mo_hinh(ten, tok):
    if ten == "qwen35":
        return tao_qwen35_nho(len(tok)), [1, 3, 5]
    return tao_gemma4_nho(len(tok)), [1, 3, 5]


@pytest.mark.parametrize("ten", ["qwen35", "gemma4"])
def test_hook_tong_quat(ten, tok_qwen):
    m, lop = _mo_hinh(ten, tok_qwen)
    eos = (tok_qwen.convert_tokens_to_ids("<|im_end|>"),)
    cau = [q for q in doc_cau("harmless_val") if q.split()[0] in tok_qwen.get_vocab()][:3]
    enc = ma_hoa(tok_qwen, cau, QWEN35_9B)
    assert enc.vi_tri_lay == -5
    # (ii) α = 0 khớp tuyệt đối
    a = sinh_tham_lam(m, tok_qwen, enc, 8, eos, None, lay_logit=True)
    ct0 = can_thiep_chinh(lop[1], torch.ones(32) / 32 ** 0.5, 2.0, 0.0, "a", 1, enc.so_token_khoi)
    b = sinh_tham_lam(m, tok_qwen, enc, 8, eos, [ct0], lay_logit=True)
    assert torch.equal(a.token_day_du, b.token_day_du) and torch.equal(a.logits, b.logits)
    # chiếu bỏ kiểu Arditi tìm được attention và MLP ở mọi lớp
    with gan_can_thiep(m, chieu_bo_moi_lop(len(tim_cac_lop(m)), torch.ones(32))):
        m(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    # (i), (iii), kiểm âm ở float32
    g = torch.Generator().manual_seed(2)
    rh = {l: torch.nn.functional.normalize(torch.randn(32, generator=g), dim=0) for l in lop}
    # nhiễu đo trên tập lớn hơn tập kiểm, như script bước 1 (xem docstring kiem_hook)
    nhieu_a = [q for q in doc_cau("harmless_train") if q.split()[0] in tok_qwen.get_vocab()][:30]
    nhieu_b = [q for q in doc_cau("harmful_train") if q.split()[0] in tok_qwen.get_vocab()][:30]
    dv = DauVaoKiemHook(cau, cau[::-1], lop, rh, {l: 3.0 for l in lop}, eos, max_new_tokens=8, co_lo=3,
                        cau_hoi_nhieu_a=nhieu_a, cau_hoi_nhieu_b=nhieu_b)
    t = chay_kiem_hook(m, tok_qwen, QWEN35_9B, dv)["tong_ket"]
    assert t["ii_dat"] and t["i_dat"] and t["iii_dat"] and t["kiem_am_dat"], t
