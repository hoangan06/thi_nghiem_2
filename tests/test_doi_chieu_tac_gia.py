"""Hai phép can thiệp của phần 1 cho kết quả trùng từng bit với hook của tác giả (tests/tac_gia/hook_utils.py),
và sinh dưới chiếu bỏ cho cùng chuỗi token như generate theo cách của tác giả."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from conftest import doc_cau, tao_llama_nho
from tn2.can_thiep import chieu_bo_moi_lop, cong_vao_khoi_moi_vi_tri, gan_can_thiep
from tn2.khuon import ma_hoa
from tn2.mo_hinh import LLAMA3_8B, tim_cac_lop
from tn2.sinh import cau_hinh_greedy, sinh_tham_lam


def _nap_hook_tac_gia():
    if "jaxtyping" not in sys.modules:          # file của tác giả chỉ dùng jaxtyping để chú thích kiểu
        jt = types.ModuleType("jaxtyping")

        class _A:
            def __getitem__(self, k):
                return object
        jt.Float = _A()
        jt.Int = _A()
        sys.modules["jaxtyping"] = jt
    f = Path(__file__).parent / "tac_gia" / "hook_utils.py"
    sp = importlib.util.spec_from_file_location("hook_utils_tac_gia", f)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_trung_tung_bit_voi_hook_tac_gia(tok_llama, dtype):
    H = _nap_hook_tac_gia()
    m = tao_llama_nho(dtype=dtype)
    lop = tim_cac_lop(m)
    L = len(lop)
    enc = ma_hoa(tok_llama, doc_cau("harmful_val", 6), LLAMA3_8B)
    d = torch.randn(32, dtype=torch.float64, generator=torch.Generator().manual_seed(5)) * 3

    def chay(pre, fw):
        with H.add_hooks(pre, fw):
            return m(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, -1].float()

    # chiếu bỏ ở mọi lớp (get_all_direction_ablation_hooks)
    pre = [(lop[l], H.get_direction_ablation_input_pre_hook(direction=d)) for l in range(L)]
    fw = [(lop[l].self_attn, H.get_direction_ablation_output_hook(direction=d)) for l in range(L)]
    fw += [(lop[l].mlp, H.get_direction_ablation_output_hook(direction=d)) for l in range(L)]
    a = chay(pre, fw)
    with gan_can_thiep(m, chieu_bo_moi_lop(L, d)):
        b = m(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, -1].float()
    assert torch.equal(a, b)
    # cộng vào đầu vào lớp 12 (get_activation_addition_input_pre_hook, coeff = 1)
    a2 = chay([(lop[12], H.get_activation_addition_input_pre_hook(vector=d, coeff=torch.tensor(1.0)))], [])
    with gan_can_thiep(m, cong_vao_khoi_moi_vi_tri(12, d, 1.0)):
        b2 = m(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, -1].float()
    assert torch.equal(a2, b2)
    # sinh dưới chiếu bỏ
    gc = cau_hinh_greedy(10, (128001,), tok_llama.pad_token_id)
    with H.add_hooks(pre, fw):
        ga = m.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, generation_config=gc)
    gb = sinh_tham_lam(m, tok_llama, enc, 10, (128001,), chieu_bo_moi_lop(L, d), giai_ma="arditi")
    assert torch.equal(ga[:, enc.do_dai_dem:], gb.token_day_du)
