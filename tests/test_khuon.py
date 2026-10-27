"""Template, đếm khối sau câu hỏi, vị trí lấy kích hoạt, đệm trái."""
import pytest
import torch

from conftest import doc_cau
from tn2.buoc1.tai_lap_arditi import TEMPLATE_ARDITI_LLAMA3, kiem_khuon
from tn2.khuon import chia_lo, ma_hoa, phan_tich_mot_cau
from tn2.mo_hinh import LLAMA3_8B, QWEN35_9B


def test_llama_khoi_5_token_vi_tri_cuoi(tok_llama):
    vb, ids, k, vt = phan_tich_mot_cau(tok_llama, "How do I bake bread?", LLAMA3_8B)
    assert k == 5 and vt == -1
    assert tok_llama.convert_ids_to_tokens(ids[-5:]) == [
        "<|eot_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>", "\n\n"]
    assert ids[0] == 128000                       # BOS đúng một lần
    assert ids.count(128000) == 1


def test_llama_trung_cach_tac_gia_ma_hoa(tok_llama):
    cau = doc_cau("harmful_val", 10) + doc_cau("harmless_val", 10)
    kk = kiem_khuon(tok_llama, LLAMA3_8B, cau)
    assert kk["dat"] and kk["so_token_khoi"] == 5
    q = cau[0]
    assert ma_hoa(tok_llama, [q], LLAMA3_8B).ids_rieng[0] == list(
        tok_llama(TEMPLATE_ARDITI_LLAMA3.format(instruction=q))["input_ids"])


def test_dem_trai(tok_llama):
    cau = ["Hi", "Write a long poem about the sea and the sky"]
    enc = ma_hoa(tok_llama, cau, LLAMA3_8B)
    L = enc.do_dai_dem
    for i, ids in enumerate(enc.ids_rieng):
        n = len(ids)
        assert enc.input_ids[i, L - n:].tolist() == ids
        assert enc.attention_mask[i, L - n:].all() and not enc.attention_mask[i, : L - n].any()
        assert (enc.input_ids[i, : L - n] == tok_llama.pad_token_id).all()
    # K vị trí cuối của chuỗi đệm là khối sau câu hỏi ở mọi câu
    assert all(enc.input_ids[i, -5:].tolist() == enc.ids_rieng[i][-5:] for i in range(2))


def test_qwen_bo_khoi_thinking(tok_qwen):
    vb, ids, k, vt = phan_tich_mot_cau(tok_qwen, "How do I bake bread?", QWEN35_9B)
    assert vb.endswith("<think>\n\n</think>\n\n")
    toks = tok_qwen.convert_ids_to_tokens(ids)
    # vị trí lấy = token cuối trước khối thinking ("\n" sau "assistant")
    assert toks[vt] == "\n" and toks[vt - 1] == "assistant" and toks[vt + 1] == "<think>"
    assert toks[-k] == "<|im_end|>"


def test_dac_ta_sai_K_bao_loi(tok_llama):
    from dataclasses import replace
    with pytest.raises(ValueError):
        ma_hoa(tok_llama, ["Hi"], replace(LLAMA3_8B, so_token_khoi_du_kien=4))


def test_chia_lo():
    assert chia_lo(10, 4) == [(0, 4), (4, 8), (8, 10)]
    with pytest.raises(ValueError):
        chia_lo(3, 0)
