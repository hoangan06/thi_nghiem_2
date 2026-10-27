"""Sinh greedy: bỏ qua generation_config gây nhiễu, xử lý token kết thúc, hai cách giải mã."""
import torch

from conftest import doc_cau
from tn2.khuon import ma_hoa
from tn2.mo_hinh import LLAMA3_8B
from tn2.sinh import sinh_tham_lam, sinh_theo_lo

EOS = (128001, 128009)


def test_greedy_bang_argmax_tay(tok_llama, llama_f32):
    m = llama_f32
    assert m.generation_config.do_sample and m.generation_config.repetition_penalty == 1.5   # cố ý gây nhiễu
    cau = doc_cau("harmless_val", 3)
    enc = ma_hoa(tok_llama, cau, LLAMA3_8B)
    r = sinh_tham_lam(m, tok_llama, enc, 30, EOS)
    for i in range(3):
        ids = list(enc.ids_rieng[i])
        ra = []
        for _ in range(30):
            t = int(m(input_ids=torch.tensor([ids])).logits[0, -1].argmax())
            ra.append(t)
            ids.append(t)
            if t in EOS:
                break
        assert r.token_hop_le(i) == ra


def test_khong_can_thiep_thi_khong_gan_hook_nao(tok_llama, llama_f32, monkeypatch):
    """Phép (ii) và kiểm "không gắn hook" ở pilot: không có can thiệp thì không đăng ký cả bộ đếm vị trí."""
    import tn2.sinh as S

    def cam(*a, **k):
        raise AssertionError("gan_can_thiep không được gọi khi không có can thiệp")
    monkeypatch.setattr(S, "gan_can_thiep", cam)
    enc = ma_hoa(tok_llama, doc_cau("harmless_val", 2), LLAMA3_8B)
    sinh_tham_lam(llama_f32, tok_llama, enc, 4, EOS)
    sinh_tham_lam(llama_f32, tok_llama, enc, 4, EOS, [])


def test_dung_o_token_ket_thuc(tok_llama, llama_f32):
    m = llama_f32
    enc = ma_hoa(tok_llama, doc_cau("harmless_val", 2), LLAMA3_8B)
    r0 = sinh_tham_lam(m, tok_llama, enc, 10, EOS)
    t0 = int(r0.token_day_du[0, 2])          # dùng token thứ 3 của câu 0 làm "token kết thúc"
    r = sinh_tham_lam(m, tok_llama, enc, 10, (t0,), lay_logit=True)
    assert r.cham_eos[0] and r.so_hop_le[0] == 3
    assert r.token_hop_le(0) == r0.token_hop_le(0)[:3]
    assert r.logits.shape[1] == r.token_day_du.shape[1]
    # "den_eos": văn bản bỏ token kết thúc
    assert r.van_ban[0] == tok_llama.decode(r0.token_day_du[0, :2], skip_special_tokens=False)


def test_giai_ma_arditi_bo_token_dac_biet(tok_llama, llama_f32):
    m = llama_f32
    kq = sinh_theo_lo(m, tok_llama, LLAMA3_8B, doc_cau("harmless_val", 3), 2, 6, (128001,), giai_ma="arditi")
    assert len(kq) == 3
    for r in kq:
        assert "<|" not in r["van_ban"] and r["van_ban"] == r["van_ban"].strip()


def test_theo_lo_giu_thu_tu(tok_llama, llama_f32):
    cau = doc_cau("harmless_val", 5)
    a = sinh_theo_lo(llama_f32, tok_llama, LLAMA3_8B, cau, 2, 5, EOS)
    assert [r["cau_hoi"] for r in a] == cau
