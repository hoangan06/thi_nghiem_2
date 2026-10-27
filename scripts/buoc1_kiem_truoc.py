"""Bước 1, kiểm trước khi chạy (vài giây, không nạp trọng số, không cần GPU).

Chạy:
    python scripts/buoc1_kiem_truoc.py --mo_hinh meta-llama/Meta-Llama-3-8B-Instruct

Kiểm trên tokenizer và file cấu hình THẬT (test của kho chỉ dùng tokenizer đồ chơi):
1. Khuôn chat của ta trùng cách tác giả mã hoá ở mọi câu của 6 tập Arditi (520 câu khác nhau);
2. khối sau câu hỏi đúng 5 token, vị trí lấy kích hoạt là −1; in các token này;
3. eos, pad của tokenizer; eos dùng cho pipeline chính và cho phần tái lập;
4. generation_config của mô hình, và cấu hình sinh HIỆU LỰC sau khi transformers trộn cấu hình greedy của ta
   với generation_config của mô hình (dựng mô hình trên thiết bị "meta", không tốn bộ nhớ).
Ghi tất cả vào results/buoc1/<ten_luot>/kiem_truoc.json. Mã thoát 0 nếu (1), (2) đạt và cấu hình hiệu lực là greedy.
"""
from __future__ import annotations

import argparse
import sys

from _chung import GOC, NhatKy, ten_luot_mac_dinh

import torch

from tn2 import nguon_goc as NG
from tn2.buoc1.tai_lap_arditi import kiem_khuon, nap_du_lieu
from tn2.khuon import ma_hoa
from tn2.mo_hinh import EOS_ARDITI_LLAMA3, LLAMA3_8B
from tn2.sinh import cau_hinh_greedy


def cau_hinh_hieu_luc(nguon: str, revision: str | None, eos: tuple[int, ...], pad_id: int) -> dict:
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    cfg = AutoConfig.from_pretrained(nguon, revision=revision)
    with torch.device("meta"):
        m = AutoModelForCausalLM.from_config(cfg)
    try:
        m.generation_config = GenerationConfig.from_pretrained(nguon, revision=revision)
    except OSError:
        pass
    gc_mo_hinh = m.generation_config.to_dict()
    gc, _ = m._prepare_generation_config(cau_hinh_greedy(512, eos, pad_id))
    hl = gc.to_dict()
    greedy = (hl.get("do_sample") is False and hl.get("num_beams") == 1 and hl.get("repetition_penalty") == 1.0
              and not hl.get("no_repeat_ngram_size") and not hl.get("min_new_tokens")
              and not hl.get("bad_words_ids") and not hl.get("suppress_tokens")
              and not hl.get("begin_suppress_tokens") and hl.get("forced_bos_token_id") is None
              and hl.get("forced_eos_token_id") is None and not hl.get("stop_strings") and not hl.get("max_time")
              and not hl.get("sequence_bias") and not hl.get("min_length")
              and not hl.get("exponential_decay_length_penalty") and not hl.get("watermarking_config") and list(hl.get("eos_token_id") or []) == list(eos))
    return {"generation_config_mo_hinh": gc_mo_hinh, "hieu_luc": hl, "la_greedy_dung": bool(greedy),
            "so_lop_config": getattr(cfg, "num_hidden_layers", None), "commit_hf": getattr(cfg, "_commit_hash", None)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh", default=LLAMA3_8B.hf_id)
    p.add_argument("--revision", default=None)
    p.add_argument("--ra", default=str(GOC / "results" / "buoc1"))
    p.add_argument("--ten_luot", default=None)
    a = p.parse_args(argv)

    from transformers import AutoTokenizer

    thu_muc = NG.thu_muc_moi(a.ra, a.ten_luot or ten_luot_mac_dinh("kiem_truoc"))
    log = NhatKy(thu_muc / "nhat_ky.txt")
    tok = AutoTokenizer.from_pretrained(a.mo_hinh, revision=a.revision)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token                      # như nap_mo_hinh với LLAMA3_8B (pad_bang_eos)
    d = nap_du_lieu()
    tat_ca = sorted(set(d.harmful_train + d.harmless_train + d.harmful_val + d.harmless_val
                        + d.harmless_test + d.jailbreakbench))
    kk = kiem_khuon(tok, LLAMA3_8B, tat_ca)
    enc = ma_hoa(tok, [tat_ca[0]], LLAMA3_8B)
    khoi_dung = enc.so_token_khoi == 5 and enc.vi_tri_lay == -1
    eos_chinh = tuple(LLAMA3_8B.eos_chinh)
    gc_chinh = cau_hinh_hieu_luc(a.mo_hinh, a.revision, eos_chinh, tok.pad_token_id)
    gc_arditi = cau_hinh_hieu_luc(a.mo_hinh, a.revision, EOS_ARDITI_LLAMA3, tok.pad_token_id)
    kq = {
        "moi_truong": NG.thong_tin_moi_truong(),
        "kiem_khuon": kk, "khoi_5_token_vi_tri_cuoi": khoi_dung,
        "tokenizer": {"lop": type(tok).__name__, "bos": [tok.bos_token, tok.bos_token_id],
                      "eos": [tok.eos_token, tok.eos_token_id], "pad": [tok.pad_token, tok.pad_token_id],
                      "id_I": tok.convert_tokens_to_ids("I"),
                      "token_128001_128009": tok.convert_ids_to_tokens([128001, 128009])},
        "vi_du_loi_nhac": enc.van_ban[0], "vi_du_ids_cuoi": enc.ids_rieng[0][-8:],
        "cau_hinh_sinh_chinh": gc_chinh, "cau_hinh_sinh_arditi": gc_arditi,
    }
    NG.ghi_json(thu_muc / "kiem_truoc.json", kq)
    log(f"Khuôn chat: {kk['so_cau_lech']}/{kk['so_cau']} câu lệch cách tác giả; khối sau câu hỏi {kk['token_khoi']}")
    log(f"Khối 5 token, vị trí lấy −1: {khoi_dung}; id của 'I' = {kq['tokenizer']['id_I']} (phải là 40)")
    log(f"Tokenizer: {kq['tokenizer']}")
    log(f"generation_config của mô hình: {gc_chinh['generation_config_mo_hinh']}")
    log(f"Cấu hình hiệu lực là greedy đúng: chính {gc_chinh['la_greedy_dung']}, Arditi {gc_arditi['la_greedy_dung']}")
    dat = bool(kk["dat"] and khoi_dung and kq["tokenizer"]["id_I"] == 40
               and gc_chinh["la_greedy_dung"] and gc_arditi["la_greedy_dung"])
    log(f"ĐẠT: {dat}")
    return 0 if dat else 1


if __name__ == "__main__":
    sys.exit(main())
