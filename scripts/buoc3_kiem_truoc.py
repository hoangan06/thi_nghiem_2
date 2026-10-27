"""Bước 3, kiểm trước khi nạp trọng số (vài giây, không cần GPU), cho MỘT mô hình chính.

Chạy:
    python scripts/buoc3_kiem_truoc.py --mo_hinh_dac_ta qwen35_9b
    python scripts/buoc3_kiem_truoc.py --mo_hinh_dac_ta gemma4_12b

Kiểm trên tokenizer và file cấu hình THẬT (test của kho chỉ dùng tokenizer đồ chơi); Ghi chú kỹ thuật hai mô hình
chính, mục 5, việc 2 và 4:
1. Khối sau câu hỏi trên cả 512 câu dựng (bốn file en/vi): đúng 9 token, đúng chuỗi token dự kiến của đặc tả,
   vị trí lấy kích hoạt −5 (token cuối phần mở đầu lượt trả lời, không tính khối thinking rỗng); template tắt
   thinking (kết thúc bằng khối thinking rỗng của đặc tả).
2. Token kết thúc: Qwen3.5-9B 248046 = <|im_end|>, 248044 = <|endoftext|>; Gemma 4 12B 106 = <turn|> (Nhap 3.5).
3. Cấu hình: số lớp, danh sách lớp full attention trong layer_types so với đặc tả.
4. Cấu hình sinh HIỆU LỰC (greedy) sau khi transformers trộn với generation_config của mô hình.
Ghi vào results/buoc3/<mô hình>/<ten_luot>/kiem_truoc.json.
Mã thoát: 0 đạt; 2 khối sau câu hỏi hoặc token kết thúc lệch dự kiến (DỪNG, không nạp trọng số); 1 lỗi khác.
"""
from __future__ import annotations

import argparse
import hashlib
import sys

from _chung import GOC, NhatKy, chon_dac_ta, ten_luot_mac_dinh

from tn2 import nguon_goc as NG
from tn2.buoc3.nap_kiem import kiem_eos_tokenizer, kiem_khoi_tokenizer
from tn2.du_lieu import nap_tap_dung
from tn2.khuon import ap_template


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo_hinh_dac_ta", required=True)
    p.add_argument("--mo_hinh", default=None, help="thư mục cục bộ hoặc id Hugging Face; mặc định hf_id của đặc tả")
    p.add_argument("--revision", default=None)
    p.add_argument("--ra", default=str(GOC / "results" / "buoc3"))
    p.add_argument("--ten_luot", default=None)
    a = p.parse_args(argv)
    dac_ta = chon_dac_ta(a.mo_hinh_dac_ta)
    nguon = a.mo_hinh or dac_ta.hf_id

    from transformers import AutoConfig, AutoTokenizer
    from buoc1_kiem_truoc import cau_hinh_hieu_luc

    thu_muc = NG.thu_muc_moi(f"{a.ra}/{dac_ta.ten}", a.ten_luot or ten_luot_mac_dinh("kiem_truoc"))
    log = NhatKy(thu_muc / "nhat_ky.txt")
    tok = AutoTokenizer.from_pretrained(nguon, revision=a.revision)
    tok.padding_side = "left"
    if dac_ta.pad_bang_eos or tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tap = nap_tap_dung()
    tat_ca = tap.harm_en + tap.harm_vi + tap.harmless_en + tap.harmless_vi
    khoi = kiem_khoi_tokenizer(tok, dac_ta, tat_ca)
    eos = kiem_eos_tokenizer(tok, dac_ta)
    tpl = getattr(tok, "chat_template", None) or ""
    ket_thuc_thinking = all(ap_template(tok, q, dac_ta).endswith(dac_ta.chuoi_thinking or "")
                            for q in (tap.harm_en[0], tap.harmless_vi[0]))

    cfg = AutoConfig.from_pretrained(nguon, revision=a.revision)
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    lt = list(getattr(tcfg, "layer_types", []) or [])
    full = [i for i, x in enumerate(lt) if x == "full_attention"]
    cau_truc = {"lop_config": type(cfg).__name__, "model_type": getattr(cfg, "model_type", None),
                "so_lop": getattr(tcfg, "num_hidden_layers", None), "hidden_size": getattr(tcfg, "hidden_size", None),
                "vocab_size": getattr(tcfg, "vocab_size", None), "layer_types": lt, "full_attention": full,
                "full_attention_dac_ta": list(dac_ta.lop_full_attention),
                "sliding_window": getattr(tcfg, "sliding_window", None), "commit_hf": getattr(cfg, "_commit_hash", None)}
    cau_truc["dat"] = (cau_truc["so_lop"] == dac_ta.so_lop and full == list(dac_ta.lop_full_attention))
    gc = cau_hinh_hieu_luc(nguon, a.revision, tuple(dac_ta.eos_chinh), tok.pad_token_id)
    kq = {"moi_truong": NG.thong_tin_moi_truong(), "dac_ta": dac_ta.ten, "nguon": nguon, "revision": a.revision,
          "khoi_sau_cau_hoi": khoi, "token_ket_thuc": eos, "template_tat_thinking": ket_thuc_thinking,
          "template_kwargs": dac_ta.template_kwargs, "chat_template_sha256": hashlib.sha256(tpl.encode()).hexdigest(),
          "chat_template": tpl, "cau_truc_config": cau_truc, "cau_hinh_sinh": gc,
          "tokenizer": {"lop": type(tok).__name__, "so_token": len(tok)},
          "vi_du": {"en": ap_template(tok, tap.harm_en[0], dac_ta), "vi": ap_template(tok, tap.harm_vi[0], dac_ta)}}
    NG.ghi_json(thu_muc / "kiem_truoc.json", kq)
    log(f"[{dac_ta.ten}] khối sau câu hỏi: {khoi['khoi_theo_so_cau']} | K {khoi['K_theo_so_cau']} | "
        f"vị trí lấy {khoi['vi_tri_lay_theo_so_cau']} | {khoi['so_cau_loi']} câu lỗi → đạt {khoi['dat']}")
    log(f"[{dac_ta.ten}] token kết thúc: {eos['token_cua_eos']} | sai {eos['sai']} | pad {eos['pad']} → đạt {eos['dat']}")
    log(f"[{dac_ta.ten}] template kết thúc bằng khối thinking rỗng: {ket_thuc_thinking}")
    log(f"[{dac_ta.ten}] cấu trúc config: {cau_truc['so_lop']} lớp, full attention {full} → đạt {cau_truc['dat']}")
    log(f"[{dac_ta.ten}] generation_config của mô hình: {gc['generation_config_mo_hinh']}")
    log(f"[{dac_ta.ten}] cấu hình hiệu lực là greedy đúng: {gc['la_greedy_dung']}")
    if not (khoi["dat"] and eos["dat"] and ket_thuc_thinking):
        log("DỪNG: khối sau câu hỏi, template hoặc token kết thúc lệch dự kiến — không nạp trọng số. "
            "Xem kiem_truoc.json, mục khoi_sau_cau_hoi và token_ket_thuc.")
        return 2
    dat = bool(cau_truc["dat"] and gc["la_greedy_dung"])
    log(f"ĐẠT: {dat}")
    return 0 if dat else 1


if __name__ == "__main__":
    sys.exit(main())
