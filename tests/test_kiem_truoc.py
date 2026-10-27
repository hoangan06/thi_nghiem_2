"""Script kiểm trước khi chạy: chạy thông trên tokenizer đồ chơi và mô hình nhỏ lưu ra đĩa."""
import json

from conftest import tao_llama_nho


def test_kiem_truoc_chay_thong(tok_llama, tmp_path):
    import buoc1_kiem_truoc as S

    d = tmp_path / "mo_hinh"
    m = tao_llama_nho(so_lop=2)
    tok_llama.save_pretrained(d)
    m.save_pretrained(d)
    rc = S.main(["--mo_hinh", str(d), "--ra", str(tmp_path / "ra"), "--ten_luot", "x"])
    assert rc == 0
    k = json.loads((tmp_path / "ra" / "x" / "kiem_truoc.json").read_text(encoding="utf-8"))
    hl = k["cau_hinh_sinh_chinh"]["hieu_luc"]
    assert hl["do_sample"] is False and hl["repetition_penalty"] == 1.0      # không bị generation_config đè
    assert k["cau_hinh_sinh_chinh"]["generation_config_mo_hinh"]["repetition_penalty"] == 1.5
    assert k["cau_hinh_sinh_arditi"]["hieu_luc"]["eos_token_id"] == [128001]
