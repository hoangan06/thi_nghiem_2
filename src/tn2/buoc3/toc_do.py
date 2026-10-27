"""Đo tốc độ sinh và bộ nhớ đỉnh (Nhap, Trình tự chạy bước 3 và "Quy mô và chi phí": tốc độ sinh đo khi nạp
từng mô hình chính; tính lại các bảng chi phí trước lượt hiệu chỉnh).

Đo bằng đúng hàm sinh của pipeline chính (tn2.sinh.sinh_theo_lo: greedy, bf16, đệm trái, 256 token), ở vài cỡ lô.
Mỗi cấu hình chạy cùng một danh sách câu; ghi thời gian, số token sinh, câu mỗi giờ, bộ nhớ GPU đỉnh. Hết bộ nhớ
thì ghi "het_bo_nho" và đi tiếp cấu hình sau (không dừng cả lượt).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import torch

from ..mo_hinh import DacTaMoHinh
from ..sinh import sinh_theo_lo

# Số câu cần sinh theo Nhap ("Quy mô và chi phí", ước lượng ban đầu, tổng cho cả hai mô hình): hiệu chỉnh khoảng
# 25 900; pilot khoảng 13 000; chạy chính khoảng 158 100; hướng ngẫu nhiên khoảng 17 600; kiểm α Gemma khoảng 9 800.
SO_CAU_THEO_NHAP = {"hieu_chinh": 25_900, "pilot": 13_000, "chay_chinh": 158_100, "huong_ngau_nhien": 17_600,
                    "kiem_alpha_gemma": 9_800}


def _dong_bo():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def do_toc_do(model, tok, dac_ta: DacTaMoHinh, cau_hoi: Sequence[str], cac_co_lo: Sequence[int],
              max_new_tokens: int, eos_ids: Sequence[int], tao_can_thiep: Callable | None = None,
              nhan: str = "", bao_tien_do=None) -> list[dict[str, Any]]:
    ra = []
    for co_lo in cac_co_lo:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        _dong_bo()
        t0 = time.time()
        try:
            kq = sinh_theo_lo(model, tok, dac_ta, list(cau_hoi), co_lo, max_new_tokens, eos_ids, tao_can_thiep)
            _dong_bo()
            dt = time.time() - t0
            so_token = sum(r["so_token"] for r in kq)
            d = {"nhan": nhan, "co_lo": co_lo, "so_cau": len(kq), "thoi_gian_giay": dt,
                 "cau_moi_gio": len(kq) / dt * 3600 if dt > 0 else None,
                 "token_sinh": so_token, "token_moi_giay": so_token / dt if dt > 0 else None,
                 "ti_le_cham_gioi_han": sum(1 for r in kq if not r["cham_eos"]) / len(kq), "het_bo_nho": False}
        except torch.cuda.OutOfMemoryError:
            d = {"nhan": nhan, "co_lo": co_lo, "het_bo_nho": True}
        if torch.cuda.is_available():
            d["bo_nho_dinh_GiB"] = torch.cuda.max_memory_allocated() / 2**30
            d["bo_nho_dat_truoc_dinh_GiB"] = torch.cuda.max_memory_reserved() / 2**30
        ra.append(d)
        if bao_tien_do:
            bao_tien_do(f"tốc độ [{nhan}] lô {co_lo}: {d}")
    return ra


# Chạy chính quét MỌI lớp nên số câu phụ thuộc số lớp của mô hình (rà soát code bước 3, R2 B3):
# 244 câu × 4 chiều × 2 kiểu × L lớp + 244 câu × 4 lượt nền. Qwen3.5-9B (32 lớp) 63 440; Gemma 4 12B (48 lớp)
# 94 672; tổng 158 112, khớp con số làm tròn 158 100 của Nhap. Hiệu chỉnh (5 lớp đại diện mỗi mô hình) và hướng
# ngẫu nhiên (tối đa 3 lớp mỗi mô hình, mỗi kiểu) không phụ thuộc số lớp nên chia đều hai mô hình. Pilot chỉ chạy
# trên Qwen3.5-9B, kiểm α chỉ trên Gemma 4 12B. Giữ cả kiểu (b); bỏ kiểu (b) ở bước 9 thì tính lại.
SO_CAU_CHINH, SO_CHIEU, SO_KIEU, SO_LUOT_NEN = 244, 4, 2, 4


def so_cau_theo_mo_hinh(dac_ta: DacTaMoHinh) -> dict[str, int]:
    """Số câu cần sinh cho MỘT mô hình theo Nhap (ước lượng, giữ cả hai kiểu)."""
    ra = {"hieu_chinh": SO_CAU_THEO_NHAP["hieu_chinh"] // 2,
          "chay_chinh": SO_CAU_CHINH * SO_CHIEU * SO_KIEU * dac_ta.so_lop + SO_CAU_CHINH * SO_LUOT_NEN,
          "huong_ngau_nhien": SO_CAU_THEO_NHAP["huong_ngau_nhien"] // 2}
    if dac_ta.ten.startswith("qwen"):
        ra["pilot"] = SO_CAU_THEO_NHAP["pilot"]
    if dac_ta.ten.startswith("gemma"):
        ra["kiem_alpha"] = SO_CAU_THEO_NHAP["kiem_alpha_gemma"]
    return ra


def uoc_gio_gpu(cau_moi_gio: float, dac_ta: DacTaMoHinh) -> dict[str, Any]:
    """Giờ GPU ước cho MỘT mô hình từ tốc độ đo được (ước lượng thô, để tính lại chi phí)."""
    if not cau_moi_gio:
        return {}
    phan = so_cau_theo_mo_hinh(dac_ta)
    gio = {k: v / cau_moi_gio for k, v in phan.items()}
    gio["tong"] = sum(gio.values())
    return {"cau_moi_gio": cau_moi_gio, "so_cau_theo_nhap": phan, "gio_gpu": gio,
            "chu_thich": "Chạy chính tính theo số lớp của mô hình (244 × 4 chiều × 2 kiểu × L + 244 × 4 lượt nền); "
                         "hiệu chỉnh, hướng ngẫu nhiên chia đều hai mô hình; pilot chỉ Qwen3.5-9B, kiểm α chỉ "
                         "Gemma 4 12B. Giữ cả kiểu (b). Tốc độ đo ở lô nhanh nhất trên máy 80 GB; ước lượng thô."}
