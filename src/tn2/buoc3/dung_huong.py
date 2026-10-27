"""Dựng hướng và s_ℓ cho một mô hình chính (Nhap, mục Dựng hướng, điểm 1–8; mục 3.3; mục 7.3).

1. Không lọc tập dựng theo hành vi: dùng đủ 128 câu mỗi tập (tn2.du_lieu.nap_tap_dung).
2. Vị trí lấy kích hoạt = token cuối của phần mở đầu lượt trả lời, không tính khối thinking rỗng (đo bằng
   tn2.khuon; với Qwen3.5-9B và Gemma 4 12B dự kiến là −5).
3. Vector riêng cho mọi lớp, lấy ở ĐẦU RA khối decoder ℓ (hook trên lớp, không lấy hidden_states cuối vì đã qua
   chuẩn hoá cuối; với Gemma 4, đầu ra lớp đã nhân layer_scalar và cũng là đầu vào của lớp sau).
4. r_ℓ = trung bình (câu có hại) − trung bình (câu vô hại), riêng từng ngôn ngữ; r̂_ℓ = r_ℓ / ‖r_ℓ‖; lưu ‖r_ℓ‖.
5. Tắt thinking: qua template_kwargs của đặc tả (enable_thinking=False).
6. Phép kiểm chia đôi: chia mỗi tập dựng thành hai nửa (cùng một phép chia cho bốn file, theo chỉ số cặp
   en–vi; seed ghi lại), dựng hướng trên từng nửa, cosine theo lớp, riêng từng ngôn ngữ. Đây cũng là "đường
   tham chiếu" của C3 (Nhap 7.3).
7. Cosine giữa hướng dựng ở hai vị trí: vị trí lấy (điểm 2) và token cuối thật của lời nhắc (sau khối thinking
   rỗng, −1). Chỉ để chẩn đoán và báo cáo; vị trí lấy giữ như đã chốt dù cosine thấp.
8. Độ lớn tương đối ‖r_ℓ‖ / s_ℓ.

s_ℓ (Nhap 3.3) = độ lớn trung bình ‖h‖ của luồng dư ở lớp ℓ, tại vị trí lấy, trên tập dựng GỘP hai ngôn ngữ
(512 câu). Một s_ℓ chung cho cả hai hướng (định nghĩa "cùng liều" của C4, Nhap 7.4).

Ngoài ra lưu kích hoạt từng câu ở vị trí lấy (kiểu số của mô hình), để bootstrap của C3 (Nhap 7.3: rút lại theo
cặp en–vi, dựng lại cả hai hướng) chạy trên CPU về sau mà không cần GPU. Cosine en–vi tính ở đây chỉ là ước lượng
điểm để mô tả; phân tích C3 chính thức (khoảng tin cậy, vùng S) làm ở bước phân tích.
"""
from __future__ import annotations

from typing import Any

import torch

from ..du_lieu import NGON_NGU, TapDung
from ..huong import chia_doi, cosine
from ..khuon import ma_hoa
from ..kich_hoat import kich_hoat_tung_cau
from ..mo_hinh import DacTaMoHinh

SEED_CHIA_DOI = 20261027
VI_TRI_CUOI_THAT = -1


def _chuan_hoa_kiem(r: torch.Tensor, ten: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = r.norm(dim=-1)
    if bool((n == 0).any()):
        raise ValueError(f"{ten}: hướng có độ lớn 0 ở lớp {torch.nonzero(n == 0).flatten().tolist()}.")
    return r / n.unsqueeze(-1), n


def _chuan_hoa_chan_doan(r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Như _chuan_hoa_kiem nhưng cho hướng chỉ dùng để chẩn đoán (Nhap, Dựng hướng điểm 7): lớp có độ lớn 0 thì
    hướng chuẩn hoá là NaN (ghi null trong JSON), không dừng việc dựng hướng chính (rà soát code bước 3, R2 B4)."""
    n = r.norm(dim=-1)
    bang_0 = n == 0
    r_hat = r / n.clamp_min(1e-300).unsqueeze(-1)
    r_hat[bang_0] = float("nan")
    return r_hat, n, torch.nonzero(bang_0).flatten().tolist()


def _nho_nhat(x: torch.Tensor) -> float | None:
    """Giá trị nhỏ nhất bỏ NaN; None nếu mọi giá trị đều NaN."""
    x = x[~torch.isnan(x)]
    return float(x.min()) if x.numel() else None


@torch.no_grad()
def dung_huong(model, tok, dac_ta: DacTaMoHinh, tap: TapDung, co_lo: int = 32,
               position_ids_theo_mat_na: bool = True, seed_chia_doi: int = SEED_CHIA_DOI,
               bao_tien_do=None) -> dict[str, Any]:
    """Trả về dict gồm 'tensor' (để torch.save) và 'tom_tat' (để ghi JSON).

    position_ids_theo_mat_na mặc định True: pipeline chính luôn truyền position_ids = cumsum(mask) − 1 như
    generate (Ân duyệt Q1, 27/10/2026); False chỉ dùng trong test để so với hàm trung bình của bước 1."""
    vt = ma_hoa(tok, [tap.harm_en[0]], dac_ta).vi_tri_lay
    if vt == VI_TRI_CUOI_THAT:
        raise ValueError("Vị trí lấy trùng token cuối thật: mô hình này không có khối thinking rỗng?")
    vi_tri = [vt, VI_TRI_CUOI_THAT]
    kh: dict[tuple[str, str], torch.Tensor] = {}
    for tap_ten in ("harm", "harmless"):
        for ng in NGON_NGU:
            cau = getattr(tap, f"{tap_ten}_{ng}")
            kh[(tap_ten, ng)] = kich_hoat_tung_cau(model, tok, dac_ta, cau, vi_tri, co_lo=co_lo,
                                                   position_ids_theo_mat_na=position_ids_theo_mat_na)
            if bao_tien_do:
                bao_tien_do(f"dựng hướng: xong kích hoạt {tap_ten}_{ng} ({len(cau)} câu)")
    n_cau = {k: int(v.shape[0]) for k, v in kh.items()}
    so_lop = int(next(iter(kh.values())).shape[2])

    def tb(t: torch.Tensor, p: int, idx=None) -> torch.Tensor:      # [nL, d] float64
        x = t[:, p] if idx is None else t[idx, p]
        return x.to(torch.float64).mean(dim=0)

    tensor: dict[str, Any] = {"vi_tri_lay": vt, "vi_tri_cuoi_that": VI_TRI_CUOI_THAT, "so_lop": so_lop}
    tom: dict[str, Any] = {"vi_tri_lay": vt, "vi_tri_cuoi_that": VI_TRI_CUOI_THAT, "so_cau": {f"{a}_{b}": n for (a, b), n in n_cau.items()},
                           "position_ids_theo_mat_na": position_ids_theo_mat_na, "seed_chia_doi": seed_chia_doi}
    # s_ℓ: trung bình ‖h‖ ở vị trí lấy trên 512 câu gộp (Nhap 3.3).
    do_lon = torch.cat([v[:, 0].to(torch.float64).norm(dim=-1) for v in kh.values()], dim=0)   # [512, nL]
    s = do_lon.mean(dim=0)
    tensor["s"] = s
    tensor["s_theo_tap"] = {f"{a}_{b}": v[:, 0].to(torch.float64).norm(dim=-1).mean(dim=0) for (a, b), v in kh.items()}
    tom["so_cau_tinh_s"] = int(do_lon.shape[0])
    nua1, nua2 = chia_doi(n_cau[("harm", "en")], seed_chia_doi)
    if n_cau[("harmless", "en")] != n_cau[("harm", "en")]:
        raise ValueError("Hai tập dựng khác số câu; phép chia đôi dùng chung một phép chia.")
    tensor["chia_doi"] = {"nua_1": nua1, "nua_2": nua2}
    for ng in NGON_NGU:
        h, v = kh[("harm", ng)], kh[("harmless", ng)]
        r = tb(h, 0) - tb(v, 0)
        r_hat, n = _chuan_hoa_kiem(r, f"hướng {ng} ở vị trí lấy")
        rB = tb(h, 1) - tb(v, 1)
        rB_hat, nB, lop_rB_0 = _chuan_hoa_chan_doan(rB)
        r1 = tb(h, 0, nua1) - tb(v, 0, nua1)
        r2 = tb(h, 0, nua2) - tb(v, 0, nua2)
        tensor[f"r_{ng}"] = r
        tensor[f"r_hat_{ng}"] = r_hat
        tensor[f"do_lon_tho_{ng}"] = n
        tensor[f"do_lon_tuong_doi_{ng}"] = n / s
        tensor[f"r_hat_{ng}_cuoi_that"] = rB_hat
        tensor[f"do_lon_tho_{ng}_cuoi_that"] = nB
        cos_chia_doi = cosine(r1, r2)
        cos_hai_vi_tri = cosine(r_hat, rB_hat)
        tensor[f"cos_chia_doi_{ng}"] = cos_chia_doi
        tensor[f"cos_hai_vi_tri_{ng}"] = cos_hai_vi_tri
        tom[ng] = {"do_lon_tho": n.tolist(), "do_lon_tuong_doi": (n / s).tolist(),
                   "cos_chia_doi": cos_chia_doi.tolist(), "cos_hai_vi_tri": cos_hai_vi_tri.tolist(),
                   "cos_chia_doi_nho_nhat": _nho_nhat(cos_chia_doi),
                   "cos_hai_vi_tri_nho_nhat": _nho_nhat(cos_hai_vi_tri),
                   "lop_huong_cuoi_that_do_lon_0": lop_rB_0}
    cos_en_vi = cosine(tensor["r_hat_en"], tensor["r_hat_vi"])
    tensor["cos_en_vi_uoc_luong_diem"] = cos_en_vi
    tom["s"] = s.tolist()
    tom["cos_en_vi_uoc_luong_diem"] = cos_en_vi.tolist()
    tom["ghi_chu_cos_en_vi"] = ("Ước lượng điểm, chỉ để mô tả; phân tích C3 chính thức (bootstrap 10 000 lần trên "
                                "tập dựng, vùng S) làm ở bước phân tích (Nhap 7.3) từ file kich_hoat_vi_tri_lay.pt.")
    kich_hoat_luu = {f"{a}_{b}": v[:, 0].contiguous() for (a, b), v in kh.items()}    # [128, nL, d], kiểu số mô hình
    return {"tensor": tensor, "tom_tat": tom, "kich_hoat_vi_tri_lay": kich_hoat_luu}
