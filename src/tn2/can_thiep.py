"""Hook can thiệp vào luồng dư: cộng một vector, hoặc chiếu bỏ một hướng.

Một can thiệp gồm: các lớp, điểm đặt, kiểu, vector và vùng token.

Điểm đặt:
- "ra_khoi"  : đầu ra của khối decoder ℓ (pipeline chính; Nhap Dựng hướng điểm 3).
- "vao_khoi" : đầu vào của khối decoder ℓ (= resid_pre, cách code của Arditi et al. đặt hook).
- "ra_attn", "ra_mlp" : đầu ra khối attention / MLP trong lớp ℓ (chỉ dùng cho chiếu bỏ kiểu Arditi).

Kiểu:
- "cong"     : h ← h + v (v đã nhân sẵn hệ số và dấu, ví dụ ±α·s_ℓ·r̂_ℓ).
- "chieu_bo" : h ← h − (h·d̂) d̂, với d̂ = d / (‖d‖ + 1e-8) tính như code của Arditi et al.

Vùng token (Nhap mục 3.2):
- "moi_vi_tri"   : mọi vị trí, kể cả vị trí đệm (đúng như code của Arditi et al.).
- "khoi_va_sinh" : K token cuối của lời nhắc (khối template sau câu hỏi) cộng với các token sinh ra.
  so_token_sinh = None: mọi token sinh (phương án 1); = 1: chỉ token sinh đầu tiên (phương án 3);
  = 0: không token sinh nào.

Theo dõi vị trí: một forward hook trên lớp embedding đầu vào đếm số token đã đi qua mô hình trong
ngữ cảnh `gan_can_thiep`. Lượt forward thứ nhất (prefill) mang các vị trí 0..L−1 của chuỗi đã đệm;
mỗi lượt sau (sinh có KV cache) mang đúng một vị trí tiếp theo. Do đó vị trí p ≥ L là token sinh thứ
g = p − L (g = 0 là token sinh đầu tiên, được đưa vào mô hình ở lượt forward thứ hai). Cơ chế này không
phụ thuộc kiểu lớp hay kiểu cache của từng mô hình. Điều kiện dùng: trong một ngữ cảnh chỉ gọi đúng một
lần generate (hoặc một chuỗi forward nối tiếp có cache), bắt đầu từ lời nhắc dài L đã đệm trái.

Hook không đổi dtype và không sửa tensor tại chỗ: trả về tensor mới. Ở các vị trí ngoài vùng, giá trị
giữ nguyên tuyệt đối (dùng torch.where). Vì vậy hook với vector 0 cho đầu ra giống hệt khi không gắn hook
(phép kiểm (ii) của Nhap).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch

from .mo_hinh import khoi_attn, khoi_mlp, tim_cac_lop

DIEM_HOP_LE = ("ra_khoi", "vao_khoi", "ra_attn", "ra_mlp")
KIEU_HOP_LE = ("cong", "chieu_bo")
LOI_GIA_LAP_HOP_LE = (None, "lech_vung", "bo_cong_khi_sinh")


@dataclass(frozen=True)
class Vung:
    loai: str = "khoi_va_sinh"          # "moi_vi_tri" | "khoi_va_sinh"
    so_token_khoi: int = 0              # K
    so_token_sinh: int | None = None    # None = mọi token sinh; n = n token sinh đầu

    def __post_init__(self):
        if self.loai not in ("moi_vi_tri", "khoi_va_sinh"):
            raise ValueError(f"Vùng không hợp lệ: {self.loai}")
        if self.loai == "khoi_va_sinh" and self.so_token_khoi <= 0:
            raise ValueError("Vùng khoi_va_sinh cần so_token_khoi > 0.")
        if self.so_token_sinh is not None and self.so_token_sinh < 0:
            raise ValueError("so_token_sinh phải ≥ 0 hoặc None.")


def vung_phuong_an(phuong_an: int, so_token_khoi: int) -> Vung:
    """Vùng của phương án 1 (mọi token sinh) hoặc phương án 3 (N = 1 token sinh đầu) — Nhap mục 3.2."""
    if phuong_an == 1:
        return Vung("khoi_va_sinh", so_token_khoi, None)
    if phuong_an == 3:
        return Vung("khoi_va_sinh", so_token_khoi, 1)
    raise ValueError("Chỉ có phương án 1 và 3 (phương án 2 đã bị loại).")


def mat_na_vung(vung: Vung, vi_tri: torch.Tensor, do_dai_loi_nhac: int | None,
                loi_gia_lap: str | None = None) -> torch.Tensor:
    """Mặt nạ bool [s] cho các vị trí tuyệt đối vi_tri (trong chuỗi đã đệm trái)."""
    if vung.loai == "moi_vi_tri":
        return torch.ones_like(vi_tri, dtype=torch.bool)
    if do_dai_loi_nhac is None:
        raise ValueError("Vùng khoi_va_sinh cần độ dài lời nhắc L.")
    L = int(do_dai_loi_nhac)
    if loi_gia_lap == "lech_vung":      # lỗi cài vào cho kiểm âm: tính vùng với L − 1 thay cho L
        # Phương án 3: vùng lệch sang trái một token (bỏ mất token sinh đầu tiên).
        # Phương án 1: vùng bắt đầu sớm hơn một token (thêm token ngay trước khối), phần sinh giữ nguyên.
        L = L - 1
    khoi = (vi_tri >= L - vung.so_token_khoi) & (vi_tri < L)
    g = vi_tri - L
    if loi_gia_lap == "bo_cong_khi_sinh" or vung.so_token_sinh == 0:
        sinh = torch.zeros_like(khoi)
    elif vung.so_token_sinh is None:
        sinh = g >= 0
    else:
        sinh = (g >= 0) & (g < vung.so_token_sinh)
    return khoi | sinh


@dataclass
class CanThiep:
    lop: tuple[int, ...]
    diem: str
    kieu: str
    vector: torch.Tensor
    vung: Vung

    def __post_init__(self):
        if self.diem not in DIEM_HOP_LE:
            raise ValueError(f"Điểm đặt không hợp lệ: {self.diem}")
        if self.kieu not in KIEU_HOP_LE:
            raise ValueError(f"Kiểu can thiệp không hợp lệ: {self.kieu}")
        if self.vector.ndim != 1:
            raise ValueError("Vector phải một chiều [d_model].")
        self.lop = tuple(int(x) for x in self.lop)


class _BoDemViTri:
    """Đếm vị trí tuyệt đối của các token trong lượt forward hiện tại."""

    def __init__(self):
        self.da_qua = 0
        self.vi_tri: torch.Tensor | None = None

    def hook_embedding(self, module, args, output):
        s = output.shape[1]
        self.vi_tri = torch.arange(self.da_qua, self.da_qua + s, device=output.device)
        self.da_qua += s


def _ap_dung(h: torch.Tensor, ct: CanThiep, mask: torch.Tensor) -> torch.Tensor:
    """h [B, s, d] → tensor mới. mask [s] bool."""
    if ct.kieu == "cong":
        v = ct.vector.to(device=h.device, dtype=h.dtype)
        moi = h + v
    else:
        d = ct.vector
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)   # đúng như get_direction_ablation_* của tác giả
        d = d.to(device=h.device, dtype=h.dtype)
        moi = h - (h @ d).unsqueeze(-1) * d
    if bool(mask.all()):
        return moi
    return torch.where(mask.view(1, -1, 1), moi, h)


@contextlib.contextmanager
def gan_can_thiep(model: torch.nn.Module, cac_can_thiep: Sequence[CanThiep] | None,
                  do_dai_loi_nhac: int | None = None, loi_gia_lap: str | None = None) -> Iterator[_BoDemViTri]:
    """Gắn các can thiệp trong phạm vi khối `with`; luôn gỡ hook khi ra khỏi khối.

    do_dai_loi_nhac: L = độ dài lời nhắc đã đệm trái (bắt buộc với vùng khoi_va_sinh).
    loi_gia_lap: chỉ dùng cho phép kiểm âm (kiem_hook); None trong mọi lượt chạy thật.
    """
    if loi_gia_lap not in LOI_GIA_LAP_HOP_LE:
        raise ValueError(f"loi_gia_lap không hợp lệ: {loi_gia_lap}")
    cac_lop = tim_cac_lop(model)
    dem = _BoDemViTri()
    handles = [model.get_input_embeddings().register_forward_hook(dem.hook_embedding)]

    def _mask(ct: CanThiep, s: int, device) -> torch.Tensor:
        if ct.vung.loai == "moi_vi_tri":
            return torch.ones(s, dtype=torch.bool, device=device)
        if dem.vi_tri is None or dem.vi_tri.shape[0] != s:
            raise RuntimeError("Bộ đếm vị trí không khớp độ dài chuỗi của lớp (mô hình không đi qua "
                               "embedding đầu vào bằng input_ids?).")
        return mat_na_vung(ct.vung, dem.vi_tri.to(device), do_dai_loi_nhac, loi_gia_lap)

    def tao_hook_ra(ct: CanThiep):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            moi = _ap_dung(h, ct, _mask(ct, h.shape[1], h.device))
            if isinstance(output, tuple):
                return (moi, *output[1:])
            return moi
        return hook

    def tao_hook_vao(ct: CanThiep):
        def hook(module, args, kwargs):
            if len(args) > 0:
                h = args[0]
                moi = _ap_dung(h, ct, _mask(ct, h.shape[1], h.device))
                return (moi, *args[1:]), kwargs
            if "hidden_states" in kwargs:
                h = kwargs["hidden_states"]
                kw = dict(kwargs)
                kw["hidden_states"] = _ap_dung(h, ct, _mask(ct, h.shape[1], h.device))
                return args, kw
            raise RuntimeError("Không tìm thấy hidden_states trong đầu vào của lớp.")
        return hook

    try:
        for ct in (cac_can_thiep or []):
            for l in ct.lop:
                if not 0 <= l < len(cac_lop):
                    raise ValueError(f"Lớp {l} ngoài khoảng [0, {len(cac_lop)}).")
                lop = cac_lop[l]
                if ct.diem == "ra_khoi":
                    handles.append(lop.register_forward_hook(tao_hook_ra(ct)))
                elif ct.diem == "vao_khoi":
                    handles.append(lop.register_forward_pre_hook(tao_hook_vao(ct), with_kwargs=True))
                elif ct.diem == "ra_attn":
                    handles.append(khoi_attn(lop).register_forward_hook(tao_hook_ra(ct)))
                else:
                    handles.append(khoi_mlp(lop).register_forward_hook(tao_hook_ra(ct)))
        yield dem
    finally:
        for h in handles:
            h.remove()


def chieu_bo_moi_lop(so_lop: int, huong: torch.Tensor) -> list[CanThiep]:
    """Chiếu bỏ hướng ở mọi lớp, mọi vị trí: đầu vào mỗi khối + đầu ra attention + đầu ra MLP
    (đúng get_all_direction_ablation_hooks của Arditi et al.)."""
    tat_ca = tuple(range(so_lop))
    vung = Vung("moi_vi_tri")
    return [CanThiep(tat_ca, "vao_khoi", "chieu_bo", huong, vung),
            CanThiep(tat_ca, "ra_attn", "chieu_bo", huong, vung),
            CanThiep(tat_ca, "ra_mlp", "chieu_bo", huong, vung)]


def cong_vao_khoi_moi_vi_tri(lop: int, vector: torch.Tensor, he_so: float = 1.0) -> list[CanThiep]:
    """Cộng he_so·vector vào đầu vào khối `lop` ở mọi vị trí (get_activation_addition_input_pre_hook)."""
    return [CanThiep((lop,), "vao_khoi", "cong", vector * he_so, Vung("moi_vi_tri"))]


def vector_can_thiep(r_hat: torch.Tensor, s: float, alpha: float, kieu_ab: str) -> torch.Tensor:
    """±α·s_ℓ·r̂_ℓ: dấu + cho kiểu (a), dấu − cho kiểu (b) — Nhap mục 3.3. Tính ở float32."""
    if kieu_ab not in ("a", "b"):
        raise ValueError("kieu_ab phải là 'a' hoặc 'b'.")
    dau = 1.0 if kieu_ab == "a" else -1.0
    return (dau * float(alpha) * float(s)) * r_hat.to(torch.float32)


def can_thiep_chinh(lop: int, r_hat: torch.Tensor, s: float, alpha: float, kieu_ab: str,
                    phuong_an: int, so_token_khoi: int) -> CanThiep:
    """Can thiệp của pipeline chính: đầu ra khối ℓ, vùng theo phương án, vector ±α·s_ℓ·r̂_ℓ."""
    return CanThiep((lop,), "ra_khoi", "cong", vector_can_thiep(r_hat, s, alpha, kieu_ab),
                    vung_phuong_an(phuong_an, so_token_khoi))
