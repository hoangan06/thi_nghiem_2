"""Sinh theo kế hoạch, ghi bền và chạy tiếp được sau khi bị ngắt (dùng cho bước 4; dùng lại cho pilot, lượt kiểm α
của Gemma, chạy chính và hướng ngẫu nhiên).

Một ĐƠN VỊ (DonVi) là một ô của thiết kế: (tập, phần, kiểu, ngôn ngữ câu hỏi, hướng, lớp, α, phương án). Mỗi đơn
vị sinh câu trả lời cho mọi câu của phần đó và ghi ra MỘT file JSONL riêng. Can thiệp theo Nhap 3.3:
h ← h ± α · s_ℓ · r̂_ℓ ở đầu ra khối ℓ, dấu + cho kiểu (a), − cho kiểu (b), vùng theo phương án (3.2), hướng r̂ của
ngôn ngữ dựng hướng ("chiều X→Y": hướng X, câu hỏi Y), s_ℓ chung hai ngôn ngữ. Sinh greedy, đệm trái, 256 token,
token kết thúc đặt rõ (3.5), giải mã "den_eos" (giữ token đặc biệt để chấm vỡ / rò thinking về sau).
Lượt nền (3.6): gắn hook với α = 0 (vector 0, khớp tuyệt đối với không gắn hook — phép (ii) của kiểm pipeline).

Bền và chạy tiếp (yêu cầu của Ân, 27/10/2026: không ghi đè, không mất file, máy dừng giữa chừng thì chạy lại
cùng lệnh là tiếp từ chỗ dừng):
- Khoá: mỗi lượt chỉ một tiến trình (fcntl.flock trên .<tên lượt>.khoa cạnh thư mục lượt); tiến trình thứ hai dừng.
- Thư mục lượt có ke_hoach.json (kế hoạch + dấu vân tay của mọi thứ quyết định đầu ra: đặc tả, commit HF của mô
  hình, SHA file hướng, SHA cách chia tập và dữ liệu, tham số sinh, danh sách đơn vị, mã nguồn). Chạy lại cùng
  thư mục: dấu vân tay phải khớp; không khớp mà đã có đơn vị sinh xong thì dừng, không đụng vào file nào.
- moi_truong_goc.json: môi trường số học của phiên đầu tiên sinh đơn vị (torch, transformers, CUDA, GPU, backend
  lớp attention tuyến tính, kiểu số). Phiên sau khác môi trường thì dừng (lượt nền và lượt can thiệp được so theo
  cặp trên từng câu), trừ khi cho phép rõ ràng; mỗi .xong.json ghi số phiên và dấu vân tay môi trường.
- Mỗi đơn vị ghi <ma>.jsonl qua file tạm + fsync + os.replace (không bao giờ có file kết quả viết dở), rồi ghi
  <ma>.xong.json (SHA, số dòng). Đơn vị chỉ tính là xong khi có cả hai và SHA, số dòng, id khớp.
- File hỏng hay viết dở không bị xoá hay ghi đè: chuyển sang hong/ kèm dấu thời gian, rồi sinh lại.
- Lô của một đơn vị cố định theo kế hoạch (cùng câu, cùng thứ tự, cùng cỡ lô), nên lượt nền và lượt can thiệp có
  cùng cách đệm (so theo cặp trên từng câu, Nhap 4.5).
- Gộp theo phần: xong nền + lưới chính thì ghi ngay cau_tra_loi.jsonl, tom_tat.json, HOAN_TAT_CHINH.json (dùng
  được cho bước 5, 6 kể cả khi phần mở rộng chưa xong); xong mức α mở rộng thì ghi cau_tra_loi_mo_rong.jsonl,
  tom_tat_mo_rong.json, HOAN_TAT_MO_RONG.json (để riêng, Nhap 3.3); xong cả hai: sha256.txt và HOAN_TAT.json.
  File tổng hợp đã có mà khác nội dung thì bản cũ được chuyển sang hong/ trước.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from .can_thiep import can_thiep_chinh
from .khuon import chia_lo, ma_hoa
from .mo_hinh import DacTaMoHinh
from .sinh import sinh_tham_lam

PHIEN_BAN_DINH_DANG = 1
NGON_NGU = ("en", "vi")
# Bốn chiều (Nhap, quy ước "chiều X→Y": hướng dựng từ ngôn ngữ X, can thiệp trên câu hỏi ngôn ngữ Y).
CHIEU = (("en", "en"), ("en", "vi"), ("vi", "en"), ("vi", "vi"))
TAP_THEO_KIEU = {"a": "D_plain_test", "b": "D_harm_test"}
LUOI_ALPHA_HIEU_CHINH = (0.1, 0.2, 0.4, 0.8)        # Nhap 3.3
ALPHA_MO_RONG = (0.05, 1.6)                          # Nhap 3.3, luật mở rộng lưới (sinh sẵn: Ân duyệt 27/10/2026)
PHUONG_AN = (1, 3)                                   # Nhap 3.2
MAX_NEW_TOKENS = 256                                 # Nhap 3.5


@dataclass(frozen=True)
class DonVi:
    loai: str                 # "nen" | "can_thiep" | "nen_khong_hook"
    nhom: str                 # "nen" | "luoi_chinh" | "nen_khong_hook" | "mo_rong"
    tap: str                  # "D_plain_test" | "D_harm_test"
    phan: str                 # "hieu_chinh" | "pilot" | "chinh"
    kieu: str                 # "a" | "b"
    ngon_ngu_cau_hoi: str     # "en" | "vi"
    huong: str | None         # ngôn ngữ dựng hướng; None ở lượt nền
    lop: int | None           # lớp gắn hook (lượt nền: lớp gắn hook α = 0; lượt nền không hook: None)
    alpha: float
    phuong_an: int | None     # None ở lượt nền không hook

    @property
    def ma(self) -> str:
        a = f"{self.alpha:g}".replace(".", "p")
        h = self.huong or "0"
        if self.lop is None:                         # lượt nền không gắn hook (chẩn đoán)
            return f"{self.loai}__{self.tap}__{self.phan}__k{self.kieu}__q{self.ngon_ngu_cau_hoi}"
        return (f"{self.loai}__{self.tap}__{self.phan}__k{self.kieu}__q{self.ngon_ngu_cau_hoi}__h{h}"
                f"__l{self.lop:02d}__a{a}__pa{self.phuong_an}")

    @property
    def chieu(self) -> str | None:
        return None if self.huong is None else f"{self.huong}->{self.ngon_ngu_cau_hoi}"


def ke_hoach_hieu_chinh(dac_ta: DacTaMoHinh, luoi: Sequence[float] = LUOI_ALPHA_HIEU_CHINH,
                        mo_rong: Sequence[float] = ALPHA_MO_RONG, phan: str = "hieu_chinh",
                        chan_doan: bool = True) -> list[DonVi]:
    """Kế hoạch lượt hiệu chỉnh (Nhap 3.3, 3.4, 3.2, 3.6) cho một mô hình.

    Thứ tự chạy: lượt nền (4 đơn vị) → lưới chính (theo kiểu, α tăng dần, lớp, phương án, chiều) → lượt nền KHÔNG
    gắn hook (4 đơn vị, chẩn đoán; Ân duyệt Q-B ngày 28/10/2026) → mức mở rộng. Như vậy nếu máy dừng hẳn, phần cần
    nhất (nền và lưới chính) đã có trước.
    """
    lop = tuple(dac_ta.lop_hieu_chinh)
    if len(lop) != 5:
        raise ValueError(f"{dac_ta.ten}: cần 5 lớp đại diện (Nhap 3.4), đặc tả có {lop}.")
    lop_nen = lop[len(lop) // 2]
    kh: list[DonVi] = []
    for kieu, tap in TAP_THEO_KIEU.items():
        for ng in NGON_NGU:
            kh.append(DonVi("nen", "nen", tap, phan, kieu, ng, None, lop_nen, 0.0, 1))
    def luoi_cua(nhom: str, cac_alpha: Sequence[float]) -> None:
        for kieu, tap in TAP_THEO_KIEU.items():
            for alpha in sorted(cac_alpha):
                for l in lop:
                    for pa in PHUONG_AN:
                        for huong, ng in CHIEU:
                            kh.append(DonVi("can_thiep", nhom, tap, phan, kieu, ng, huong, l, float(alpha), pa))
    luoi_cua("luoi_chinh", luoi)
    if chan_doan:
        for kieu, tap in TAP_THEO_KIEU.items():
            for ng in NGON_NGU:
                kh.append(DonVi("nen_khong_hook", "nen_khong_hook", tap, phan, kieu, ng, None, None, 0.0, None))
    luoi_cua("mo_rong", mo_rong)
    ma = [d.ma for d in kh]
    if len(set(ma)) != len(ma):
        raise ValueError("Mã đơn vị bị trùng.")
    return kh


# ─────────────────────────────────────── ghi bền ───────────────────────────────────────
def _fsync_thu_muc(thu_muc: Path) -> None:
    try:
        fd = os.open(str(thu_muc), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def ghi_nguyen_tu(duong: Path, noi_dung: str | bytes) -> None:
    """Ghi qua file tạm cùng thư mục + fsync + os.replace: file đích hoặc là bản cũ, hoặc là bản mới đầy đủ.
    File tạm tên ".<tên>.tam-<pid>"; nếu tiến trình chết trước os.replace thì file tạm còn lại (KhoKetQua.mo xử lý
    trường hợp này cho ke_hoach.json; các nơi khác bỏ qua file ẩn)."""
    duong.parent.mkdir(parents=True, exist_ok=True)
    tam = duong.with_name(f".{duong.name}.tam-{os.getpid()}")
    if isinstance(noi_dung, str):
        noi_dung = noi_dung.encode("utf-8")
    with open(tam, "wb") as f:
        f.write(noi_dung)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tam, duong)
    _fsync_thu_muc(duong.parent)


def sha256_chuoi(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for k in iter(lambda: f.read(1 << 20), b""):
            h.update(k)
    return h.hexdigest()


from .nguon_goc import dau_van_tay_ma_nguon  # noqa: E402,F401  (chuyển sang nguon_goc để bước 3 cũng ghi)


def _bay_gio() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def tach_dong_jsonl(noi_dung: str) -> list[dict[str, Any]]:
    """Đọc JSONL chỉ tách ở "\n". Không dùng str.splitlines(): nó tách cả ở U+2028, U+2029, U+0085 mà json.dumps
    (ensure_ascii=False) để nguyên trong chuỗi, nên một câu trả lời chứa các ký tự này sẽ làm đơn vị không bao giờ
    được tính là xong (phát hiện khi rà soát độc lập code bước 4)."""
    return [json.loads(x) for x in noi_dung.split("\n") if x.strip()]


# Các phần được gộp riêng: nền + lưới chính (đầu vào bước 5, 6) và mức α mở rộng (để riêng, Nhap 3.3).
PHAN_TONG_HOP = {
    "chinh": {"nhom": ("nen", "luoi_chinh"), "file": "cau_tra_loi.jsonl", "tom_tat": "tom_tat.json",
              "hoan_tat": "HOAN_TAT_CHINH.json"},
    "chan_doan": {"nhom": ("nen_khong_hook",), "file": "cau_tra_loi_nen_khong_hook.jsonl",
                  "tom_tat": "tom_tat_nen_khong_hook.json", "hoan_tat": "HOAN_TAT_NEN_KHONG_HOOK.json"},
    "mo_rong": {"nhom": ("mo_rong",), "file": "cau_tra_loi_mo_rong.jsonl", "tom_tat": "tom_tat_mo_rong.json",
                "hoan_tat": "HOAN_TAT_MO_RONG.json"},
}


def bam_moi_truong(moi_truong: dict[str, Any]) -> str:
    return sha256_chuoi(json.dumps(moi_truong, sort_keys=True, ensure_ascii=False))


def phan_cua(dv: "DonVi") -> str:
    return {"mo_rong": "mo_rong", "nen_khong_hook": "chan_doan"}.get(dv.nhom, "chinh")


class DangChay(RuntimeError):
    """Một tiến trình khác đang chạy cùng thư mục lượt."""


class KhoKetQua:
    """Thư mục của một lượt: ke_hoach.json, don_vi/, hong/, phien.jsonl, moi_truong_goc.json, các file tổng hợp."""

    def __init__(self, thu_muc: str | Path):
        self.goc = Path(thu_muc)
        self.don_vi = self.goc / "don_vi"
        self.hong = self.goc / "hong"
        self.file_ke_hoach = self.goc / "ke_hoach.json"
        self.file_hoan_tat = self.goc / "HOAN_TAT.json"
        self.file_moi_truong = self.goc / "moi_truong_goc.json"
        self._khoa_fd: int | None = None
        self.canh_bao_khoa: str | None = None

    def jsonl(self, dv: DonVi) -> Path:
        return self.don_vi / f"{dv.ma}.jsonl"

    def xong(self, dv: DonVi) -> Path:
        return self.don_vi / f"{dv.ma}.xong.json"

    # ── khoá: không cho hai tiến trình chạy cùng một lượt ──
    def khoa(self) -> None:
        """Khoá độc quyền (fcntl.flock) trên file .<tên lượt>.khoa cạnh thư mục lượt. Đã có tiến trình khác giữ
        khoá → DangChay. Khoá tự nhả khi tiến trình kết thúc (kể cả khi bị giết)."""
        import fcntl

        self.goc.parent.mkdir(parents=True, exist_ok=True)
        p = self.goc.parent / f".{self.goc.name}.khoa"
        fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise DangChay(f"Một tiến trình khác đang chạy lượt {self.goc} (khoá {p}). Kiểm tmux ls / nvidia-smi.")
        except OSError as e:
            # Hệ file không hỗ trợ flock (vd. /mnt/d của WSL khi gộp trên máy cá nhân; rà soát bước 4, R1 C1):
            # không phải dấu hiệu có tiến trình khác. Không khoá, có cảnh báo; script CHỈ cho chạy không khoá ở nhánh
            # gộp (mọi đơn vị đã xong), còn cần sinh thì dừng mã 9.
            os.close(fd)
            self.canh_bao_khoa = f"hệ file không hỗ trợ khoá ({type(e).__name__}: {e}); chạy không khoá"
            return
        self._khoa_fd = fd

    def nha_khoa(self) -> None:
        if self._khoa_fd is not None:
            os.close(self._khoa_fd)
            self._khoa_fd = None

    def chuyen_vao_hong(self, p: Path, nhan: str) -> str:
        """Chuyển một file sang hong/ (không xoá, không đè file đã có trong hong/). Trả về tên mới."""
        self.hong.mkdir(parents=True, exist_ok=True)
        goc_ten = f"{p.name}.{_bay_gio()}.{nhan}"
        dich, n = self.hong / goc_ten, 0
        while dich.exists():
            n += 1
            dich = self.hong / f"{goc_ten}.{n}"
        os.replace(p, dich)
        _fsync_thu_muc(p.parent)
        return dich.name

    def sao_vao_hong(self, p: Path, nhan: str) -> str:
        """Sao (không chuyển) một file sang hong/; dùng trước khi ghi đè nguyên tử file đó, để không có lúc nào
        thiếu file."""
        import shutil

        self.hong.mkdir(parents=True, exist_ok=True)
        goc_ten = f"{p.name}.{_bay_gio()}.{nhan}"
        dich, n = self.hong / goc_ten, 0
        while dich.exists():
            n += 1
            dich = self.hong / f"{goc_ten}.{n}"
        shutil.copy2(p, dich)
        with open(dich, "rb") as f:
            os.fsync(f.fileno())
        _fsync_thu_muc(self.hong)
        return dich.name

    def co_don_vi_nao(self) -> bool:
        return self.don_vi.exists() and any(x.suffix in (".jsonl", ".json") for x in self.don_vi.iterdir())

    # ── kế hoạch ──
    def mo(self, ke_hoach: dict[str, Any], cho_phep_doi_ma_nguon: bool = False) -> str:
        """Tạo lượt mới, hoặc mở lại lượt cũ nếu dấu vân tay khớp. Trả về "moi", "tiep" hoặc "lam_lai_ke_hoach".

        Dấu vân tay khác: nếu CHƯA có đơn vị nào (lượt trước dừng ở bước kiểm, chưa sinh gì) thì chuyển kế hoạch cũ
        sang hong/ và tạo kế hoạch mới ("lam_lai_ke_hoach"); nếu đã có đơn vị thì ValueError, không đụng file nào."""
        if self.file_ke_hoach.exists():
            try:
                cu = json.loads(self.file_ke_hoach.read_text(encoding="utf-8"))
            except (OSError, ValueError):          # ValueError gồm JSONDecodeError và UnicodeDecodeError
                cu = None
            if not isinstance(cu, dict) or not isinstance(cu.get("dau_van_tay"), dict):
                cu = {"dau_van_tay": {"_ke_hoach_hong": True}}
            khac = [k for k in ke_hoach["dau_van_tay"] if cu.get("dau_van_tay", {}).get(k) != ke_hoach["dau_van_tay"][k]]
            if "ma_nguon" in khac and cho_phep_doi_ma_nguon:
                khac.remove("ma_nguon")
            if not khac:
                return "tiep"
            if self.co_don_vi_nao():
                if cu["dau_van_tay"].get("_ke_hoach_hong"):
                    raise ValueError(f"ke_hoach.json trong {self.goc} bị hỏng (không đọc được) mà đã có đơn vị sinh "
                                     f"xong. Không ghi đè; dừng lại, chép thư mục về và hỏi trước khi làm gì tiếp.")
                raise ValueError(f"Thư mục {self.goc} đã có một lượt KHÁC (khác ở: {khac}) và đã có đơn vị sinh xong. "
                                 f"Không ghi đè; dùng thư mục khác, hoặc chạy lại đúng cấu hình cũ.")
            # Sao bản cũ sang hong/ rồi ghi đè nguyên tử: không có lúc nào thư mục thiếu ke_hoach.json.
            self.sao_vao_hong(self.file_ke_hoach, "ke_hoach_cu_chua_sinh")
            ghi_nguyen_tu(self.file_ke_hoach, json.dumps(ke_hoach, ensure_ascii=False, indent=2) + "\n")
            if self.file_moi_truong.exists():
                self.chuyen_vao_hong(self.file_moi_truong, "ke_hoach_cu_chua_sinh")
            return "lam_lai_ke_hoach"
        if self.goc.exists():
            # Chết sau khi ghi file tạm của kế hoạch đầu tiên, trước os.replace (rà soát bước 4, R2 C1): chỉ khi thư
            # mục KHÔNG có gì khác ngoài file tạm đúng mẫu (và hong/) mới chuyển file tạm sang hong/ rồi lập kế hoạch.
            co = list(self.goc.iterdir())
            tam = [x for x in co if x.is_file() and x.name.startswith(f".{self.file_ke_hoach.name}.tam-")]
            khac_tam = [x.name for x in co if x not in tam and x.name != self.hong.name]
            if khac_tam:
                raise ValueError(f"Thư mục {self.goc} đã có file nhưng không có ke_hoach.json: không ghi vào "
                                 f"({sorted(khac_tam)[:5]}).")
            for x in tam:
                self.chuyen_vao_hong(x, "ke_hoach_tam_do_dang")
        self.goc.mkdir(parents=True, exist_ok=True)
        ghi_nguyen_tu(self.file_ke_hoach, json.dumps(ke_hoach, ensure_ascii=False, indent=2) + "\n")
        return "moi"

    def ghi_phien(self, thong_tin: dict[str, Any]) -> int:
        """Thêm một dòng vào phien.jsonl (mỗi phiên một dòng; không xoá dòng cũ). Trả về số thứ tự phiên (từ 1)."""
        p = self.goc / "phien.jsonl"
        cu = p.read_bytes() if p.exists() else b""          # theo byte: một dòng hỏng không chặn phiên mới
        if cu and not cu.endswith(b"\n"):
            cu += b"\n"
        # Số phiên lớn hơn mọi số đã dùng, kể cả số ghi trong .xong.json (phien.jsonl có thể cũ hơn các đơn vị, vd.
        # khi đẩy một bản kéo về giữa chừng lên máy mới; rà lại pipeline 28/10/2026).
        # Lấy cả số "phien" ghi trong từng dòng (phien.jsonl có thể còn dòng có số lớn hơn số dòng, vd. sau khi ghép
        # bản kéo về; đối chiếu lần cuối, R2 F4, 28/10/2026). Dòng hỏng chỉ được đếm, không chặn.
        dong_cu = [x for x in cu.split(b"\n") if x.strip()]
        da_dung = len(dong_cu)
        for x in dong_cu:
            try:
                d = json.loads(x.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if isinstance(d, dict) and isinstance(d.get("phien"), int) and not isinstance(d.get("phien"), bool):
                da_dung = max(da_dung, d["phien"])
        if self.don_vi.exists():
            for x in self.don_vi.glob("*.xong.json"):
                try:
                    d = json.loads(x.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(d, dict) and isinstance(d.get("phien"), int):
                    da_dung = max(da_dung, d["phien"])
        so = da_dung + 1
        ghi_nguyen_tu(p, cu + (json.dumps({"phien": so, **thong_tin}, ensure_ascii=False) + "\n").encode("utf-8"))
        return so

    def kiem_moi_truong(self, moi_truong: dict[str, Any], cho_phep_doi: bool = False) -> tuple[bool, list[str]]:
        """So môi trường của phiên này với phiên đầu tiên đã sinh đơn vị (moi_truong_goc.json).

        Chưa có file gốc → ghi (phiên này thành gốc). Có mà khác: nếu chưa có đơn vị nào thì thay file gốc (bản cũ
        sang hong/); nếu đã có đơn vị thì không đạt, trừ khi cho_phep_doi (được ghi lại ở phien.jsonl).
        Lý do: lượt nền và lượt can thiệp được so theo cặp trên từng câu, nên phải cùng môi trường số học."""
        if self.file_moi_truong.exists():
            try:
                goc = json.loads(self.file_moi_truong.read_text(encoding="utf-8"))
                if not isinstance(goc, dict):
                    raise ValueError("moi_truong_goc.json không phải object")
            except (OSError, ValueError):
                # File gốc hỏng: chuyển sang hong/ rồi xử lý như mất file gốc (nhánh dưới).
                self.chuyen_vao_hong(self.file_moi_truong, "moi_truong_goc_hong")
                return self.kiem_moi_truong(moi_truong, cho_phep_doi)
            khac = sorted(k for k in set(goc) | set(moi_truong) if goc.get(k) != moi_truong.get(k))
            if not khac:
                return True, []
            if self.co_don_vi_nao():
                return cho_phep_doi, khac
            self.sao_vao_hong(self.file_moi_truong, "moi_truong_cu_chua_sinh")
        elif self.co_don_vi_nao():
            # Mất moi_truong_goc.json mà đã có đơn vị (vd. chỉ chép don_vi/ sang máy mới): so với dấu vân tay môi
            # trường đã ghi trong .xong.json của các đơn vị cũ.
            bam = bam_moi_truong(moi_truong)
            cu = set()
            for x in self.don_vi.glob("*.xong.json"):
                try:
                    d = json.loads(x.read_text(encoding="utf-8"))
                    cu.add(d.get("moi_truong_so_hoc_sha256") if isinstance(d, dict) else None)
                except (OSError, ValueError):
                    cu.add(None)
            if cu != {bam}:
                return cho_phep_doi, ["thieu_moi_truong_goc_va_khac_don_vi_cu"]
        ghi_nguyen_tu(self.file_moi_truong, json.dumps(moi_truong, ensure_ascii=False, indent=2) + "\n")
        return True, []

    # ── đơn vị ──
    def kiem_don_vi(self, dv: DonVi, id_cau: Sequence[str]) -> tuple[bool, str]:
        """(xong, lý do). Xong khi có cả .jsonl và .xong.json, SHA khớp, đủ dòng, đúng id theo thứ tự."""
        j, x = self.jsonl(dv), self.xong(dv)
        if not j.exists() and not x.exists():
            return False, "chua_co"
        if not j.exists() or not x.exists():
            return False, "thieu_mot_trong_hai_file"
        try:
            meta = json.loads(x.read_text(encoding="utf-8"))
        except (OSError, ValueError):       # gồm JSONDecodeError và UnicodeDecodeError (rà soát bước 4, R2 C2)
            return False, "xong_json_hong"
        if not isinstance(meta, dict):
            return False, "xong_json_hong"
        if sha256_file(j) != meta.get("sha256"):
            return False, "sai_sha256"
        try:
            dong = tach_dong_jsonl(j.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, "jsonl_hong"
        if [d.get("id") for d in dong] != list(id_cau) or meta.get("so_dong") != len(dong):
            return False, "sai_id_hoac_so_dong"
        if any(d.get("ma_don_vi") != dv.ma for d in dong):
            return False, "sai_ma_don_vi"
        return True, "xong"

    def chuyen_hong(self, dv: DonVi, ly_do: str) -> list[str]:
        """Chuyển file của một đơn vị chưa hợp lệ sang hong/ (không xoá). Trả về tên file đã chuyển."""
        return [self.chuyen_vao_hong(p, ly_do) for p in (self.jsonl(dv), self.xong(dv)) if p.exists()]

    def ghi_don_vi(self, dv: DonVi, dong: list[dict[str, Any]], meta: dict[str, Any]) -> None:
        noi_dung = "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dong)
        ghi_nguyen_tu(self.jsonl(dv), noi_dung)
        meta = {**meta, "sha256": sha256_chuoi(noi_dung), "so_dong": len(dong), "ma_don_vi": dv.ma}
        ghi_nguyen_tu(self.xong(dv), json.dumps(meta, ensure_ascii=False, indent=2) + "\n")


# ─────────────────────────────────────── sinh ───────────────────────────────────────
def sinh_don_vi(model, tok, dac_ta: DacTaMoHinh, dv: DonVi, cau: Sequence[dict[str, Any]],
                huong: dict[str, torch.Tensor], s: torch.Tensor, eos: Sequence[int], max_new_tokens: int,
                co_lo: int) -> list[dict[str, Any]]:
    """Sinh cho mọi câu của một đơn vị, theo lô cỡ co_lo (giữ thứ tự). Trả về các dòng JSONL.

    Lượt nền (huong None, α = 0): vector 0, lấy theo hướng tiếng Anh cho có hình dạng. Dòng của lượt nền ghi
    lop / phuong_an là None; lớp và phương án của hook α = 0 nằm ở lop_hook / phuong_an_hook (Ân duyệt Q-C ngày
    28/10/2026), để phép nhóm theo lớp ở bước 6 không lẫn dòng nền vào một ô. Lượt nền không hook: không gắn hook
    nào (chẩn đoán, Q-B)."""
    khong_hook = dv.loai == "nen_khong_hook"
    if not khong_hook:
        r_hat = huong["en" if dv.huong is None else dv.huong][dv.lop].to(torch.float32)
        s_l = float(s[dv.lop])
    truong = asdict(dv)
    truong["lop_hook"], truong["phuong_an_hook"] = dv.lop, dv.phuong_an
    if dv.loai != "can_thiep":
        truong["lop"], truong["phuong_an"] = None, None
    ra: list[dict[str, Any]] = []
    for a, b in chia_lo(len(cau), co_lo):
        phan = cau[a:b]
        enc = ma_hoa(tok, [c[dv.ngon_ngu_cau_hoi] for c in phan], dac_ta)
        ct = [] if khong_hook else [can_thiep_chinh(dv.lop, r_hat, s_l, dv.alpha, dv.kieu, dv.phuong_an,
                                                     enc.so_token_khoi)]
        kq = sinh_tham_lam(model, tok, enc, max_new_tokens, eos, can_thiep=ct, giai_ma="den_eos")
        for i, c in enumerate(phan):
            token = kq.token_hop_le(i)
            ra.append({"ma_don_vi": dv.ma, "id": c["id"], "mo_hinh": dac_ta.ten, **truong, "chieu": dv.chieu,
                       "nhan_doi_chieu": c.get("nhan_doi_chieu"),
                       "cau_hoi": c[dv.ngon_ngu_cau_hoi], "van_ban": kq.van_ban[i], "token": token,
                       "so_token": kq.so_hop_le[i], "cham_eos": bool(kq.cham_eos[i]),
                       "bi_cat": not bool(kq.cham_eos[i]),
                       "token_ket_thuc": (int(token[-1]) if kq.cham_eos[i] else None),
                       "so_token_khoi": enc.so_token_khoi, "vi_tri_trong_lo": i, "lo": [a, b]})
    return ra


def kiem_ke_hoach(kho: KhoKetQua, ke_hoach: Sequence[DonVi], cau_theo_tap: dict[str, list[dict[str, Any]]]
                  ) -> tuple[list[DonVi], list[tuple[DonVi, str]]]:
    """(đơn vị đã xong, [(đơn vị chưa xong, lý do)])."""
    xong, chua = [], []
    for dv in ke_hoach:
        ok, ly_do = kho.kiem_don_vi(dv, [c["id"] for c in cau_theo_tap[dv.tap]])
        (xong.append(dv) if ok else chua.append((dv, ly_do)))
    return xong, chua


def chay_ke_hoach(model, tok, dac_ta: DacTaMoHinh, ke_hoach: Sequence[DonVi],
                  cau_theo_tap: dict[str, list[dict[str, Any]]], huong: dict[str, torch.Tensor], s: torch.Tensor,
                  eos: Sequence[int], kho: KhoKetQua, max_new_tokens: int = MAX_NEW_TOKENS, co_lo: int = 40,
                  log: Callable[[str], None] = print, so_don_vi_toi_da: int | None = None,
                  meta_phien: dict[str, Any] | None = None) -> dict[str, Any]:
    """Chạy mọi đơn vị chưa xong theo thứ tự kế hoạch. Đơn vị xong từ trước được giữ nguyên (không sinh lại).

    so_don_vi_toi_da: dừng sau khi sinh từng ấy đơn vị trong lần gọi này (None = chạy hết).
    meta_phien: ghi thêm vào .xong.json của mỗi đơn vị (số phiên, dấu vân tay môi trường).
    Đơn vị mức mở rộng: nhật ký không in số câu bị cắt hay thời gian (Nhap 3.3: để riêng, không đọc trước)."""
    xong, chua = kiem_ke_hoach(kho, ke_hoach, cau_theo_tap)
    hong = []
    for dv, ly_do in chua:
        if ly_do != "chua_co":
            hong.append({"ma": dv.ma, "ly_do": ly_do, "chuyen_sang": kho.chuyen_hong(dv, ly_do)})
            log(f"Đơn vị {dv.ma} chưa hợp lệ ({ly_do}): chuyển file sang hong/, sinh lại.")
    can_lam = [dv for dv, _ in chua]
    log(f"{len(ke_hoach)} đơn vị: đã xong {len(xong)}, cần sinh {len(can_lam)}."
        + (" Đơn vị mức mở rộng không ghi nhật ký từng đơn vị; xem tiến độ bằng cách đếm file *.xong.json trong don_vi/."
           if any(dv.nhom == "mo_rong" for dv in can_lam) else ""))
    bat_dau = time.time()
    da_lam = 0
    for k, dv in enumerate(can_lam):
        if so_don_vi_toi_da is not None and da_lam >= so_don_vi_toi_da:
            break
        t0 = time.time()
        dong = sinh_don_vi(model, tok, dac_ta, dv, cau_theo_tap[dv.tap], huong, s, eos, max_new_tokens, co_lo)
        dt = time.time() - t0
        kho.ghi_don_vi(dv, dong, {"thoi_gian_giay": dt, "ket_thuc": _bay_gio(),
                                  "dtype": str(next(model.parameters()).dtype), **(meta_phien or {})})
        da_lam += 1
        con = len(can_lam) - (k + 1)
        tb = (time.time() - bat_dau) / da_lam
        if dv.nhom == "mo_rong":
            # Không ghi gì theo từng đơn vị mức mở rộng: số câu bị cắt, thời gian, kể cả dấu giờ của dòng nhật ký, đều
            # lộ độ dài câu trả lời (Nhap 3.3: để riêng, không đọc trước; rà độc lập 28/10/2026). Thời gian vẫn ghi
            # trong .xong.json. Xem tiến độ: đếm file trong don_vi/.
            pass
        else:
            log(f"[{len(xong) + k + 1}/{len(ke_hoach)}] {dv.ma}: {len(dong)} câu, {dt:.1f} giây, bị cắt "
                f"{sum(d['bi_cat'] for d in dong)}; còn {con} đơn vị, ước {con * tb / 60:.0f} phút.")
    return {"tong": len(ke_hoach), "da_co_tu_truoc": len(xong), "sinh_trong_phien": da_lam, "hong": hong,
            "con_lai": len(can_lam) - da_lam}


def _ghi_neu_khac(kho: KhoKetQua, p: Path, noi_dung: str, nhan: str) -> None:
    """Ghi file tổng hợp; đã có mà khác nội dung thì chuyển bản cũ sang hong/ trước (không ghi đè)."""
    if p.exists():
        if sha256_file(p) == sha256_chuoi(noi_dung):
            return
        kho.chuyen_vao_hong(p, nhan)
    ghi_nguyen_tu(p, noi_dung)


def tong_hop_phan(kho: KhoKetQua, ke_hoach: Sequence[DonVi], cau_theo_tap: dict[str, list[dict[str, Any]]],
                  phan: str, dau_hieu_thinking: Sequence[str] = ()) -> dict[str, Any] | None:
    """Gộp một phần ("chinh" hoặc "mo_rong") khi mọi đơn vị của phần đó đã xong. Phần không có đơn vị → None.
    Ghi <file gộp>, <tom_tat> và <HOAN_TAT_...> của phần; file đã có cùng nội dung thì để nguyên."""
    cau_hinh = PHAN_TONG_HOP[phan]
    ds = [dv for dv in ke_hoach if dv.nhom in cau_hinh["nhom"]]
    if not ds:
        return None
    _, chua = kiem_ke_hoach(kho, ds, cau_theo_tap)
    if chua:
        raise RuntimeError(f"Phần {phan}: còn {len(chua)} đơn vị chưa xong (ví dụ {chua[0][0].ma}).")
    gop: list[str] = []
    tom: dict[str, Any] = {"phan": phan, "so_don_vi": len(ds), "theo_nhom": {}, "theo_don_vi": {}}
    for dv in ds:
        noi = kho.jsonl(dv).read_text(encoding="utf-8")
        gop.append(noi)
        dong = tach_dong_jsonl(noi)
        ro = sum(any(d in r["van_ban"] for d in dau_hieu_thinking) for r in dong) if dau_hieu_thinking else None
        kt: dict[str, int] = {}
        for r in dong:
            kt[str(r["token_ket_thuc"])] = kt.get(str(r["token_ket_thuc"]), 0) + 1
        tom["theo_don_vi"][dv.ma] = {"so_cau": len(dong), "bi_cat": sum(r["bi_cat"] for r in dong),
                                    "ro_thinking": ro, "token_ket_thuc": kt}
        n = tom["theo_nhom"].setdefault(dv.nhom, {"so_don_vi": 0, "so_cau": 0, "bi_cat": 0, "ro_thinking": 0})
        n["so_don_vi"] += 1
        n["so_cau"] += len(dong)
        n["bi_cat"] += sum(r["bi_cat"] for r in dong)
        n["ro_thinking"] += ro or 0
    tom["so_cau_tra_loi"] = sum(v["so_cau"] for v in tom["theo_don_vi"].values())
    if phan == "chan_doan":
        tom["so_khop_voi_nen_co_hook"] = so_khop_nen_khong_hook(kho, ke_hoach, ds)
    noi_gop = "".join(gop)
    _ghi_neu_khac(kho, kho.goc / cau_hinh["file"], noi_gop, "khac_ban_gop_moi")
    _ghi_neu_khac(kho, kho.goc / cau_hinh["tom_tat"], json.dumps(tom, ensure_ascii=False, indent=2) + "\n",
                  "khac_ban_moi")
    ht = {"phan": phan, "so_don_vi": len(ds), "so_cau_tra_loi": tom["so_cau_tra_loi"], "file": cau_hinh["file"],
          "sha256": sha256_chuoi(noi_gop)}
    _ghi_neu_khac(kho, kho.goc / cau_hinh["hoan_tat"], json.dumps(ht, ensure_ascii=False, indent=2) + "\n",
                  "khac_ban_moi")
    return {**ht, "tom_tat": tom}


def so_khop_nen_khong_hook(kho: KhoKetQua, ke_hoach: Sequence[DonVi], ds: Sequence[DonVi]) -> dict[str, Any]:
    """So từng câu của lượt nền không hook với lượt nền có hook α = 0 cùng tập, kiểu, ngôn ngữ câu hỏi: dãy token
    phải giống hệt (Nhap 3.6: "đầu ra α = 0 giống hệt lượt không gắn hook"; ở đây chỉ để chẩn đoán sớm, ở đúng cỡ lô
    và 256 token; phép kiểm chính thức vẫn ở pilot)."""
    ra: dict[str, Any] = {"tong_cau": 0, "giong_het": 0, "theo_don_vi": {}}
    for dv in ds:
        nen = [x for x in ke_hoach if x.loai == "nen" and x.tap == dv.tap and x.kieu == dv.kieu
               and x.ngon_ngu_cau_hoi == dv.ngon_ngu_cau_hoi]
        if len(nen) != 1:
            ra["theo_don_vi"][dv.ma] = {"loi": "không tìm thấy đúng một đơn vị nền tương ứng"}
            continue
        a = {r["id"]: r["token"] for r in tach_dong_jsonl(kho.jsonl(nen[0]).read_text(encoding="utf-8"))}
        b = {r["id"]: r["token"] for r in tach_dong_jsonl(kho.jsonl(dv).read_text(encoding="utf-8"))}
        khac = sorted(i for i in b if a.get(i) != b[i])
        ra["theo_don_vi"][dv.ma] = {"don_vi_nen": nen[0].ma, "so_cau": len(b), "giong_het": len(b) - len(khac),
                                    "id_khac": khac}
        ra["tong_cau"] += len(b)
        ra["giong_het"] += len(b) - len(khac)
    ra["tat_ca_giong_het"] = ra["tong_cau"] > 0 and ra["giong_het"] == ra["tong_cau"]
    return ra


def _bang_ke_cuoi(kho: KhoKetQua, ke_hoach: Sequence[DonVi]) -> tuple[dict[str, Any], str]:
    """Tính (nội dung HOAN_TAT.json, nội dung sha256.txt) từ các file đang có, không ghi gì.
    sha256.txt liệt kê mọi file trừ nhat_ky.txt, sha256.txt, HOAN_TAT.json, file ẩn và hong/."""
    ht: dict[str, Any] = {"hoan_tat": True, "so_don_vi": len(ke_hoach), "phan": {}}
    for phan, ch in PHAN_TONG_HOP.items():
        if any(dv.nhom in ch["nhom"] for dv in ke_hoach):
            p = kho.goc / ch["hoan_tat"]
            if not p.exists():
                raise RuntimeError(f"Chưa gộp phần {phan}.")
            ht["phan"][phan] = json.loads(p.read_text(encoding="utf-8"))
    ht["so_cau_tra_loi"] = sum(v["so_cau_tra_loi"] for v in ht["phan"].values())
    tep = sorted(p for p in kho.goc.rglob("*") if p.is_file() and "hong" not in p.relative_to(kho.goc).parts
                 and p.name not in ("sha256.txt", "HOAN_TAT.json", "nhat_ky.txt") and not p.name.startswith("."))
    dong_sha = "".join(f"{sha256_file(p)}  {p.relative_to(kho.goc).as_posix()}\n" for p in tep)
    ht["sha256_txt_sha256"] = sha256_chuoi(dong_sha)
    return ht, dong_sha


def _noi_dung_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def tong_hop_cuoi(kho: KhoKetQua, ke_hoach: Sequence[DonVi]) -> dict[str, Any]:
    """Khi mọi phần đã gộp: sha256.txt rồi HOAN_TAT.json (HOAN_TAT.json ghi cuối cùng)."""
    ht, dong_sha = _bang_ke_cuoi(kho, ke_hoach)
    _ghi_neu_khac(kho, kho.goc / "sha256.txt", dong_sha, "khac_ban_moi")
    _ghi_neu_khac(kho, kho.file_hoan_tat, _noi_dung_json(ht), "khac_ban_moi")
    return ht


def doi_sau_hoan_tat(kho: KhoKetQua) -> list[str]:
    """Lượt từng hoàn tất (có HOAN_TAT.json) mà nay không còn nguyên: so sha256.txt cũ với các file đang có, trả về
    danh sách thay đổi ("khác: x", "thêm: y", "mất: z"; hoặc "sha256.txt không đọc được") để ghi CHÚ Ý trước khi gộp
    lại. Chỉ để báo, không quyết định gì."""
    p = kho.goc / "sha256.txt"
    try:
        cu = {}
        for dong in p.read_text(encoding="utf-8").split("\n"):
            if dong.strip():
                h, ten = dong.split("  ", 1)
                cu[ten] = h
    except (OSError, ValueError):
        return ["sha256.txt không đọc được"]
    nay = {q.relative_to(kho.goc).as_posix(): q for q in kho.goc.rglob("*")
           if q.is_file() and "hong" not in q.relative_to(kho.goc).parts
           and q.name not in ("sha256.txt", "HOAN_TAT.json", "nhat_ky.txt") and not q.name.startswith(".")}
    ra = [f"mất: {t}" for t in sorted(set(cu) - set(nay))]
    ra += [f"thêm: {t}" for t in sorted(set(nay) - set(cu))]
    ra += [f"khác: {t}" for t in sorted(set(cu) & set(nay)) if sha256_file(nay[t]) != cu[t]]
    return ra


def da_hoan_tat(kho: KhoKetQua, ke_hoach: Sequence[DonVi], cau_theo_tap: dict[str, list[dict[str, Any]]]) -> bool:
    """Lượt đã hoàn tất và còn nguyên: mọi đơn vị hợp lệ; mỗi file gộp đúng bằng ghép các đơn vị của phần và khớp
    SHA trong HOAN_TAT của phần; sha256.txt và HOAN_TAT.json trùng từng byte với bản tính lại từ các file đang có
    (nên HOAN_TAT.json hỏng, sha256.txt rỗng hay thiếu dòng, một file đổi sau khi xong đều là chưa hoàn tất → gộp lại
    từ đơn vị, bản cũ sang hong/; đối chiếu lần cuối, R2 F5, 28/10/2026)."""
    try:
        return _da_hoan_tat(kho, ke_hoach, cau_theo_tap)
    except (OSError, ValueError, AttributeError, TypeError, KeyError, RuntimeError):
        return False                       # một file tổng hợp hỏng: chưa hoàn tất, gộp lại từ các đơn vị


def _da_hoan_tat(kho: KhoKetQua, ke_hoach: Sequence[DonVi], cau_theo_tap: dict[str, list[dict[str, Any]]]) -> bool:
    if not kho.file_hoan_tat.exists() or kiem_ke_hoach(kho, ke_hoach, cau_theo_tap)[1]:
        return False
    for phan, ch in PHAN_TONG_HOP.items():
        ds = [dv for dv in ke_hoach if dv.nhom in ch["nhom"]]
        if not ds:
            continue
        p_ht, p_gop = kho.goc / ch["hoan_tat"], kho.goc / ch["file"]
        if not p_ht.exists() or not p_gop.exists():
            return False
        ht_phan = json.loads(p_ht.read_text(encoding="utf-8"))
        sha_gop = sha256_file(p_gop)
        sha_tu_don_vi = sha256_chuoi("".join(kho.jsonl(dv).read_text(encoding="utf-8") for dv in ds))
        if not isinstance(ht_phan, dict) or ht_phan.get("sha256") != sha_gop or sha_gop != sha_tu_don_vi \
                or ht_phan.get("so_don_vi") != len(ds) or ht_phan.get("file") != ch["file"]:
            return False
    ht, dong_sha = _bang_ke_cuoi(kho, ke_hoach)
    p_sha = kho.goc / "sha256.txt"
    return (p_sha.exists() and p_sha.read_bytes() == dong_sha.encode("utf-8")
            and kho.file_hoan_tat.read_bytes() == _noi_dung_json(ht).encode("utf-8"))
