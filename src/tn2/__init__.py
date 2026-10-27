"""tn2 — code của thí nghiệm 2 (can thiệp bằng hướng từ chối theo từng lớp, tiếng Anh và tiếng Việt).

Thiết kế chuẩn: doc_thi_nghiem_2/Nhap.docx (bản khoá 27/10/2026). Bản trình bày gọn:
doc_thi_nghiem_2/Thiet_ke_thi_nghiem_27_10_2026.md. Sơ đồ module ↔ mục thiết kế: docs/CAU_TRUC_CODE.md.

Các module dùng chung cho mọi bước (theo nguyên tắc viết code ở mục "Kiểm pipeline" của Nhap:
phần tái lập Arditi et al. gọi đúng các hàm của pipeline chính cho template, lấy kích hoạt,
tính hướng, hook và sinh):

- mo_hinh   : đặc tả mô hình, nạp mô hình, tìm danh sách lớp.
- khuon     : áp chat template, mã hoá, xác định khối template sau câu hỏi và vị trí lấy kích hoạt.
- kich_hoat : lấy trung bình kích hoạt (và độ lớn trung bình) ở các lớp, các vị trí.
- huong     : hiệu trung bình, chuẩn hoá, cosine, chia đôi.
- can_thiep : hook can thiệp (cộng vector / chiếu bỏ) theo lớp, điểm đặt và vùng token.
- sinh      : sinh greedy theo lô, có hoặc không có can thiệp.
- diem      : điểm từ chối theo logit, KL, so khớp chuỗi.
- kiem_hook : phần 2 của kiểm pipeline (ba phép kiểm đường can thiệp + kiểm âm).
- buoc1     : phần 1 của kiểm pipeline (tái lập Arditi et al. 2024 trên Llama-3-8B-Instruct).
"""

__version__ = "0.1.0"
