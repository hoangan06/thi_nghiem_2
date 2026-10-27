"""Bước 1 của trình tự chạy (Nhap): kiểm pipeline trên Llama-3-8B-Instruct.

- tai_lap_arditi: phần 1, tái lập Arditi et al. (2024) — chỉ phần riêng của bài (lọc, chọn hướng theo KL và
  điểm từ chối, so khớp chuỗi, bốn tiêu chí); mọi khâu dùng chung gọi pipeline chính trong tn2.
- Phần 2 (kiểm đường can thiệp) nằm ở tn2.kiem_hook vì bước 3 dùng lại cho Qwen3.5-9B và Gemma 4 12B.
"""
