# Mã của tác giả dùng làm đối chứng trong test

`hook_utils.py` chép nguyên văn từ kho andyrdt/refusal_direction, commit 9d852fae,
`pipeline/utils/hook_utils.py` (Apache License 2.0, xem https://github.com/andyrdt/refusal_direction/blob/9d852fae/LICENSE).
Chỉ dùng trong `tests/test_doi_chieu_tac_gia.py` để kiểm hai phép can thiệp của phần 1 cho kết quả trùng từng bit
với hook của tác giả. Không dùng trong code chạy thật.
