# Nguồn dữ liệu

## `data/arditi/` — dữ liệu của Arditi et al. (2024)

Nguồn: kho andyrdt/refusal_direction, commit 9d852fae (Apache License 2.0). Rút đúng như code của tác giả
(`dataset/load_dataset.py`, `pipeline/run_pipeline.py`): `random.seed(42)`, rồi lần lượt 128 câu có hại và 128 câu
vô hại để dựng hướng, 32 + 32 câu kiểm định, 100 câu Alpaca vô hại (harmless_test) để chấm; JailbreakBench lấy
nguyên 100 câu. Đã kiểm (26–27/10): rút lại theo code tác giả cho đúng từng câu và đúng thứ tự; 100 câu
harmless_test trùng các câu trong `harmless_baseline_completions.json` tác giả công bố.

Mỗi dòng là một JSON có trường `instruction` (và các trường gốc khác nếu có).

| File | Số câu | SHA-256 |
|---|---|---|
| harmful_train.jsonl | 128 | 9014d764bbfb030b51755c42343393a1bcd1338794d4cbbacba8d454c0b21d6e |
| harmless_train.jsonl | 128 | d9aeb4d2d028a225a36d64e16d8838cafaabd3bb6e8a0abc1aac26359c3b2547 |
| harmful_val.jsonl | 32 | c1fa41c627300338fc527c2f6efd76bed8dcba7090bc703c1e62802a4b397024 |
| harmless_val.jsonl | 32 | ba03a7275c91473c637cc30b5f47bf1af509b5e6205ac37ade63b26436499d18 |
| harmless_test.jsonl | 100 | 91cba9c6b4f49369aa26af544cad0a543f8ba1be4e82c5d0803ee2d921051d6b |
| jailbreakbench.jsonl | 100 | 2b9e604c7e605162c8cd59f852406342c113c0cb672e96becd96e67ce4090d4c |

## `data/arditi/tac_gia_llama3/` — sản phẩm chạy của tác giả cho Llama-3-8B-Instruct

Chép từ `pipeline/runs/meta-llama-3-8b-instruct/` của cùng commit. Chỉ dùng để chẩn đoán (Nhap: các sản phẩm
này được tạo bằng khuôn chat lỗi, nên không đặt ngưỡng).

| File | Nguồn trong kho tác giả | SHA-256 |
|---|---|---|
| direction_metadata.json | `direction_metadata.json` ({"pos": −5, "layer": 12}) | d130c5240eceabe4dc0135e373f7fac51fe85082e74cc03eca87800842c7128e |
| direction_evaluations.json | `select_direction/direction_evaluations.json` (160 ứng viên) | 257f881e6045e497628957822667f02e0c8aca342469c72e9b70a835bed0bfda |
| mean_diffs.pt | `generate_directions/mean_diffs.pt` ([5, 32, 4096] float64) | 0e441a0af6b8f25cb30af2e072b0e988fc5192443434e10a590a1ed563a70b9e |

Số tác giả công bố trong `completions/*_evaluations.json` (so khớp chuỗi; tỉ lệ từ chối = 1 − success_rate):
JailbreakBench gốc 0,95; sau chiếu bỏ 0,00; Alpaca gốc 0,01; sau khi cộng 1,00.
Số của bài dùng cho tiêu chí (4) theo Nhap: 0,01 (chiếu bỏ) và 1,00 (cộng).
