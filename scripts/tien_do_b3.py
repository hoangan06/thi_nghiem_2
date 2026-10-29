"""Xem tiến độ bước 3 trên máy thuê (chỉ đọc thư mục kết quả và danh sách tiến trình; không đụng lượt đang chạy).

Chạy trong thư mục kho:
    python3 scripts/tien_do_b3.py

In từng bước con (4a–4e) của hai mô hình: đã xong sau bao nhiêu phút, đang chạy bao lâu (kèm hai dòng log cuối), hay
dừng mà chưa xong; và khoảng thời gian còn lại ước theo thời gian ước của runbook (ước, chưa đo). Bước 4d không in
tiến độ theo lô, nên chỉ ước được một khoảng; đọc hai dòng log cuối để biết đang ở pha bf16 hay float32. Chỉ tính
các lượt của lần thuê này (sau giờ sửa cuối của /workspace/env_b3.sh). Với `b3 qwen35_9b kiem_nap`, 4c–4e của Qwen hiện "chưa
chạy" và được cộng vào thời gian còn lại dù sẽ không chạy.
"""
import datetime as dt
import glob
import os

B = [("kiem_truoc", "4a", "buoc3_kiem_truoc", "kiem_truoc.json", (0, 1)),
     ("nap_kiem", "4b", "buoc3_nap_kiem", "nap_kiem.json", (2, 8)),
     ("dung_huong", "4c", "buoc3_dung_huong", "huong.pt", (1, 8)),
     ("kiem_hook", "4d", "buoc3_kiem_hook", "ket_luan.json", (35, 60)),
     ("toc_do", "4e", "buoc3_do_toc_do", "toc_do.json", (7, 15))]


def main() -> None:
    lenh = []
    for p in glob.glob("/proc/[0-9]*/cmdline"):
        try:
            lenh.append(open(p, "rb").read().replace(b"\0", b" ").decode("utf-8", "replace"))
        except OSError:
            pass
    now = dt.datetime.now()
    # Chỉ tính các lượt của lần thuê này: thư mục có giờ trong tên sau lần sửa cuối của /workspace/env_b3.sh (tạo khi
    # dựng máy, trước mọi lệnh b3). results/ của lượt thuê trước đã chép lên máy thì không tính là xong.
    env = "/workspace/env_b3.sh"
    moc = f"{dt.datetime.fromtimestamp(os.path.getmtime(env)):%Y%m%d_%H%M%S}" if os.path.exists(env) else ""
    if moc:
        print(f"Chỉ tính các lượt từ {moc} (giờ sửa cuối của {env}).")
    con = [0.0, 0.0]
    print(f"Bây giờ: {now:%Y-%m-%d %H:%M:%S}")
    for M in ("qwen35_9b", "gemma4_12b"):
        print(f"== {M}")
        for loai, ma, script, xong, (lo, hi) in B:
            ds = sorted(d for d in glob.glob(f"results/buoc3/{M}/{loai}_2*") if os.path.isdir(d) and d[-15:] >= moc)
            chay = any(script + ".py" in x and M in x for x in lenh)
            if not ds:
                print(f"  {ma} {loai}: chưa chạy (ước {lo}–{hi} phút)")
                con[0] += lo
                con[1] += hi
                continue
            d = ds[-1]
            bd = dt.datetime.strptime(d[-15:], "%Y%m%d_%H%M%S")
            if os.path.exists(os.path.join(d, xong)):
                kt = dt.datetime.fromtimestamp(max(os.path.getmtime(p) for p in glob.glob(d + "/*")))
                print(f"  {ma} {loai}: XONG sau {(kt - bd).total_seconds() / 60:.1f} phút ({os.path.basename(d)})")
                continue
            da = (now - bd).total_seconds() / 60
            if chay:
                print(f"  {ma} {loai}: ĐANG CHẠY {da:.1f} phút (ước {lo}–{hi} phút)")
                con[0] += max(0, lo - da)
                con[1] += max(0, hi - da)
            else:
                print(f"  {ma} {loai}: DỪNG, chưa xong ({os.path.basename(d)}): xem log_b3")
            nk = os.path.join(d, "nhat_ky.txt")
            if os.path.exists(nk):
                for dong in open(nk, encoding="utf-8", errors="replace").read().splitlines()[-2:]:
                    print("      " + dong)
    print(f"Còn lại ước {con[0]:.0f}–{con[1]:.0f} phút → xong khoảng {now + dt.timedelta(minutes=con[0]):%H:%M} – "
          f"{now + dt.timedelta(minutes=con[1]):%H:%M} (giờ máy thuê; ước, chưa đo; giả định chạy cả hai mô hình)")


if __name__ == "__main__":
    main()
