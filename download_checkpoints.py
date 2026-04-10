"""跨平台 SAM 2.1 checkpoint 下载脚本（替代 checkpoints/download_chckpts.sh）。

用法：
    python download_checkpoints.py            # 下载全部 4 个 ckpt 到 checkpoints/
    python download_checkpoints.py --only large   # 只下载 large（默认 annotator.py 用的那个）

依赖：仅标准库（urllib），无需 wget/curl/tqdm。
"""

import argparse
import os
import sys
import time
import urllib.request

BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"

CKPTS = {
    "tiny":  "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "base":  "sam2.1_hiera_base_plus.pt",
    "large": "sam2.1_hiera_large.pt",
}


def _progress(blocks, block_size, total):
    downloaded = blocks * block_size
    if total > 0:
        pct = min(100.0, downloaded * 100.0 / total)
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_done:7.1f} / {mb_total:7.1f} MB")
    else:
        sys.stdout.write(f"\r  downloaded {downloaded // (1024 * 1024)} MB")
    sys.stdout.flush()


def download_one(filename: str, dest_dir: str) -> bool:
    url = f"{BASE_URL}/{filename}"
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {filename} already exists ({os.path.getsize(dest) // (1024*1024)} MB)")
        return True
    print(f"[download] {filename}")
    print(f"  from {url}")
    tmp = dest + ".part"
    try:
        t0 = time.time()
        urllib.request.urlretrieve(url, tmp, reporthook=_progress)
        sys.stdout.write("\n")
        os.replace(tmp, dest)
        print(f"  done in {time.time() - t0:.1f}s\n")
        return True
    except Exception as e:
        sys.stdout.write("\n")
        print(f"  FAILED: {e}\n")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download SAM 2.1 checkpoints")
    parser.add_argument(
        "--only",
        choices=list(CKPTS.keys()) + ["all"],
        default="all",
        help="只下载某个尺寸；默认全部 4 个",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(dest_dir, exist_ok=True)

    targets = list(CKPTS.values()) if args.only == "all" else [CKPTS[args.only]]

    failed = []
    for fn in targets:
        if not download_one(fn, dest_dir):
            failed.append(fn)

    if failed:
        print("以下 ckpt 下载失败，请检查网络后重试：")
        for fn in failed:
            print(f"  - {fn}")
        sys.exit(1)
    print("全部 ckpt 下载完成。")


if __name__ == "__main__":
    main()
