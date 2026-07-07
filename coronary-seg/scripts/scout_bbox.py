#!/usr/bin/env python3
"""
scripts/scout_bbox_tri.py — 三方向血管边界框侦察
==================================================================
用于三方向 2.5D 训练前检查 crop-size 是否会切掉血管。

三方向含义：
  axis=0: 沿 H 方向切片，二维平面是 W×D
  axis=1: 沿 W 方向切片，二维平面是 H×D
  axis=2: 沿 D 方向切片，二维平面是 H×W

本脚本扫描训练集 label，分别统计三个方向下：
  1. 血管在二维平面中的 bbox 跨度
  2. 血管中心相对图像中心的偏移
  3. 中心裁 320/384/448/512 是否会切到血管

只读缓存，不训练，不改数据。

用法：
  PYTHONPATH=. python scripts/scout_bbox_tri.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --max-cases 0 \
      --test-sizes 320,384,448,512
"""

from monai.data import PersistentDataset
from src.data import build_preprocess, load_split
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--max-cases", type=int, default=0,
                   help="0=全部训练集")
    p.add_argument("--test-sizes", default="320,384,448,512",
                   help="要评估的候选中心裁剪尺寸，例如 320,384,448,512")
    return p.parse_args()


def project_foreground(fg, axis):
    """
    fg: bool array, shape = (H, W, D)

    返回某个切片方向对应的二维投影：
      axis=0 -> 平面 W×D，投影掉 H
      axis=1 -> 平面 H×D，投影掉 W
      axis=2 -> 平面 H×W，投影掉 D
    """
    if axis == 0:
        proj = fg.any(axis=0)   # (W, D)
        plane_name = "axis=0, plane=W×D"
        dim0_name = "W"
        dim1_name = "D"
    elif axis == 1:
        proj = fg.any(axis=1)   # (H, D)
        plane_name = "axis=1, plane=H×D"
        dim0_name = "H"
        dim1_name = "D"
    elif axis == 2:
        proj = fg.any(axis=2)   # (H, W)
        plane_name = "axis=2, plane=H×W"
        dim0_name = "H"
        dim1_name = "W"
    else:
        raise ValueError(f"Unsupported axis: {axis}")

    return proj, plane_name, dim0_name, dim1_name


def stat_line(name, arr):
    arr = np.asarray(arr)
    print(
        f"  {name}: "
        f"均值={arr.mean():.1f}  "
        f"中位={np.median(arr):.1f}  "
        f"95分位={np.percentile(arr, 95):.1f}  "
        f"最大={arr.max():.1f}"
    )


def analyse_one_axis(cache, n_cases, axis, test_sizes):
    bbox_dim0 = []
    bbox_dim1 = []
    center_off_dim0 = []
    center_off_dim1 = []
    valid_cases = 0
    empty_cases = 0

    plane_name = None
    dim0_name = None
    dim1_name = None

    for ci in range(n_cases):
        vol = cache[ci]
        label = np.asarray(vol["label"])[0]  # (H, W, D)
        fg = label > 0

        proj, plane_name, dim0_name, dim1_name = project_foreground(fg, axis)

        dim0, dim1 = proj.shape
        rows = np.where(proj.any(axis=1))[0]
        cols = np.where(proj.any(axis=0))[0]

        if len(rows) == 0 or len(cols) == 0:
            empty_cases += 1
            continue

        valid_cases += 1

        span0 = rows.max() - rows.min() + 1
        span1 = cols.max() - cols.min() + 1

        center0 = (rows.max() + rows.min()) / 2.0
        center1 = (cols.max() + cols.min()) / 2.0

        bbox_dim0.append(span0)
        bbox_dim1.append(span1)
        center_off_dim0.append(center0 - dim0 / 2.0)
        center_off_dim1.append(center1 - dim1 / 2.0)

        if (ci + 1) % 50 == 0:
            print(f"    ...axis={axis}: {ci + 1}/{n_cases}")

    bbox_dim0 = np.asarray(bbox_dim0)
    bbox_dim1 = np.asarray(bbox_dim1)
    off0 = np.asarray(center_off_dim0)
    off1 = np.asarray(center_off_dim1)

    print("\n" + "=" * 70)
    print(f"  {plane_name}")
    print("=" * 70)
    print(f"  有效病例: {valid_cases}/{n_cases}，空标注病例: {empty_cases}")

    if valid_cases == 0:
        print("  没有找到前景，跳过。")
        return

    print("\n--- 1. 血管 bbox 跨度，单位：像素 ---")
    stat_line(f"{dim0_name} 方向跨度", bbox_dim0)
    stat_line(f"{dim1_name} 方向跨度", bbox_dim1)

    print("\n--- 2. 血管中心相对图像中心偏移，单位：像素，取绝对值 ---")
    stat_line(f"{dim0_name} 偏移", np.abs(off0))
    stat_line(f"{dim1_name} 偏移", np.abs(off1))

    print("\n--- 3. 各候选中心裁剪尺寸覆盖情况 ---")
    for size in test_sizes:
        half = size / 2.0
        miss = 0
        max_cut = 0.0
        cuts = []

        for span0, span1, o0, o1 in zip(bbox_dim0, bbox_dim1, off0, off1):
            need0 = abs(o0) + span0 / 2.0
            need1 = abs(o1) + span1 / 2.0

            cut = max(need0 - half, need1 - half, 0.0)
            if cut > 0:
                miss += 1
                cuts.append(cut)
                max_cut = max(max_cut, cut)

        pct = 100.0 * miss / valid_cases
        mean_cut = float(np.mean(cuts)) if cuts else 0.0

        print(
            f"  中心裁 {size}×{size}: "
            f"{miss}/{valid_cases} 病例会切到血管 "
            f"({pct:.1f}%)，"
            f"平均切入={mean_cut:.1f} 像素，"
            f"最严重切入={max_cut:.1f} 像素"
        )

    print("\n--- 4. 建议判读 ---")
    print("  - 如果某个 crop-size 在三个方向均接近 0% 切到，说明中心裁比较安全。")
    print("  - 如果 axis=0 或 axis=1 的 D 方向切到很多，说明三方向训练不适合太小 crop。")
    print("  - 如果 384 切到较多，而 448/512 明显改善，应优先考虑更大 crop 或动态 ROI crop。")


def main():
    args = parse_args()

    test_sizes = [int(s.strip())
                  for s in args.test_sizes.split(",") if s.strip()]
    preprocess = build_preprocess(args.spacing, args.hu_min, args.hu_max)

    train_rec, _, _ = load_split(args.split_json)
    if args.max_cases and args.max_cases > 0:
        train_rec = train_rec[:args.max_cases]

    cache = PersistentDataset(
        data=train_rec,
        transform=preprocess,
        cache_dir=args.cache_dir
    )

    print("=" * 70)
    print("  三方向血管边界框侦察")
    print("=" * 70)
    print(f"  病例数: {len(train_rec)}")
    print(f"  test crop sizes: {test_sizes}")
    print("  axes: 0=W×D, 1=H×D, 2=H×W")

    for axis in (0, 1, 2):
        analyse_one_axis(cache, len(train_rec), axis, test_sizes)

    print("\n" + "=" * 70)
    print("最终判读：")
    print("  1. 三方向训练时，crop-size 必须同时适合 W×D、H×D、H×W 三个平面。")
    print("  2. 如果 384 在某方向切到病例较多，建议改用 448/512 或改动态 ROI crop。")
    print("  3. 如果显存不够，优先增大 crop-size，降低 batch-size。")
    print("=" * 70)


if __name__ == "__main__":
    main()
