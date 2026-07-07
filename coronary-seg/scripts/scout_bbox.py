#!/usr/bin/env python3
"""
scripts/scout_bbox.py — 血管边界框侦察
==================================================================
扫描训练集所有病例的标注（用已建好的预处理缓存），统计血管在 x/y
平面的分布，回答：
  1. 血管边界框在 x/y 方向多大？（决定裁剪尺寸下限）
  2. 血管中心是否都在图像中心附近？（决定能否中心裁）
  3. 若中心裁 320/384/448，会切掉多少病例的血管？切多严重？

只读缓存的 label，不训练、不改数据。

用法（GPU 节点或 CPU 均可）：
  PYTHONPATH=. python scripts/scout_bbox.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --max-cases 0     # 0=全部训练集
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_preprocess, load_split
from monai.data import PersistentDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--max-cases", type=int, default=0,
                   help="0=全部训练集")
    p.add_argument("--test-sizes", default="320,384,448",
                   help="要评估的候选中心裁剪尺寸")
    return p.parse_args()


def main():
    args = parse_args()
    preprocess = build_preprocess(args.spacing, args.hu_min, args.hu_max)
    train_rec, _, _ = load_split(args.split_json)
    if args.max_cases and args.max_cases > 0:
        train_rec = train_rec[:args.max_cases]

    cache = PersistentDataset(data=train_rec, transform=preprocess,
                              cache_dir=args.cache_dir)

    print("=" * 60)
    print(f"  血管边界框侦察（{len(train_rec)} 个病例）")
    print("=" * 60)

    bbox_h, bbox_w = [], []          # 每个病例血管在 H/W 方向的跨度
    img_hw = []                      # 每个病例图像的 H,W
    center_off_h, center_off_w = [], []  # 血管中心相对图像中心的偏移

    for ci in range(len(train_rec)):
        vol = cache[ci]
        label = np.asarray(vol["label"])[0]      # (H, W, D)
        H, W, D = label.shape
        fg = label > 0
        # 在 H/W 平面上，血管出现过的行/列
        proj = fg.any(axis=2)                    # (H, W) 任意 z 有前景
        rows = np.where(proj.any(axis=1))[0]
        cols = np.where(proj.any(axis=0))[0]
        if len(rows) == 0:
            continue
        h_span = rows.max() - rows.min() + 1
        w_span = cols.max() - cols.min() + 1
        h_center = (rows.max() + rows.min()) / 2
        w_center = (cols.max() + cols.min()) / 2

        bbox_h.append(h_span)
        bbox_w.append(w_span)
        img_hw.append((H, W))
        center_off_h.append(h_center - H / 2)
        center_off_w.append(w_center - W / 2)

        if (ci + 1) % 50 == 0:
            print(f"  ...{ci + 1}/{len(train_rec)}")

    bbox_h = np.array(bbox_h)
    bbox_w = np.array(bbox_w)
    off_h = np.array(center_off_h)
    off_w = np.array(center_off_w)

    def stat(name, arr):
        print(f"  {name}: 均值={arr.mean():.0f}  中位={np.median(arr):.0f}  "
              f"最大={arr.max():.0f}  95分位={np.percentile(arr, 95):.0f}")

    print("\n--- 1. 血管边界框跨度（像素）---")
    stat("H 方向跨度", bbox_h)
    stat("W 方向跨度", bbox_w)

    print("\n--- 2. 血管中心相对图像中心的偏移（像素，正=偏右/下）---")
    stat("H 偏移(绝对值)", np.abs(off_h))
    stat("W 偏移(绝对值)", np.abs(off_w))
    print(f"  说明：偏移越小说明血管越居中，中心裁越安全")

    print("\n--- 3. 各候选尺寸中心裁的覆盖情况 ---")
    sizes = [int(s) for s in args.test_sizes.split(",")]
    for size in sizes:
        half = size / 2
        # 从图像中心裁 size×size，血管是否完全落入？
        # 血管中心偏移 + 血管半跨度 是否超过 half
        miss = 0
        max_cut = 0
        for hspan, wspan, oh, ow in zip(bbox_h, bbox_w, off_h, off_w):
            need_h = abs(oh) + hspan / 2   # 从图像中心到血管最远边
            need_w = abs(ow) + wspan / 2
            if need_h > half or need_w > half:
                miss += 1
                cut = max(need_h - half, need_w - half)
                max_cut = max(max_cut, cut)
        pct = 100 * miss / len(bbox_h)
        print(f"  中心裁 {size}×{size}: {miss}/{len(bbox_h)} 病例血管被切到 "
              f"({pct:.1f}%)，最严重切入 {max_cut:.0f} 像素")

    print("\n" + "=" * 60)
    print("判读：")
    print("  - 看第1项：裁剪尺寸至少要 > 血管最大跨度")
    print("  - 看第2项：偏移小(如<30像素)=可中心裁；偏移大=需动态定位")
    print("  - 看第3项：选一个'切到病例%很低、切入像素很小'的尺寸")
    print("=" * 60)


if __name__ == "__main__":
    main()
