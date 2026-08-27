#!/usr/bin/env python3
"""
scripts/check_fig_assets.py — 核对 export_nii.py 导出的掩膜能否复现 csv 数字
==================================================================
定性图（Fig. 5）的图注要写 Betti-0，这个数字必须和论文表格一致。
去 3D Slicer 渲染之前先跑这个，别等图做完了才发现对不上。

写成文件而不是让人粘贴多行命令，理由见 .kb/decisions.md DEC-013。

用法：
  PYTHONPATH=. python scripts/check_fig_assets.py 702
  PYTHONPATH=. python scripts/check_fig_assets.py 702 --dir vis_nii
"""

import os
import sys
import argparse
import nibabel as nib
from scipy import ndimage


def human(n):
    for u in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}T"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case_ids", nargs="+")
    p.add_argument("--dir", default="vis_nii")
    args = p.parse_args()

    for cid in args.case_ids:
        gt_path = os.path.join(args.dir, f"{cid}_gt.nii.gz")
        if not os.path.exists(gt_path):
            print(f"[跳过] 找不到 {gt_path}")
            continue

        gt = nib.load(gt_path).get_fdata() > 0
        n_gt = ndimage.label(gt)[1]
        print(f"\n=== case {cid} ===")
        print(f"GT 连通域 = {n_gt}")
        print(f"{'列':<12}{'连通域':>7}{'B0误差':>8}{'Dice':>9}")

        for name in ("singleaxis", "triaxial", "stage2"):
            path = os.path.join(args.dir, f"{cid}_{name}.nii.gz")
            if not os.path.exists(path):
                print(f"{name:<12}{'—— 未导出':>24}")
                continue
            m = nib.load(path).get_fdata() > 0
            k = ndimage.label(m)[1]
            denom = m.sum() + gt.sum()
            dice = (2 * (m & gt).sum() / denom) if denom else float("nan")
            print(f"{name:<12}{k:>7d}{abs(k - n_gt):>8d}{dice:>9.4f}")

    print(f"\n=== {args.dir}/ 文件大小（scp 前心里有数）===")
    total = 0
    for f in sorted(os.listdir(args.dir)):
        sz = os.path.getsize(os.path.join(args.dir, f))
        total += sz
        print(f"  {f:<28}{human(sz):>8}")
    print(f"  {'合计':<28}{human(total):>8}")


if __name__ == "__main__":
    main()
