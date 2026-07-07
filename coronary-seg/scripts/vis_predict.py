#!/usr/bin/env python3
"""
scripts/vis_predict.py — 可视化指定病例的预测 vs 金标准
==================================================================
对指定病例推理，用最大密度投影(MIP)把 3D 血管压成 2D 看整体结构：
  - 金标准血管树 vs 预测血管树，三个投影方向并排
  - 一眼看出：是"整段漏检"(预测缺一大块) 还是"断裂"(断成几段)
输出 PNG。

用法：
  PYTHONPATH=. python scripts/vis_predict.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --ckpt runs/exp_2p5d/best.pth \
      --case-ids 931,728,630,39,449 \
      --out-dir vis_cases \
      --thr 0.5 --min-voxels 300 --tta
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_preprocess, load_split
from src.model import build_model
from monai.data import PersistentDataset
from scripts.predict import predict_volume, remove_small_components


def mip(vol, axis):
    """最大密度投影：沿某轴取最大值，3D->2D。"""
    return vol.max(axis=axis)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--case-ids", required=True, help="逗号分隔的病例id")
    p.add_argument("--out-dir", default="vis_cases")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--backbone", default="segresnet")
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min-voxels", type=int, default=300)
    p.add_argument("--pad-multiple", type=int, default=32)
    p.add_argument("--tta", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = {"k": args.k, "backbone": args.backbone,
           "init_filters": args.init_filters, "out_channels": 1}
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preprocess = build_preprocess(args.spacing, args.hu_min, args.hu_max)
    _, _, test_rec = load_split(args.split_json)
    id2rec = {str(r.get("id")): r for r in test_rec}

    want_ids = [x.strip() for x in args.case_ids.split(",")]
    for cid in want_ids:
        if cid not in id2rec:
            print(f"[!] 病例 {cid} 不在测试集，跳过")
            continue
        rec = id2rec[cid]
        cache = PersistentDataset(data=[rec], transform=preprocess,
                                  cache_dir=args.cache_dir)
        vol = cache[0]
        gt = np.asarray(vol["label"])[0].astype(np.uint8)   # (H,W,D)
        image3d = np.asarray(vol["image"])

        pred = predict_volume(model, image3d, args.k, device, args.thr,
                              pad_multiple=args.pad_multiple,
                              use_tta=args.tta)
        pred = remove_small_components(pred, args.min_voxels)

        # 三个方向的 MIP
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        views = [("Axial", 2), ("Coronal", 1), ("Sagittal", 0)]
        for col, (vname, ax_i) in enumerate(views):
            gt_mip = mip(gt, ax_i)
            pr_mip = mip(pred, ax_i)
            axes[0, col].imshow(gt_mip.T, cmap="Reds", origin="lower")
            axes[0, col].set_title(f"GT - {vname}", fontsize=11)
            axes[0, col].axis("off")
            axes[1, col].imshow(pr_mip.T, cmap="Blues", origin="lower")
            axes[1, col].set_title(f"Pred - {vname}", fontsize=11)
            axes[1, col].axis("off")

        d = (2 * (pred & gt).sum()) / (pred.sum() + gt.sum() + 1e-6)
        fig.suptitle(f"Case {cid}   Dice={d:.3f}   "
                     f"GT voxels={gt.sum()}  Pred voxels={pred.sum()}",
                     fontsize=13)
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"case{cid}_mip.png")
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"[{cid}] Dice={d:.3f}  saved: {out_path}")

    print(f"\n完成。看 {args.out_dir}/ 下的 PNG：")
    print("  上排红色=金标准, 下排蓝色=预测")
    print("  对比看: 预测是'缺一大段'(漏检) 还是'断成几截'(断裂)")


if __name__ == "__main__":
    main()
