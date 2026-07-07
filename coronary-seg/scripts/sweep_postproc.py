#!/usr/bin/env python3
"""
scripts/sweep_postproc.py — 后处理参数扫描
==================================================================
思路：推理很慢，所以每个病例只推理一次，缓存原始二值预测，
然后在缓存的预测上扫描 min_voxels × max_gap 的所有组合，
找出让 Betti-0 最低、同时 Dice/clDice 不掉的最优后处理配置。

用法：
  PYTHONPATH=. python scripts/sweep_postproc.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --ckpt runs/exp_2p5d/best.pth \
      --out-csv runs/exp_2p5d/sweep_postproc.csv \
      --max-cases 20 \
      --min-voxels-list 50,100,200,300,500 \
      --max-gap-list 0,10,15,20,25
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_preprocess, load_split
from src.model import build_model
from monai.data import PersistentDataset

# 复用 predict.py 里的函数
from scripts.predict import (
    dice_coef, cldice_coef, betti0_error, hd95, precision_recall,
    remove_small_components, reconnect_endpoints, predict_volume,
)


def evaluate(pred, gt):
    d = dice_coef(pred, gt)
    cl = cldice_coef(pred, gt)
    b0, _, _ = betti0_error(pred, gt)
    h = hd95(pred, gt)
    return d, cl, b0, h


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-csv", default="runs/exp_2p5d/sweep_postproc.csv")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--backbone", default="segresnet")
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--pad-multiple", type=int, default=32)
    p.add_argument("--max-cases", type=int, default=20)
    p.add_argument("--min-voxels-list", default="50,100,200,300,500")
    p.add_argument("--max-gap-list", default="0,10,15,20,25")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, ckpt={args.ckpt}")

    cfg = {"k": args.k, "backbone": args.backbone,
           "init_filters": args.init_filters, "out_channels": 1}
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"加载 checkpoint（val_dice={ckpt.get('val_dice')}）")

    preprocess = build_preprocess(args.spacing, args.hu_min, args.hu_max)
    _, _, test_rec = load_split(args.split_json)
    if args.max_cases and args.max_cases > 0:
        test_rec = test_rec[:args.max_cases]
    cache = PersistentDataset(data=test_rec, transform=preprocess,
                              cache_dir=args.cache_dir)

    min_voxels_list = [int(x) for x in args.min_voxels_list.split(",")]
    max_gap_list = [int(x) for x in args.max_gap_list.split(",")]

    # ---- 第一步：每个病例推理一次，缓存原始预测 + gt ----
    print(f"\n[1/2] 推理 {len(test_rec)} 个病例（每个只推一次，缓存预测）...")
    cached = []   # list of (pred_raw, gt)
    for ci in range(len(test_rec)):
        vol = cache[ci]
        image3d = np.asarray(vol["image"])
        gt = np.asarray(vol["label"])[0].astype(np.uint8)
        pred_raw = predict_volume(model, image3d, args.k, device,
                                  args.thr, pad_multiple=args.pad_multiple)
        cached.append((pred_raw, gt))
        print(f"  推理 [{ci+1}/{len(test_rec)}] case {test_rec[ci].get('id')}")

    # ---- 第二步：在缓存预测上扫描所有后处理组合 ----
    print(f"\n[2/2] 扫描 {len(min_voxels_list)}×{len(max_gap_list)} "
          f"= {len(min_voxels_list)*len(max_gap_list)} 组后处理参数...")

    results = []
    # 先记录一组"不做后处理"的 baseline
    for label, mv, mg in ([("raw", None, None)] +
                          [(f"mv{mv}_gap{mg}", mv, mg)
                           for mv in min_voxels_list
                           for mg in max_gap_list]):
        dices, cldices, b0s, hds = [], [], [], []
        for pred_raw, gt in cached:
            if mv is None:
                pred = pred_raw
            else:
                pred = remove_small_components(pred_raw, mv)
                if mg > 0:
                    pred = reconnect_endpoints(pred, mg)
            d, cl, b0, h = evaluate(pred, gt)
            dices.append(d); cldices.append(cl); b0s.append(b0)
            if not np.isnan(h):
                hds.append(h)
        row = {"config": label,
               "min_voxels": mv if mv is not None else "-",
               "max_gap": mg if mg is not None else "-",
               "dice": np.mean(dices),
               "cldice": np.mean(cldices),
               "betti0_err": np.mean(b0s),
               "hd95": np.mean(hds) if hds else float("nan")}
        results.append(row)
        print(f"  {label:16s}  dice={row['dice']:.4f}  "
              f"clDice={row['cldice']:.4f}  B0={row['betti0_err']:.2f}  "
              f"HD95={row['hd95']:.2f}")

    # 写 csv
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # ---- 找最优：Betti-0 最低且 Dice 不低于 raw-0.005 ----
    raw = results[0]
    valid = [r for r in results[1:] if r["dice"] >= raw["dice"] - 0.005]
    if valid:
        best = min(valid, key=lambda r: r["betti0_err"])
        print(f"\n===== 推荐配置 =====")
        print(f"  {best['config']}: min_voxels={best['min_voxels']}, "
              f"max_gap={best['max_gap']}")
        print(f"  dice={best['dice']:.4f}  clDice={best['cldice']:.4f}  "
              f"B0={best['betti0_err']:.2f}  HD95={best['hd95']:.2f}")
        print(f"  （对比 raw: dice={raw['dice']:.4f} B0={raw['betti0_err']:.2f}）")
    print(f"\n结果已存: {args.out_csv}")


if __name__ == "__main__":
    main()
