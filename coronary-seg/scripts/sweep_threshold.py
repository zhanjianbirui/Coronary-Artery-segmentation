#!/usr/bin/env python3
"""
scripts/sweep_threshold.py — 预测阈值扫描
==================================================================
思路：推理一次得到"概率图"（不二值化），缓存下来，然后扫描不同
阈值 thr，每个 thr 把概率图二值化 + 固定后处理，评估指标，找最优阈值。

固定后处理用上一步 sweep 得到的最优：min_voxels=300, max_gap=0（不重连）。

用法：
  PYTHONPATH=. python scripts/sweep_threshold.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --ckpt runs/exp_2p5d/best.pth \
      --out-csv runs/exp_2p5d/sweep_threshold.csv \
      --max-cases 20 \
      --thr-list 0.3,0.35,0.4,0.45,0.5,0.55,0.6 \
      --min-voxels 300 --max-gap 0
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_preprocess, load_split
from src.model import build_model
from monai.data import PersistentDataset

from scripts.predict import (
    dice_coef, cldice_coef, betti0_error, hd95, precision_recall,
    remove_small_components, reconnect_endpoints, pad_to_multiple_2d,
)


@torch.no_grad()
def predict_prob_volume(model, image3d, k, device, batch=16, pad_multiple=32):
    """推理返回概率图 (H,W,D)，不二值化。"""
    img = torch.as_tensor(np.asarray(image3d))[0]
    H, W, D = img.shape
    prob_vol = np.zeros((H, W, D), dtype=np.float32)
    for start in range(0, D, batch):
        zc = list(range(D))[start:start + batch]
        stacks = []
        for z in zc:
            idx = [int(np.clip(z + off, 0, D - 1)) for off in range(-k, k + 1)]
            stacks.append(img[:, :, idx].permute(2, 0, 1))
        xb = torch.stack(stacks).float().to(device)
        xb, oh, ow = pad_to_multiple_2d(xb, multiple=pad_multiple)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits = model(xb)
        logits = logits[..., :oh, :ow]
        probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        for j, z in enumerate(zc):
            prob_vol[:, :, z] = probs[j]
    return prob_vol


def apply_postproc(binary, min_voxels, max_gap):
    m = remove_small_components(binary, min_voxels)
    if max_gap > 0:
        m = reconnect_endpoints(m, max_gap)
    return m


def evaluate(pred, gt):
    d = dice_coef(pred, gt)
    cl = cldice_coef(pred, gt)
    b0, _, _ = betti0_error(pred, gt)
    h = hd95(pred, gt)
    prec, rec = precision_recall(pred, gt)
    return d, cl, b0, h, prec, rec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-csv", default="runs/exp_2p5d/sweep_threshold.csv")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--backbone", default="segresnet")
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--pad-multiple", type=int, default=32)
    p.add_argument("--max-cases", type=int, default=20)
    p.add_argument("--thr-list", default="0.3,0.35,0.4,0.45,0.5,0.55,0.6")
    p.add_argument("--min-voxels", type=int, default=300)
    p.add_argument("--max-gap", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, ckpt={args.ckpt}")
    print(f"固定后处理: min_voxels={args.min_voxels}, max_gap={args.max_gap}")

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

    thr_list = [float(x) for x in args.thr_list.split(",")]

    # 第一步：推理，缓存概率图 + gt
    print(f"\n[1/2] 推理 {len(test_rec)} 个病例（缓存概率图）...")
    cached = []
    for ci in range(len(test_rec)):
        vol = cache[ci]
        prob = predict_prob_volume(model, np.asarray(vol["image"]),
                                   args.k, device,
                                   pad_multiple=args.pad_multiple)
        gt = np.asarray(vol["label"])[0].astype(np.uint8)
        cached.append((prob, gt))
        print(f"  推理 [{ci+1}/{len(test_rec)}] case {test_rec[ci].get('id')}")

    # 第二步：扫描阈值
    print(f"\n[2/2] 扫描 {len(thr_list)} 个阈值...")
    results = []
    for thr in thr_list:
        ds, cls, b0s, hds, precs, recs = [], [], [], [], [], []
        for prob, gt in cached:
            binary = (prob > thr).astype(np.uint8)
            pred = apply_postproc(binary, args.min_voxels, args.max_gap)
            d, cl, b0, h, pr, rc = evaluate(pred, gt)
            ds.append(d); cls.append(cl); b0s.append(b0)
            precs.append(pr); recs.append(rc)
            if not np.isnan(h):
                hds.append(h)
        row = {"thr": thr, "dice": np.mean(ds), "cldice": np.mean(cls),
               "precision": np.mean(precs), "recall": np.mean(recs),
               "betti0_err": np.mean(b0s),
               "hd95": np.mean(hds) if hds else float("nan")}
        results.append(row)
        print(f"  thr={thr:.2f}  dice={row['dice']:.4f}  "
              f"clDice={row['cldice']:.4f}  P={row['precision']:.3f}  "
              f"R={row['recall']:.3f}  B0={row['betti0_err']:.2f}  "
              f"HD95={row['hd95']:.2f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    best = max(results, key=lambda r: r["dice"])
    print(f"\n===== 最高 Dice 的阈值 =====")
    print(f"  thr={best['thr']:.2f}  dice={best['dice']:.4f}  "
          f"clDice={best['cldice']:.4f}  B0={best['betti0_err']:.2f}  "
          f"HD95={best['hd95']:.2f}")
    print(f"\n结果已存: {args.out_csv}")


if __name__ == "__main__":
    main()
