#!/usr/bin/env python3
"""
scripts/sweep_spatial_prior.py — 空间先验后处理参数扫描
==================================================================
思路和 sweep_postproc.py 一样：推理很慢，所以每个病例只推理一次、
缓存原始二值预测，然后在缓存上零成本扫描空间先验的参数网格
（max_dist_mm × n_anchor），找出让 HD95 / precision 最好、
同时 Dice/clDice/Betti-0 不掉的配置。

为什么要有这一步（见 src/spatial_prior.py 头部注释与 .kb 诊断）：
  测试集 HD95 重尾，corr(HD95, precision) = -0.665 而 corr(HD95, recall)
  只有 -0.229 —— 长尾是"离血管树很远的大块假阳"造成的，不是漏检。
  现有后处理只有体积阈值，删不掉这种"大而远"的假阳。

除了常规均值，本脚本额外输出两个专门盯长尾的指标：
  hd95_p90  —— HD95 的 90 分位（长尾直接体现在这里）
  n_hd95_gt50 —— HD95>50mm 的病例数（诊断里那 12 例）
均值容易被少数极端值主导，看分位数才知道是真的修好了还是被平均掉了。

用法（单轴）：
  PYTHONPATH=. python scripts/sweep_spatial_prior.py \
      --cache-dir /path/to/cache/preproc \
      --ckpt runs/exp_2p5d/best.pth \
      --out-csv runs/exp_2p5d/sweep_spatial_prior.csv \
      --max-cases 40

用法（三正交，推荐 —— 当前最优方案就是三正交）：
  PYTHONPATH=. python scripts/sweep_spatial_prior.py \
      --cache-dir /path/to/cache/preproc \
      --ckpt runs/exp_tri2p5d/best.pth --tri --fuse mean --thr 0.50 \
      --out-csv runs/exp_tri2p5d/sweep_spatial_prior.csv \
      --max-cases 40

建议先用 --case-ids 931 630 595 741 728 把诊断出的 5 个难例单独跑一遍，
确认这些病例的 HD95 真的掉下来了，再跑更大的子集看整体有没有副作用。
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
from src.spatial_prior import spatial_prior_filter
from monai.data import PersistentDataset

from scripts.predict import (
    dice_coef, cldice_coef, betti0_error, hd95, precision_recall,
    remove_small_components, predict_volume,
)
from scripts.predict_tri import predict_tri_fused


def evaluate(pred, gt):
    """与 predict.py / predict_tri.py 完全同口径的指标。"""
    d = dice_coef(pred, gt)
    cl = cldice_coef(pred, gt)
    b0, _, _ = betti0_error(pred, gt)
    h = hd95(pred, gt)
    p, r = precision_recall(pred, gt)
    return d, cl, b0, h, p, r


def summarize(per_case):
    """把逐病例指标聚合成一行，额外给长尾统计。"""
    arr = {k: np.array([c[k] for c in per_case], dtype=np.float64)
           for k in ("dice", "cldice", "betti0_err", "precision", "recall")}
    hds = np.array([c["hd95"] for c in per_case], dtype=np.float64)
    hds = hds[~np.isnan(hds)]
    return {
        "dice": arr["dice"].mean(),
        "cldice": arr["cldice"].mean(),
        "betti0_err": arr["betti0_err"].mean(),
        "precision": arr["precision"].mean(),
        "recall": arr["recall"].mean(),
        "hd95": hds.mean() if len(hds) else float("nan"),
        "hd95_median": np.median(hds) if len(hds) else float("nan"),
        "hd95_p90": np.percentile(hds, 90) if len(hds) else float("nan"),
        "n_hd95_gt50": int((hds > 50).sum()),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-csv",
                   default="runs/exp_tri2p5d/sweep_spatial_prior.csv")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--backbone", default="segresnet")
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--pad-multiple", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-px-per-batch", type=int, default=4_000_000)
    # 推理方式
    p.add_argument("--tri", action="store_true",
                   help="用三正交融合推理（当前最优方案）")
    p.add_argument("--axes", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--fuse", default="mean", choices=["mean", "max"])
    # 病例选择
    p.add_argument("--max-cases", type=int, default=40, help="0=全部测试集")
    p.add_argument("--case-ids", nargs="*", default=None,
                   help="指定病例 id（如诊断出的难例 931 630 595 741 728）")
    # 固定的体积阈值（空间先验之前先做，与现有最优后处理一致）
    p.add_argument("--min-voxels", type=int, default=300)
    # 扫描网格
    p.add_argument("--dist-list", default="10,15,20,30,40,60",
                   help="max_dist_mm 网格，逗号分隔")
    p.add_argument("--anchor-list", default="1,2,3",
                   help="n_anchor 网格，逗号分隔")
    p.add_argument("--no-chain", action="store_true",
                   help="关闭链式生长（默认开启）")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, ckpt={args.ckpt}, tri={args.tri}")

    cfg = {"k": args.k, "backbone": args.backbone,
           "init_filters": args.init_filters, "out_channels": 1}
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"加载 checkpoint（epoch={ckpt.get('epoch')}, "
          f"val_dice={ckpt.get('val_dice')}）")

    preprocess = build_preprocess(args.spacing, args.hu_min, args.hu_max)
    _, _, test_rec = load_split(args.split_json)

    if args.case_ids:
        want = {str(c) for c in args.case_ids}
        idxs = [i for i, r in enumerate(test_rec) if str(r.get("id")) in want]
        print(f"[指定病例] {len(idxs)} 例: {[test_rec[i].get('id') for i in idxs]}")
    else:
        n = len(test_rec) if args.max_cases <= 0 else min(args.max_cases,
                                                          len(test_rec))
        idxs = list(range(n))
        print(f"[子集] 前 {len(idxs)} 例")

    cache = PersistentDataset(data=test_rec, transform=preprocess,
                              cache_dir=args.cache_dir)

    dist_list = [float(x) for x in args.dist_list.split(",")]
    anchor_list = [int(x) for x in args.anchor_list.split(",")]
    chain = not args.no_chain

    # ---- 第一步：每个病例推理一次，缓存"体积阈值之后"的预测 + gt ----
    # 缓存的是已经过 min_voxels 的结果：空间先验是叠加在现有最优后处理之上的，
    # 不是替代它。这样扫描出来的增量就是空间先验的纯贡献。
    print(f"\n[1/2] 推理 {len(idxs)} 个病例（每个只推一次）...")
    cached = []   # list of (case_id, pred_mv, gt)
    for n_done, ci in enumerate(idxs, 1):
        vol = cache[ci]
        image3d = np.asarray(vol["image"])
        gt = np.asarray(vol["label"])[0].astype(np.uint8)
        if args.tri:
            fused = predict_tri_fused(
                model, image3d, args.k, device, axes=tuple(args.axes),
                methods=(args.fuse,), batch=args.batch,
                pad_multiple=args.pad_multiple,
                max_px_per_batch=args.max_px_per_batch)
            pred_raw = (fused[args.fuse] >= args.thr).astype(np.uint8)
        else:
            pred_raw = predict_volume(model, image3d, args.k, device,
                                      args.thr, batch=args.batch,
                                      pad_multiple=args.pad_multiple)
        pred_mv = remove_small_components(pred_raw, args.min_voxels)
        cid = test_rec[ci].get("id", ci)
        cached.append((cid, pred_mv, gt))
        print(f"  [{n_done}/{len(idxs)}] case {cid}")

    # ---- 第二步：在缓存预测上扫描空间先验参数 ----
    n_cfg = 1 + len(dist_list) * len(anchor_list)
    print(f"\n[2/2] 扫描 {len(dist_list)}×{len(anchor_list)} + baseline "
          f"= {n_cfg} 组配置...")

    rows = []
    configs = ([("baseline_mv%d" % args.min_voxels, None, None)] +
               [(f"d{d:g}_a{a}", d, a) for a in anchor_list for d in dist_list])

    for label, dist, n_anchor in configs:
        per_case = []
        n_dropped_total = 0
        for cid, pred_mv, gt in cached:
            if dist is None:
                pred = pred_mv
                info = {"n_dropped": 0}
            else:
                pred, info = spatial_prior_filter(
                    pred_mv, spacing=args.spacing, n_anchor=n_anchor,
                    max_dist_mm=dist, chain=chain, return_info=True)
            d, cl, b0, h, p, r = evaluate(pred, gt)
            per_case.append({"dice": d, "cldice": cl, "betti0_err": b0,
                             "hd95": h, "precision": p, "recall": r})
            n_dropped_total += info["n_dropped"]
        row = {"config": label,
               "max_dist_mm": dist if dist is not None else "-",
               "n_anchor": n_anchor if n_anchor is not None else "-",
               "chain": chain if dist is not None else "-",
               "n_comp_dropped": n_dropped_total}
        row.update(summarize(per_case))
        rows.append(row)
        print(f"  {label:14s} dice={row['dice']:.4f} clDice={row['cldice']:.4f} "
              f"B0={row['betti0_err']:.2f} HD95={row['hd95']:.2f} "
              f"(中位{row['hd95_median']:.1f}/P90 {row['hd95_p90']:.1f}, "
              f">50 的 {row['n_hd95_gt50']} 例) "
              f"P={row['precision']:.4f} R={row['recall']:.4f} "
              f"删了 {row['n_comp_dropped']} 个分量")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 汇总：跟 baseline 比，谁真的赢了 ----
    base = rows[0]
    print(f"\n===== 相对 baseline（{base['config']}）的变化 =====")
    print(f"  {'config':14s} {'ΔDice':>8} {'ΔclDice':>8} {'ΔB0':>7} "
          f"{'ΔHD95':>8} {'ΔP90':>8} {'ΔPrec':>8}")
    for r in rows[1:]:
        print(f"  {r['config']:14s} {r['dice']-base['dice']:+8.4f} "
              f"{r['cldice']-base['cldice']:+8.4f} "
              f"{r['betti0_err']-base['betti0_err']:+7.2f} "
              f"{r['hd95']-base['hd95']:+8.2f} "
              f"{r['hd95_p90']-base['hd95_p90']:+8.2f} "
              f"{r['precision']-base['precision']:+8.4f}")
    print(f"\n  结果已存: {args.out_csv}")
    print("  选参建议：优先看 ΔHD95/ΔP90 明显为负、而 ΔDice/ΔclDice 不掉"
          "（≥ -0.002）的那一行；\n"
          "  若某行 ΔPrec 涨了但 ΔclDice 掉不少，说明 D 太小、砍到真分支了。")


if __name__ == "__main__":
    main()
