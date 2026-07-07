#!/usr/bin/env python3
"""
scripts/analyze_cases.py — 从预测结果 csv 分析病例表现
==================================================================
不重新推理，直接读 test_final_tta.csv，按 Dice/clDice 排序，
找出表现好/差的病例，以及"Dice和clDice不一致"的矛盾病例。

用法：
  PYTHONPATH=. python scripts/analyze_cases.py \
      --csv runs/exp_2p5d/test_final_tta.csv \
      --metric pp   # pp=后处理后, raw=原始
"""

import argparse
import csv
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--metric", default="pp", choices=["pp", "raw"],
                   help="看后处理(pp)还是原始(raw)指标")
    p.add_argument("--topn", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    pre = args.metric
    rows = list(csv.DictReader(open(args.csv)))
    for r in rows:
        for k in list(r.keys()):
            if k != "id":
                try:
                    r[k] = float(r[k])
                except (ValueError, TypeError):
                    r[k] = float("nan")
    n = len(rows)
    print(f"共 {n} 个病例，看 [{pre}] 指标\n")

    dice_k = f"{pre}_dice"
    cldice_k = f"{pre}_cldice"
    b0_k = f"{pre}_betti0_err"
    hd_k = f"{pre}_hd95"

    # 整体均值
    print("===== 整体均值 =====")
    for k, name in [(dice_k, "Dice"), (cldice_k, "clDice"),
                    (b0_k, "Betti-0"), (hd_k, "HD95")]:
        vals = [r[k] for r in rows if not np.isnan(r[k])]
        print(f"  {name}: {np.mean(vals):.4f}")

    def show(title, sorted_rows, keys):
        print(f"\n===== {title} =====")
        header = "  病例      " + "  ".join(f"{k:>8}" for k in keys)
        print(header)
        for r in sorted_rows:
            line = f"  {r['id']:<8}" + "  ".join(
                f"{r[k]:8.3f}" if isinstance(r[k], float) else f"{r[k]:>8}"
                for k in keys)
            print(line)

    keys = [dice_k, cldice_k, b0_k, hd_k]

    # Dice 最低（最差病例）
    by_dice = sorted(rows, key=lambda r: r[dice_k])
    show(f"Dice 最低 {args.topn} 个（最难病例）", by_dice[:args.topn], keys)

    # Dice 最高（最好病例）
    show(f"Dice 最高 {args.topn} 个（最好病例）",
         by_dice[-args.topn:][::-1], keys)

    # clDice 最低
    by_cldice = sorted(rows, key=lambda r: r[cldice_k])
    show(f"clDice 最低 {args.topn} 个", by_cldice[:args.topn], keys)

    # 矛盾病例：Dice和clDice排名差异大
    # 计算每个病例 Dice排名 - clDice排名
    dice_rank = {r["id"]: i for i, r in enumerate(by_dice)}
    cldice_rank = {r["id"]: i for i, r in enumerate(by_cldice)}
    for r in rows:
        r["_rank_diff"] = dice_rank[r["id"]] - cldice_rank[r["id"]]

    # Dice排名远高于clDice排名（Dice好但clDice差）：体素重叠好但连通差
    by_diff = sorted(rows, key=lambda r: r["_rank_diff"], reverse=True)
    show("Dice相对好但clDice相对差（覆盖够但碎/断）",
         by_diff[:args.topn], keys + ["_rank_diff"])

    # 反过来：clDice好但Dice差（连通好但边界/细节差）
    show("clDice相对好但Dice相对差（连通够但边界/粗细差）",
         by_diff[-args.topn:][::-1], keys + ["_rank_diff"])

    # 建议看哪些病例可视化
    worst_ids = [r["id"] for r in by_dice[:5]]
    best_ids = [r["id"] for r in by_dice[-3:]]
    print(f"\n===== 建议可视化的病例 =====")
    print(f"  最差(看怎么失败): {','.join(worst_ids)}")
    print(f"  最好(作对照):     {','.join(best_ids)}")
    print(f"\n  下一步可视化命令示例:")
    print(f"  PYTHONPATH=. python scripts/vis_predict.py \\")
    print(f"      --cache-dir <cache> --ckpt runs/exp_2p5d/best.pth \\")
    print(f"      --case-ids {','.join(worst_ids)} --out-dir vis_worst")


if __name__ == "__main__":
    main()
