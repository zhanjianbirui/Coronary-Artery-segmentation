#!/usr/bin/env python3
"""
scripts/compare_runs.py — 多方案逐病例配对显著性比较
==================================================================
为什么需要这个：均值差不等于有提升。2026-08-24 那次按均值几乎把三正交
相对单轴+TTA 的 Dice 说成"打平"，配对检验才看出那是**统计显著的小幅
下降**（p=0.015），而看起来更大的 HD95 改善反而**不显著**（p=0.064，
均值被重尾拉动）。论文里每个 Δ 都该配 p 值。

本脚本只读结果 csv，不需要 GPU / torch / monai —— **login 节点直接跑**。

支持两种 csv 列名格式，用 `路径:前缀` 指定：
  - predict.py / predict_tri.py:  raw_dice / pp_dice / ...   → 前缀 raw 或 pp
  - predict_stage2.py:            s1_dice / s2_dice / ...    → 前缀 s1 或 s2

用法：
  PYTHONPATH=. python scripts/compare_runs.py \\
      --baseline "三正交=runs/exp_tri2p5d/test_metrics_tri_mean050.csv:pp" \\
      --runs "单轴=runs/exp_2p5d/test_metrics_optimal.csv:pp" \\
             "单轴+TTA=runs/exp_2p5d/test_final_tta.csv:pp" \\
             "Stage2=runs/stage2/test_metrics_stage2.csv:s2"

检验：Wilcoxon 符号秩（配对、非参数，不假设正态；HD95 重尾时比 t 检验稳）。
多重比较用 Holm-Bonferroni 校正 —— 一次比 4 个指标 × 若干方案，
不校正会虚增假阳性，论文里这是会被问的点。
"""

import os
import sys
import csv
import argparse
import numpy as np

# 指标方向：True = 越大越好
METRICS = {
    "dice": True,
    "cldice": True,
    "betti0_err": False,
    "hd95": False,
}


def load_run(spec):
    """解析 'LABEL=path:prefix'，返回 (label, {case_id: {metric: value}})。"""
    if "=" not in spec:
        raise ValueError(f"缺少 LABEL=：{spec}")
    label, rest = spec.split("=", 1)
    if ":" not in rest:
        raise ValueError(f"缺少 :前缀（如 :pp / :s2）：{spec}")
    path, prefix = rest.rsplit(":", 1)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    data = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cid = str(r.get("id"))
            vals = {}
            for m in METRICS:
                key = f"{prefix}_{m}"
                if key not in r:
                    raise KeyError(
                        f"{path} 里没有列 {key}。该文件的列是：{list(r.keys())}")
                try:
                    v = float(r[key])
                except (TypeError, ValueError):
                    v = float("nan")
                vals[m] = v
            data[cid] = vals
    return label.strip(), data


def paired_test(a, b):
    """a, b 为配对样本。返回 (p 值, 检验名)。b 是基线。"""
    diff = np.asarray(a) - np.asarray(b)
    nz = diff[diff != 0]
    if len(nz) == 0:
        return float("nan"), "全部相同"
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(a, b).pvalue), "Wilcoxon"
    except ImportError:
        # 退化为符号检验（二项），无 scipy 时仍能给结论
        from math import comb
        n = len(nz)
        k = int((nz > 0).sum())
        k = min(k, n - k)
        p = 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        return float(min(1.0, p)), "符号检验"


def holm(pvals):
    """Holm-Bonferroni 校正，返回与输入等长的校正后 p。"""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    if not idx:
        return list(pvals)
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    out = list(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)      # 保证单调不减
        out[i] = running
    return out


def compare(base_label, base_data, run_label, run_data, alpha=0.05):
    ids = sorted(set(base_data) & set(run_data))
    rows = []
    for m, higher_better in METRICS.items():
        a = [run_data[i][m] for i in ids]
        b = [base_data[i][m] for i in ids]
        keep = [j for j in range(len(ids))
                if np.isfinite(a[j]) and np.isfinite(b[j])]
        a = [a[j] for j in keep]
        b = [b[j] for j in keep]
        if not a:
            rows.append({"metric": m, "n": 0})
            continue
        diff = np.array(a) - np.array(b)
        better = int((diff > 0).sum() if higher_better else (diff < 0).sum())
        worse = int((diff < 0).sum() if higher_better else (diff > 0).sum())
        p, test = paired_test(a, b)
        rows.append({
            "metric": m, "n": len(a),
            "base": float(np.mean(b)), "run": float(np.mean(a)),
            "delta": float(np.mean(diff)),
            "better": better, "worse": worse, "tie": len(a) - better - worse,
            "p_raw": p, "test": test,
            "improved": (diff.mean() > 0) == higher_better,
        })
    ps = [r.get("p_raw", float("nan")) for r in rows]
    for r, pc in zip(rows, holm(ps)):
        r["p_holm"] = pc
    return ids, rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True,
                   help="基线，格式 LABEL=path:prefix")
    p.add_argument("--runs", nargs="+", required=True,
                   help="待比较方案，可多个，格式同上")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-csv", default=None, help="可选，把结果表存成 csv")
    return p.parse_args()


def main():
    args = parse_args()
    base_label, base_data = load_run(args.baseline)
    print(f"基线: {base_label}  ({len(base_data)} 例)")

    all_rows = []
    for spec in args.runs:
        label, data = load_run(spec)
        ids, rows = compare(base_label, base_data, label, data, args.alpha)
        print("\n" + "=" * 78)
        print(f"  {label}  vs  {base_label}   （配对 {len(ids)} 例）")
        print("=" * 78)
        print(f"  {'指标':<12} {'基线':>9} {'本方案':>9} {'Δ':>9} "
              f"{'优/劣':>9} {'p(原始)':>10} {'p(Holm)':>10}  结论")
        print("  " + "-" * 74)
        for r in rows:
            if r["n"] == 0:
                print(f"  {r['metric']:<12} 无可配对数据")
                continue
            sig = r["p_holm"] < args.alpha and not np.isnan(r["p_holm"])
            if not sig:
                verdict = "不显著"
            else:
                verdict = "显著提升 ✓" if r["improved"] else "显著变差 ✗"
            print(f"  {r['metric']:<12} {r['base']:>9.4f} {r['run']:>9.4f} "
                  f"{r['delta']:>+9.4f} {r['better']:>4}/{r['worse']:<4} "
                  f"{r['p_raw']:>10.2e} {r['p_holm']:>10.2e}  {verdict}")
            all_rows.append({"run": label, "baseline": base_label, **r})
        print(f"\n  检验: {rows[0].get('test','-')}（配对非参数）"
              f" + Holm-Bonferroni 校正（{len(METRICS)} 个指标）")
        print(f"  优/劣 = 本方案在多少例上更好 / 更差（平局不计）")

    if args.out_csv and all_rows:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n结果表已存: {args.out_csv}")

    print("\n提醒：Δ 为正不代表有提升 —— 看 p(Holm) 是否 < alpha。"
          "\n      HD95 是重尾分布，均值易被少数极端病例主导，"
          "配对检验比看均值可靠。")


if __name__ == "__main__":
    main()
