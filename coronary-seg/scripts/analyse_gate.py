#!/usr/bin/env python3
"""
scripts/analyse_gate.py — 残差门控在做什么？

动机
====
消融只能回答「有门控 vs 没门控，哪个指标更好」，回答不了「门控**是否按设计
工作**」。DEC-008 的设计意图是：

    stage-1 已经确信且正确的地方 gate→0（保持不动），
    stage-1 出错或模糊的地方 gate→1（大胆修正）。

本脚本用现有数据直接检验这个假设，**不需要重新训练或重新推理 stage-1**。

四个分区（按 stage-1 的对错切分，阈值 0.5）
    TP: label=1, prob>=0.5   stage-1 正确命中
    TN: label=0, prob< 0.5   stage-1 正确排除
    FN: label=1, prob< 0.5   stage-1 漏检   <- 需要修正
    FP: label=0, prob>=0.5   stage-1 假阳   <- 需要修正

判读
====
若 gate(FN), gate(FP) 显著高于 gate(TP), gate(TN)，说明门控确实学会了
「只在该改的地方施加修正」——这是机制性证据，独立于消融的指标输赢。
即便消融显示 no-gate 的 Dice 更高，该结论仍然成立并值得报告。

同时给出 gate 与**预测熵** H(p) = -p·log p - (1-p)·log(1-p) 的相关性：
熵高 = stage-1 拿不准。若二者正相关，说明门控利用了不确定性信号。

用法
====
    PYTHONPATH=. python scripts/analyse_gate.py \\
        --data-root /path/to/cache/stage2_tri/test \\
        --ckpt runs/stage2_tri/best.pth \\
        --out-csv runs/stage2_tri/gate_analysis.csv

    # 纯函数自测（不需要 torch/monai/数据）
    python scripts/analyse_gate.py --self-test
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REGIONS = ("TP", "TN", "FN", "FP")


def region_masks(prob, label, thr=0.5):
    """按 stage-1 的对错把体素切成四个分区。纯 numpy，可单测。"""
    pos_pred = prob >= thr
    pos_true = label > 0
    return {
        "TP": pos_true & pos_pred,
        "TN": (~pos_true) & (~pos_pred),
        "FN": pos_true & (~pos_pred),
        "FP": (~pos_true) & pos_pred,
    }


def gate_stats_by_region(gate, prob, label, thr=0.5):
    """各分区的 gate 均值与体素数。返回 {区域: (mean, count)}。

    空分区返回 (nan, 0) —— 例如 stage-1 完美的 patch 没有 FN/FP。
    """
    out = {}
    for name, m in region_masks(prob, label, thr).items():
        n = int(m.sum())
        out[name] = (float(gate[m].mean()) if n else float("nan"), n)
    return out


def binary_entropy(p, eps=1e-6):
    """逐体素预测熵，衡量 stage-1 的不确定程度。p=0.5 时最大。"""
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def corr(a, b):
    """Pearson 相关系数；退化输入返回 nan。"""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------
#  以下需要 torch / monai / 实际数据
# ----------------------------------------------------------------------
def collect_gate(model, image, prob, device, roi=128, n_patch=8, seed=0):
    """在若干随机 patch 上前向，取回 gate。

    用采样而非全图滑窗：本脚本做的是统计分析，采样足够且快得多。
    优先采样含前景的位置，否则绝大多数 patch 是纯背景，FN/FP 分区会太空。
    """
    import torch

    rng = np.random.default_rng(seed)
    D, H, W = prob.shape          # 实际是 (H,W,D)，三轴对称处理，命名不影响逻辑
    r = min(roi, D, H, W)

    # 优先在 stage-1 认为有血管的位置附近采样
    fg = np.argwhere(prob >= 0.5)
    centres = []
    for i in range(n_patch):
        if len(fg) and i < n_patch - 1:
            c = fg[rng.integers(len(fg))]
        else:                       # 留一个纯随机 patch 作为对照
            c = np.array([rng.integers(D), rng.integers(H), rng.integers(W)])
        centres.append(c)

    gates, probs, labels_idx = [], [], []
    for c in centres:
        s = [int(np.clip(c[k] - r // 2, 0, (D, H, W)[k] - r)) for k in range(3)]
        sl = tuple(slice(s[k], s[k] + r) for k in range(3))
        x = np.stack([image[sl], prob[sl]], axis=0)[None]      # (1,2,r,r,r)
        with torch.no_grad():
            xt = torch.from_numpy(x).float().to(device)
            _, parts = model(xt, return_parts=True)
        g = parts["gate"]
        if g is None:                                          # --no-gate 模型
            return None, None, None
        gates.append(g[0, 0].cpu().numpy())
        probs.append(prob[sl])
        labels_idx.append(sl)
    return gates, probs, labels_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", help="stage-2 npz 目录（含 image/prob/label）")
    ap.add_argument("--ckpt", help="stage-2 checkpoint")
    ap.add_argument("--out-csv", default="runs/gate_analysis.csv")
    ap.add_argument("--init-filters", type=int, default=16)
    ap.add_argument("--roi", type=int, default=128)
    ap.add_argument("--n-patch", type=int, default=8)
    ap.add_argument("--max-cases", type=int, default=30)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.data_root or not args.ckpt:
        print("需要 --data-root 与 --ckpt（或用 --self-test）", file=sys.stderr)
        return 1

    import torch
    from src.stage2_model import build_stage2_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_stage2_model({"init_filters": args.init_filters,
                                "use_gate": True}).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"device={device}  ckpt epoch={ckpt.get('epoch')} "
          f"val_dice={ckpt.get('val_dice')}")

    files = sorted(glob.glob(os.path.join(args.data_root, "*.npz")))
    if args.max_cases:
        files = files[:args.max_cases]
    print(f"分析 {len(files)} 个病例，每例 {args.n_patch} 个 {args.roi}³ patch")

    rows = []
    agg = {k: [] for k in REGIONS}
    corrs = []
    for i, f in enumerate(files, 1):
        d = np.load(f)
        image, prob, label = d["image"], d["prob"], d["label"]
        if image.ndim == 4:
            image = image[0]
        res = collect_gate(model, image, prob, device, args.roi,
                           args.n_patch, seed=i)
        if res[0] is None:
            print("该 checkpoint 不含门控（--no-gate 训练），无法分析")
            return 1
        gates, probs, slices = res

        for g, p, sl in zip(gates, probs, slices):
            st = gate_stats_by_region(g, p, label[sl], args.thr)
            c = corr(g, binary_entropy(p))
            if not np.isnan(c):
                corrs.append(c)
            for k in REGIONS:
                if st[k][1]:
                    agg[k].append(st[k][0])
            rows.append({"case": os.path.basename(f), "corr_gate_entropy": c,
                         **{f"gate_{k}": st[k][0] for k in REGIONS},
                         **{f"n_{k}": st[k][1] for k in REGIONS}})
        if i % 10 == 0:
            print(f"  [{i}/{len(files)}]")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 62)
    print("  门控在各分区的平均强度（按 stage-1 的对错切分）")
    print("=" * 62)
    desc = {"TP": "stage-1 正确命中", "TN": "stage-1 正确排除",
            "FN": "stage-1 漏检  <- 该修", "FP": "stage-1 假阳  <- 该修"}
    for k in REGIONS:
        v = agg[k]
        if v:
            print(f"  gate({k})  {np.mean(v):.4f}   {desc[k]}   (n={len(v)} patch)")
    need = [np.mean(agg[k]) for k in ("FN", "FP") if agg[k]]
    keep = [np.mean(agg[k]) for k in ("TP", "TN") if agg[k]]
    if need and keep:
        ratio = np.mean(need) / max(np.mean(keep), 1e-8)
        print(f"\n  「该修区域」/「该留区域」的门控强度比 = {ratio:.2f}")
        print("  > 1 表示门控确实更多作用在 stage-1 出错处（符合 DEC-008 的设计意图）")
        print("  ≈ 1 表示门控接近均匀施加，未体现选择性")
    if corrs:
        print(f"\n  gate 与预测熵的相关系数 = {np.mean(corrs):+.4f}")
        print("  正值表示门控在 stage-1 拿不准的地方更活跃")
    print("=" * 62)
    print(f"逐 patch 明细已写入 {args.out_csv}")
    return 0


def self_test():
    """纯函数自测：不需要 torch/monai/数据。"""
    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fail += 1
            print(f"  FAIL  {name}")

    print("\n[1] region_masks 的四分区")
    prob = np.array([0.9, 0.1, 0.2, 0.8])
    label = np.array([1, 0, 1, 0])
    m = region_masks(prob, label)
    chk("TP = 高概率且是血管", m["TP"].tolist() == [True, False, False, False])
    chk("TN = 低概率且非血管", m["TN"].tolist() == [False, True, False, False])
    chk("FN = 低概率但是血管（漏检）", m["FN"].tolist() == [False, False, True, False])
    chk("FP = 高概率但非血管（假阳）", m["FP"].tolist() == [False, False, False, True])
    chk("四分区互斥且完备",
        sum(int(m[k].sum()) for k in REGIONS) == prob.size)

    print("\n[2] gate_stats_by_region")
    gate = np.array([0.1, 0.1, 0.9, 0.9])          # 只在 FN/FP 上高
    st = gate_stats_by_region(gate, prob, label)
    chk("gate(FN) 高", abs(st["FN"][0] - 0.9) < 1e-9)
    chk("gate(TP) 低", abs(st["TP"][0] - 0.1) < 1e-9)
    chk("计数正确", all(st[k][1] == 1 for k in REGIONS))
    # 全部命中的 patch：TP 有值，其余三区为空 -> 应返回 nan 而非报错
    st2 = gate_stats_by_region(np.array([0.5]), np.array([0.9]), np.array([1]))
    chk("空分区返回 nan 而非报错",
        np.isnan(st2["FN"][0]) and st2["FN"][1] == 0 and st2["TP"][1] == 1)

    print("\n[3] binary_entropy")
    e = binary_entropy(np.array([0.5, 0.01, 0.99]))
    chk("p=0.5 时熵最大", e[0] > e[1] and e[0] > e[2])
    chk("熵非负", (e >= 0).all())
    chk("p=0/1 不产生 inf/nan", np.isfinite(binary_entropy(np.array([0.0, 1.0]))).all())

    print("\n[4] corr")
    x = np.arange(10.0)
    chk("完全正相关 = 1", abs(corr(x, 2 * x) - 1.0) < 1e-9)
    chk("完全负相关 = -1", abs(corr(x, -x) + 1.0) < 1e-9)
    chk("常量输入返回 nan（不崩）", np.isnan(corr(x, np.ones(10))))

    print(f"\n{'=' * 46}\n  通过 {ok}   失败 {fail}\n{'=' * 46}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
