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


def gate_stats_by_region(gate, prob, label, thr=0.5, delta=None):
    """各分区的门控与修正量统计。返回 {区域: dict}。

    只看 gate 是不够的：实际施加的修正是 **g x delta**。
    背景区域可能 gate 很高但 delta≈0，那样 gate 高就无害 —— 这种情况下
    结论应当是「门控冗余」而非「门控失效」，二者对论文的含义不同。

    每个区域返回:
        gate  —— 门控强度均值
        adel  —— |delta| 均值（网络想改多少）
        acorr —— |g x delta| 均值（实际改了多少）
        n     —— 体素数
    空分区返回 nan —— 例如 stage-1 完美的 patch 没有 FN/FP。
    """
    out = {}
    for name, m in region_masks(prob, label, thr).items():
        n = int(m.sum())
        if not n:
            out[name] = {"gate": float("nan"), "adel": float("nan"),
                         "acorr": float("nan"), "n": 0}
            continue
        rec = {"gate": float(gate[m].mean()), "n": n}
        if delta is not None:
            ad = np.abs(delta)
            rec["adel"] = float(ad[m].mean())
            rec["acorr"] = float((gate * ad)[m].mean())
        else:
            rec["adel"] = rec["acorr"] = float("nan")
        out[name] = rec
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


def pick_device(pref="auto"):
    """选设备。

    login 节点上 torch.cuda.is_available() 可能返回 True，但真正把张量搬上去
    会抛 "CUDA-capable device(s) is/are busy or unavailable"。所以 auto 模式
    实际做一次搬运测试，失败就回退 CPU，而不是信任那个标志位。
    """
    import torch

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(1).to("cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"[device] CUDA 不可用（{type(e).__name__}），回退 CPU。"
              f"如需 GPU 请用 sbatch slurm/analyse_gate.sbatch 提交作业")
        return "cpu"


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

    gates, deltas, probs, labels_idx = [], [], [], []
    for c in centres:
        s = [int(np.clip(c[k] - r // 2, 0, (D, H, W)[k] - r)) for k in range(3)]
        sl = tuple(slice(s[k], s[k] + r) for k in range(3))
        x = np.stack([image[sl], prob[sl]], axis=0)[None]      # (1,2,r,r,r)
        with torch.no_grad():
            xt = torch.from_numpy(x).float().to(device)
            _, parts = model(xt, return_parts=True)
        g = parts["gate"]
        if g is None:                                          # --no-gate 模型
            return None, None, None, None
        gates.append(g[0, 0].cpu().numpy())
        deltas.append(parts["delta"][0, 0].cpu().numpy())
        probs.append(prob[sl])
        labels_idx.append(sl)
    return gates, deltas, probs, labels_idx


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
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="auto 会实际试一次 CUDA 初始化，失败则回退 CPU —— "
                         "login 节点上 torch.cuda.is_available() 可能返回 True "
                         "但设备其实不可用")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if not args.data_root or not args.ckpt:
        print("需要 --data-root 与 --ckpt（或用 --self-test）", file=sys.stderr)
        return 1

    import torch
    from src.stage2_model import build_stage2_model

    device = pick_device(args.device)
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
    agg = {k: {"gate": [], "adel": [], "acorr": []} for k in REGIONS}
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
        gates, deltas, probs, slices = res

        for g, dl, p, sl in zip(gates, deltas, probs, slices):
            st = gate_stats_by_region(g, p, label[sl], args.thr, delta=dl)
            c = corr(g, binary_entropy(p))
            if not np.isnan(c):
                corrs.append(c)
            for k in REGIONS:
                if st[k]["n"]:
                    for metric in ("gate", "adel", "acorr"):
                        agg[k][metric].append(st[k][metric])
            rows.append({"case": os.path.basename(f), "corr_gate_entropy": c,
                         **{f"{m}_{k}": st[k][m] for k in REGIONS
                            for m in ("gate", "adel", "acorr")},
                         **{f"n_{k}": st[k]["n"] for k in REGIONS}})
        if i % 10 == 0:
            print(f"  [{i}/{len(files)}]")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    import csv
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 74)
    print("  门控与修正量（按 stage-1 的对错切分）")
    print("=" * 74)
    desc = {"TP": "stage-1 正确命中", "TN": "stage-1 正确排除",
            "FN": "stage-1 漏检  <- 该修", "FP": "stage-1 假阳  <- 该修"}
    print(f"  {'区域':<6}{'gate':>8}{'|delta|':>10}{'|g*delta|':>12}   说明")
    for k in REGIONS:
        a = agg[k]
        if a["gate"]:
            print(f"  {k:<6}{np.mean(a['gate']):>8.4f}{np.mean(a['adel']):>10.4f}"
                  f"{np.mean(a['acorr']):>12.4f}   {desc[k]}")

    def mean_of(regs, metric):
        v = [np.mean(agg[k][metric]) for k in regs if agg[k][metric]]
        return np.mean(v) if v else float("nan")

    print("\n  ---- 判读 ----")
    g_need, g_keep = mean_of(("FN", "FP"), "gate"), mean_of(("TP", "TN"), "gate")
    c_need, c_keep = mean_of(("FN", "FP"), "acorr"), mean_of(("TP", "TN"), "acorr")
    if np.isfinite(g_need) and np.isfinite(g_keep):
        print(f"  门控强度比  gate(该修)/gate(该留) = {g_need / max(g_keep, 1e-8):.2f}")
    if np.isfinite(c_need) and np.isfinite(c_keep):
        print(f"  实际修正比  |g*d|(该修)/|g*d|(该留) = {c_need / max(c_keep, 1e-8):.2f}")
    print("  > 1 表示修正确实集中在 stage-1 出错处（符合 DEC-008 设计意图）")

    all_gate = [x for k in REGIONS for x in agg[k]["gate"]]
    if all_gate:
        gm = np.mean(all_gate)
        print(f"\n  全局 gate 均值 = {gm:.4f}")
        if gm > 0.9:
            print("  ** gate 处处接近 1 -> refined = prob + g*delta 退化为 prob + delta，")
            print("     门控在数学上等价于无门控。这解释了消融中 no-gate 不劣的结果。")
        elif gm < 0.2:
            print("  ** gate 处处接近 0 -> 精修几乎未生效，网络退化为恒等映射。")
    if corrs:
        print(f"\n  gate 与预测熵的相关系数 = {np.mean(corrs):+.4f}")
        print("  正值表示门控在 stage-1 拿不准的地方更活跃；≈0 表示未利用不确定性信号")
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
    chk("gate(FN) 高", abs(st["FN"]["gate"] - 0.9) < 1e-9)
    chk("gate(TP) 低", abs(st["TP"]["gate"] - 0.1) < 1e-9)
    chk("计数正确", all(st[k]["n"] == 1 for k in REGIONS))
    chk("不传 delta 时 adel/acorr 为 nan", np.isnan(st["TP"]["adel"]))
    # 全部命中的 patch：TP 有值，其余三区为空 -> 应返回 nan 而非报错
    st2 = gate_stats_by_region(np.array([0.5]), np.array([0.9]), np.array([1]))
    chk("空分区返回 nan 而非报错",
        np.isnan(st2["FN"]["gate"]) and st2["FN"]["n"] == 0 and st2["TP"]["n"] == 1)

    print("\n[3] delta 统计 —— 区分「门控失效」与「门控冗余」")
    # 场景：gate 处处接近 1，但 delta 只在 FN/FP 上非零
    #       -> 门控本身无选择性，但实际修正仍集中在该修处 = 冗余而非失效
    g_hi = np.array([0.95, 0.95, 0.95, 0.95])
    d_sel = np.array([0.0, 0.0, 2.0, 2.0])          # 只想改 FN/FP
    st3 = gate_stats_by_region(g_hi, prob, label, delta=d_sel)
    chk("|delta| 在该改处大", st3["FN"]["adel"] > st3["TP"]["adel"])
    chk("实际修正量 = gate x |delta|",
        abs(st3["FN"]["acorr"] - 0.95 * 2.0) < 1e-9)
    chk("该留处实际修正为 0（门控冗余而非失效）",
        abs(st3["TP"]["acorr"]) < 1e-9)
    # 场景：delta 处处相同 -> 修正无选择性，才是真正的失效
    d_flat = np.full(4, 1.0)
    st4 = gate_stats_by_region(g_hi, prob, label, delta=d_flat)
    chk("delta 均匀时实际修正也均匀（真·失效）",
        abs(st4["FN"]["acorr"] - st4["TP"]["acorr"]) < 1e-9)

    print("\n[4] binary_entropy")
    e = binary_entropy(np.array([0.5, 0.01, 0.99]))
    chk("p=0.5 时熵最大", e[0] > e[1] and e[0] > e[2])
    chk("熵非负", (e >= 0).all())
    chk("p=0/1 不产生 inf/nan", np.isfinite(binary_entropy(np.array([0.0, 1.0]))).all())

    print("\n[5] corr")
    x = np.arange(10.0)
    chk("完全正相关 = 1", abs(corr(x, 2 * x) - 1.0) < 1e-9)
    chk("完全负相关 = -1", abs(corr(x, -x) + 1.0) < 1e-9)
    chk("常量输入返回 nan（不崩）", np.isnan(corr(x, np.ones(10))))

    print(f"\n{'=' * 46}\n  通过 {ok}   失败 {fail}\n{'=' * 46}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
