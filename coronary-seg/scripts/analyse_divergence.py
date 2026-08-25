#!/usr/bin/env python3
"""
scripts/analyse_divergence.py — 三正交融合丢掉了什么？

动机
====
现在的融合是 mean：三个方向的概率逐体素取平均。这一步把三个数压成一个，
**方向间的分歧（std）被完全丢弃**。但分歧本身是信息：

    三个方向都说是血管   -> 一致，可信
    只有一个方向说是     -> 分歧大，很可能是该方向特有的假阳

而 analyse_gate.py 的结果显示，stage-2 的残差门控与预测熵的相关系数只有
-0.065，即网络无法自行从融合后的概率里推断出「这里该不该信」。
若分歧确实标记了出错位置，把它显式提供给下游就有明确依据。

本脚本回答两个问题
==================
A. 分歧是否标记了 stage-1 的错误？
   按 std 分桶，比较各桶的错误率（FN+FP 占比）。若单调上升，则分歧有判别力。
   同时给出 AUC：用 std 单独预测「此处 stage-1 是否出错」的能力。

B. 用分歧做自适应阈值能否改善指标？
   thr(x) = thr0 + alpha * std(x)
     alpha > 0  分歧大处更严格（抑制假阳）
     alpha < 0  分歧大处更宽松（保住细分支）
   扫描 alpha，与固定阈值基线对比 Dice/clDice/Betti-0/HD95。

注意：本脚本**不修改任何既有推理路径**，只是把 predict_tri_probs 已经
算出、但随后被 mean 丢弃的信息捡回来分析。

用法
====
    PYTHONPATH=. python scripts/analyse_divergence.py \\
        --cache-dir /path/to/cache --ckpt runs/exp_tri2p5d/best.pth \\
        --max-cases 20 --out-csv runs/exp_tri2p5d/divergence.csv

    python scripts/analyse_divergence.py --self-test
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ALPHAS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)


def divergence_stats(probs, label, thr=0.5):
    """分歧与 stage-1 错误的关系。

    probs: list of (H,W,D)，每个方向一个概率体
    返回 dict，含分桶错误率与 AUC。
    """
    # 中间累加用 float64：概率的方向间标准差本身量级很小（典型 1e-2 以下），
    # float32 累加的舍入噪声可达 1e-7，会在「三方向一致」处产生虚假分歧。
    # 结果转回 float32 存储，额外内存只在临时量上。
    stack = np.stack(probs, axis=0).astype(np.float32)
    if stack.ndim != 4:
        raise ValueError(
            f"期望每个方向一个 3D 概率体，堆叠后应为 4D，实得 {stack.shape}。"
            f"常见原因：predict_tri_probs 返回 dict，直接迭代拿到的是 axis 编号。")
    mean = stack.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = stack.std(axis=0, dtype=np.float64).astype(np.float32)

    pred = mean >= thr
    gt = label > 0
    wrong = pred != gt                       # stage-1 在此处出错

    return mean, std, wrong


def analysis_roi(mean, label, min_prob=0.05):
    """限定分析区域：模型有响应处 或 真值处。

    为什么不能用全体素：冠脉占体积不足 1%，其余是三方向一致认定为背景
    (std=0) 且预测正确的区域。把它们计入会造成两个问题 ——
      1. std 的分位数大量重复，低位分桶为空 -> nanmean 全 NaN
      2. AUC 被这些"无争议且正确"的体素主导而虚高，失去意义
    只在模型有响应或真值所在的区域评估分歧的判别力，才是有信息量的问题。
    """
    return (mean >= min_prob) | (label > 0)


def bucket_error_rate(std, wrong, n_bins=5):
    """按 std 分位数分桶，给出每桶的错误率。

    若错误率随 std 单调上升，说明分歧确实标记了出错位置。
    """
    qs = np.quantile(std, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-6
    out = []
    for i in range(n_bins):
        m = (std >= qs[i]) & (std < qs[i + 1])
        n = int(m.sum())
        out.append({"bin": i, "lo": float(qs[i]), "hi": float(qs[i + 1]),
                    "n": n,
                    "err_rate": float(wrong[m].mean()) if n else float("nan")})
    return out


def auc_score(score, positive):
    """用 score 预测 positive 的 AUC（Mann-Whitney U，无需 sklearn）。

    0.5 = 无判别力；越接近 1 说明 score 越能指出出错位置。
    大体积时对负样本下采样以控制内存。
    """
    s_pos = score[positive]
    s_neg = score[~positive]
    if s_pos.size == 0 or s_neg.size == 0:
        return float("nan")
    cap = 200_000
    rng = np.random.default_rng(0)
    if s_pos.size > cap:
        s_pos = rng.choice(s_pos, cap, replace=False)
    if s_neg.size > cap:
        s_neg = rng.choice(s_neg, cap, replace=False)
    allv = np.concatenate([s_pos, s_neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    r_pos = ranks[:s_pos.size].sum()
    n1, n2 = float(s_pos.size), float(s_neg.size)
    return float((r_pos - n1 * (n1 + 1) / 2) / (n1 * n2))


def adaptive_threshold_mask(mean, std, thr0=0.5, alpha=0.0):
    """thr(x) = thr0 + alpha * std(x)，逐体素阈值。

    alpha>0: 分歧大处更严格（抑制只在单方向出现的假阳）
    alpha<0: 分歧大处更宽松（保住可能被平均稀释的细分支）
    阈值裁剪到 (0,1) 开区间，避免全 0 或全 1。
    """
    thr = np.clip(thr0 + alpha * std, 1e-3, 1.0 - 1e-3)
    return mean >= thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir")
    ap.add_argument("--ckpt")
    ap.add_argument("--split-json", default="splits/split.json")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--backbone", default="segresnet")
    ap.add_argument("--init-filters", type=int, default=32)
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--hu-min", type=float, default=-200.0)
    ap.add_argument("--hu-max", type=float, default=800.0)
    ap.add_argument("--axes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--thr", type=float, default=0.50)
    ap.add_argument("--min-voxels", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pad-multiple", type=int, default=32)
    ap.add_argument("--max-cases", type=int, default=20)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--roi-min-prob", type=float, default=0.05,
                    help="分析区域下限：低于该概率且非真值的体素视为"
                         "无争议背景，排除在分歧分析之外")
    ap.add_argument("--out-csv", default="runs/exp_tri2p5d/divergence.csv")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.cache_dir or not args.ckpt:
        print("需要 --cache-dir 与 --ckpt（或 --self-test）", file=sys.stderr)
        return 1

    import torch
    from monai.data import PersistentDataset

    from src.data import load_split, build_preprocess
    from src.model import build_model
    # 复用 predict_tri.py 的推理与指标实现，保证与主流程口径完全一致。
    # 它有 if __name__ == "__main__" 保护，import 不会触发 argparse。
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from predict_tri import (predict_tri_probs, postprocess, dice_coef,
                             cldice_coef, betti0_error, hd95)

    device = args.device
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            try:
                torch.zeros(1).to("cuda")
                device = "cuda"
            except RuntimeError:
                print("[device] CUDA 不可用，回退 CPU")
    print(f"device={device}  ckpt={args.ckpt}  axes={args.axes}")

    cfg = {"k": args.k, "backbone": args.backbone,
           "init_filters": args.init_filters, "out_channels": 1}
    model = build_model(cfg).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"checkpoint epoch={ck.get('epoch')} val_dice={ck.get('val_dice')}")

    preprocess = build_preprocess(args.spacing, args.hu_min, args.hu_max)
    _, _, test_rec = load_split(args.split_json)
    cache = PersistentDataset(data=test_rec, transform=preprocess,
                              cache_dir=args.cache_dir)
    n = min(args.max_cases, len(test_rec)) if args.max_cases else len(test_rec)
    print(f"分析 {n} 例\n")

    rows, buckets_all, aucs = [], [], []
    roi_fracs, zero_fracs = [], []
    alpha_metrics = {a: [] for a in ALPHAS}

    for i in range(n):
        item = cache[i]
        image = item["image"]
        label = np.asarray(item["label"])[0].astype(np.uint8)

        # predict_tri_probs 返回的是 dict{axis: prob_vol}，不是 list ——
        # 直接迭代会拿到 axis 编号而非概率体。
        prob_by_axis = predict_tri_probs(model, image, args.k, device,
                                         axes=tuple(args.axes),
                                         batch=args.batch,
                                         pad_multiple=args.pad_multiple)
        probs = [np.asarray(prob_by_axis[ax], dtype=np.float32)
                 for ax in args.axes]
        mean, std, wrong = divergence_stats(probs, label, args.thr)

        # ---- A. 分歧是否标记了错误（只在有争议的区域内评估）----
        roi = analysis_roi(mean, label, args.roi_min_prob)
        n_roi, n_all = int(roi.sum()), int(roi.size)
        zero_frac = float((std[roi] == 0).mean()) if n_roi else float("nan")
        std_r, wrong_r = std[roi], wrong[roi]

        bk = bucket_error_rate(std_r, wrong_r, args.n_bins)
        buckets_all.append([b["err_rate"] for b in bk])
        a = auc_score(std_r, wrong_r)
        aucs.append(a)
        roi_fracs.append(n_roi / n_all)
        zero_fracs.append(zero_frac)

        # ---- B. 自适应阈值 ----
        for alpha in ALPHAS:
            m = adaptive_threshold_mask(mean, std, args.thr, alpha)
            m = postprocess(m.astype(np.uint8), min_voxels=args.min_voxels,
                            max_gap=0)
            be, _, _ = betti0_error(m, label)
            alpha_metrics[alpha].append(
                (dice_coef(m, label), cldice_coef(m, label), be,
                 hd95(m, label)))

        rows.append({"case": test_rec[i].get("id", i), "auc_std_vs_error": a,
                     "roi_frac": n_roi / n_all, "std_zero_frac_in_roi": zero_frac,
                     "std_mean_roi": float(std_r.mean()) if n_roi else float("nan"),
                     "std_max": float(std.max()),
                     **{f"err_bin{b['bin']}": b["err_rate"] for b in bk}})
        print(f"  [{i + 1}/{n}] AUC={a:.3f}  ROI={n_roi / n_all:.3%}  "
              f"ROI内std均值={std_r.mean() if n_roi else float('nan'):.4f}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 70)
    print("  A. 方向分歧(std)是否标记了 stage-1 的错误？")
    print("=" * 70)
    print(f"  分析区域 = (mean >= {args.roi_min_prob}) 或 真值处，"
          f"平均占全体积 {np.mean(roi_fracs):.2%}")
    print(f"  该区域内 std 恰为 0 的体素占 {np.nanmean(zero_fracs):.1%}"
          f"（三方向完全一致）\n")
    with np.errstate(invalid="ignore"):
        bm = np.nanmean(np.array(buckets_all, dtype=np.float64), axis=0)
    print(f"  按 std 分位数分 {args.n_bins} 桶，各桶的错误率：")
    valid = [(i, v) for i, v in enumerate(bm) if np.isfinite(v)]
    for i, v in valid:
        lab = "最低" if i == 0 else ("最高" if i == len(bm) - 1 else "  中")
        print(f"    第{i + 1}桶(std {lab})  错误率 {v:.4f}  {'#' * int(v * 200)}")
    n_empty = len(bm) - len(valid)
    if n_empty:
        print(f"    （{n_empty} 个桶为空：std 取值高度集中，分位数重复）")
    if len(valid) >= 2:
        lo, hi = valid[0][1], valid[-1][1]
        print(f"\n  最高桶/最低桶 错误率之比 = {hi / max(lo, 1e-9):.1f}")
    print(f"  AUC(用 std 预测'此处出错') = {np.nanmean(aucs):.4f}")
    print("  0.5=无判别力；>0.7 说明分歧确实指向了出错位置")

    print("\n" + "=" * 70)
    print("  B. 用分歧做自适应阈值  thr(x) = 0.50 + alpha * std(x)")
    print("=" * 70)
    print(f"  {'alpha':>7}{'Dice':>9}{'clDice':>9}{'Betti0':>9}{'HD95':>9}   说明")
    base = None
    for alpha in ALPHAS:
        v = np.array(alpha_metrics[alpha], dtype=np.float64)
        m = np.nanmean(v, axis=0)
        if alpha == 0.0:
            base = m
        tag = "  <- 基线(固定阈值)" if alpha == 0.0 else ""
        print(f"  {alpha:>7.2f}{m[0]:>9.4f}{m[1]:>9.4f}{m[2]:>9.2f}{m[3]:>9.2f}{tag}")
    if base is not None:
        print("\n  相对基线的变化：")
        for alpha in ALPHAS:
            if alpha == 0.0:
                continue
            m = np.nanmean(np.array(alpha_metrics[alpha]), axis=0)
            d = m - base
            print(f"  alpha={alpha:>5.2f}  dDice={d[0]:+.4f}  dclDice={d[1]:+.4f}"
                  f"  dBetti0={d[2]:+.2f}  dHD95={d[3]:+.2f}")
    print("=" * 70)
    print(f"\n逐病例明细写入 {args.out_csv}")
    print("注意：这是子集结果，只用于判断方向，绝对增益需全量验证。")
    return 0


def self_test():
    ok = fail = 0

    def chk(name, cond):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fail += 1
            print(f"  FAIL  {name}")

    print("\n[1] divergence_stats")
    # 三个方向完全一致 -> std 应为 0
    p = np.full((4, 4, 4), 0.9, dtype=np.float32)
    mean, std, wrong = divergence_stats([p, p, p], np.ones((4, 4, 4), np.uint8))
    chk("三方向一致时 std=0（float64 累加，无虚假分歧）", np.allclose(std, 0, atol=1e-9))
    chk("mean 等于原值", np.allclose(mean, 0.9))
    chk("全部命中时无错误", not wrong.any())
    # 一个方向不同 -> std > 0
    q = np.full((4, 4, 4), 0.1, dtype=np.float32)
    _, std2, _ = divergence_stats([p, p, q], np.ones((4, 4, 4), np.uint8))
    chk("单方向分歧产生 std>0", (std2 > 0).all())

    # 这个坑真实发生过：predict_tri_probs 返回 dict，直接迭代得到的是 axis
    # 编号（标量），堆叠后 std 归约成 0-d，一路静默到 auc_score 才报
    # "invalid index to scalar variable"。护栏让它在源头就报清楚。
    try:
        divergence_stats([0, 1, 2], np.ones((4, 4, 4), np.uint8))
        chk("误传标量列表时被护栏拦下", False)
    except ValueError as e:
        chk("误传标量列表时被护栏拦下（而非静默到 AUC 才炸）",
            "4D" in str(e))

    print("\n[2] analysis_roi —— 排除无争议背景")
    mean_v = np.array([0.9, 0.5, 0.01, 0.01])
    lab_v = np.array([1, 0, 1, 0], dtype=np.uint8)
    roi = analysis_roi(mean_v, lab_v, 0.05)
    chk("高概率处纳入", roi[0] and roi[1])
    chk("低概率但是真值 -> 纳入（漏检必须能被看到）", roi[2])
    chk("低概率且非真值 -> 排除（无争议背景）", not roi[3])

    print("\n[3] 空桶不再导致崩溃")
    # 真实数据里背景 std 恒为 0，分位数大量重复 -> 低位桶为空
    std_z = np.concatenate([np.zeros(800), np.linspace(0.01, 0.5, 200)])
    wr = np.zeros(1000, dtype=bool); wr[900:] = True
    bk_z = bucket_error_rate(std_z, wr, 5)
    chk("分位数重复时仍返回 n_bins 个桶", len(bk_z) == 5)
    empty = [b for b in bk_z if b["n"] == 0]
    chk("空桶的 err_rate 为 nan 而非报错",
        all(np.isnan(b["err_rate"]) for b in empty))
    arr = np.array([[b["err_rate"] for b in bk_z]], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        bm_z = np.nanmean(arr, axis=0)
    chk("汇总时 nan 可被 isfinite 过滤（不会 int(nan) 崩）",
        all(np.isfinite(v) for v in bm_z if np.isfinite(v)))

    print("\n[4] bucket_error_rate")
    std = np.linspace(0, 1, 1000)
    wrong = std > 0.8                       # 只有高 std 处出错
    bk = bucket_error_rate(std, wrong, 5)
    chk("分 5 桶", len(bk) == 5)
    chk("错误率随 std 单调不减",
        all(bk[i]["err_rate"] <= bk[i + 1]["err_rate"] + 1e-9 for i in range(4)))
    chk("最低桶错误率为 0", abs(bk[0]["err_rate"]) < 1e-9)
    chk("最高桶错误率最大", bk[-1]["err_rate"] > 0.5)

    print("\n[5] auc_score")
    score = np.array([0.1, 0.2, 0.8, 0.9])
    pos = np.array([False, False, True, True])
    chk("完美区分 AUC=1", abs(auc_score(score, pos) - 1.0) < 1e-9)
    chk("反向区分 AUC=0", abs(auc_score(-score, pos)) < 1e-9)
    chk("无正样本返回 nan",
        np.isnan(auc_score(score, np.zeros(4, dtype=bool))))

    print("\n[6] adaptive_threshold_mask")
    mean = np.array([0.6, 0.6])
    std = np.array([0.0, 0.3])
    m0 = adaptive_threshold_mask(mean, std, 0.5, 0.0)
    chk("alpha=0 等价固定阈值", m0.tolist() == [True, True])
    mp = adaptive_threshold_mask(mean, std, 0.5, 0.5)
    chk("alpha>0 时高分歧处被抑制", mp.tolist() == [True, False])
    mn = adaptive_threshold_mask(mean, std, 0.5, -1.0)
    chk("alpha<0 时高分歧处更宽松（本例两者都保留）", mn.all())
    big = adaptive_threshold_mask(np.array([0.5]), np.array([10.0]), 0.5, 1.0)
    chk("极端 std 不会使阈值越界导致全空", big.size == 1)

    print(f"\n{'=' * 46}\n  通过 {ok}   失败 {fail}\n{'=' * 46}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
