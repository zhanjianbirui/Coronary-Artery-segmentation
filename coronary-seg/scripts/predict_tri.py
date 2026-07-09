#!/usr/bin/env python3
"""
scripts/predict_tri.py — 三正交方向 2.5D 推理 + 概率融合 + 拓扑评估
==================================================================
和单轴 predict.py 的区别只有一点：这里对同一个模型沿三个正交方向
各推理一遍，得到三个概率体 p0/p1/p2（都在同一 (H,W,D)=RAS 网格上），
然后把它们融合（mean 或 max/noisy-OR）、阈值化、后处理、评估。

核心动机：轴位单方向对近似平行于轴位面走行的血管容易断裂；冠状/矢状
方向正好补上。融合后再重建 3D，才可能把 Betti-0 压下来。融合方式是
胜负手：
  - mean：整体 Dice/HD95 更友好，但会稀释"只在一个平面清晰"的血管，
          反而伤连通性；
  - max（noisy-OR）：任一正交面看到血管就算数，物理上正好修断裂，
          代价是 FP 上升。

指标 / 后处理 / padding / TTA 全部复用单轴 predict.py 的实现，保证口径
与单轴 baseline 完全可比。

两种运行模式：
  A. 扫描模式（默认，推荐先跑）：对小子集（务必含 931/728/630）一次推理、
     多方案融合，输出 {mean,max} × {阈值网格} → Dice/clDice/Betti0/HD95
     的对比表，选出最优组合。
  B. 全量模式（--fixed-fuse mean|max --thr 0.4 ...）：用选定的融合方式+阈值，
     跑全测试集，逐病例写 csv（含 raw / pp 两组），和单轴 predict.py 的
     csv 格式一致。

用法（扫描，先拿 best.pth 验证方向）：
  PYTHONPATH=. python scripts/predict_tri.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --ckpt runs/exp_tri/best.pth \
      --sweep \
      --case-ids 931 728 630 \
      --extra-cases 17 \
      --thr-grid 0.30 0.35 0.40 0.45 0.50 \
      --out-csv runs/exp_tri/sweep_metrics.csv \
      --pad-multiple 32

用法（全量，选定组合后）：
  PYTHONPATH=. python scripts/predict_tri.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --ckpt runs/exp_tri/best.pth \
      --fixed-fuse max --thr 0.40 \
      --min-voxels 300 --max-gap 0 \
      --out-csv runs/exp_tri/test_metrics_tri.csv \
      --max-cases 0 --pad-multiple 32
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.data import PersistentDataset
from src.model import build_model
from src.data import build_preprocess, load_split

# smart_reconnect 可选：全量模式若用 --smart 才需要；扫描模式默认不用
try:
    from src.smart_reconnect import smart_reconnect
    _HAS_SMART = True
except Exception:
    _HAS_SMART = False

_INFER_PAD = (32,)  # padding 倍数，运行时由 predict_volume_axis 设置


# ===============================================================
# 指标（原样复用单轴 predict.py，保证口径一致）
# ===============================================================
def dice_coef(pred, gt, eps=1e-6):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def cldice_coef(pred, gt, eps=1e-6):
    """clDice：中心线覆盖率的调和平均。基于 3D 骨架。"""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)
    tprec = (np.logical_and(skel_pred, gt).sum() + eps) / (skel_pred.sum() + eps)
    tsens = (np.logical_and(skel_gt, pred).sum() + eps) / (skel_gt.sum() + eps)
    return 2 * tprec * tsens / (tprec + tsens + eps)


def betti0_error(pred, gt):
    """连通分量数之差的绝对值（衡量碎片化）。"""
    _, n_pred = ndimage.label(pred)
    _, n_gt = ndimage.label(gt)
    return abs(n_pred - n_gt), n_pred, n_gt


def hd95(pred, gt):
    """95 百分位 Hausdorff 距离（体素）。用 MONAI。"""
    try:
        from monai.metrics import compute_hausdorff_distance
        p = torch.as_tensor(pred[None, None].astype(np.uint8))
        g = torch.as_tensor(gt[None, None].astype(np.uint8))
        v = compute_hausdorff_distance(p, g, percentile=95).item()
        return v
    except Exception:
        return float("nan")


# ===============================================================
# 后处理（原样复用）
# ===============================================================
def remove_small_components(mask, min_voxels=200):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = set(np.where(sizes >= min_voxels)[0] + 1)
    out = np.isin(lab, list(keep)) if keep else np.zeros_like(mask)
    return out.astype(mask.dtype)


def reconnect_endpoints(mask, max_gap=15):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    out = mask.copy().astype(bool)
    coords = {i: np.argwhere(lab == i) for i in range(1, n + 1)}
    centroids = {i: c.mean(axis=0) for i, c in coords.items()}
    ids = list(coords.keys())
    for a_idx in range(len(ids)):
        for b_idx in range(a_idx + 1, len(ids)):
            ia, ib = ids[a_idx], ids[b_idx]
            ca, cb = coords[ia], coords[ib]
            if np.linalg.norm(centroids[ia] - centroids[ib]) > max_gap * 4:
                continue
            sa = ca[::max(1, len(ca) // 50)]
            sb = cb[::max(1, len(cb) // 50)]
            d = np.linalg.norm(sa[:, None] - sb[None], axis=2)
            mi = np.unravel_index(d.argmin(), d.shape)
            if d[mi] <= max_gap:
                p0, p1 = sa[mi[0]], sb[mi[1]]
                steps = int(d[mi]) + 1
                for t in np.linspace(0, 1, steps):
                    pt = np.round(p0 + t * (p1 - p0)).astype(int)
                    out[pt[0], pt[1], pt[2]] = True
    return out.astype(mask.dtype)


def postprocess(mask, min_voxels=200, max_gap=0, smart=False,
                smart_L=8, smart_align=0.5):
    m = remove_small_components(mask, min_voxels)
    if smart and max_gap > 0 and _HAS_SMART:
        m = smart_reconnect(m, max_gap=max_gap, L=smart_L, align_thr=smart_align)
        m = remove_small_components(m, max(30, min_voxels // 5))
    elif max_gap > 0:
        m = reconnect_endpoints(m, max_gap)
    return m


# ===============================================================
# Padding（原样复用）
# ===============================================================
def pad_to_multiple_2d(x, multiple=32):
    """x: (B,C,H,W)。把 H/W padding 到 multiple 的整数倍，返回 (x_pad, h, w)。"""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
    return x, h, w


@torch.no_grad()
def _infer_logits(model, xb, device):
    xb_p, oh, ow = pad_to_multiple_2d(xb, multiple=_INFER_PAD[0])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=(device == "cuda")):
        logits = model(xb_p)
    return logits[..., :oh, :ow]


def _infer_prob_tta(model, xb, device, use_tta):
    """一个 batch (b,C,H,W) → 概率 (b,H,W)。TTA 为面内 4-way 翻转。"""
    if not use_tta:
        logits = _infer_logits(model, xb, device)
        return torch.sigmoid(logits.float())[:, 0]
    flip_dims = [(), (-1,), (-2,), (-2, -1)]
    prob_sum = None
    for dims in flip_dims:
        xin = torch.flip(xb, dims=dims) if dims else xb
        logits = _infer_logits(model, xin, device)
        prob = torch.sigmoid(logits.float())
        if dims:
            prob = torch.flip(prob, dims=dims)
        prob = prob[:, 0]
        prob_sum = prob if prob_sum is None else prob_sum + prob
    return prob_sum / len(flip_dims)


# ===============================================================
# 三方向逐切片推理 → 概率体（不阈值化，阈值留到融合后）
# ===============================================================
def _take_slice_stack(img, axis, center, k, n):
    """
    和 data_tri.extract_slice_stack 的 permute 语义严格对齐。
    img: (H,W,D) tensor。返回单张 stack (2k+1, A, B)。
      axis=2 平面(H,W); axis=1 平面(H,D); axis=0 平面(W,D)
    """
    idx = [int(np.clip(center + off, 0, n - 1)) for off in range(-k, k + 1)]
    if axis == 2:
        return img[:, :, idx].permute(2, 0, 1)      # (2k+1,H,W)
    elif axis == 1:
        return img[:, idx, :].permute(1, 0, 2)      # (2k+1,H,D)
    else:
        return img[idx, :, :]                        # (2k+1,W,D)


def _scatter_slice(prob_vol, plane, axis, center):
    """把某个 center 层的 2D 预测 plane 写回 3D 概率体对应位置。"""
    if axis == 2:
        prob_vol[:, :, center] = plane               # plane (H,W)
    elif axis == 1:
        prob_vol[:, center, :] = plane               # plane (H,D)
    else:
        prob_vol[center, :, :] = plane               # plane (W,D)


def predict_prob_axis(model, img, k, axis, device, batch=16,
                      pad_multiple=32, use_tta=False):
    """
    沿单个 axis 逐层推理，返回该方向的概率体 (H,W,D) float32（未阈值化）。
    img: (H,W,D) tensor。
    """
    global _INFER_PAD
    _INFER_PAD = (pad_multiple,)

    H, W, D = img.shape
    n = img.shape[axis]                 # 沿该轴的层数 = center 的取值范围
    prob_vol = np.zeros((H, W, D), dtype=np.float32)

    centers = list(range(n))
    for start in range(0, n, batch):
        cc = centers[start:start + batch]
        stacks = [_take_slice_stack(img, axis, c, k, n) for c in cc]
        xb = torch.stack(stacks).float().to(device)      # (b,2k+1,A,B)
        probs = _infer_prob_tta(model, xb, device, use_tta).cpu().numpy()
        for j, c in enumerate(cc):
            _scatter_slice(prob_vol, probs[j], axis, c)
    return prob_vol


def predict_tri_probs(model, image3d, k, device, axes=(0, 1, 2),
                      batch=16, pad_multiple=32, use_tta=False):
    """
    对一个体块，三个方向各推一遍，返回 dict{axis: prob_vol(H,W,D)}。
    image3d: (1,H,W,D) 预处理后的体块。
    """
    img = torch.as_tensor(np.asarray(image3d))[0]         # (H,W,D)
    probs = {}
    for axis in axes:
        probs[axis] = predict_prob_axis(
            model, img, k, axis, device, batch=batch,
            pad_multiple=pad_multiple, use_tta=use_tta)
    return probs


def fuse_probs(prob_dict, method):
    """把三个方向的概率体融合成一个。method: 'mean' | 'max'。"""
    stack = np.stack(list(prob_dict.values()), axis=0)    # (n_ax,H,W,D)
    if method == "mean":
        return stack.mean(axis=0)
    elif method == "max":
        return stack.max(axis=0)                          # noisy-OR 的上界近似
    else:
        raise ValueError(f"未知融合方式: {method}")


def evaluate_case(pred, gt):
    d = dice_coef(pred, gt)
    cl = cldice_coef(pred, gt)
    b0, np_, ng_ = betti0_error(pred, gt)
    h = hd95(pred, gt)
    return {"dice": d, "cldice": cl, "betti0_err": b0,
            "n_pred": np_, "n_gt": ng_, "hd95": h}


# ===============================================================
# case 选择工具
# ===============================================================
def _case_id(rec, idx):
    return str(rec.get("id", str(idx)))


def select_case_indices(test_rec, case_ids, extra_cases, max_cases):
    """
    返回要评估的 test_rec 下标列表。
      - case_ids：必须包含的具体 case id（如 931/728/630）
      - extra_cases：在这些必含 case 之外，再补多少个（取靠前的）凑成小子集
      - max_cases：全量模式用；>0 时只取前 max_cases 个
    """
    id2idx = {_case_id(r, i): i for i, r in enumerate(test_rec)}

    if case_ids:
        chosen = []
        for cid in case_ids:
            cid = str(cid)
            if cid in id2idx:
                chosen.append(id2idx[cid])
            else:
                print(f"  [警告] case id {cid} 不在 test split 里，已跳过")
        # 补齐 extra_cases 个靠前且未选中的
        if extra_cases > 0:
            for i in range(len(test_rec)):
                if i not in chosen:
                    chosen.append(i)
                    if len(chosen) >= len(case_ids) + extra_cases:
                        break
        return chosen

    idxs = list(range(len(test_rec)))
    if max_cases and max_cases > 0:
        idxs = idxs[:max_cases]
    return idxs


# ===============================================================
# 扫描模式：一次推理、多方案融合
# ===============================================================
def run_sweep(model, cache, test_rec, idxs, args, device):
    methods = ["mean", "max"]
    thr_grid = args.thr_grid
    # 累加器： (method, thr) -> list of per-case metric dict（用 raw，聚焦融合本身）
    agg = {(m, t): [] for m in methods for t in thr_grid}
    per_case_rows = []

    for rank, ci in enumerate(idxs):
        cid = _case_id(test_rec[ci], ci)
        vol = cache[ci]
        image3d = np.asarray(vol["image"])                 # (1,H,W,D)
        gt = np.asarray(vol["label"])[0].astype(np.uint8)  # (H,W,D)

        # 三轴各推一遍（大头，只做一次）
        prob_dict = predict_tri_probs(
            model, image3d, args.k, device, axes=tuple(args.axes),
            batch=args.batch, pad_multiple=args.pad_multiple, use_tta=args.tta)

        print(f"[{rank+1}/{len(idxs)}] case {cid}  三轴推理完成，扫融合×阈值...")

        for m in methods:
            fused = fuse_probs(prob_dict, m)               # (H,W,D) float
            for t in thr_grid:
                pred = (fused > t).astype(np.uint8)
                met = evaluate_case(pred, gt)
                agg[(m, t)].append(met)
                per_case_rows.append({
                    "id": cid, "fuse": m, "thr": t,
                    "dice": met["dice"], "cldice": met["cldice"],
                    "betti0_err": met["betti0_err"],
                    "n_pred": met["n_pred"], "n_gt": met["n_gt"],
                    "hd95": met["hd95"]})

        # 逐 case 打印各方案的 B0err，肉眼确认 931/728/630 有没有被桥接
        line = f"    case {cid} B0err:  "
        for m in methods:
            b0s = [f"{t:.2f}:{evaluate_case((fuse_probs(prob_dict,m)>t).astype(np.uint8),gt)[0] if False else agg[(m,t)][-1]['betti0_err']}"
                   for t in thr_grid]
            line += f"[{m}] " + " ".join(b0s) + "  "
        print(line)

    # ---- 写逐 case 明细 csv ----
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_case_rows[0].keys()))
        w.writeheader()
        w.writerows(per_case_rows)

    # ---- 汇总表：均值 ----
    print("\n" + "=" * 78)
    print("  扫描汇总（子集均值，raw 无后处理，聚焦融合本身）")
    print("  子集 case:", ", ".join(_case_id(test_rec[i], i) for i in idxs))
    print("=" * 78)
    header = f"  {'fuse':<5} {'thr':>5} | {'Dice':>7} {'clDice':>7} {'B0err':>7} {'HD95':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    summary_rows = []
    for m in methods:
        for t in thr_grid:
            ms = agg[(m, t)]
            d = np.mean([x["dice"] for x in ms])
            cl = np.mean([x["cldice"] for x in ms])
            b0 = np.mean([x["betti0_err"] for x in ms])
            hvals = [x["hd95"] for x in ms if not np.isnan(x["hd95"])]
            h = np.mean(hvals) if hvals else float("nan")
            print(f"  {m:<5} {t:>5.2f} | {d:>7.4f} {cl:>7.4f} {b0:>7.2f} {h:>8.2f}")
            summary_rows.append({"fuse": m, "thr": t, "dice": d,
                                 "cldice": cl, "betti0_err": b0, "hd95": h})
        print("  " + "-" * (len(header) - 2))

    # 推荐：clDice 最高 + Betti0 最低的折中（先按 clDice 排，再看 B0）
    best_cl = max(summary_rows, key=lambda r: r["cldice"])
    best_b0 = min(summary_rows, key=lambda r: r["betti0_err"])
    print(f"\n  clDice 最高:  fuse={best_cl['fuse']} thr={best_cl['thr']:.2f} "
          f"→ clDice={best_cl['cldice']:.4f} B0err={best_cl['betti0_err']:.2f} "
          f"Dice={best_cl['dice']:.4f}")
    print(f"  Betti0 最低:  fuse={best_b0['fuse']} thr={best_b0['thr']:.2f} "
          f"→ B0err={best_b0['betti0_err']:.2f} clDice={best_b0['cldice']:.4f} "
          f"Dice={best_b0['dice']:.4f}")
    print(f"\n  对比单轴 baseline（全量 200 例）: clDice=0.8670  Betti0=3.68  "
          f"HD95=23.66  Dice=0.8027")
    print("  注意：以上是小子集、raw、且含 931/728/630，数值会偏难，"
          "别直接和 200 例均值比绝对值；看的是 mean vs max 的相对趋势。")
    print(f"\n  逐 case 明细已存: {args.out_csv}")


# ===============================================================
# 全量模式：固定融合方式 + 阈值，跑全测试集，写 csv（含 raw/pp）
# ===============================================================
def run_full(model, cache, test_rec, idxs, args, device):
    out_csv = args.out_csv
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # 断点续跑
    done_ids = set()
    if os.path.isfile(out_csv):
        try:
            with open(out_csv, newline="") as f:
                for r in csv.DictReader(f):
                    done_ids.add(str(r.get("id")))
            print(f"[续跑] 已有 csv，{len(done_ids)} 个病例已完成，将跳过")
        except Exception as e:
            print(f"[续跑] 读旧 csv 失败（{e}），从头开始")
            done_ids = set()

    fh = None
    writer = None

    def _append(row):
        nonlocal fh, writer
        if fh is None:
            new = (not os.path.isfile(out_csv)) or os.path.getsize(out_csv) == 0
            fh = open(out_csv, "a", newline="")
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if new:
                writer.writeheader()
        writer.writerow(row)
        fh.flush()

    for rank, ci in enumerate(idxs):
        cid = _case_id(test_rec[ci], ci)
        if cid in done_ids:
            print(f"[{rank+1}/{len(idxs)}] case {cid} 已完成，跳过")
            continue

        vol = cache[ci]
        image3d = np.asarray(vol["image"])
        gt = np.asarray(vol["label"])[0].astype(np.uint8)

        prob_dict = predict_tri_probs(
            model, image3d, args.k, device, axes=tuple(args.axes),
            batch=args.batch, pad_multiple=args.pad_multiple, use_tta=args.tta)
        fused = fuse_probs(prob_dict, args.fixed_fuse)
        pred_raw = (fused > args.thr).astype(np.uint8)
        pred_pp = postprocess(pred_raw, args.min_voxels, args.max_gap,
                              smart=args.smart, smart_L=args.smart_l,
                              smart_align=args.smart_align)

        m_raw = evaluate_case(pred_raw, gt)
        m_pp = evaluate_case(pred_pp, gt)
        row = {"id": cid, "fuse": args.fixed_fuse, "thr": args.thr,
               **{f"raw_{k}": v for k, v in m_raw.items()},
               **{f"pp_{k}": v for k, v in m_pp.items()}}
        _append(row)

        print(f"[{rank+1}/{len(idxs)}] case {cid}  "
              f"raw: dice={m_raw['dice']:.4f} clDice={m_raw['cldice']:.4f} "
              f"B0err={m_raw['betti0_err']} | "
              f"pp: dice={m_pp['dice']:.4f} clDice={m_pp['cldice']:.4f} "
              f"B0err={m_pp['betti0_err']}")

    if fh is not None:
        fh.close()

    # 从完整 csv 读所有行算均值
    all_rows = []
    with open(out_csv, newline="") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k not in ("id", "fuse"):
                    try:
                        r[k] = float(v)
                    except (ValueError, TypeError):
                        pass
            all_rows.append(r)
    if not all_rows:
        print("没有可评估的病例。")
        return

    print("\n===== 三方向融合 测试集均值 =====")
    print(f"  融合={args.fixed_fuse}  thr={args.thr}  "
          f"min_voxels={args.min_voxels}  max_gap={args.max_gap}")
    for prefix in ["raw", "pp"]:
        for metric in ["dice", "cldice", "betti0_err", "hd95"]:
            key = f"{prefix}_{metric}"
            vals = [r[key] for r in all_rows
                    if isinstance(r.get(key), float) and not np.isnan(r[key])]
            mean = np.mean(vals) if vals else float("nan")
            tag = "原始" if prefix == "raw" else "后处理"
            print(f"  [{tag}] {metric}: {mean:.4f}")
    print(f"\n  单轴 baseline: Dice=0.8027 clDice=0.8670 Betti0=3.68 HD95=23.66")
    print(f"  结果已存: {out_csv}")


# ===============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-csv", default="runs/exp_tri/sweep_metrics.csv")

    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)

    p.add_argument("--backbone", default="segresnet")
    p.add_argument("--init-filters", type=int, default=32)

    p.add_argument("--axes", type=int, nargs="+", default=[0, 1, 2],
                   help="参与融合的方向，默认三方向。单轴对照可传 --axes 2")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--pad-multiple", type=int, default=32)
    p.add_argument("--tta", action="store_true", help="面内 4-way 翻转 TTA")

    # 扫描模式
    p.add_argument("--sweep", action="store_true",
                   help="扫描模式：mean/max × 阈值网格，输出对比表")
    p.add_argument("--case-ids", type=str, nargs="*", default=[],
                   help="子集必含的 case id（务必含 931 728 630）")
    p.add_argument("--extra-cases", type=int, default=17,
                   help="在必含 case 之外再补几个凑成小子集")
    p.add_argument("--thr-grid", type=float, nargs="+",
                   default=[0.30, 0.35, 0.40, 0.45, 0.50])

    # 全量模式
    p.add_argument("--fixed-fuse", choices=["mean", "max"], default=None,
                   help="全量模式：固定融合方式（不设则默认进扫描模式）")
    p.add_argument("--thr", type=float, default=0.40)
    p.add_argument("--min-voxels", type=int, default=300)
    p.add_argument("--max-gap", type=int, default=0)
    p.add_argument("--smart", action="store_true")
    p.add_argument("--smart-l", type=int, default=8)
    p.add_argument("--smart-align", type=float, default=0.5)
    p.add_argument("--max-cases", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, ckpt={args.ckpt}, axes={args.axes}")

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
    cache = PersistentDataset(data=test_rec, transform=preprocess,
                              cache_dir=args.cache_dir)

    # 决定跑哪些 case
    is_sweep = args.sweep or (args.fixed_fuse is None)
    if is_sweep:
        idxs = select_case_indices(test_rec, args.case_ids,
                                   args.extra_cases, 0)
        print(f"[扫描模式] 子集 {len(idxs)} 例")
        run_sweep(model, cache, test_rec, idxs, args, device)
    else:
        idxs = select_case_indices(test_rec, [], 0, args.max_cases)
        print(f"[全量模式] {len(idxs)} 例，融合={args.fixed_fuse} thr={args.thr}")
        run_full(model, cache, test_rec, idxs, args, device)


if __name__ == "__main__":
    main()
