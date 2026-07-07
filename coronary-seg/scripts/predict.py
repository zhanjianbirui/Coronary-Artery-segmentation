#!/usr/bin/env python3
"""
scripts/predict.py — 2.5D 推理 + 后处理 + 拓扑评估
==================================================================
对测试集每个病例：
  1. 预处理成 3D 体块（复用 data.py 的预处理链 + 缓存）
  2. 逐 z 切片做 2.5D 推理（相邻 2k+1 层），预测中心层，堆叠回 3D
  3. 后处理：小连通分量过滤 + 端点重连
  4. 拓扑评估：Dice / clDice / Betti-0 误差 / HD95
     —— 同时输出"不带后处理"和"带后处理"两组
  5. 结果存 csv（每病例 + 整体均值）

用法：
  PYTHONPATH=. python scripts/predict.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --ckpt runs/exp_2p5d/best.pth \
      --out-csv runs/exp_2p5d/test_metrics.csv \
      --max-cases 0 \
      --pad-multiple 32
"""

from monai.data import PersistentDataset
from src.smart_reconnect import smart_reconnect
from src.model import build_model
from src.data import build_preprocess, load_split
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

_INFER_PAD = (32,)  # padding 倍数，运行时由 predict_volume 设置


# ---------------------------------------------------------------
# 指标
# ---------------------------------------------------------------
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

    # tprec: pred骨架落在gt里的比例; tsens: gt骨架落在pred里的比例
    tprec = (np.logical_and(skel_pred, gt).sum() + eps) / \
        (skel_pred.sum() + eps)
    tsens = (np.logical_and(skel_gt, pred).sum() + eps) / (skel_gt.sum() + eps)

    return 2 * tprec * tsens / (tprec + tsens + eps)


def betti0_error(pred, gt):
    """连通分量数之差的绝对值（衡量碎片化）。"""
    _, n_pred = ndimage.label(pred)
    _, n_gt = ndimage.label(gt)
    return abs(n_pred - n_gt), n_pred, n_gt


def hd95(pred, gt):
    """95百分位 Hausdorff 距离（体素）。用 MONAI。"""
    try:
        from monai.metrics import compute_hausdorff_distance

        p = torch.as_tensor(pred[None, None].astype(np.uint8))
        g = torch.as_tensor(gt[None, None].astype(np.uint8))
        v = compute_hausdorff_distance(p, g, percentile=95).item()
        return v
    except Exception:
        return float("nan")


# ---------------------------------------------------------------
# 后处理
# ---------------------------------------------------------------
def remove_small_components(mask, min_voxels=200):
    lab, n = ndimage.label(mask)

    if n == 0:
        return mask

    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = set(np.where(sizes >= min_voxels)[0] + 1)

    out = np.isin(lab, list(keep)) if keep else np.zeros_like(mask)

    return out.astype(mask.dtype)


def reconnect_endpoints(mask, max_gap=15):
    """
    简化版端点重连：对每个连通分量的端点，若与另一分量端点距离 < max_gap，
    用直线体素连接。这是轻量近似（完整版可用测地距离/最短路径）。
    """
    lab, n = ndimage.label(mask)

    if n <= 1:
        return mask

    out = mask.copy().astype(bool)

    # 每个连通分量的体素坐标
    coords = {i: np.argwhere(lab == i) for i in range(1, n + 1)}

    # 每个连通分量的质心
    centroids = {i: c.mean(axis=0) for i, c in coords.items()}

    ids = list(coords.keys())

    for a_idx in range(len(ids)):
        for b_idx in range(a_idx + 1, len(ids)):
            ia, ib = ids[a_idx], ids[b_idx]
            ca, cb = coords[ia], coords[ib]

            # 若质心距离太远，跳过，避免计算量过大
            if np.linalg.norm(centroids[ia] - centroids[ib]) > max_gap * 4:
                continue

            # 下采样，避免 O(N*M) 过大
            sa = ca[::max(1, len(ca) // 50)]
            sb = cb[::max(1, len(cb) // 50)]

            d = np.linalg.norm(sa[:, None] - sb[None], axis=2)
            mi = np.unravel_index(d.argmin(), d.shape)

            if d[mi] <= max_gap:
                p0, p1 = sa[mi[0]], sb[mi[1]]

                # 用直线体素连接两个近端点
                steps = int(d[mi]) + 1

                for t in np.linspace(0, 1, steps):
                    pt = np.round(p0 + t * (p1 - p0)).astype(int)
                    out[pt[0], pt[1], pt[2]] = True

    return out.astype(mask.dtype)


def postprocess(mask, min_voxels=200, max_gap=15, smart=False,
                smart_L=8, smart_align=0.5):
    m = remove_small_components(mask, min_voxels)
    if smart and max_gap > 0:
        m = smart_reconnect(m, max_gap=max_gap, L=smart_L,
                            align_thr=smart_align)
        m = remove_small_components(m, max(30, min_voxels // 5))
    elif max_gap > 0:
        m = reconnect_endpoints(m, max_gap)
    return m


# ---------------------------------------------------------------
# Padding 工具函数
# ---------------------------------------------------------------
def pad_to_multiple_2d(x, multiple=32):
    """
    x: Tensor, shape = (B, C, H, W)

    将 H/W padding 到 multiple 的整数倍，避免 SegResNet 在 encoder-decoder
    skip connection 相加时出现尺寸不匹配，例如 94 vs 93。

    返回：
      x_pad: padding 后的 tensor
      h: 原始高度
      w: 原始宽度
    """
    _, _, h, w = x.shape

    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h > 0 or pad_w > 0:
        # F.pad 参数顺序是: left, right, top, bottom
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

    return x, h, w


# ---------------------------------------------------------------
# 2.5D 推理：逐切片预测，堆回 3D
# ---------------------------------------------------------------
@torch.no_grad()
def _infer_logits(model, xb, device):
    """单次前向，返回 logits（已在原尺寸）。"""
    xb_p, oh, ow = pad_to_multiple_2d(xb, multiple=_INFER_PAD[0])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        enabled=(device == "cuda")):
        logits = model(xb_p)
    return logits[..., :oh, :ow]


def _infer_prob_tta(model, xb, device, use_tta):
    """
    对一个 batch (b,C,H,W) 推理，返回概率 (b,H,W)。
    use_tta=True 时做 4-way 翻转 TTA：原图 + 水平翻 + 垂直翻 + 双翻，
    各自预测后反翻转回来，概率平均。
    """
    if not use_tta:
        logits = _infer_logits(model, xb, device)
        return torch.sigmoid(logits.float())[:, 0]

    # flips: (空, 水平dim=-1, 垂直dim=-2, 双翻)
    flip_dims = [(), (-1,), (-2,), (-2, -1)]
    prob_sum = None
    for dims in flip_dims:
        xin = torch.flip(xb, dims=dims) if dims else xb
        logits = _infer_logits(model, xin, device)
        prob = torch.sigmoid(logits.float())
        # 反翻转回原方向
        if dims:
            prob = torch.flip(prob, dims=dims)
        prob = prob[:, 0]
        prob_sum = prob if prob_sum is None else prob_sum + prob
    return prob_sum / len(flip_dims)


def predict_volume(model, image3d, k, device, thr=0.5, batch=16, pad_multiple=32, use_tta=False):
    """
    image3d: (1,H,W,D) 预处理后的体块。
    返回二值 3D 预测 (H,W,D)。

    注意：
    SegResNet 对输入 H/W 尺寸比较敏感。
    如果 H/W 不能被下采样倍数整除，decoder 里 skip connection 可能出现：
        RuntimeError: The size of tensor a (...) must match tensor b (...)
    所以这里先 padding 到 32 的倍数，预测后再 crop 回原始尺寸。
    """
    img = torch.as_tensor(np.asarray(image3d))[0]     # (H,W,D)
    H, W, D = img.shape

    prob_vol = np.zeros((H, W, D), dtype=np.float32)

    zs = list(range(D))

    for start in range(0, D, batch):
        zc = zs[start:start + batch]
        stacks = []

        for z in zc:
            idx = [
                int(np.clip(z + off, 0, D - 1))
                for off in range(-k, k + 1)
            ]

            # img[:, :, idx] = (H,W,2k+1)
            # permute 后变成 (2k+1,H,W)
            stacks.append(img[:, :, idx].permute(2, 0, 1))

        xb = torch.stack(stacks).float().to(device)   # (b,2k+1,H,W)

        # 推理（padding 在 _infer_logits 内部处理；可选 TTA）
        global _INFER_PAD
        _INFER_PAD = (pad_multiple,)
        probs_t = _infer_prob_tta(model, xb, device, use_tta)  # (b,H,W)
        probs = probs_t.cpu().numpy()

        for j, z in enumerate(zc):
            prob_vol[:, :, z] = probs[j]

    return (prob_vol > thr).astype(np.uint8)


def evaluate_case(pred, gt):
    d = dice_coef(pred, gt)
    cl = cldice_coef(pred, gt)
    b0, np_, ng_ = betti0_error(pred, gt)
    h = hd95(pred, gt)

    return {
        "dice": d,
        "cldice": cl,
        "betti0_err": b0,
        "n_pred": np_,
        "n_gt": ng_,
        "hd95": h
    }


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-csv", default="runs/exp_2p5d/test_metrics.csv")

    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)

    p.add_argument("--backbone", default="segresnet")
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--thr", type=float, default=0.5)

    # 新增参数：控制 padding 到多少的倍数
    p.add_argument(
        "--pad-multiple",
        type=int,
        default=32,
        help="将 H/W padding 到该数的整数倍，避免 SegResNet skip connection 尺寸不匹配"
    )

    p.add_argument("--min-voxels", type=int, default=200)
    p.add_argument("--max-gap", type=int, default=15)
    p.add_argument("--smart", action="store_true", help="用方向感知智能重连")
    p.add_argument("--smart-l", type=int, default=8, help="局部方向估计半径")
    p.add_argument("--smart-align", type=float, default=0.5, help="方向一致性阈值")
    p.add_argument("--max-cases", type=int, default=0, help="0=全部测试集")
    p.add_argument("--tta", action="store_true", help="开启4-way翻转TTA")

    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, ckpt={args.ckpt}")

    # 模型
    cfg = {
        "k": args.k,
        "backbone": args.backbone,
        "init_filters": args.init_filters,
        "out_channels": 1
    }

    model = build_model(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(
        f"加载 checkpoint（epoch={ckpt.get('epoch')}, "
        f"val_dice={ckpt.get('val_dice')}）"
    )

    # 测试集
    preprocess = build_preprocess(args.spacing, args.hu_min, args.hu_max)
    _, _, test_rec = load_split(args.split_json)

    if args.max_cases and args.max_cases > 0:
        test_rec = test_rec[:args.max_cases]

    cache = PersistentDataset(
        data=test_rec,
        transform=preprocess,
        cache_dir=args.cache_dir
    )

    rows = []

    # ---- 断点续跑：读已有 csv，跳过已完成病例 ----
    import csv as _csv
    done_ids = set()
    csv_exists = os.path.isfile(args.out_csv)
    if csv_exists:
        try:
            with open(args.out_csv, newline="") as _f:
                for _r in _csv.DictReader(_f):
                    done_ids.add(str(_r.get("id")))
            print(f"[续跑] 已有 csv，{len(done_ids)} 个病例已完成，将跳过")
        except Exception as _e:
            print(f"[续跑] 读旧 csv 失败（{_e}），从头开始")
            done_ids = set()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    rows = []
    _fieldnames = None
    _csv_fh = None

    def _append_row(row):
        nonlocal _fieldnames, _csv_fh
        need_header = not (csv_exists or _csv_fh is not None)
        if _csv_fh is None:
            _fieldnames = list(row.keys())
            _csv_fh = open(args.out_csv, "a", newline="")
            _writer = _csv.DictWriter(_csv_fh, fieldnames=_fieldnames)
            if need_header and os.path.getsize(args.out_csv) == 0:
                _writer.writeheader()
            _append_row._writer = _writer
        _append_row._writer.writerow(row)
        _csv_fh.flush()

    for ci in range(len(test_rec)):
        cid_check = str(test_rec[ci].get("id", str(ci)))
        if cid_check in done_ids:
            print(f"[{ci+1}/{len(test_rec)}] case {cid_check} 已完成，跳过")
            continue
        vol = cache[ci]

        image3d = np.asarray(vol["image"])                  # (1,H,W,D)
        gt = np.asarray(vol["label"])[0].astype(np.uint8)   # (H,W,D)

        pred_raw = predict_volume(
            model,
            image3d,
            args.k,
            device,
            args.thr,
            pad_multiple=args.pad_multiple,
            use_tta=args.tta
        )

        pred_pp = postprocess(
            pred_raw,
            args.min_voxels,
            args.max_gap,
            smart=args.smart,
            smart_L=args.smart_l,
            smart_align=args.smart_align
        )

        m_raw = evaluate_case(pred_raw, gt)
        m_pp = evaluate_case(pred_pp, gt)

        cid = test_rec[ci].get("id", str(ci))

        _row = {
            "id": cid,
            **{f"raw_{k}": v for k, v in m_raw.items()},
            **{f"pp_{k}": v for k, v in m_pp.items()}
        }
        rows.append(_row)
        _append_row(_row)   # 边跑边存，防中断丢失

        print(
            f"[{ci + 1}/{len(test_rec)}] case {cid}  "
            f"raw: dice={m_raw['dice']:.4f} "
            f"clDice={m_raw['cldice']:.4f} "
            f"B0err={m_raw['betti0_err']} | "
            f"pp: dice={m_pp['dice']:.4f} "
            f"clDice={m_pp['cldice']:.4f} "
            f"B0err={m_pp['betti0_err']}"
        )

    if len(rows) == 0:
        print("没有可评估的测试病例，请检查 split.json 或 --max-cases 设置。")
        return

    # csv 已边跑边写完；从完整 csv 读全部行（含续跑前已完成的）算均值
    if _csv_fh is not None:
        _csv_fh.close()

    all_rows = []
    with open(args.out_csv, newline="") as _f:
        for _r in _csv.DictReader(_f):
            for _k, _v in _r.items():
                if _k != "id":
                    try:
                        _r[_k] = float(_v)
                    except (ValueError, TypeError):
                        pass
            all_rows.append(_r)
    rows = all_rows
    if len(rows) == 0:
        print("没有可评估的测试病例。")
        return

    keys = list(rows[0].keys())

    # （csv 已在循环中增量写入，无需重复写）

    # 打印均值
    print("\n===== 测试集均值 =====")

    for prefix in ["raw", "pp"]:
        for metric in ["dice", "cldice", "betti0_err", "hd95"]:
            vals = [
                r[f"{prefix}_{metric}"]
                for r in rows
                if not (
                    isinstance(r[f"{prefix}_{metric}"], float)
                    and np.isnan(r[f"{prefix}_{metric}"])
                )
            ]

            mean = np.mean(vals) if vals else float("nan")
            tag = "原始" if prefix == "raw" else "后处理"

            print(f"  [{tag}] {metric}: {mean:.4f}")

    print(f"\n结果已存: {args.out_csv}")


if __name__ == "__main__":
    main()
