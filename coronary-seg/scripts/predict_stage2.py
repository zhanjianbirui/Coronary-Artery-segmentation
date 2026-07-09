#!/usr/bin/env python3
"""
scripts/predict_stage2.py — stage-2 3D 精修推理 + 评估
==================================================================
对每个 test case：
  1. 载入 stage2_prepare 存的 npz（含 image + stage-1 概率 prob + label）
  2. 拼成 2 通道输入 [image, prob]
  3. 用 MONAI sliding_window_inference 做 128³ 滑窗精修（重叠加权平均拼回整图）
  4. 阈值化 + 后处理 + 评估 Dice/clDice/Betti0/HD95

指标实现与单轴/三方向 predict 完全一致，保证与 baseline 3.68 / clDice
0.8670 / HD95 23.66 可比。

用法：
  PYTHONPATH=. python scripts/predict_stage2.py \
      --data-root /net/scratch/z67253xh/cache/stage2_npz \
      --ckpt runs/stage2/best.pth \
      --thr 0.50 --min-voxels 300 --max-gap 0 \
      --roi 128 --overlap 0.5 \
      --out-csv runs/stage2/test_metrics_stage2.csv
"""

import os
import sys
import csv
import glob
import argparse
import numpy as np
import torch
from scipy import ndimage
from skimage.morphology import skeletonize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monai.inferers import sliding_window_inference
from src.stage2_model import build_stage2_model


# ===============================================================
# 指标（与 predict_tri.py 完全一致的实现）
# ===============================================================
def dice_coef(pred, gt, eps=1e-6):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def cldice_coef(pred, gt, eps=1e-6):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    sp, sg = skeletonize(pred), skeletonize(gt)
    tprec = (np.logical_and(sp, gt).sum() + eps) / (sp.sum() + eps)
    tsens = (np.logical_and(sg, pred).sum() + eps) / (sg.sum() + eps)
    return 2 * tprec * tsens / (tprec + tsens + eps)


def betti0_error(pred, gt):
    _, np_ = ndimage.label(pred)
    _, ng_ = ndimage.label(gt)
    return abs(np_ - ng_), np_, ng_


def hd95(pred, gt):
    try:
        from monai.metrics import compute_hausdorff_distance
        p = torch.as_tensor(pred[None, None].astype(np.uint8))
        g = torch.as_tensor(gt[None, None].astype(np.uint8))
        return compute_hausdorff_distance(p, g, percentile=95).item()
    except Exception:
        return float("nan")


def remove_small_components(mask, min_voxels=300):
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
    cents = {i: c.mean(axis=0) for i, c in coords.items()}
    ids = list(coords.keys())
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            ia, ib = ids[a], ids[b]
            if np.linalg.norm(cents[ia] - cents[ib]) > max_gap * 4:
                continue
            ca, cb = coords[ia], coords[ib]
            sa = ca[::max(1, len(ca) // 50)]
            sb = cb[::max(1, len(cb) // 50)]
            d = np.linalg.norm(sa[:, None] - sb[None], axis=2)
            mi = np.unravel_index(d.argmin(), d.shape)
            if d[mi] <= max_gap:
                p0, p1 = sa[mi[0]], sb[mi[1]]
                for t in np.linspace(0, 1, int(d[mi]) + 1):
                    pt = np.round(p0 + t * (p1 - p0)).astype(int)
                    out[pt[0], pt[1], pt[2]] = True
    return out.astype(mask.dtype)


def postprocess(mask, min_voxels=300, max_gap=0):
    m = remove_small_components(mask, min_voxels)
    if max_gap > 0:
        m = reconnect_endpoints(m, max_gap)
    return m


def evaluate_case(pred, gt):
    b0, np_, ng_ = betti0_error(pred, gt)
    return {"dice": dice_coef(pred, gt), "cldice": cldice_coef(pred, gt),
            "betti0_err": b0, "n_pred": np_, "n_gt": ng_,
            "hd95": hd95(pred, gt)}


# ===============================================================
# npz 载入：兼容不同 key 命名
# ===============================================================
def load_npz(path):
    d = np.load(path)
    keys = set(d.files)
    # image
    img_key = next((k for k in ("image", "img", "ct") if k in keys), None)
    prob_key = next((k for k in ("prob", "stage1_prob", "p1") if k in keys), None)
    lab_key = next((k for k in ("label", "lab", "gt", "mask") if k in keys), None)
    if img_key is None or prob_key is None:
        raise KeyError(f"{os.path.basename(path)} 缺 image/prob，实际keys={d.files}")
    image = np.asarray(d[img_key]).astype(np.float32)
    prob = np.asarray(d[prob_key]).astype(np.float32)
    label = (np.asarray(d[lab_key]).astype(np.uint8)
             if lab_key is not None else None)
    # 去掉可能的通道维 -> (D,H,W)
    image = np.squeeze(image)
    prob = np.squeeze(prob)
    if label is not None:
        label = np.squeeze(label)
    return image, prob, label


def _case_id_from_path(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


# ===============================================================
# 推理一个 case
# ===============================================================
@torch.no_grad()
def infer_case(model, image, prob, device, roi, overlap, use_amp, sw_batch=2):
    # 组 2 通道输入 (1,2,D,H,W)
    x = np.stack([image, prob], axis=0)[None]              # (1,2,D,H,W)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)

    def _pred(patch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp):
            return model(patch)                            # logits

    logits = sliding_window_inference(
        xt, roi_size=(roi, roi, roi), sw_batch_size=sw_batch,
        predictor=_pred, overlap=overlap, mode="gaussian")
    prob_out = torch.sigmoid(logits.float())[0, 0].cpu().numpy()   # (D,H,W)
    return prob_out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True,
                   help="stage2 npz 目录（每个 test case 一个 .npz）")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-csv", default="runs/stage2/test_metrics_stage2.csv")
    p.add_argument("--glob", default="*.npz",
                   help="匹配 test npz 的通配（如只测子集可改）")
    p.add_argument("--init-filters", type=int, default=16)
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--roi", type=int, default=128)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--sw-batch", type=int, default=2)
    p.add_argument("--thr", type=float, default=0.50)
    p.add_argument("--min-voxels", type=int, default=300)
    p.add_argument("--max-gap", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--case-ids", type=str, nargs="*", default=[],
                   help="只测这些 case id（文件名去扩展名），务必含 931 728 630")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_amp)

    cfg = {"init_filters": args.init_filters, "use_gate": not args.no_gate}
    model = build_stage2_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"device={device} amp={use_amp}  "
          f"ckpt(epoch={ckpt.get('epoch')}, val_dice={ckpt.get('val_dice')})")

    files = sorted(glob.glob(os.path.join(args.data_root, args.glob)))
    if args.case_ids:
        want = set(str(c) for c in args.case_ids)
        files = [f for f in files if _case_id_from_path(f) in want] + \
                [f for f in files if _case_id_from_path(f) not in want]
        if args.max_cases <= 0:
            args.max_cases = len(want) + 5   # 必含 + 少量补充
    if args.max_cases and args.max_cases > 0:
        files = files[:args.max_cases]
    print(f"待测 {len(files)} 个 case")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fh = open(args.out_csv, "w", newline="")
    writer = None
    rows = []

    for i, path in enumerate(files):
        cid = _case_id_from_path(path)
        try:
            image, prob, label = load_npz(path)
            if label is None:
                print(f"[{i+1}/{len(files)}] {cid} 无 label，跳过评估")
                continue

            # stage-1 基线（精修前）指标
            s1 = (prob > args.thr).astype(np.uint8)
            s1_pp = postprocess(s1, args.min_voxels, args.max_gap)
            m_s1 = evaluate_case(s1_pp, label)

            # stage-2 精修
            prob2 = infer_case(model, image, prob, device, args.roi,
                               args.overlap, use_amp, args.sw_batch)
            s2 = (prob2 > args.thr).astype(np.uint8)
            s2_pp = postprocess(s2, args.min_voxels, args.max_gap)
            m_s2 = evaluate_case(s2_pp, label)

            row = {"id": cid,
                   **{f"s1_{k}": v for k, v in m_s1.items()},
                   **{f"s2_{k}": v for k, v in m_s2.items()}}
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
            rows.append(row)

            print(f"[{i+1}/{len(files)}] {cid}  "
                  f"stage1: dice={m_s1['dice']:.4f} clD={m_s1['cldice']:.4f} "
                  f"B0={m_s1['betti0_err']} | "
                  f"stage2: dice={m_s2['dice']:.4f} clD={m_s2['cldice']:.4f} "
                  f"B0={m_s2['betti0_err']}")
        except Exception as e:
            print(f"[{i+1}/{len(files)}] {cid} 出错跳过: {repr(e)}")
            continue

    fh.close()
    if not rows:
        print("无可评估 case。")
        return

    # 汇总：stage1 vs stage2 均值
    def _mean(key):
        vals = [r[key] for r in rows
                if isinstance(r[key], float) and not np.isnan(r[key])]
        return np.mean(vals) if vals else float("nan")

    print("\n" + "=" * 68)
    print("  stage-1（精修前） vs stage-2（精修后）  测试集均值")
    print("=" * 68)
    print(f"  {'metric':<12} {'stage1':>10} {'stage2':>10} {'Δ':>10}")
    print("  " + "-" * 44)
    for m in ["dice", "cldice", "betti0_err", "hd95"]:
        s1v, s2v = _mean(f"s1_{m}"), _mean(f"s2_{m}")
        better = "↓好" if m in ("betti0_err", "hd95") else "↑好"
        print(f"  {m:<12} {s1v:>10.4f} {s2v:>10.4f} {s2v-s1v:>+10.4f}  ({better})")
    print(f"\n  单轴 baseline: Dice=0.8027 clDice=0.8670 Betti0=3.68 HD95=23.66")
    print(f"  结果已存: {args.out_csv}")


if __name__ == "__main__":
    main()
