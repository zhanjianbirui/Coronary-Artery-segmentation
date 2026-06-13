#!/usr/bin/env python
"""推理: 载入 best.pth, 对测试集滑窗推理(可选 TTA), 后处理去碎片, 保存预测 nii.gz.

用法:
    python scripts/predict.py --config configs/default.yaml \
        --ckpt runs/exp_segresnet/best.pth --split test
关键开关 (在 yaml 的 infer 段, 也可命令行覆盖):
    infer.tta=true                  8 向翻转平均, 更准但慢 8 倍
    infer.min_component_voxels=30   删除小于该体素数的连通域, 0 关闭
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoint import load_checkpoint
from src.config import Config
from src.data import build_val_transforms, load_split
from src.model import build_model
from src.utils import AverageMeter, get_logger


_FLIP_AXES = [(), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]  # 8 向


@torch.no_grad()
def infer_probs(model, image, roi, sw_bs, overlap, device, amp, tta):
    """返回前景概率图 [1,C,D,H,W]. tta=True 时做 8 向翻转平均."""
    axes_list = _FLIP_AXES if tta else [()]
    prob_sum = None
    for ax in axes_list:
        x = torch.flip(image, dims=ax) if ax else image
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = sliding_window_inference(x, roi, sw_bs, model, overlap=overlap)
        prob = torch.softmax(logits.float(), dim=1)
        if ax:
            prob = torch.flip(prob, dims=ax)
        prob_sum = prob if prob_sum is None else prob_sum + prob
    return prob_sum / len(axes_list)


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """删除体素数小于 min_voxels 的前景连通域(去碎片假阳性). min_voxels<=0 时不处理."""
    if min_voxels <= 0:
        return mask
    try:
        from scipy.ndimage import label
    except ImportError:
        return mask
    lab, n = label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # 背景不算
    keep = sizes >= min_voxels
    return keep[lab].astype(np.uint8)


def fg_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """前景二值 Dice."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return float(2.0 * inter / (pred.sum() + gt.sum() + eps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True, help="模型权重 (通常 best.pth)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--save_dir", default="", help="预测输出目录, 默认 work_dir/predictions")
    ap.add_argument("opts", nargs="*")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config).apply_overrides(args.opts)
    log = get_logger("predict")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = args.save_dir or str(Path(cfg.output.work_dir) / "predictions")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    tta = bool(getattr(cfg.infer, "tta", False))
    min_cc = int(getattr(cfg.infer, "min_component_voxels", 0))
    amp = cfg.train.amp

    # ---- 模型 ----
    model = build_model(cfg.model).to(device)
    meta = load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()
    log.info(f"载入 {args.ckpt} (epoch {meta['epoch']}, best Dice {meta['best_metric']:.4f})")
    log.info(f"推理设置: TTA={'开(8向)' if tta else '关'} | 后处理最小连通域={min_cc} 体素")

    # ---- 数据 ----
    split = load_split(cfg.data.split_json)
    val_tf = build_val_transforms(cfg.preprocess)
    ds = Dataset(data=split[args.split], transform=val_tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers)
    roi = tuple(cfg.preprocess.patch_size)

    dice_meter = AverageMeter()
    has_label = False

    for i, batch in enumerate(loader):
        images = batch["image"].to(device)
        probs = infer_probs(model, images, roi, cfg.infer.sw_batch_size,
                            cfg.infer.overlap, device, amp, tta)
        pred = torch.argmax(probs[0], dim=0).cpu().numpy().astype(np.uint8)  # [D,H,W]
        pred = remove_small_components(pred, min_cc)                          # 后处理

        # 评估 (若有标签)
        if "label" in batch:
            has_label = True
            gt = batch["label"][0].squeeze().cpu().numpy().astype(np.uint8)
            dice_meter.update(fg_dice(pred, gt), n=1)

        # 保存
        case_id = split[args.split][i].get("id", f"case_{i:04d}")
        out_path = Path(save_dir) / f"{case_id}_seg.nii.gz"
        _save_like(pred, batch, out_path)
        log.info(f"[{i+1}/{len(loader)}] {case_id} -> {out_path.name}")

    if has_label:
        log.info(f"{args.split} 集平均 Dice = {dice_meter.avg:.4f} "
                 f"(TTA={'on' if tta else 'off'}, 后处理={min_cc})")


def _save_like(pred: np.ndarray, batch, out_path: Path) -> None:
    """用原图 affine 保存预测, 保证空间对齐."""
    import nibabel as nib
    arr = np.asarray(pred).squeeze().astype(np.uint8)
    affine = None
    img = batch["image"]
    if hasattr(img, "meta") and "affine" in img.meta:
        aff = img.meta["affine"]
        affine = np.asarray(aff[0]) if getattr(aff, "ndim", 2) == 3 else np.asarray(aff)
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(arr, affine), str(out_path))


if __name__ == "__main__":
    main()
