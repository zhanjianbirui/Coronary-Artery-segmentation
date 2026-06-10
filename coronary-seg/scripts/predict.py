#!/usr/bin/env python
"""推理: 载入 best.pth, 对测试集滑窗推理, 保存预测 nii.gz.

用法:
    python scripts/predict.py --config configs/default.yaml \
        --ckpt runs/exp_default/best.pth --split test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Invertd, SaveImaged

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoint import load_checkpoint
from src.config import Config
from src.data import build_val_transforms, load_split
from src.model import build_metric, build_model, build_post_transforms
from src.utils import get_logger


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

    # ---- 模型 ----
    model = build_model(cfg.model).to(device)
    meta = load_checkpoint(args.ckpt, model, map_location=device)
    model.eval()
    log.info(f"载入 {args.ckpt} (训练到 epoch {meta['epoch']}, "
             f"best Dice {meta['best_metric']:.4f})")

    # ---- 数据 ----
    split = load_split(cfg.data.split_json)
    val_tf = build_val_transforms(cfg.preprocess)
    ds = Dataset(data=split[args.split], transform=val_tf)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers)

    # 把预测变换回原始空间再存盘 (Invertd 撤销 Spacing/Crop 等)
    post = AsDiscrete(argmax=True)
    saver = SaveImaged(
        keys="pred", output_dir=save_dir, output_postfix="seg",
        resample=False, separate_folder=False, print_log=False,
    )

    metric = build_metric()
    post_pred, post_label = build_post_transforms(cfg.model.out_channels)
    roi = tuple(cfg.preprocess.patch_size)

    metric.reset()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch["image"].to(device)
            with torch.autocast(device_type=device.type,
                                enabled=cfg.train.amp and device.type == "cuda"):
                logits = sliding_window_inference(
                    images, roi, cfg.infer.sw_batch_size, model, overlap=cfg.infer.overlap
                )
            # 评估 (若有标签)
            if "label" in batch:
                labels = batch["label"].to(device)
                p = [post_pred(logits[j]) for j in range(logits.shape[0])]
                g = [post_label(labels[j]) for j in range(labels.shape[0])]
                metric(p, g)

            # 保存预测掩码
            pred = post(logits[0]).cpu()
            case_id = split[args.split][i].get("id", f"case_{i:04d}")
            out_path = Path(save_dir) / f"{case_id}_seg.nii.gz"
            _save_like(pred, batch, out_path)
            log.info(f"[{i+1}/{len(loader)}] {case_id} -> {out_path.name}")

    if metric._buffers is not None and len(metric.get_buffer()) > 0:
        dice = float(metric.aggregate().item())
        log.info(f"{args.split} 集平均 Dice = {dice:.4f}")


def _save_like(pred: torch.Tensor, batch, out_path: Path) -> None:
    """用原图的 affine 保存预测, 保证空间对齐."""
    import nibabel as nib
    import numpy as np
    arr = pred.squeeze().numpy().astype(np.uint8)
    # 从 MONAI meta tensor 取 affine
    affine = None
    img = batch["image"]
    if hasattr(img, "meta") and "affine" in img.meta:
        affine = np.asarray(img.meta["affine"][0]) if img.meta["affine"].ndim == 3 \
            else np.asarray(img.meta["affine"])
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(arr, affine), str(out_path))


if __name__ == "__main__":
    main()
