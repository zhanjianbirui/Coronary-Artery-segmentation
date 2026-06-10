#!/usr/bin/env python
"""训练主入口.

用法:
    python scripts/train.py --config configs/default.yaml \
        data.data_root=/path/to/ImageCAS

    # 断点续训 (4天上限被杀后重新提交):
    python scripts/train.py --config configs/default.yaml --resume
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import Config
from src.data import build_dataloaders, load_split
from src.engine import train_one_epoch, validate
from src.model import build_loss, build_metric, build_model, build_post_transforms
from src.utils import count_parameters, get_logger, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--resume", action="store_true", help="从 last.pth 续训")
    ap.add_argument("opts", nargs="*", help="配置覆盖, 如 train.lr=0.001")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config).apply_overrides(args.opts)
    work_dir = Path(cfg.output.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("train", work_dir / "train.log")
    cfg.save(work_dir / "resolved_config.yaml")  # 存最终配置, 可复现

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"设备: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 数据 ----
    split = load_split(cfg.data.split_json)
    if cfg.data.data_root:  # 用配置里的根目录覆盖 split 中的绝对路径前缀(可选)
        pass
    train_loader, val_loader = build_dataloaders(cfg, split)
    log.info(f"训练样本 {len(split['train'])} | 验证样本 {len(split['val'])}")

    # ---- 模型 / 损失 / 优化 ----
    model = build_model(cfg.model).to(device)
    log.info(f"模型 {cfg.model.name}, 参数量 {count_parameters(model):,}")
    loss_fn = build_loss()
    metric = build_metric()
    post_pred, post_label = build_post_transforms(cfg.model.out_channels)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.max_epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")

    # ---- 续训 ----
    start_epoch, best_metric = 0, -1.0
    last_ckpt = work_dir / "last.pth"
    if args.resume and last_ckpt.is_file():
        meta = load_checkpoint(last_ckpt, model, optimizer, scheduler, map_location=device)
        start_epoch = meta["epoch"] + 1
        best_metric = meta["best_metric"]
        log.info(f"从 epoch {start_epoch} 续训, 当前 best Dice={best_metric:.4f}")

    # ---- 主循环 ----
    roi = tuple(cfg.preprocess.patch_size)
    for epoch in range(start_epoch, cfg.train.max_epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scaler=scaler if scaler.is_enabled() else None,
            grad_clip=cfg.train.grad_clip,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        log.info(f"Epoch {epoch:04d} | loss {train_loss:.4f} | lr {lr:.2e} "
                 f"| {time.time()-t0:.1f}s")

        # 每个 epoch 都存 last (供续训), 原子写防损坏
        save_checkpoint(last_ckpt, model, optimizer, scheduler, epoch,
                        best_metric, cfg.to_dict())

        # 定期验证
        if (epoch + 1) % cfg.train.val_interval == 0 or epoch == cfg.train.max_epochs - 1:
            dice = validate(
                model, val_loader, metric, post_pred, post_label, device,
                roi_size=roi, sw_batch_size=cfg.infer.sw_batch_size,
                overlap=cfg.infer.overlap, use_amp=cfg.train.amp and device.type == "cuda",
            )
            log.info(f"           >>> val Dice {dice:.4f} (best {best_metric:.4f})")
            if dice > best_metric:
                best_metric = dice
                save_checkpoint(work_dir / "best.pth", model, optimizer, scheduler,
                                epoch, best_metric, cfg.to_dict())
                log.info(f"           *** 刷新最佳, 保存 best.pth (Dice={dice:.4f})")

    log.info(f"训练完成. 最佳验证 Dice = {best_metric:.4f}")


if __name__ == "__main__":
    main()
