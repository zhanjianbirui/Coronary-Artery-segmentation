"""训练引擎: 单 epoch 训练 + 验证.

抽成纯函数, 与训练脚本解耦, 便于测试和复用.
- train_one_epoch: AMP 混合精度 + 梯度裁剪
- validate: 滑窗推理整图 + Dice 指标
"""
from __future__ import annotations

from typing import Any

import torch
from monai.inferers import sliding_window_inference

from .utils import AverageMeter


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float = 0.0,
) -> float:
    """跑一个训练 epoch, 返回平均 loss.

    scaler 非 None 时启用 AMP. grad_clip>0 时裁剪梯度.
    """
    model.train()
    meter = AverageMeter()
    use_amp = scaler is not None

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        meter.update(loss.item(), n=images.size(0))
    return meter.avg


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: Any,
    metric: Any,
    post_pred: Any,
    post_label: Any,
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int = 4,
    overlap: float = 0.5,
    use_amp: bool = True,
) -> float:
    """滑窗推理整张验证图, 累计 Dice, 返回平均值."""
    model.eval()
    metric.reset()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = sliding_window_inference(
                inputs=images,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
            )
        # decollate -> 逐样本后处理 -> 喂指标
        preds = [post_pred(logits[i]) for i in range(logits.shape[0])]
        gts = [post_label(labels[i]) for i in range(labels.shape[0])]
        metric(preds, gts)

    result = metric.aggregate()
    if isinstance(result, (list, tuple)):
        result = result[0]
    return float(result.item())
