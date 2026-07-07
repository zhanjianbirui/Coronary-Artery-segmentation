"""断点续训.

保存完整训练状态 (model + optimizer + scheduler + epoch + best_metric + config),
使得作业被 SLURM 超时杀掉后, 重新提交能从上次精确续训, 而不是从头来.

约定:
- last.pth : 每个 epoch 覆盖写, 用于续训
- best.pth : 验证指标刷新时写, 用于最终推理
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config_dict: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config_dict,
    }
    # 先写临时文件再原子替换, 防止写一半被杀导致 checkpoint 损坏
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """载入 checkpoint. 返回 meta 信息 {'epoch':..., 'best_metric':..., 'config':...}.

    optimizer / scheduler 给定时一并恢复 (续训用);
    只为推理加载权重时可不传, 仅恢复 model.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {
        "epoch": ckpt.get("epoch", 0),
        "best_metric": ckpt.get("best_metric", -1.0),
        "config": ckpt.get("config"),
    }
