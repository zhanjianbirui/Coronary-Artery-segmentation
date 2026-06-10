"""通用工具: 随机种子, 日志, 指标统计."""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """统一所有随机源, 保证实验可复现.

    deterministic=True 会启用 cudnn 确定性算法 (更可复现但更慢);
    分割训练通常关掉以换取速度.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # 输入尺寸固定时 benchmark 能自动挑最快卷积算法
        torch.backends.cudnn.benchmark = True


def get_logger(name: str = "coronary", log_file: str | Path | None = None) -> logging.Logger:
    """同时输出到 stdout 和文件 (若给定). 重复调用不会叠加 handler."""
    logger = logging.getLogger(name)
    if logger.handlers:  # 已配置过, 直接返回
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class AverageMeter:
    """累计平均值. 用于 epoch 内汇总 loss / dice."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def count_parameters(model: torch.nn.Module) -> int:
    """可训练参数量."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
