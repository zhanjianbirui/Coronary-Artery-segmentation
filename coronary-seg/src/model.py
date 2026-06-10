"""模型 / 损失 / 指标.

工厂模式: 改 cfg.model.name 即可换网络, 训练代码无需改动.
损失 DiceCELoss: Dice 项抗类别不平衡, CE 项稳定梯度, 二者互补.
指标 DiceMetric: 分割任务的标准评估, 忽略背景只看血管.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete, Compose


def build_model(model_cfg: Any) -> nn.Module:
    """根据配置造网络. 新增模型在此登记即可."""
    name = model_cfg.name.lower()
    if name == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            channels=tuple(model_cfg.channels),
            strides=tuple(model_cfg.strides),
            num_res_units=model_cfg.num_res_units,
            dropout=model_cfg.dropout,
        )
    raise ValueError(f"未知模型: {model_cfg.name!r} (目前支持: 'unet')")


def build_loss() -> nn.Module:
    """DiceCE: softmax 多类. to_onehot_y 把整数标签转 one-hot."""
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,  # 不把背景算进 Dice 损失, 聚焦血管
    )


def build_metric() -> DiceMetric:
    """前景 Dice. reduction=mean_batch 后取均值."""
    return DiceMetric(include_background=False, reduction="mean", get_not_nans=False)


def build_post_transforms(out_channels: int):
    """验证时把网络输出和标签转成离散 one-hot, 喂给 DiceMetric.

    返回 (post_pred, post_label).
    """
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=out_channels)])
    post_label = Compose([AsDiscrete(to_onehot=out_channels)])
    return post_pred, post_label
