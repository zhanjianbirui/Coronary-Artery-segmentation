"""模型 / 损失 / 指标.

工厂模式: 改 cfg.model.name 即可换网络, 训练代码无需改动.
损失:
  - dice_ce        : DiceCELoss (Dice 抗类别不平衡 + CE 稳梯度)
  - dice_ce_cldice : DiceCE + clDice(软骨架中心线 Dice), 针对细血管拓扑连通性
指标 DiceMetric: 分割任务标准评估, 忽略背景只看血管.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, SegResNet
from monai.transforms import AsDiscrete, Compose


# ============================================================
# 网络工厂
# ============================================================
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
    if name == "segresnet":
        # 残差编解码 + 组归一化, 在 3D 医学分割上通常稳超朴素 UNet.
        return SegResNet(
            spatial_dims=3,
            in_channels=model_cfg.in_channels,
            out_channels=model_cfg.out_channels,
            init_filters=getattr(model_cfg, "init_filters", 32),
            blocks_down=tuple(getattr(model_cfg, "blocks_down", [1, 2, 2, 4])),
            blocks_up=tuple(getattr(model_cfg, "blocks_up", [1, 1, 1])),
            dropout_prob=(model_cfg.dropout if model_cfg.dropout > 0 else None),
        )
    raise ValueError(f"未知模型: {model_cfg.name!r} (支持: 'unet', 'segresnet')")


# ============================================================
# clDice 软骨架损失 (Shit et al., CVPR 2021)
# 原理: 对预测/标签做"软骨架化"(可微的形态学细化), 用中心线上的
#       precision/sensitivity 算 clDice. 直接奖励血管树的"连通不断裂",
#       而普通 Dice 对细长血管断几体素几乎无感. 二者互补.
# ============================================================
def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    # 3D 软腐蚀 = 三个方向 min-pool (用 -maxpool(-x) 实现)
    p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
    p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
    p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def _soft_skel(img: torch.Tensor, iters: int) -> torch.Tensor:
    """可微软骨架: 反复腐蚀并提取被开运算去掉的"脊"."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(prob_fg: torch.Tensor, target_fg: torch.Tensor,
                iters: int = 3, smooth: float = 1.0) -> torch.Tensor:
    """prob_fg, target_fg: [B,1,D,H,W], 取值 [0,1]. 返回标量 clDice 损失."""
    skel_pred = _soft_skel(prob_fg, iters)
    skel_true = _soft_skel(target_fg, iters)
    dims = (1, 2, 3, 4)
    tprec = (torch.sum(skel_pred * target_fg, dims) + smooth) / \
            (torch.sum(skel_pred, dims) + smooth)            # 中心线精确率
    tsens = (torch.sum(skel_true * prob_fg, dims) + smooth) / \
            (torch.sum(skel_true, dims) + smooth)            # 中心线敏感度
    cl = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)
    return cl.mean()


class DiceCEclDiceLoss(nn.Module):
    """总损失 = DiceCE + w * clDice. 二分类(背景+前景)下取前景通道算 clDice."""

    def __init__(self, cldice_weight: float = 0.5, cldice_iters: int = 3):
        super().__init__()
        self.dicece = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
        self.w = float(cldice_weight)
        self.iters = int(cldice_iters)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.dicece(logits, labels)
        if self.w > 0:
            prob_fg = torch.softmax(logits.float(), dim=1)[:, 1:2]   # 前景概率
            target_fg = (labels == 1).float()                        # [B,1,...]
            if target_fg.dim() == logits.dim() - 1:                  # 兜底 [B,...]
                target_fg = target_fg.unsqueeze(1)
            loss = loss + self.w * soft_cldice(prob_fg, target_fg, self.iters)
        return loss


def build_loss(loss_cfg: Any = None) -> nn.Module:
    """按配置造损失. loss_cfg 为 None 时退回纯 DiceCE (向后兼容)."""
    name = getattr(loss_cfg, "name", "dice_ce") if loss_cfg is not None else "dice_ce"
    if name == "dice_ce":
        return DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    if name == "dice_ce_cldice":
        return DiceCEclDiceLoss(
            cldice_weight=getattr(loss_cfg, "cldice_weight", 0.5),
            cldice_iters=getattr(loss_cfg, "cldice_iters", 3),
        )
    raise ValueError(f"未知损失: {name!r} (支持: 'dice_ce', 'dice_ce_cldice')")


# ============================================================
# 指标 / 后处理变换
# ============================================================
def build_metric() -> DiceMetric:
    return DiceMetric(include_background=False, reduction="mean", get_not_nans=False)


def build_post_transforms(out_channels: int):
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=out_channels)])
    post_label = Compose([AsDiscrete(to_onehot=out_channels)])
    return post_pred, post_label
