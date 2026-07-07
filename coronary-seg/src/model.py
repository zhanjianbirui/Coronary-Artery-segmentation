#!/usr/bin/env python3
"""
src/model.py — 2.5D 分割网络 + 损失
==================================================================
适配 2.5D：
  - spatial_dims = 2（2D 卷积网络）
  - in_channels  = 2k+1（相邻切片堆叠成通道）
  - out_channels = 1（只预测中心层血管概率）

骨干可配置（默认 SegResNet，可切 UNet），损失先用 DiceCE。
clDice 等 3D 拓扑损失不在切片级算（单张 2D 切片无骨架可言），
留到推理拼回 3D 后在评估阶段处理。

自测：
  python src/model.py --k 2 --backbone segresnet
"""

import argparse
import torch
import torch.nn as nn

from monai.networks.nets import SegResNet, UNet
from monai.losses import DiceCELoss


# ----------------------------------------------------------------------
# 网络工厂
# ----------------------------------------------------------------------
def build_model(cfg):
    in_ch = 2 * cfg["k"] + 1
    out_ch = cfg.get("out_channels", 1)
    backbone = cfg.get("backbone", "segresnet").lower()

    if backbone == "segresnet":
        model = SegResNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=out_ch,
            init_filters=cfg.get("init_filters", 32),
            blocks_down=cfg.get("blocks_down", (1, 2, 2, 4)),
            blocks_up=cfg.get("blocks_up", (1, 1, 1)),
            dropout_prob=cfg.get("dropout", 0.0),
        )
    elif backbone == "unet":
        model = UNet(
            spatial_dims=2,
            in_channels=in_ch,
            out_channels=out_ch,
            channels=cfg.get("unet_channels", (32, 64, 128, 256, 512)),
            strides=(2, 2, 2, 2),
            num_res_units=cfg.get("num_res_units", 2),
        )
    else:
        raise ValueError(f"未知 backbone: {backbone}")
    return model


# ----------------------------------------------------------------------
# 损失：DiceCE（sigmoid，单通道二分类）
# ----------------------------------------------------------------------
def build_loss(cfg):
    # out_channels=1 -> 用 sigmoid + 二分类 Dice
    # smooth_nr/smooth_dr: 分子分母平滑项，前景极稀疏时防止 0/0 -> nan
    return DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        smooth_nr=cfg.get("smooth_nr", 1e-5),
        smooth_dr=cfg.get("smooth_dr", 1e-5),
        lambda_dice=cfg.get("lambda_dice", 1.0),
        lambda_ce=cfg.get("lambda_ce", 1.0),
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# 自测
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--backbone", default="segresnet",
                   choices=["segresnet", "unet"])
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--init-filters", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {"k": args.k, "backbone": args.backbone,
           "init_filters": args.init_filters, "out_channels": 1}

    print("=" * 60)
    print("  Step 4 自测: 2.5D 网络 + 损失")
    print("=" * 60)
    in_ch = 2 * args.k + 1
    print(f"  backbone={args.backbone}, in_channels={in_ch}, out_channels=1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device={device}")

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    print(f"  可训练参数量: {count_params(model):,}")

    # 造假输入，走一遍前向 + 损失 + 反向
    B, H, W = args.batch_size, args.crop_size, args.crop_size
    x = torch.randn(B, in_ch, H, W, device=device)
    y = (torch.rand(B, 1, H, W, device=device) > 0.99).float()  # 稀疏前景

    print(f"\n  输入 x: {tuple(x.shape)}")
    logits = model(x)
    print(f"  输出 logits: {tuple(logits.shape)}  "
          f"（应为 ({B}, 1, {H}, {W})）")

    loss = loss_fn(logits, y)
    print(f"  loss: {loss.item():.4f}")

    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    print(f"  反向传播: {'[有梯度，OK]' if has_grad else '[!! 无梯度]'}")

    ok = tuple(logits.shape) == (B, 1, H, W) and has_grad
    print(f"\n  检查: {'[通过]' if ok else '[!! 不通过]'}")
    print("\n" + "=" * 60)
    print("确认输出形状 (B,1,H,W)、loss 正常、有梯度，就写第五步"
          "（训练循环 engine + train.py，先过拟合一个 batch）。")
    print("=" * 60)


if __name__ == "__main__":
    main()