#!/usr/bin/env python3
"""
src/stage2_model.py — stage-2 残差门控 3D 精修网络
==================================================================
输入 2 通道 [原图, stage-1 概率]，输出精修后的分割 logits。

核心：残差门控（residual gating），不让网络从零重画分割，而是在
stage-1 已有证据上做「增减信心」的修正：

    stage1_logit = logit(clamp(prob, eps, 1-eps))     # 概率反解回 logit 空间
    delta        = Conv(backbone_feature)             # 网络学的修正量
    gate         = sigmoid(GateConv(backbone_feature))# 每体素的修正门控 [0,1]
    final_logit  = stage1_logit + gate * delta

为什么在 logit 空间做残差：
  - 概率空间直接相加会溢出 [0,1]、梯度病态；
  - logit 空间相加天然对应「在原有证据上增/减置信度」，数学干净；
  - 若 delta=0，则 final=stage1，网络退化为恒等，训练起点稳定
    （比从零学分割更容易，且不会一上来破坏 stage-1 的好结果）。

门控 gate 让网络自己决定「哪里该改」：stage-1 已经很确信的区域 gate→0
保持不动，断裂/模糊区域 gate→1 大胆修正。这正是 refinement 的本意，
避免网络退化成「照着 prob 描边」而修不了断裂。

坑（已处理）：
  - prob 存盘可能有恰好 0/1 的体素，logit(0/1)=∓inf。先 clamp 到
    [eps, 1-eps] 再取 logit，否则第一步就 NaN。
"""

import torch
import torch.nn as nn

from monai.networks.nets import SegResNet


def prob_to_logit(prob, eps=1e-5):
    """概率 -> logit，先 clamp 防 ±inf。"""
    prob = prob.clamp(min=eps, max=1.0 - eps)
    return torch.log(prob) - torch.log1p(-prob)


class ResidualGatedSegResNet(nn.Module):
    """
    2 通道输入的残差门控 3D 精修网络。
      in_channels=2  (原图 + stage1 概率)
      out: 精修 logits (B,1,D,H,W)
    """
    def __init__(self, init_filters=16, blocks_down=(1, 2, 2, 4),
                 blocks_up=(1, 1, 1), dropout=0.0, use_gate=True,
                 prob_channel=1):
        super().__init__()
        self.use_gate = use_gate
        self.prob_channel = prob_channel   # 输入里哪个通道是 stage1 概率

        # backbone：SegResNet 输出 init_filters 通道的特征
        # 用 out_channels=init_filters 拿特征，再自己接 delta/gate 头
        self.backbone = SegResNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=init_filters,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout if dropout > 0 else None,
        )
        self.delta_head = nn.Conv3d(init_filters, 1, kernel_size=1)
        if use_gate:
            self.gate_head = nn.Conv3d(init_filters, 1, kernel_size=1)
            # gate 初始偏置设负，使初始 sigmoid(gate)≈小值 → 起步接近恒等，
            # 训练早期先信任 stage-1，稳定收敛
            nn.init.constant_(self.gate_head.bias, -2.0)
        # delta 头初始输出≈0，进一步保证起点=stage1（恒等映射）
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, x):
        # x: (B,2,D,H,W)，通道 prob_channel 是 stage1 概率
        prob = x[:, self.prob_channel:self.prob_channel + 1]     # (B,1,...)
        stage1_logit = prob_to_logit(prob)

        feat = self.backbone(x)                                  # (B,F,...)
        delta = self.delta_head(feat)                            # (B,1,...)
        if self.use_gate:
            gate = torch.sigmoid(self.gate_head(feat))           # (B,1,...)
            final_logit = stage1_logit + gate * delta
        else:
            final_logit = stage1_logit + delta
        return final_logit


def build_stage2_model(cfg):
    return ResidualGatedSegResNet(
        init_filters=cfg.get("init_filters", 16),
        blocks_down=tuple(cfg.get("blocks_down", (1, 2, 2, 4))),
        blocks_up=tuple(cfg.get("blocks_up", (1, 1, 1))),
        dropout=cfg.get("dropout", 0.0),
        use_gate=cfg.get("use_gate", True),
        prob_channel=cfg.get("prob_channel", 1),
    )


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# 自测
# ----------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    print("=" * 60)
    print("  ResidualGatedSegResNet 自测")
    print("=" * 60)

    B, P = 2, 64          # 自测用 64³ 省显存/内存，实际训练 128³
    # 构造输入：通道0原图，通道1是一个"有断裂"的概率体
    img = torch.rand(B, 1, P, P, P)
    prob = torch.zeros(B, 1, P, P, P)
    # 画一条中间断开的"血管"概率
    for i in range(P):
        if 28 <= i <= 32:      # 故意留一段低概率（断裂）
            prob[:, :, i, P // 2, P // 2] = 0.1
        else:
            prob[:, :, i, P // 2, P // 2] = 0.9
    x = torch.cat([img, prob], dim=1)     # (B,2,P,P,P)

    model = build_stage2_model({"init_filters": 16})
    print(f"  参数量: {count_params(model):,}")
    print(f"  输入: {tuple(x.shape)} (通道0=原图, 通道1=stage1概率)")

    # 1) 前向形状
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"  输出: {tuple(out.shape)} (应为 B,1,P,P,P)")

    # 2) 恒等起点检查：delta 头初始化为 0，输出应≈stage1_logit
    with torch.no_grad():
        stage1_logit = prob_to_logit(prob)
        diff = (out - stage1_logit).abs().max().item()
    print(f"  恒等起点: |out - stage1_logit|_max = {diff:.6f} "
          f"(delta初始化为0，应≈0 → 起步=stage1)")

    # 3) clamp 防 inf：prob 含 0 和 1 时不 NaN
    prob_extreme = torch.zeros(B, 1, P, P, P)
    prob_extreme[:, :, :P // 2] = 1.0    # 一半恰好=1，一半恰好=0
    x_ext = torch.cat([img, prob_extreme], dim=1)
    with torch.no_grad():
        out_ext = model(x_ext)
    print(f"  极端prob(0/1): 输出无NaN={not torch.isnan(out_ext).any().item()} "
          f"无Inf={not torch.isinf(out_ext).any().item()}")

    # 4) 反向传播
    model.train()
    out = model(x)
    loss = out.mean()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    print(f"  反向传播: 有非零梯度={has_grad}")

    ok = (out.shape == (B, 1, P, P, P)
          and diff < 1e-4
          and not torch.isnan(out_ext).any().item()
          and has_grad)
    print(f"\n  自测: {'[通过]' if ok else '[!! 不符预期]'}")
    if ok:
        print("  关键性质确认：起步=恒等(不破坏stage1) + 极端prob不炸 + 可训练")


if __name__ == "__main__":
    main()
