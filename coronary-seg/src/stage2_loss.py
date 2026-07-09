#!/usr/bin/env python3
"""
src/stage2_loss.py — stage-2 精修损失：DiceFocal + soft-clDice
==================================================================
目标是「修断裂」，所以在标准区域损失(DiceFocal)之外，额外加一个直接
优化中心线连通性的 soft-clDice 项。总损失：

    L = w_region * DiceFocal(logits, y)  +  w_cldice * softclDice(prob, y)

设计要点 / 已处理的坑：
  1. soft-clDice 基于可微 soft-skeleton（迭代 min/max-pool 近似形态学细化）。
     血管细，默认迭代 k=5。
  2. 前景极稀疏（~0.9%），采样到的 patch 可能整块无前景。这类 patch 的
     soft-clDice 分母为 0，会产生 NaN 或假梯度 —— 因此对「GT 前景为空」的
     样本，本 batch 的 clDice 项按样本屏蔽（只对有前景的样本求平均）。
  3. soft-clDice 绝不单独用；必须与 Dice/CE 组合，否则训练不稳。
  4. 全部在概率/логits 上用 torch 算子实现，可微。

自测见文件末尾 main()。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.losses import DiceFocalLoss
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False


# ----------------------------------------------------------------------
# soft-skeleton：3D 可微软骨架
# ----------------------------------------------------------------------
def soft_erode3d(x):
    """3D 软腐蚀 = 3 个正交方向 min-pool 的最小值（用 -maxpool(-x) 实现 min）。"""
    # x: (B,1,D,H,W)
    p1 = -F.max_pool3d(-x, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-x, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-x, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def soft_dilate3d(x):
    """3D 软膨胀 = 3x3x3 max-pool。"""
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)


def soft_open3d(x):
    return soft_dilate3d(soft_erode3d(x))


def soft_skeleton3d(x, k=5):
    """
    可微 soft-skeleton（Shit et al. clDice）。
    x: (B,1,D,H,W)，值域[0,1]。返回同形状的软骨架。
    """
    x1 = soft_open3d(x)
    skel = F.relu(x - x1)
    for _ in range(k):
        x = soft_erode3d(x)
        x1 = soft_open3d(x)
        delta = F.relu(x - x1)
        # skel = skel + delta - skel*delta （软并集，避免重复计数）
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ----------------------------------------------------------------------
# soft-clDice
# ----------------------------------------------------------------------
def soft_cldice_loss(prob, target, k=5, eps=1e-5, per_sample_mask=True):
    """
    prob:   (B,1,D,H,W) 预测概率 [0,1]
    target: (B,1,D,H,W) GT {0,1}
    返回标量损失 = 1 - clDice。
    对无前景样本按样本屏蔽（不参与平均），避免空 patch 的 NaN。
    """
    B = prob.shape[0]
    s_pred = soft_skeleton3d(prob, k)
    s_true = soft_skeleton3d(target, k)

    # 逐样本展平求和
    def _flat_sum(a):
        return a.view(B, -1).sum(dim=1)

    # tprec：预测骨架落在 GT 内的比例；tsens：GT 骨架落在预测内的比例
    tprec = (_flat_sum(s_pred * target) + eps) / (_flat_sum(s_pred) + eps)
    tsens = (_flat_sum(s_true * prob) + eps) / (_flat_sum(s_true) + eps)
    cldice = 2.0 * tprec * tsens / (tprec + tsens + eps)   # (B,)
    loss_per = 1.0 - cldice                                 # (B,)

    if per_sample_mask:
        has_fg = (_flat_sum(target) > 0).float()            # (B,)
        denom = has_fg.sum().clamp(min=1.0)
        return (loss_per * has_fg).sum() / denom
    return loss_per.mean()


# ----------------------------------------------------------------------
# 组合损失
# ----------------------------------------------------------------------
class Stage2Loss(nn.Module):
    """
    L = w_region * DiceFocal(logits, y) + w_cldice * softclDice(sigmoid(logits), y)

    logits: (B,1,D,H,W) 网络原始输出（未过 sigmoid）
    y:      (B,1,D,H,W) GT {0,1}
    """
    def __init__(self, w_region=1.0, w_cldice=0.5, cldice_k=5,
                 focal_gamma=2.0, warmup_steps=0):
        super().__init__()
        self.w_region = w_region
        self.w_cldice = w_cldice
        self.cldice_k = cldice_k
        self.warmup_steps = warmup_steps
        self._step = 0

        if _HAS_MONAI:
            # sigmoid=True 表示内部对 logits 过 sigmoid；含 Dice + Focal
            self.region = DiceFocalLoss(sigmoid=True, gamma=focal_gamma,
                                        squared_pred=True)
        else:
            self.region = None

    def _region_loss(self, logits, y):
        if self.region is not None:
            return self.region(logits, y)
        # 退化实现（无 monai 时）：soft Dice + BCE
        prob = torch.sigmoid(logits)
        num = 2 * (prob * y).sum() + 1.0
        den = (prob + y).sum() + 1.0
        dice = 1 - num / den
        bce = F.binary_cross_entropy_with_logits(logits, y)
        return dice + bce

    def forward(self, logits, y):
        region = self._region_loss(logits, y)

        # clDice warmup：前 warmup_steps 步只用区域损失，等预测不再是噪声
        # 再引入 clDice（骨架在纯噪声上无意义，过早引入会拖慢收敛）
        w_cl = self.w_cldice
        if self.warmup_steps > 0 and self._step < self.warmup_steps:
            w_cl = 0.0
        self._step += 1

        if w_cl > 0:
            prob = torch.sigmoid(logits)
            cldice = soft_cldice_loss(prob, y, k=self.cldice_k)
        else:
            cldice = torch.zeros((), device=logits.device)

        total = self.w_region * region + w_cl * cldice
        return total, {"region": region.detach(),
                       "cldice": cldice.detach(),
                       "total": total.detach()}


def build_stage2_loss(cfg):
    return Stage2Loss(
        w_region=cfg.get("w_region", 1.0),
        w_cldice=cfg.get("w_cldice", 0.5),
        cldice_k=cfg.get("cldice_k", 5),
        focal_gamma=cfg.get("focal_gamma", 2.0),
        warmup_steps=cfg.get("cldice_warmup", 0),
    )


# ----------------------------------------------------------------------
# 自测
# ----------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    B, D, H, W = 2, 32, 32, 32
    print("=" * 60)
    print("  Stage2Loss 自测")
    print("=" * 60)

    # 构造一个「有血管」的合成 GT：一条对角线管
    y = torch.zeros(B, 1, D, H, W)
    for i in range(D):
        y[:, :, i, i % H, i % W] = 1
        if i % H + 1 < H:
            y[:, :, i, i % H + 1, i % W] = 1
    print(f"  GT 前景占比: {100*y.mean().item():.3f}%")

    loss_fn = Stage2Loss(w_region=1.0, w_cldice=0.5, cldice_k=5)

    # 1) 完美预测（logits 很大处=前景）：loss 应很小
    logits_perfect = (y * 20 - 10).clone().requires_grad_(True)
    l, parts = loss_fn(logits_perfect, y)
    print(f"\n  [完美预测] total={parts['total']:.4f} "
          f"region={parts['region']:.4f} cldice={parts['cldice']:.4f}")

    # 2) 全背景预测：loss 应较大，且不能 NaN
    logits_bg = torch.full((B, 1, D, H, W), -10.0, requires_grad=True)
    l2, parts2 = loss_fn(logits_bg, y)
    print(f"  [全背景]   total={parts2['total']:.4f} "
          f"region={parts2['region']:.4f} cldice={parts2['cldice']:.4f}")

    # 3) 反向传播能跑通、梯度非 NaN
    l2.backward()
    g = logits_bg.grad
    print(f"  [反向]     grad 非NaN={not torch.isnan(g).any().item()}, "
          f"grad范数={g.norm().item():.4f}")

    # 4) 空 GT patch（全背景 GT）：clDice 应被屏蔽、不 NaN
    y_empty = torch.zeros(B, 1, D, H, W)
    logits_rand = torch.randn(B, 1, D, H, W, requires_grad=True)
    l3, parts3 = loss_fn(logits_rand, y_empty)
    print(f"\n  [空GT]     total={parts3['total']:.4f} "
          f"cldice={parts3['cldice']:.4f} "
          f"(空patch应被屏蔽, cldice≈0, 无NaN={not torch.isnan(l3).any().item()})")

    # 5) soft-skeleton 形状检查
    sk = soft_skeleton3d(y, k=5)
    print(f"\n  soft_skeleton 输出形状: {tuple(sk.shape)} "
          f"(应={tuple(y.shape)}), 值域=[{sk.min():.3f},{sk.max():.3f}]")

    # 判据
    ok = (parts['total'] < parts2['total']            # 完美 < 全背景
          and not torch.isnan(g).any().item()          # 梯度不 NaN
          and abs(parts3['cldice'].item()) < 1e-6      # 空 GT 屏蔽
          and sk.shape == y.shape)
    print(f"\n  自测: {'[通过]' if ok else '[!! 不符预期]'}")


if __name__ == "__main__":
    main()
