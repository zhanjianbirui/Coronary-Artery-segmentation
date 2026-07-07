#!/usr/bin/env python3
"""
src/engine.py — 训练 / 验证循环核心（数值稳定强化版）
==================================================================
针对"训练中途突然全 nan"做的稳定性强化：
  1. AMP 用 bfloat16 而非 float16
     —— bf16 数值范围与 fp32 相同，几乎不会溢出成 inf/nan（根治）。
     A100 原生支持。bf16 下不需要 GradScaler。
  2. 梯度级 nan/inf 检查
     —— 不只查 loss，还在 optimizer.step() 前查梯度是否有限；
     梯度只要出现 nan/inf 就跳过这一步，绝不让参数被污染。
  3. 梯度裁剪照常保留。

真正的拓扑指标（clDice/Betti-0/HD95）在推理拼回 3D 后评估，不在这里。
"""

import torch


def dice_score(logits, label, thr=0.5, eps=1e-6):
    """二分类 Dice（batch 平均）。logits: (B,1,H,W)，label: (B,1,H,W)。"""
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    dims = (1, 2, 3)
    inter = (pred * label).sum(dim=dims)
    denom = pred.sum(dim=dims) + label.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean().item()


def _amp_dtype(use_amp):
    """bf16 优先（稳定），不支持则退回 fp16。"""
    if not use_amp:
        return None
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _autocast(use_amp):
    dtype = _amp_dtype(use_amp)
    if dtype is None:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def make_scaler(use_amp):
    """
    只有 fp16 才需要 GradScaler；bf16 不需要（范围足够，不做 loss scaling）。
    返回 (scaler, need_scaler)。
    """
    dtype = _amp_dtype(use_amp)
    need = (dtype == torch.float16)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=need)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=need)
    return scaler, need


def _grad_is_finite(model):
    """检查所有梯度是否有限（无 nan/inf）。"""
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                return False
    return True


def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler,
                    need_scaler, use_amp, sampler=None, epoch=0,
                    log_every=20, grad_clip=1.0):
    model.train()
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    total_loss, n = 0.0, 0
    n_skipped = 0
    for i, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        # ---- 防护1：loss 非有限，跳过 ----
        if not torch.isfinite(loss):
            n_skipped += 1
            if n_skipped <= 5 or n_skipped % 200 == 0:
                print(f"    [skip-loss] iter {i+1} loss 非有限，跳过"
                      f"（累计 {n_skipped}）")
            continue

        # ---- 反向 ----
        if need_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # ---- 防护2：梯度非有限，跳过 step（绝不污染参数）----
        if not _grad_is_finite(model):
            n_skipped += 1
            if n_skipped <= 5 or n_skipped % 200 == 0:
                print(f"    [skip-grad] iter {i+1} 梯度非有限，跳过"
                      f"（累计 {n_skipped}）")
            optimizer.zero_grad(set_to_none=True)
            if need_scaler:
                scaler.update()   # 让 scaler 状态推进
            continue

        # ---- 梯度裁剪 ----
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # ---- 更新 ----
        if need_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs
        if (i + 1) % log_every == 0:
            print(f"    [train] iter {i+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}")

    if n_skipped > 0:
        print(f"    [epoch summary] 本轮跳过 {n_skipped} 个坏 batch")
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp):
    model.eval()
    total_loss, total_dice, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with _autocast(use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        bs = x.size(0)
        lv = loss.item()
        if lv == lv and abs(lv) != float("inf"):   # 非 nan 非 inf
            total_loss += lv * bs
        total_dice += dice_score(logits.float(), y) * bs
        n += bs
    return total_loss / max(n, 1), total_dice / max(n, 1)
