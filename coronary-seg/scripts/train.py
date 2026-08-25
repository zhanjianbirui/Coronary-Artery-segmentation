#!/usr/bin/env python3
"""
scripts/train.py — 训练入口
==================================================================
两种模式：
  1. --overfit-one-batch : sanity check。对同一个真实 batch 反复训练，
     若 loss 明显下降、dice 升到接近 1，说明"数据->网络->损失->反向->
     优化器更新"整条链路是通的。上全量训练前必做。
  2. 默认（不加该 flag）: 正式训练循环（AdamW + CosineAnnealing +
     每轮验证 + 保存 best/last checkpoint）。

先在 login 节点用 CPU + 小尺寸跑过拟合：
  CUDA_VISIBLE_DEVICES="" python scripts/train.py \
      --cache-dir /path/to/cache/preproc \
      --overfit-one-batch --crop-size 128 --max-cases 5 \
      --steps 150 --num-workers 0

确认 loss 下降后，再上 SLURM 做全量训练（去掉 overfit flag，crop 512）。
"""

from src.engine import (train_one_epoch, validate, dice_score,
                        make_scaler)
from src.model import build_model, build_loss, count_params
from src.data import build_dataloaders
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--index-dir", default="splits")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--neg-per-pos", type=float, default=0.25)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=0,
                   help="限制病例数做测试；0=全量")
    p.add_argument("--pin-memory", action="store_true")
    # 模型
    p.add_argument("--backbone", default="segresnet",
                   choices=["segresnet", "unet"])
    p.add_argument("--init-filters", type=int, default=32)
    # 训练
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="梯度裁剪范数上限，<=0 关闭")
    p.add_argument("--out-dir", default="runs/exp_2p5d")
    p.add_argument("--no-amp", action="store_true",
                   help="关闭混合精度（CPU 测试时会自动关）")
    # sanity check
    p.add_argument("--overfit-one-batch", action="store_true")
    p.add_argument("--steps", type=int, default=150,
                   help="过拟合模式的迭代步数")
    # 断点续训
    p.add_argument("--resume", action="store_true",
                   help="从 out-dir/last.pth 恢复训练（用于被 SLURM 上限杀掉后接着跑）")
    p.add_argument("--init-from", default=None,
                   help="只加载该 checkpoint 的模型权重作为初始化，"
                        "optimizer/scheduler/epoch 全部重建。"
                        "用于「已跑满 --epochs 后想再多训一段」—— "
                        "这种情况不能用 --resume，见下方 validate_init_from 的说明")
    return p.parse_args()


def validate_init_from(init_from, out_dir, resume):
    """校验 --init-from 的用法，有问题就抛 ValueError。

    两条硬规则：

    1. **不能与 --resume 同用**。两者语义冲突：--resume 恢复完整训练状态
       （optimizer/scheduler/epoch），--init-from 只借权重、其余全部重来。

    2. **源 checkpoint 不能位于 out_dir 内**。否则训练第一次刷新 best.pth
       就会把当作初始化的那份权重覆盖掉 —— 而它通常正是某个已验证的最优权重
       （例如 exp_tri2p5d/best.pth，val_dice=0.8135）。必须写到新目录。

    为什么需要 --init-from 而不能直接 --resume 续训（BUG-007）：
      CosineAnnealingLR 的 T_max 等于当初的 --epochs。跑满之后 LR 已退火到
      eta_min（默认 0）。而 --resume 会 sched.load_state_dict()，PyTorch 的
      scheduler state_dict 里**连 T_max 一起存**，所以命令行上新传的
      --epochs 会被静默覆盖回旧值，last_epoch 超过 T_max 后余弦进入下一周期、
      LR 回升，行为完全不可控。
    """
    if not init_from:
        return
    if resume:
        raise ValueError(
            "--init-from 与 --resume 不能同用：前者只借权重、其余重建，"
            "后者恢复完整训练状态。想接着跑被杀掉的训练用 --resume；"
            "想在已跑满的权重上再训一段用 --init-from。")
    if not os.path.isfile(init_from):
        raise ValueError(f"--init-from 指向的文件不存在: {init_from}")

    src = os.path.realpath(init_from)
    dst = os.path.realpath(out_dir)
    if src.startswith(dst + os.sep) or os.path.dirname(src) == dst:
        raise ValueError(
            f"--init-from 的 checkpoint 位于 --out-dir 内:\n"
            f"  init-from: {src}\n"
            f"  out-dir  : {dst}\n"
            f"训练会在第一次刷新 best.pth 时覆盖掉它。请改用一个新的 --out-dir。")


def build_cfg(args):
    cfg = vars(args).copy()
    if cfg["max_cases"] == 0:
        cfg["max_cases"] = None
    return cfg


def find_batch_with_fg(loader, max_try=10):
    """从 loader 里取一个含前景的 batch（避免抽到全背景）。"""
    it = iter(loader)
    for _ in range(max_try):
        batch = next(it)
        if (batch["label"] > 0).any():
            return batch
    return batch  # 实在没有就用最后一个


def overfit_one_batch(model, loader, loss_fn, device, steps, lr):
    print("\n" + "=" * 60)
    print("  过拟合单个 batch (sanity check)")
    print("=" * 60)
    batch = find_batch_with_fg(loader)
    x = batch["image"].to(device)
    y = batch["label"].to(device)
    fg = 100 * (y > 0).float().mean().item()
    print(f"  batch: x={tuple(x.shape)}  前景占比={fg:.3f}%")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    log_every = max(1, steps // 20)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0:
            d = dice_score(logits.detach().float(), y)
            print(f"  step {step:4d}/{steps}  loss={loss.item():.4f}  "
                  f"dice={d:.4f}")

    print("\n  判读：loss 若从 ~1 明显降到 <0.1、dice 升到 >0.9，"
          "说明整条训练链路 OK。")
    print("=" * 60)


def full_train(model, train_loader, val_loader, loss_fn, device,
               cfg, use_amp):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"])
    scaler, need_scaler = make_scaler(use_amp)
    os.makedirs(cfg["out_dir"], exist_ok=True)

    best_dice = -1.0
    start_epoch = 0

    # ---- 断点续训：从 last.pth 恢复 ----
    last_path = os.path.join(cfg["out_dir"], "last.pth")
    if cfg.get("resume") and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_dice = ckpt.get("best_dice", ckpt.get("val_dice", -1.0))
        print(f"[resume] 从 {last_path} 恢复，"
              f"接续 epoch {start_epoch}，best_dice={best_dice:.4f}")
    elif cfg.get("resume"):
        print(f"[resume] 未找到 {last_path}，从头开始训练")

    # ---- 权重初始化：只借模型权重，optimizer/scheduler/epoch 全部重建 ----
    # 与 --resume 互斥（由 validate_init_from 保证），所以放在 elif 之后是安全的
    elif cfg.get("init_from"):
        ckpt = torch.load(cfg["init_from"], map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)          # strict：结构不一致直接报错，不静默跳过
        src_epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
        src_dice = ckpt.get("val_dice") if isinstance(ckpt, dict) else None
        print(f"[init-from] 载入 {cfg['init_from']} 的模型权重"
              f"（源 epoch={src_epoch}"
              + (f" val_dice={src_dice:.4f}" if isinstance(src_dice, float) else "")
              + "）")
        print(f"[init-from] optimizer/scheduler 全部重建："
              f"lr={cfg['lr']}  T_max={cfg['epochs']}  从 epoch 0 计数")
        print(f"[init-from] best_dice 重新从 0 开始评判，"
              f"新的 best.pth 写入 {cfg['out_dir']}")

    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{cfg['epochs']} "
              f"(lr={opt.param_groups[0]['lr']:.2e}) ---")
        tr_loss = train_one_epoch(
            model, train_loader, loss_fn, opt, device, scaler, need_scaler,
            use_amp, sampler=train_loader.batch_sampler, epoch=epoch,
            grad_clip=cfg.get("grad_clip", 1.0))
        val_loss, val_dice = validate(model, val_loader, loss_fn,
                                      device, use_amp)
        sched.step()
        print(f"  train_loss={tr_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            is_best = True
        else:
            is_best = False

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict() if need_scaler else None,
            "val_dice": val_dice,
            "best_dice": best_dice,
        }
        # 原子写 last.pth（先写临时文件再改名，避免被杀时写坏）
        tmp = os.path.join(cfg["out_dir"], "last.pth.tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, os.path.join(cfg["out_dir"], "last.pth"))

        if is_best:
            torch.save(ckpt, os.path.join(cfg["out_dir"], "best.pth"))
            print(f"  [best] val_dice={val_dice:.4f} 已保存")


def main():
    args = parse_args()
    cfg = build_cfg(args)

    # 参数校验放在最前：构建模型/数据要几分钟，用法错了没必要等到那时才报
    try:
        validate_init_from(cfg.get("init_from"), cfg["out_dir"], cfg.get("resume"))
    except ValueError as e:
        print(f"参数错误：{e}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_amp)
    print(f"device={device}  amp={use_amp}")

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    print(f"backbone={cfg['backbone']}  参数量={count_params(model):,}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"train batches={len(train_loader)}  "
          f"val batches={len(val_loader)}")

    if args.overfit_one_batch:
        overfit_one_batch(model, train_loader, loss_fn, device,
                          args.steps, args.lr)
    else:
        full_train(model, train_loader, val_loader, loss_fn, device,
                   cfg, use_amp)


if __name__ == "__main__":
    main()
