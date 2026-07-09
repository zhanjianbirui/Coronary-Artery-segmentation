#!/usr/bin/env python3
"""
scripts/train_stage2.py — stage-2 3D 精修训练入口
==================================================================
复用 stage-1 train.py 的框架（AdamW + CosineAnnealing + best/last +
resume + overfit-one-batch），只替换三处：
  - build_stage2_model 代替 build_model（残差门控 3D SegResNet）
  - build_stage2_loss  代替 build_loss （DiceFocal + soft-clDice）
  - loss 返回 (total, parts) 元组：反向用 total，日志打印 parts

上真机第一件事：先 --overfit-one-batch，确认 loss 下降、dice 上升，
再去掉该 flag 上全量。

依赖你的 stage-2 dataloader（返回 {"image":(B,2,P,P,P), "label":(B,1,P,P,P)}）。
下面从 src.stage2_data 导入 build_stage2_dataloaders；若你的文件/函数名不同，
改这一行的 import 即可。
"""

from src.stage2_data import build_stage2_dataloaders
from src.stage2_loss import build_stage2_loss
from src.stage2_model import build_stage2_model, count_params
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 你的 stage-2 数据流水线入口。若函数名不同，改这里。


def parse_args():
    p = argparse.ArgumentParser()
    # 数据（具体字段按你的 stage2_data 需要；这里透传 cfg）
    p.add_argument("--data-root", required=True,
                   help="stage2_prepare 输出的 npz 目录")
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pos-ratio", type=float, default=0.8,
                   help="含前景 patch 的采样比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=0)
    # 模型
    p.add_argument("--init-filters", type=int, default=16)
    p.add_argument("--no-gate", action="store_true",
                   help="关闭门控，只用纯残差")
    # loss
    p.add_argument("--w-region", type=float, default=1.0)
    p.add_argument("--w-cldice", type=float, default=0.5)
    p.add_argument("--cldice-k", type=int, default=5)
    p.add_argument("--cldice-warmup", type=int, default=500,
                   help="前 N 个 step 只用区域损失，再引入 clDice")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    # 训练
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--out-dir", default="runs/stage2")
    p.add_argument("--log-every", type=int, default=50,
                   help="每多少个 batch 打印一次训练进度")
    p.add_argument("--no-amp", action="store_true")
    # sanity
    p.add_argument("--overfit-one-batch", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    # resume
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def build_cfg(args):
    cfg = vars(args).copy()
    if cfg["max_cases"] == 0:
        cfg["max_cases"] = None
    cfg["use_gate"] = not args.no_gate
    return cfg


@torch.no_grad()
def dice_score(logits, y, thr=0.5, eps=1e-6):
    pred = (torch.sigmoid(logits) > thr).float()
    inter = (pred * y).sum()
    return (2 * inter + eps) / (pred.sum() + y.sum() + eps)


def find_batch_with_fg(loader, max_try=20):
    it = iter(loader)
    batch = None
    for _ in range(max_try):
        batch = next(it)
        if (batch["label"] > 0).any():
            return batch
    return batch


def overfit_one_batch(model, loader, loss_fn, device, steps, lr):
    print("\n" + "=" * 60)
    print("  过拟合单个 batch (stage-2 sanity check)")
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
        loss, parts = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % log_every == 0:
            d = dice_score(logits.detach(), y)
            print(f"  step {step:4d}/{steps}  total={parts['total']:.4f}  "
                  f"region={parts['region']:.4f}  cldice={parts['cldice']:.4f}  "
                  f"dice={d:.4f}")
    print("\n  判读：total 明显下降、dice 升到 >0.9 → stage-2 链路 OK。")
    print("  注意 cldice 在 warmup 结束后才非零，届时 dice 应再上一个台阶。")
    print("=" * 60)


@torch.no_grad()
def validate(model, loader, loss_fn, device, use_amp):
    model.eval()
    tot_loss, tot_dice, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp):
            logits = model(x)
            loss, _ = loss_fn(logits, y)
        tot_loss += loss.item()
        tot_dice += dice_score(logits.float(), y).item()
        n += 1
    return tot_loss / max(1, n), tot_dice / max(1, n)


def train_one_epoch(model, loader, loss_fn, opt, device, scaler,
                    use_amp, grad_clip, epoch=0, epochs=0, log_every=50):
    import time
    model.train()
    tot = 0.0
    n = 0
    n_batches = len(loader)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp):
            logits = model(x)
            loss, parts = loss_fn(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        tot += loss.item()
        n += 1

        if (bi + 1) % log_every == 0 or (bi + 1) == n_batches:
            elapsed = time.time() - t0
            rate = (bi + 1) / max(elapsed, 1e-6)          # batch/s
            eta = (n_batches - bi - 1) / max(rate, 1e-6)  # 剩余秒
            with torch.no_grad():
                pred = (torch.sigmoid(logits.float()) > 0.5).float()
                inter = (pred * y).sum()
                dice = (2 * inter / (pred.sum() + y.sum() + 1e-6)).item()
            print(f"    E{epoch+1}/{epochs} "
                  f"[{bi+1:>4}/{n_batches}] "
                  f"loss={loss.item():.4f} "
                  f"(reg={parts['region']:.4f} cl={parts['cldice']:.4f}) "
                  f"dice={dice:.3f} "
                  f"{rate:.2f}b/s ETA={eta/60:.1f}min",
                  flush=True)
    return tot / max(1, n)


def full_train(model, train_loader, val_loader, loss_fn, device, cfg, use_amp):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"])
    # bfloat16 AMP 不需要 GradScaler；保持 None
    scaler = None
    os.makedirs(cfg["out_dir"], exist_ok=True)

    best_dice = -1.0
    start_epoch = 0
    last_path = os.path.join(cfg["out_dir"], "last.pth")
    if cfg.get("resume") and os.path.isfile(last_path):
        ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_dice = ckpt.get("best_dice", -1.0)
        print(f"[resume] 从 epoch {start_epoch} 续，best_dice={best_dice:.4f}")

    for epoch in range(start_epoch, cfg["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{cfg['epochs']} "
              f"(lr={opt.param_groups[0]['lr']:.2e}) ---")
        tr = train_one_epoch(model, train_loader, loss_fn, opt, device,
                             scaler, use_amp, cfg.get("grad_clip", 1.0),
                             epoch=epoch, epochs=cfg["epochs"],
                             log_every=cfg.get("log_every", 50))
        vl, vd = validate(model, val_loader, loss_fn, device, use_amp)
        sched.step()
        print(f"  train_loss={tr:.4f}  val_loss={vl:.4f}  val_dice={vd:.4f}")

        is_best = vd > best_dice
        if is_best:
            best_dice = vd
        ckpt = {"epoch": epoch, "model": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "val_dice": vd, "best_dice": best_dice}
        tmp = last_path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, last_path)
        if is_best:
            torch.save(ckpt, os.path.join(cfg["out_dir"], "best.pth"))
            print(f"  [best] val_dice={vd:.4f} 已保存")


def main():
    args = parse_args()
    cfg = build_cfg(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_amp)
    print(f"device={device}  amp={use_amp}")

    model = build_stage2_model(cfg).to(device)
    loss_fn = build_stage2_loss(cfg)
    print(f"stage-2 残差门控网络  参数量={count_params(model):,}  "
          f"gate={cfg['use_gate']}")
    print(f"loss: w_region={cfg['w_region']} w_cldice={cfg['w_cldice']} "
          f"cldice_k={cfg['cldice_k']} warmup={cfg['cldice_warmup']}")

    train_loader, val_loader = build_stage2_dataloaders(cfg)
    print(f"train batches={len(train_loader)}  val batches={len(val_loader)}")

    if args.overfit_one_batch:
        overfit_one_batch(model, train_loader, loss_fn, device,
                          args.steps, args.lr)
    else:
        full_train(model, train_loader, val_loader, loss_fn, device,
                   cfg, use_amp)


if __name__ == "__main__":
    main()
