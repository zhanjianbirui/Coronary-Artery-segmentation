"""在已有 checkpoint 上继续训练的公共逻辑（stage-1 / stage-2 共用）。

为什么单独成模块：`--resume` 和 `--init-from` 是两个**不同场景**，
混用会造成破坏（BUG-007）。两个训练脚本各写一份容易走样，所以抽出来。

  --resume     恢复完整训练状态（model + optimizer + scheduler + epoch）。
               用于**训练被中途打断**（例如作业超出时限），此时 last_epoch < T_max，
               scheduler 接着退火，行为完全正确。

  --init-from  只借模型权重，optimizer / scheduler / epoch / best_dice 全部重建。
               用于**已跑满 --epochs 后想再训一段**。

对跑满的 checkpoint 用 --resume 会发生什么（实测，见 tests/test_init_from.py）：

    跑满 T_max=70 后 LR = 0.00e+00
    新建 CosineAnnealingLR(T_max=120)，load_state_dict 后 T_max 被覆盖回 70
    继续 step：LR 3.00e-04 -> 2.95e-02（约初始 lr 的 98 倍）

根因是 PyTorch 的 `_LRScheduler.state_dict()` 返回 `self.__dict__` 里除
optimizer 外的**全部字段，T_max 也在其中**。所以命令行上新传的 --epochs
会被 load_state_dict 静默覆盖回旧值，last_epoch 超过 T_max 后余弦进入
下一个周期，LR 一路回升。不是"白跑"，是会**摧毁已训练好的权重**。
"""
import os

import torch


def validate_init_from(init_from, out_dir, resume):
    """校验 --init-from 的用法，有问题抛 ValueError。

    两条硬规则：

    1. **不能与 --resume 同用** —— 语义冲突（见模块文档）。

    2. **源 checkpoint 不能位于 out_dir 内** —— 否则训练第一次刷新 best.pth
       就会覆盖掉当作初始化的那份权重，而它通常正是某个已验证的最优权重
       （例如 runs/exp_tri2p5d/best.pth，val_dice=0.8135）。必须写到新目录。
    """
    if not init_from:
        return
    if resume:
        raise ValueError(
            "--init-from 与 --resume 不能同用：前者只借权重、其余重建，"
            "后者恢复完整训练状态。想接着跑被打断的训练用 --resume；"
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


def pick_state_dict(ckpt):
    """从 checkpoint 里取模型权重，兼容裸 state_dict（无 "model" 外层）。"""
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def load_init_weights(model, init_from, device, lr, epochs, out_dir):
    """载入 --init-from 的模型权重并打印来源。optimizer/scheduler 由调用方新建。

    strict load：结构不一致直接报错，不静默跳过 —— 静默跳过会得到一个
    半随机初始化的模型，而训练看起来一切正常，是最难查的那类问题。
    """
    ckpt = torch.load(init_from, map_location=device)
    model.load_state_dict(pick_state_dict(ckpt))

    src_epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    src_dice = ckpt.get("val_dice") if isinstance(ckpt, dict) else None
    tail = f" val_dice={src_dice:.4f}" if isinstance(src_dice, float) else ""
    print(f"[init-from] 载入 {init_from} 的模型权重（源 epoch={src_epoch}{tail}）")
    print(f"[init-from] optimizer/scheduler 全部重建："
          f"lr={lr}  T_max={epochs}  从 epoch 0 计数")
    print(f"[init-from] best_dice 重新从 0 开始评判，新的 best.pth 写入 {out_dir}")


def add_init_from_arg(parser):
    """给 argparse 加 --init-from，两个训练脚本共用同一段帮助文本。"""
    parser.add_argument(
        "--init-from", default=None,
        help="只加载该 checkpoint 的模型权重作为初始化，"
             "optimizer/scheduler/epoch 全部重建。"
             "用于「已跑满 --epochs 后想再多训一段」—— "
             "这种情况不能用 --resume（会让 LR 冲到初始值的约 98 倍，见 BUG-007）")
