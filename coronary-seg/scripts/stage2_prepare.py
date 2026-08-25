#!/usr/bin/env python3
"""
scripts/stage2_prepare.py — 阶段2训练数据准备
==================================================================
用阶段1的 2.5D 模型对 train/val 每个病例推理，保存：
  - 阶段1粗分割概率图 (float16, 省空间)
  - 对应的原图 和 金标准（从缓存取，保证与概率图对齐）
到磁盘，供阶段2的3D精修网络训练时直接加载。

每个病例存一个 .npz：{prob, image, label}，均为 (H,W,D)。
加 --save-views 时另存 {p0,p1,p2}（三个方向各自的概率），体积约 3 倍。

支持两种阶段1推理方式：
  - 默认（单轴）：沿 axis=2 逐层推理，与最早的 stage-2 数据一致
  - `--tri`（三正交）：沿三个正交轴各推一遍再 mean 融合。
    EXP-012 显示 stage-2 从单轴起点（Dice 0.7955）能精修到 0.8117，
    而三正交起点是 0.8012 且拓扑更好 → 换成三正交有望叠加增益。

**注意**：换推理方式必须换 `--out-dir`。本脚本靠"文件已存在就跳过"做续跑，
指向旧目录会把之前单轴生成的 npz 当成已完成，得到混了两种来源的数据集。

用法（单轴，旧）：
  PYTHONPATH=. python scripts/stage2_prepare.py \
      --cache-dir /path/to/cache/preproc \
      --ckpt runs/exp_2p5d/best.pth \
      --out-dir /path/to/cache/stage2 \
      --splits train,val

用法（三正交，新）：
  PYTHONPATH=. python scripts/stage2_prepare.py \
      --cache-dir /path/to/cache/preproc \
      --ckpt runs/exp_tri2p5d/best.pth --tri --fuse mean \
      --out-dir /path/to/cache/stage2_tri \
      --splits train,val,test \
      --batch 1 --max-px-per-batch 500000
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import build_preprocess, load_split
from src.model import build_model
from monai.data import PersistentDataset
# 复用阶段1的概率推理（含padding、TTA可选）
from scripts.predict import pad_to_multiple_2d
# 三正交逐轴推理+即时融合（内存友好，不同时保留三个概率体）
from scripts.predict_tri import predict_tri_fused, predict_tri_probs


@torch.no_grad()
def predict_prob(model, image3d, k, device, batch=16, pad_multiple=32):
    """阶段1推理，返回概率图 (H,W,D) float32。"""
    img = torch.as_tensor(np.asarray(image3d))[0]
    H, W, D = img.shape
    prob_vol = np.zeros((H, W, D), dtype=np.float32)
    for start in range(0, D, batch):
        zc = list(range(D))[start:start + batch]
        stacks = []
        for z in zc:
            idx = [int(np.clip(z + off, 0, D - 1))
                   for off in range(-k, k + 1)]
            stacks.append(img[:, :, idx].permute(2, 0, 1))
        xb = torch.stack(stacks).float().to(device)
        xb, oh, ow = pad_to_multiple_2d(xb, multiple=pad_multiple)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits = model(xb)
        logits = logits[..., :oh, :ow]
        probs = torch.sigmoid(logits.float())[:, 0].cpu().numpy()
        for j, z in enumerate(zc):
            prob_vol[:, :, z] = probs[j]
    return prob_vol


def save_npz_atomic(out_path, **arrays):
    """原子写 npz：先写 .tmp 再 rename。

    BUG-004 的教训：作业被杀/超时会留下写了一半的 npz，
    而 npz 是 zip 容器，截断后解压必然报
    `zlib.error: Error -3 while decompressing data`，
    并且续跑逻辑只看"文件是否存在"，会把这个坏文件当成已完成永远跳过。
    rename 在同一文件系统上是原子的，所以要么没有文件、要么是完整文件。
    """
    # 注意后缀必须是 .npz：np.savez_compressed 遇到非 .npz 结尾会**自动追加**
    # .npz，写出的其实是 xxx.tmp.npz，随后 rename 找不到文件而失败。
    tmp_path = out_path + ".tmp.npz"
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, out_path)
    except BaseException:
        # 包括 KeyboardInterrupt / SIGTERM 引发的异常，别留垃圾
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits", default="train,val",
                   help="要准备的划分，逗号分隔")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--backbone", default="segresnet")
    p.add_argument("--init-filters", type=int, default=32)
    p.add_argument("--pad-multiple", type=int, default=32)
    p.add_argument("--max-cases", type=int, default=0)
    # ---- 三正交推理（EXP-012 后新增）----
    p.add_argument("--tri", action="store_true",
                   help="用三正交融合生成概率图（换了它必须换 --out-dir）")
    p.add_argument("--axes", type=int, nargs="+", default=[0, 1, 2],
                   help="参与融合的轴，仅 --tri 时生效")
    p.add_argument("--fuse", default="mean", choices=["mean", "max"],
                   help="融合方式，实验证明 mean 明显优于 max")
    p.add_argument("--batch", type=int, default=16,
                   help="逐层推理的 batch；三正交大切片易 OOM，建议 1")
    p.add_argument("--save-views", action="store_true",
                   help="除融合概率外，另存三个方向各自的概率体 p0/p1/p2。"
                        "npz 体积约 3 倍，用于训练多视角 stage-2。"
                        "需与 --tri 同用")
    p.add_argument("--max-px-per-batch", type=int, default=4_000_000,
                   help="batch×平面像素上限，防大切片爆显存；三正交建议 500000")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device}, 输出到 {args.out_dir}")

    # 阶段1模型
    cfg = {"k": args.k, "backbone": args.backbone,
           "init_filters": args.init_filters, "out_channels": 1}
    model = build_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mode = (f"三正交 axes={args.axes} fuse={args.fuse}" if args.tri else "单轴(axis=2)")
    print(f"阶段1模型 epoch={ckpt.get('epoch')} val_dice={ckpt.get('val_dice')}")
    print(f"推理方式: {mode}")
    print(f"输出目录: {args.out_dir}  （续跑靠'文件已存在则跳过'，"
          f"换推理方式务必换目录）")

    preprocess = build_preprocess(args.spacing, args.hu_min, args.hu_max)
    train_rec, val_rec, test_rec = load_split(args.split_json)
    split_map = {"train": train_rec, "val": val_rec, "test": test_rec}

    for split_name in args.splits.split(","):
        split_name = split_name.strip()
        recs = split_map[split_name]
        if args.max_cases and args.max_cases > 0:
            recs = recs[:args.max_cases]
        cache = PersistentDataset(data=recs, transform=preprocess,
                                  cache_dir=args.cache_dir)
        sub_dir = os.path.join(args.out_dir, split_name)
        os.makedirs(sub_dir, exist_ok=True)
        # 上次作业被杀可能留下 .tmp，清掉免得占磁盘
        stale = [f for f in os.listdir(sub_dir) if f.endswith(".tmp.npz")]
        for f in stale:
            os.remove(os.path.join(sub_dir, f))
        if stale:
            print(f"  清理了 {len(stale)} 个残留 .tmp")
        print(f"\n=== {split_name}: {len(recs)} 病例 ===")

        for ci in range(len(recs)):
            cid = recs[ci].get("id", str(ci))
            out_path = os.path.join(sub_dir, f"{cid}.npz")
            if os.path.isfile(out_path):
                print(f"  [{ci+1}/{len(recs)}] {cid} 已存在，跳过")
                continue
            vol = cache[ci]
            image3d = np.asarray(vol["image"])[0].astype(np.float16)  # (H,W,D)
            label = np.asarray(vol["label"])[0].astype(np.uint8)
            extra = {}
            if args.tri and args.save_views:
                # 保留三个方向各自的概率体，而非只存融合结果。
                # 动机：mean 融合会丢弃方向间分歧，而该分歧携带错误位置的
                # 信息（EXP-018：AUC 0.64）。存下来才能让 stage-2 自行学习
                # 如何融合，而不是用人工规则。
                # 代价：npz 体积约为单概率版的 3 倍。
                probs = predict_tri_probs(
                    model, np.asarray(vol["image"]), args.k, device,
                    axes=tuple(args.axes), batch=args.batch,
                    pad_multiple=args.pad_multiple)
                views = [probs[ax].astype(np.float16) for ax in args.axes]
                # 同时存融合结果，使同一份数据既能训练多视角模型，
                # 也能复现既有的 2 通道结果（口径完全一致）
                prob = np.mean([v.astype(np.float32) for v in views],
                               axis=0).astype(np.float16)
                extra = {f"p{i}": v for i, v in enumerate(views)}
                del probs, views
            elif args.tri:
                fused = predict_tri_fused(
                    model, np.asarray(vol["image"]), args.k, device,
                    axes=tuple(args.axes), methods=(args.fuse,),
                    batch=args.batch, pad_multiple=args.pad_multiple,
                    max_px_per_batch=args.max_px_per_batch)
                prob = fused[args.fuse].astype(np.float16)
                del fused
            else:
                prob = predict_prob(model, np.asarray(vol["image"]),
                                    args.k, device,
                                    pad_multiple=args.pad_multiple
                                    ).astype(np.float16)
            save_npz_atomic(out_path, image=image3d, prob=prob, label=label,
                            **extra)
            print(f"  [{ci+1}/{len(recs)}] {cid} 存储完成 "
                  f"shape={image3d.shape}")
            del image3d, label, prob, vol

    print(f"\n完成。阶段2数据在 {args.out_dir}/<split>/<id>.npz")
    print("每个 npz 含: image(原图), prob(阶段1概率), label(金标准)")


if __name__ == "__main__":
    main()
