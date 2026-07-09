#!/usr/bin/env python3
"""
scripts/stage2_prepare.py — 阶段2训练数据准备
==================================================================
用阶段1的 2.5D 模型对 train/val 每个病例推理，保存：
  - 阶段1粗分割概率图 (float16, 省空间)
  - 对应的原图 和 金标准（从缓存取，保证与概率图对齐）
到磁盘，供阶段2的3D精修网络训练时直接加载。

每个病例存一个 .npz：{prob, image, label}，均为 (H,W,D)。

用法（GPU节点）：
  PYTHONPATH=. python scripts/stage2_prepare.py \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --ckpt runs/exp_2p5d/best.pth \
      --out-dir /net/scratch/z67253xh/cache/stage2 \
      --splits train,val
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
    print(f"阶段1模型 val_dice={ckpt.get('val_dice')}")

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
            prob = predict_prob(model, np.asarray(vol["image"]),
                                args.k, device,
                                pad_multiple=args.pad_multiple).astype(np.float16)
            np.savez_compressed(out_path, image=image3d, prob=prob, label=label)
            print(f"  [{ci+1}/{len(recs)}] {cid} 存储完成 "
                  f"shape={image3d.shape}")

    print(f"\n完成。阶段2数据在 {args.out_dir}/<split>/<id>.npz")
    print("每个 npz 含: image(原图), prob(阶段1概率), label(金标准)")


if __name__ == "__main__":
    main()
