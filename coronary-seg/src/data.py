#!/usr/bin/env python3
"""
src/data.py — 2.5D 数据流水线（方案B：在线切片 + 磁盘/内存缓存）
==================================================================
职责：
  1. 预处理链（读盘 -> RAS 定向 -> 重采样各向同性 -> HU 加窗），用
     PersistentDataset 缓存到磁盘，只算一次。
  2. 构建"切片索引"：每个病例的哪一层 z 是一个训练样本，含血管切片
     全保留，背景切片按比例采样（切片级类别平衡）。索引缓存成 json。
  3. SliceDataset：给定 (病例, center_z)，从缓存体块取相邻 2k+1 层堆成
     多通道输入，取中心层标注，做 2D 增强 + 统一尺寸。
  4. CaseGroupedBatchSampler：每个 batch 的切片来自同一病例，配合体块
     LRU 缓存，让"在线切片"高效可用。
  5. build_dataloaders(cfg)：返回 train/val DataLoader。

自测：
  python src/data.py \
      --split-json splits/split.json \
      --cache-dir /net/scratch/z67253xh/cache/preproc \
      --max-cases 5 --k 2
  （先用 5 个病例在 login 节点验证 batch 形状，确认无误再上 SLURM 全量）
"""

import os
import json
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, Compose,
    RandFlipd, RandRotate90d, ResizeWithPadOrCropd,
)
from monai.data import PersistentDataset


# ----------------------------------------------------------------------
# 1. 预处理链（确定性、可缓存）—— 与 Step2 验证过的完全一致
# ----------------------------------------------------------------------
def build_preprocess(spacing, hu_min, hu_max):
    keys = ["image", "label"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(spacing, spacing, spacing),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max,
                             b_min=0.0, b_max=1.0, clip=True),
    ])


def load_split(split_json):
    with open(split_json) as f:
        split = json.load(f)
    # 每条已是 {"image","label","id"}，直接可喂 MONAI
    return split["train"], split["val"], split["test"]


# ----------------------------------------------------------------------
# 2. 切片索引：确定每个病例哪些 z 作为样本（含血管全留 + 背景采样）
# ----------------------------------------------------------------------
def build_slice_index(case_cache, n_cases, k, neg_per_pos, seed, tag,
                      index_path):
    """
    遍历每个病例的 label，找含血管切片（正）与背景切片（负），
    负样本按 neg_per_pos 比例采样。返回 [(case_idx, z, is_pos), ...]。
    结果缓存到 index_path，避免每次重算。
    注意：首次运行会触发所有病例的预处理（顺便预热磁盘缓存）。
    """
    if os.path.isfile(index_path):
        with open(index_path) as f:
            meta = json.load(f)
        if meta.get("k") == k and meta.get("n_cases") == n_cases \
                and meta.get("neg_per_pos") == neg_per_pos:
            print(f"  [复用] 切片索引 {index_path} "
                  f"（{len(meta['slices'])} 个切片样本）")
            return [tuple(x) for x in meta["slices"]]
        else:
            print(f"  [重建] 索引参数变了，重新生成 {index_path}")

    print(f"  [构建] {tag} 切片索引（会遍历 {n_cases} 个病例，首次较慢）...")
    rng = np.random.RandomState(seed)
    slices = []
    for ci in range(n_cases):
        vol = case_cache[ci]                    # 触发预处理/读缓存
        label = np.asarray(vol["label"])[0]     # (H, W, D)
        fg_per_z = (label > 0).sum(axis=(0, 1))  # 每层前景像素数
        pos_z = np.where(fg_per_z > 0)[0]
        neg_z = np.where(fg_per_z == 0)[0]
        n_keep_neg = int(round(neg_per_pos * len(pos_z)))
        if n_keep_neg > 0 and len(neg_z) > 0:
            keep = rng.choice(neg_z, size=min(n_keep_neg, len(neg_z)),
                              replace=False)
        else:
            keep = np.array([], dtype=int)
        for z in pos_z:
            slices.append((ci, int(z), 1))
        for z in keep:
            slices.append((ci, int(z), 0))
        if (ci + 1) % 20 == 0:
            print(f"    ...{ci + 1}/{n_cases} 病例，累计 {len(slices)} 切片")

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "w") as f:
        json.dump({"k": k, "n_cases": n_cases, "neg_per_pos": neg_per_pos,
                   "slices": slices}, f)
    n_pos = sum(1 for s in slices if s[2] == 1)
    print(f"  [完成] {tag}: {len(slices)} 切片"
          f"（正 {n_pos} / 负 {len(slices) - n_pos}）")
    return slices


# ----------------------------------------------------------------------
# 3. SliceDataset：取 2.5D 堆叠 + 中心层标注 + 2D 增强
# ----------------------------------------------------------------------
class SliceDataset(Dataset):
    def __init__(self, case_cache, slice_index, k, crop_size, train,
                 lru_size=2):
        self.case_cache = case_cache
        self.slice_index = slice_index
        self.k = k
        self.train = train
        self._lru = OrderedDict()          # case_idx -> (img3d, lab3d)
        self._lru_size = lru_size

        keys = ["image", "label"]
        if train:
            self.tf = Compose([
                RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandRotate90d(keys=keys, prob=0.5, spatial_axes=(0, 1)),
                ResizeWithPadOrCropd(keys=keys,
                                     spatial_size=(crop_size, crop_size)),
            ])
        else:
            self.tf = Compose([
                ResizeWithPadOrCropd(keys=keys,
                                     spatial_size=(crop_size, crop_size)),
            ])

    def __len__(self):
        return len(self.slice_index)

    def _get_volume(self, case_idx):
        if case_idx in self._lru:
            self._lru.move_to_end(case_idx)
            return self._lru[case_idx]
        vol = self.case_cache[case_idx]
        # image 用 float16 存（减半内存/IO），取切片时再转 float32
        img3d = torch.as_tensor(np.asarray(vol["image"]),
                                dtype=torch.float16)[0]   # (H,W,D)
        lab3d = torch.as_tensor(np.asarray(vol["label"]),
                                dtype=torch.uint8)[0]     # (H,W,D)
        self._lru[case_idx] = (img3d, lab3d)
        self._lru.move_to_end(case_idx)
        while len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)
        return img3d, lab3d

    def __getitem__(self, i):
        case_idx, z, _ = self.slice_index[i]
        img3d, lab3d = self._get_volume(case_idx)
        D = img3d.shape[-1]
        zs = [int(np.clip(z + off, 0, D - 1))
              for off in range(-self.k, self.k + 1)]
        stack = img3d[:, :, zs].permute(
            2, 0, 1).contiguous().float()  # (2k+1,H,W)
        lab = lab3d[:, :, z].unsqueeze(0).float()               # (1,H,W)

        data = self.tf({"image": stack, "label": lab})
        return {"image": data["image"].float(),
                "label": data["label"].float()}


# ----------------------------------------------------------------------
# 4. 按病例分组的 BatchSampler：同 batch 切片同病例 -> LRU 命中率高
# ----------------------------------------------------------------------
class CaseGroupedBatchSampler(Sampler):
    def __init__(self, slice_index, batch_size, shuffle=True, seed=0,
                 drop_last=False):
        self.groups = defaultdict(list)
        for pos, (c, z, ip) in enumerate(slice_index):
            self.groups[c].append(pos)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        cases = list(self.groups.keys())
        if self.shuffle:
            rng.shuffle(cases)
        for c in cases:
            pos = self.groups[c][:]
            if self.shuffle:
                rng.shuffle(pos)
            for i in range(0, len(pos), self.batch_size):
                batch = pos[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch

    def __len__(self):
        n = 0
        for pos in self.groups.values():
            if self.drop_last:
                n += len(pos) // self.batch_size
            else:
                n += (len(pos) + self.batch_size - 1) // self.batch_size
        return n


# ----------------------------------------------------------------------
# 5. 组装 DataLoader
# ----------------------------------------------------------------------
def build_dataloaders(cfg):
    preprocess = build_preprocess(cfg["spacing"], cfg["hu_min"], cfg["hu_max"])
    train_rec, val_rec, _ = load_split(cfg["split_json"])

    if cfg.get("max_cases"):
        train_rec = train_rec[:cfg["max_cases"]]
        val_rec = val_rec[:max(1, cfg["max_cases"] // 5)]

    os.makedirs(cfg["cache_dir"], exist_ok=True)
    train_cache = PersistentDataset(data=train_rec, transform=preprocess,
                                    cache_dir=cfg["cache_dir"])
    val_cache = PersistentDataset(data=val_rec, transform=preprocess,
                                  cache_dir=cfg["cache_dir"])

    idx_dir = cfg.get("index_dir", "splits")
    train_slices = build_slice_index(
        train_cache, len(train_rec), cfg["k"], cfg["neg_per_pos"],
        cfg["seed"], "train",
        os.path.join(idx_dir, f"sidx_train_k{cfg['k']}"
                              f"_n{len(train_rec)}.json"))
    val_slices = build_slice_index(
        val_cache, len(val_rec), cfg["k"], cfg["neg_per_pos"],
        cfg["seed"], "val",
        os.path.join(idx_dir, f"sidx_val_k{cfg['k']}"
                              f"_n{len(val_rec)}.json"))

    train_ds = SliceDataset(train_cache, train_slices, cfg["k"],
                            cfg["crop_size"], train=True)
    val_ds = SliceDataset(val_cache, val_slices, cfg["k"],
                          cfg["crop_size"], train=False)

    train_sampler = CaseGroupedBatchSampler(
        train_slices, cfg["batch_size"], shuffle=True, seed=cfg["seed"],
        drop_last=True)
    val_sampler = CaseGroupedBatchSampler(
        val_slices, cfg["batch_size"], shuffle=False, seed=cfg["seed"],
        drop_last=False)

    nw = cfg["num_workers"]
    # 训练 loader：用 persistent_workers + prefetch 加速
    train_kw = dict(num_workers=nw,
                    pin_memory=cfg.get("pin_memory", False))
    if nw > 0:
        train_kw["persistent_workers"] = True
        train_kw["prefetch_factor"] = cfg.get("prefetch_factor", 4)

    # 验证 loader：朴素配置，不用 persistent_workers（只跑一遍，避免死锁）
    val_kw = dict(num_workers=min(nw, 2),
                  pin_memory=cfg.get("pin_memory", False))

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              **train_kw)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler,
                            **val_kw)
    return train_loader, val_loader


# ----------------------------------------------------------------------
# 自测入口
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True,
                   help="预处理磁盘缓存目录（放 scratch，别放 home）")
    p.add_argument("--index-dir", default="splits")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--neg-per-pos", type=float, default=0.25)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=5,
                   help="只用前 N 个病例做快速自测；全量训练设 0")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = vars(args)
    if cfg["max_cases"] == 0:
        cfg["max_cases"] = None

    print("=" * 60)
    print("  Step 3 自测: 2.5D DataLoader")
    print("=" * 60)
    print(f"  k={cfg['k']} -> 输入通道={2*cfg['k']+1}, "
          f"crop={cfg['crop_size']}, batch={cfg['batch_size']}")
    print(f"  cache_dir={cfg['cache_dir']}")
    print(f"  max_cases={cfg['max_cases']}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"\n  train batches/epoch: {len(train_loader)}")
    print(f"  val   batches/epoch: {len(val_loader)}")

    print("\n  拉取一个 train batch 确认形状...")
    batch = next(iter(train_loader))
    img, lab = batch["image"], batch["label"]
    print(f"    image: {tuple(img.shape)}  dtype={img.dtype}  "
          f"范围=[{img.min():.3f}, {img.max():.3f}]")
    print(f"    label: {tuple(lab.shape)}  dtype={lab.dtype}  "
          f"唯一值={torch.unique(lab).tolist()}")
    print(f"    该 batch 前景占比: "
          f"{100 * (lab > 0).float().mean():.3f}%")

    expected_c = 2 * cfg["k"] + 1
    ok = (img.shape[1] == expected_c and lab.shape[1] == 1
          and img.shape[2] == cfg["crop_size"]
          and img.shape[3] == cfg["crop_size"])
    print(f"\n  形状检查: {'[通过]' if ok else '[!! 不符预期，检查]'}")
    print("\n" + "=" * 60)
    print("把输出贴给我。确认 image 是 (B, 2k+1, crop, crop)、"
          "label 是 (B, 1, crop, crop) 且值为 {0,1}，就写第四步（网络）。")
    print("=" * 60)


if __name__ == "__main__":
    main()
