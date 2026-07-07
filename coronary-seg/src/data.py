#!/usr/bin/env python3
"""
src/data.py — 三正交方向 2.5D 数据流水线（方案B：一个模型三方向混合）
==================================================================
相比单 z 轴版本，这里沿三个正交方向都切片：
  axis=0 沿 H 切（矢状面，平面 W×D）
  axis=1 沿 W 切（冠状面，平面 H×D）
  axis=2 沿 D 切（轴位面，平面 H×W，原来的 z 方向）
训练时一个模型见到三个方向的切片，学会处理任意血管走向。

预处理链、缓存、类别平衡逻辑与单方向版一致，只是切片索引和取切片
扩展到三个方向。
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
# 1. 预处理链（与之前完全一致）
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
    return split["train"], split["val"], split["test"]


# ----------------------------------------------------------------------
# 2. 三方向切片索引：每个方向分别找含血管切片 + 背景采样
#    切片样本表示为 (case_idx, axis, center, is_pos)
# ----------------------------------------------------------------------
def build_slice_index(case_cache, n_cases, k, neg_per_pos, seed, tag,
                      index_path, axes=(0, 1, 2)):
    if os.path.isfile(index_path):
        with open(index_path) as f:
            meta = json.load(f)
        if (meta.get("k") == k and meta.get("n_cases") == n_cases
                and meta.get("neg_per_pos") == neg_per_pos
                and meta.get("axes") == list(axes)):
            print(f"  [复用] 切片索引 {index_path} "
                  f"（{len(meta['slices'])} 个切片样本）")
            return [tuple(x) for x in meta["slices"]]
        else:
            print(f"  [重建] 索引参数变了，重新生成 {index_path}")

    print(f"  [构建] {tag} 三方向切片索引（遍历 {n_cases} 病例，首次慢）...")
    rng = np.random.RandomState(seed)
    slices = []
    for ci in range(n_cases):
        vol = case_cache[ci]
        label = np.asarray(vol["label"])[0]     # (H, W, D)
        for axis in axes:
            # 沿 axis 方向，每一层的前景像素数
            other = tuple(a for a in (0, 1, 2) if a != axis)
            fg_per = (label > 0).sum(axis=other)   # 长度 = label.shape[axis]
            pos = np.where(fg_per > 0)[0]
            neg = np.where(fg_per == 0)[0]
            n_keep = int(round(neg_per_pos * len(pos)))
            if n_keep > 0 and len(neg) > 0:
                keep = rng.choice(neg, size=min(n_keep, len(neg)),
                                  replace=False)
            else:
                keep = np.array([], dtype=int)
            for c in pos:
                slices.append((ci, int(axis), int(c), 1))
            for c in keep:
                slices.append((ci, int(axis), int(c), 0))
        if (ci + 1) % 20 == 0:
            print(f"    ...{ci + 1}/{n_cases} 病例，累计 {len(slices)} 切片")

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "w") as f:
        json.dump({"k": k, "n_cases": n_cases, "neg_per_pos": neg_per_pos,
                   "axes": list(axes), "slices": slices}, f)
    n_pos = sum(1 for s in slices if s[3] == 1)
    print(f"  [完成] {tag}: {len(slices)} 切片"
          f"（正 {n_pos} / 负 {len(slices) - n_pos}）")
    return slices


# ----------------------------------------------------------------------
# 3. 三方向取切片
# ----------------------------------------------------------------------
def extract_slice_stack(img3d, lab3d, axis, center, k):
    """
    沿 axis 取相邻 2k+1 层堆叠成通道，返回：
      stack (2k+1, A, B) float, lab (1, A, B) float
    axis=2 平面(H,W); axis=1 平面(H,D); axis=0 平面(W,D)
    """
    n = img3d.shape[axis]
    idx = [int(np.clip(center + off, 0, n - 1)) for off in range(-k, k + 1)]
    if axis == 2:
        stack = img3d[:, :, idx].permute(2, 0, 1)      # (2k+1,H,W)
        lab = lab3d[:, :, center]                        # (H,W)
    elif axis == 1:
        stack = img3d[:, idx, :].permute(1, 0, 2)      # (2k+1,H,D)
        lab = lab3d[:, center, :]                        # (H,D)
    else:
        stack = img3d[idx, :, :]                         # (2k+1,W,D)
        lab = lab3d[center, :, :]                        # (W,D)
    stack = stack.contiguous().float()
    lab = lab.unsqueeze(0).float()                       # (1,A,B)
    return stack, lab


# ----------------------------------------------------------------------
# 4. SliceDataset
# ----------------------------------------------------------------------
class SliceDataset(Dataset):
    def __init__(self, case_cache, slice_index, k, crop_size, train,
                 lru_size=2):
        self.case_cache = case_cache
        self.slice_index = slice_index
        self.k = k
        self.train = train
        self._lru = OrderedDict()
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
        case_idx, axis, center, _ = self.slice_index[i]
        img3d, lab3d = self._get_volume(case_idx)
        stack, lab = extract_slice_stack(img3d, lab3d, axis, center, self.k)
        data = self.tf({"image": stack, "label": lab})
        return {"image": data["image"].float(),
                "label": data["label"].float()}


# ----------------------------------------------------------------------
# 5. 按病例分组的 BatchSampler（同 batch 同病例，LRU 命中高）
#    注意：三方向后，同病例的不同方向切片也归到一组，仍然高效
# ----------------------------------------------------------------------
class CaseGroupedBatchSampler(Sampler):
    def __init__(self, slice_index, batch_size, shuffle=True, seed=0,
                 drop_last=False):
        self.groups = defaultdict(list)
        for pos, rec in enumerate(slice_index):
            c = rec[0]
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
# 6. 组装 DataLoader
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
    axes = tuple(cfg.get("axes", (0, 1, 2)))
    axtag = "".join(str(a) for a in axes)
    train_slices = build_slice_index(
        train_cache, len(train_rec), cfg["k"], cfg["neg_per_pos"],
        cfg["seed"], "train",
        os.path.join(idx_dir, f"sidx_tri{axtag}_train_k{cfg['k']}"
                              f"_n{len(train_rec)}.json"), axes=axes)
    val_slices = build_slice_index(
        val_cache, len(val_rec), cfg["k"], cfg["neg_per_pos"],
        cfg["seed"], "val",
        os.path.join(idx_dir, f"sidx_tri{axtag}_val_k{cfg['k']}"
                              f"_n{len(val_rec)}.json"), axes=axes)

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
    train_kw = dict(num_workers=nw, pin_memory=cfg.get("pin_memory", False))
    if nw > 0:
        train_kw["persistent_workers"] = True
        train_kw["prefetch_factor"] = cfg.get("prefetch_factor", 4)
    val_kw = dict(num_workers=min(nw, 2),
                  pin_memory=cfg.get("pin_memory", False))

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              **train_kw)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **val_kw)
    return train_loader, val_loader


# ----------------------------------------------------------------------
# 自测
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--index-dir", default="splits")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--spacing", type=float, default=0.5)
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--crop-size", type=int, default=384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--neg-per-pos", type=float, default=0.25)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-cases", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = vars(args)
    if cfg["max_cases"] == 0:
        cfg["max_cases"] = None

    print("=" * 60)
    print("  三方向 2.5D DataLoader 自测")
    print("=" * 60)
    print(f"  k={cfg['k']} -> 通道={2*cfg['k']+1}, crop={cfg['crop_size']}, "
          f"batch={cfg['batch_size']}, axes=(0,1,2)")

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"\n  train batches/epoch: {len(train_loader)}")
    print(f"  val   batches/epoch: {len(val_loader)}")

    print("\n  拉一个 train batch 确认形状...")
    batch = next(iter(train_loader))
    img, lab = batch["image"], batch["label"]
    print(
        f"    image: {tuple(img.shape)}  范围=[{img.min():.3f}, {img.max():.3f}]")
    print(f"    label: {tuple(lab.shape)}  唯一值={torch.unique(lab).tolist()}")
    exp_c = 2 * cfg["k"] + 1
    ok = (img.shape[1] == exp_c and lab.shape[1] == 1
          and img.shape[2] == cfg["crop_size"]
          and img.shape[3] == cfg["crop_size"])
    print(f"\n  形状检查: {'[通过]' if ok else '[!! 不符预期]'}")
    print("确认后：三方向索引比单方向大约3倍，训练相应变长。")


if __name__ == "__main__":
    main()
