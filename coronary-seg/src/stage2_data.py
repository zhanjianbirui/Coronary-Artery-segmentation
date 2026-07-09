#!/usr/bin/env python3
"""
src/stage2_data.py — 阶段2（3D精修）数据加载
==================================================================
读取 stage2_prepare.py 生成的 npz（原图 + 阶段1概率 + 金标准），
做 3D patch 采样，组成阶段2训练样本：
  输入 = 2通道 [原图, 阶段1概率]  (2, P, P, P)
  标签 = 金标准                    (1, P, P, P)

采样策略：pos_ratio 比例的 patch 中心落在血管前景附近（缓解不平衡）。
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Stage2PatchDataset(Dataset):
    def __init__(self, data_dir, patch_size=128, pos_ratio=0.8,
                 samples_per_case=8, train=True):
        """
        data_dir: 含 <id>.npz 的目录（如 stage2/train）
        patch_size: 3D patch 边长
        pos_ratio: patch 中心落在前景的比例
        samples_per_case: 每个病例每个 epoch 采几个 patch
        """
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        assert len(self.files) > 0, f"{data_dir} 下没有 npz"
        self.P = patch_size
        self.pos_ratio = pos_ratio
        self.spc = samples_per_case
        self.train = train
        # 简单缓存最近用过的 volume（避免反复读盘）
        self._cache = {}
        self._cache_order = []
        self._cache_max = 4

    def __len__(self):
        return len(self.files) * self.spc

    def _load(self, fpath):
        if fpath in self._cache:
            return self._cache[fpath]
        d = np.load(fpath)
        img = d["image"].astype(np.float32)   # (H,W,D)
        prob = d["prob"].astype(np.float32)
        lab = d["label"].astype(np.uint8)
        self._cache[fpath] = (img, prob, lab)
        self._cache_order.append(fpath)
        while len(self._cache_order) > self._cache_max:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)
        return img, prob, lab

    def _sample_center(self, lab):
        """采一个 patch 中心：pos_ratio 概率落在前景，否则随机。"""
        H, W, D = lab.shape
        P = self.P
        half = P // 2
        fg = np.argwhere(lab > 0)
        use_pos = self.train and len(fg) > 0 and np.random.rand() < self.pos_ratio
        if use_pos:
            c = fg[np.random.randint(len(fg))]
        else:
            c = np.array([np.random.randint(H),
                          np.random.randint(W),
                          np.random.randint(D)])
        # clip 到合法范围（保证 patch 不出界）
        c = [int(np.clip(c[i], half, [H, W, D][i] - half - 1))
             for i in range(3)]
        return c

    def _crop(self, vol, center):
        P = self.P
        half = P // 2
        x, y, z = center
        patch = vol[x - half:x - half + P,
                    y - half:y - half + P,
                    z - half:z - half + P]
        # 若边缘不足，pad 到 P
        if patch.shape != (P, P, P):
            pad = [(0, P - patch.shape[i]) for i in range(3)]
            patch = np.pad(patch, pad, mode="constant")
        return patch

    def __getitem__(self, i):
        fpath = self.files[i % len(self.files)]
        img, prob, lab = self._load(fpath)
        center = self._sample_center(lab)
        img_p = self._crop(img, center)
        prob_p = self._crop(prob, center)
        lab_p = self._crop(lab, center)
        # 2通道输入: [原图, 阶段1概率]
        x = np.stack([img_p, prob_p], axis=0).astype(np.float32)  # (2,P,P,P)
        y = lab_p[None].astype(np.float32)                        # (1,P,P,P)
        return {"image": torch.from_numpy(x),
                "label": torch.from_numpy(y)}


def build_stage2_loaders(cfg):
    train_ds = Stage2PatchDataset(
        os.path.join(cfg["data_dir"], "train"),
        patch_size=cfg["patch_size"], pos_ratio=cfg["pos_ratio"],
        samples_per_case=cfg["samples_per_case"], train=True)
    val_ds = Stage2PatchDataset(
        os.path.join(cfg["data_dir"], "val"),
        patch_size=cfg["patch_size"], pos_ratio=cfg["pos_ratio"],
        samples_per_case=max(2, cfg["samples_per_case"] // 2), train=False)

    nw = cfg["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=nw,
                              pin_memory=cfg.get("pin_memory", False),
                              persistent_workers=(nw > 0),
                              prefetch_factor=(4 if nw > 0 else None))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=min(nw, 2),
                            pin_memory=cfg.get("pin_memory", False))
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--pos-ratio", type=float, default=0.8)
    p.add_argument("--samples-per-case", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    cfg = vars(args)
    cfg["pin_memory"] = False
    print("=" * 55)
    print("  阶段2 3D DataLoader 自测")
    print("=" * 55)
    tl, vl = build_stage2_loaders(cfg)
    print(f"train batches={len(tl)}  val batches={len(vl)}")
    batch = next(iter(tl))
    print(f"image: {batch['image'].shape} (应为 B,2,P,P,P)")
    print(f"label: {batch['label'].shape} (应为 B,1,P,P,P)")
    print(f"image范围: {batch['image'].min():.3f}~{batch['image'].max():.3f}")
    print(f"label前景占比: {batch['label'].mean():.4f}")
    print("通道0=原图, 通道1=阶段1概率")
