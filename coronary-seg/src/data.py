"""数据模块: 发现 -> 划分 -> 预处理 transform -> DataLoader.

ImageCAS 标准布局 (每病例一个文件夹):
    <data_root>/<case_id>/img.nii.gz
    <data_root>/<case_id>/label.nii.gz

为鲁棒起见, 发现逻辑不假设具体命名: 递归找所有 label 文件,
再在同目录找配对的 image 文件. 这样换个布局也能用.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

# 同目录下识别 image / label 文件名的关键词
_LABEL_KEYS = ("label", "seg", "mask", "gt")
_IMAGE_KEYS = ("img", "image", "ct", "cta", "vol")


def discover_cases(data_root: str | Path) -> list[dict[str, str]]:
    """递归发现 (image, label) 配对, 返回 MONAI 风格的字典列表.

    返回: [{"image": "/path/img.nii.gz", "label": "/path/label.nii.gz", "id": "<case>"}, ...]
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root 不存在或不是目录: {data_root}")

    items: list[dict[str, str]] = []
    seen_dirs: set[Path] = set()

    for label_path in sorted(data_root.rglob("*.nii.gz")):
        name = label_path.name.lower()
        if not any(k in name for k in _LABEL_KEYS):
            continue
        folder = label_path.parent
        if folder in seen_dirs:
            continue
        image_path = _find_image_in(folder, exclude=label_path)
        if image_path is None:
            continue
        seen_dirs.add(folder)
        items.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "id": folder.name,
            }
        )

    if not items:
        raise RuntimeError(
            f"在 {data_root} 下没发现任何 image/label 配对. "
            f"检查目录结构是否为 <case>/img.nii.gz + label.nii.gz"
        )
    return items


def _find_image_in(folder: Path, exclude: Path) -> Path | None:
    candidates = [
        p for p in folder.glob("*.nii.gz")
        if p != exclude and any(k in p.name.lower() for k in _IMAGE_KEYS)
    ]
    if candidates:
        return sorted(candidates)[0]
    # 退路: 同目录里除 label 外只有一个 nii.gz, 那就是它
    others = [p for p in folder.glob("*.nii.gz") if p != exclude]
    return sorted(others)[0] if len(others) == 1 else None


def make_split(
    cases: list[dict[str, str]],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> dict[str, list[dict[str, str]]]:
    """随机划分 train/val/test. 比例之和需为 1."""
    assert abs(sum(ratios) - 1.0) < 1e-6, f"比例之和必须为 1, 收到 {ratios}"
    cases = list(cases)
    random.Random(seed).shuffle(cases)
    n = len(cases)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": cases[:n_train],
        "val": cases[n_train:n_train + n_val],
        "test": cases[n_train + n_val:],
    }


def save_split(split: dict[str, list[dict[str, str]]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)


def load_split(path: str | Path) -> dict[str, list[dict[str, str]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def _base_transforms(pre: Any) -> list:
    """train / val 共用的确定性预处理 (读图 -> 通道 -> 朝向 -> 间距 -> 窗位 -> 裁前景)."""
    return [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=tuple(pre.target_spacing),
            mode=("bilinear", "nearest"),  # 图用双线性, 标签用最近邻(不能插值出小数类别)
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=pre.a_min, a_max=pre.a_max,
            b_min=0.0, b_max=1.0,
            clip=pre.clip,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]


def build_train_transforms(pre: Any, train_cfg: Any) -> Compose:
    """训练: 基础预处理 + 类别均衡 patch 采样 + 轻量增强."""
    t = _base_transforms(pre)
    t += [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=tuple(pre.patch_size),
            pos=train_cfg.pos_ratio,           # 含血管的 patch 占比
            neg=1.0 - train_cfg.pos_ratio,
            num_samples=train_cfg.samples_per_image,
            image_key="image",
            image_threshold=0.0,
            allow_smaller=True,                # 体积小于 patch 时自动 pad
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        EnsureTyped(keys=["image", "label"]),
    ]
    return Compose(t)


def build_val_transforms(pre: Any) -> Compose:
    """验证/推理: 只做确定性预处理, 不裁 patch (滑窗推理处理整图)."""
    t = _base_transforms(pre)
    t += [EnsureTyped(keys=["image", "label"])]
    return Compose(t)


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def build_dataloaders(cfg: Any, split: dict[str, list[dict[str, str]]]):
    """返回 (train_loader, val_loader)."""
    train_tf = build_train_transforms(cfg.preprocess, cfg.train)
    val_tf = build_val_transforms(cfg.preprocess)

    ds_cls = CacheDataset if cfg.data.cache_rate > 0 else Dataset
    train_kwargs = {"transform": train_tf}
    val_kwargs = {"transform": val_tf}
    if cfg.data.cache_rate > 0:
        train_kwargs["cache_rate"] = cfg.data.cache_rate
        val_kwargs["cache_rate"] = cfg.data.cache_rate
        train_kwargs["num_workers"] = cfg.data.num_workers
        val_kwargs["num_workers"] = cfg.data.num_workers

    train_ds = ds_cls(data=split["train"], **train_kwargs)
    val_ds = ds_cls(data=split["val"], **val_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=list_data_collate,
        pin_memory=True,
        drop_last=True,
    )
    # 验证整图尺寸不一, batch_size 必须为 1
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
