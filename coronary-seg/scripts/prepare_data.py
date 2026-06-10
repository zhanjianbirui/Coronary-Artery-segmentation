#!/usr/bin/env python
"""下载 ImageCAS + 生成数据划分. 在 login 节点运行 (需要联网).

用法:
    python scripts/prepare_data.py --config configs/default.yaml
    # 或指定已下载好的目录, 跳过下载:
    python scripts/prepare_data.py --data_root /path/to/ImageCAS
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data import discover_cases, make_split, save_split
from src.utils import get_logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data_root", default="", help="已有数据目录; 给定则跳过下载")
    ap.add_argument("--ratios", default="0.7,0.1,0.2", help="train,val,test 比例")
    ap.add_argument("opts", nargs="*", help="配置覆盖, 如 data.split_json=...")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config).apply_overrides(args.opts)
    log = get_logger("prepare")

    data_root = args.data_root or cfg.data.data_root
    if not data_root:
        log.info("未指定 data_root, 通过 kagglehub 下载 ImageCAS (体积较大, 请耐心)...")
        import kagglehub
        data_root = kagglehub.dataset_download("xiaoweixumedicalai/imagecas")
        log.info(f"下载完成: {data_root}")

    # kagglehub 有时会多套一层目录; 自动定位真正含病例文件夹的根
    data_root = _locate_dataset_root(Path(data_root))
    log.info(f"数据根目录: {data_root}")

    cases = discover_cases(data_root)
    log.info(f"发现 {len(cases)} 个病例")

    ratios = tuple(float(x) for x in args.ratios.split(","))
    split = make_split(cases, ratios=ratios, seed=cfg.seed)
    log.info(f"划分: train={len(split['train'])} "
             f"val={len(split['val'])} test={len(split['test'])}")

    out = cfg.data.split_json
    save_split(split, out)
    log.info(f"划分已保存到 {out}")
    log.info(f"训练时请确保 data.data_root 指向: {data_root}")


def _locate_dataset_root(path: Path) -> Path:
    """若 path 下没有直接的 nii.gz, 向下找含 nii.gz 的层级."""
    if any(path.rglob("*.nii.gz")):
        # 找到任意 label 文件, 其父目录的父目录通常就是数据根
        for p in path.rglob("*.nii.gz"):
            if "label" in p.name.lower() or "seg" in p.name.lower():
                return p.parent.parent
        return path
    raise FileNotFoundError(f"{path} 下找不到 nii.gz 文件")


if __name__ == "__main__":
    main()
