#!/usr/bin/env python3
"""
Step 1: 数据核对 + 划分核对/生成
------------------------------------------------
这个脚本做三件事，全部是只读或只写 split.json，不碰原始数据：
  1. 核对 1000 例数据是否齐全、命名是否符合预期
  2. 抽查一个样本，打印 shape / spacing / affine（确认能正常读）
  3. 复用旧的 split.json（若存在则校验），否则用 seed=42 重新生成

用法（在集群 coronary 环境下）：
  python scripts/check_data.py \
      --data-root /mnt/iusers01/fse-ugpgt01/compsci01/z67253xh/scratch/ImageCAS \
      --split-json splits/split.json
"""

import os
import re
import json
import glob
import argparse
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="ImageCAS 数据根目录")
    p.add_argument("--split-json", default="splits/split.json",
                   help="划分文件路径（存在则校验，不存在则生成）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", default="700,100,200",
                   help="train,val,test 例数，逗号分隔")
    p.add_argument("--regenerate", action="store_true",
                   help="强制重新生成 split.json（即使已存在）")
    return p.parse_args()


def discover_cases(data_root):
    """
    扫描数据根目录，找出所有病例。
    预期文件命名：<id>.img.nii.gz + <id>.label.nii.gz
    预期分组子目录：1-200/, 201-400/, ... （也兼容平铺）
    返回：{case_id: {"img": path, "label": path}}
    """
    img_files = glob.glob(os.path.join(data_root, "**", "*.img.nii.gz"),
                          recursive=True)
    cases = {}
    for img_path in img_files:
        base = os.path.basename(img_path)              # e.g. "1.img.nii.gz"
        m = re.match(r"^(.+)\.img\.nii\.gz$", base)
        if not m:
            continue
        case_id = m.group(1)
        label_path = img_path.replace(".img.nii.gz", ".label.nii.gz")
        cases[case_id] = {
            "img": img_path,
            "label": label_path,
            "label_exists": os.path.isfile(label_path),
        }
    return cases


def check_integrity(cases):
    print("\n" + "=" * 60)
    print("  1. 数据完整性核对")
    print("=" * 60)
    total = len(cases)
    print(f"发现病例数: {total}")

    missing_label = [cid for cid, c in cases.items() if not c["label_exists"]]
    if missing_label:
        print(f"[!] 缺少 label 的病例 ({len(missing_label)} 个):")
        for cid in missing_label[:10]:
            print(f"      {cid}")
        if len(missing_label) > 10:
            print(f"      ... 还有 {len(missing_label) - 10} 个")
    else:
        print("[ok] 每个病例都有对应的 img 和 label")

    if total != 1000:
        print(f"[!] 注意：病例数是 {total}，不是预期的 1000。请检查数据路径。")
    else:
        print("[ok] 病例总数 = 1000，符合预期")

    return total, missing_label


def inspect_sample(cases):
    print("\n" + "=" * 60)
    print("  2. 抽查一个样本的几何信息")
    print("=" * 60)
    try:
        import nibabel as nib
    except ImportError:
        print("[--] nibabel 未安装，跳过。")
        return

    sample_id = sorted(cases.keys())[0]
    img_path = cases[sample_id]["img"]
    label_path = cases[sample_id]["label"]
    print(f"样本病例: {sample_id}")
    print(f"  img:   {img_path}")

    img = nib.load(img_path)
    print(f"  shape:      {img.shape}")
    print(f"  dtype:      {img.get_data_dtype()}")
    print(f"  spacing:    {tuple(round(z, 3) for z in img.header.get_zooms())} mm")
    print(f"  affine:\n{img.affine}")

    if cases[sample_id]["label_exists"]:
        import numpy as np
        lab = nib.load(label_path)
        lab_data = lab.get_fdata()
        fg = float((lab_data > 0).mean()) * 100
        uniq = np.unique(lab_data)
        print(f"  label shape:   {lab.shape}")
        print(f"  label values:  {uniq[:10]}")
        print(f"  前景占比:      {fg:.3f}%  （确认极度不平衡）")


def handle_split(cases, split_json, seed, ratios, regenerate):
    print("\n" + "=" * 60)
    print("  3. 划分文件 (split.json)")
    print("=" * 60)

    all_ids = sorted(cases.keys(),
                     key=lambda x: int(x) if x.isdigit() else x)

    if os.path.isfile(split_json) and not regenerate:
        print(f"[已存在] {split_json} —— 校验其与当前数据是否一致")
        with open(split_json) as f:
            split = json.load(f)
        for name in ("train", "val", "test"):
            if name not in split:
                print(f"[!] split.json 缺少 '{name}' 键")
                return
        n_tr, n_va, n_te = (len(split["train"]), len(split["val"]),
                            len(split["test"]))
        print(f"  train={n_tr}, val={n_va}, test={n_te}, 合计={n_tr+n_va+n_te}")
        # 检查划分里的病例是否都能在数据里找到
        split_ids = set()
        for name in ("train", "val", "test"):
            for item in split[name]:
                # 兼容两种格式：纯 id 列表，或 {"img":..,"label":..} 列表
                if isinstance(item, dict):
                    m = re.search(r"([^/]+)\.img\.nii\.gz$", item.get("img", ""))
                    if m:
                        split_ids.add(m.group(1))
                else:
                    split_ids.add(str(item))
        data_ids = set(all_ids)
        missing = split_ids - data_ids
        if missing:
            print(f"[!] split 里有 {len(missing)} 个病例在数据中找不到（前5个）:")
            for cid in list(missing)[:5]:
                print(f"      {cid}")
        else:
            print("[ok] split.json 里的所有病例都能在数据中找到，可直接复用")
        return

    # 生成新的 split
    print(f"[生成] 未找到 split.json 或指定了 --regenerate，用 seed={seed} 生成")
    n_train, n_val, n_test = [int(x) for x in ratios.split(",")]
    if n_train + n_val + n_test != len(all_ids):
        print(f"[!] 划分例数 {n_train}+{n_val}+{n_test} "
              f"!= 病例总数 {len(all_ids)}，请调整 --ratios")
        return

    rng = random.Random(seed)
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    train_ids = sorted(shuffled[:n_train])
    val_ids = sorted(shuffled[n_train:n_train + n_val])
    test_ids = sorted(shuffled[n_train + n_val:])

    def to_records(ids):
        return [{"id": cid, "img": cases[cid]["img"],
                 "label": cases[cid]["label"]} for cid in ids]

    split = {
        "seed": seed,
        "train": to_records(train_ids),
        "val": to_records(val_ids),
        "test": to_records(test_ids),
    }
    os.makedirs(os.path.dirname(split_json) or ".", exist_ok=True)
    with open(split_json, "w") as f:
        json.dump(split, f, indent=2)
    print(f"[ok] 已写入 {split_json}")
    print(f"  train={n_train}, val={n_val}, test={n_test}")


def main():
    args = parse_args()
    print("=" * 60)
    print("  Step 1: 数据核对 + 划分")
    print("=" * 60)
    print(f"DATA_ROOT: {args.data_root}")

    if not os.path.isdir(args.data_root):
        print(f"[!] 数据根目录不存在: {args.data_root}")
        return

    cases = discover_cases(args.data_root)
    if not cases:
        print("[!] 没有发现任何 *.img.nii.gz 文件，请检查路径和命名。")
        return

    check_integrity(cases)
    inspect_sample(cases)
    handle_split(cases, args.split_json, args.seed, args.ratios,
                 args.regenerate)

    print("\n" + "=" * 60)
    print("完成。把上面的输出贴给我，确认无误后我们写第二步（预处理+2.5D切片）。")
    print("=" * 60)


if __name__ == "__main__":
    main()
