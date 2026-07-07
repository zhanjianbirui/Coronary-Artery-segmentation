#!/usr/bin/env python3
"""
Step 2: 预处理 + 2.5D 切片（单病例验证）
------------------------------------------------
用一个病例跑通完整预处理链，并把结果存成 PNG 供肉眼确认：
  1. 读病例 -> RAS 定向 -> 重采样到各向同性 0.5mm -> HU 加窗归一化
  2. 沿 z 轴切片，找一张含血管的切片，画出它的 (2k+1) 个通道 + 标注
  3. 打印：处理后 shape、总切片数、含血管切片数、前景占比

不训练、不改数据，只读一个病例 + 写 PNG 到 --out-dir。

用法（在集群 coronary 环境下）：
  python scripts/vis_slices.py \
      --split-json splits/split.json \
      --k 2 \
      --out-dir debug_vis

  # 想对比不同 k：
  python scripts/vis_slices.py --split-json splits/split.json --k 3 --out-dir debug_vis
"""

import os
import json
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-json", default="splits/split.json")
    p.add_argument("--case-index", type=int, default=0,
                   help="用 train 列表里的第几个病例（默认第0个）")
    p.add_argument("--k", type=int, default=2,
                   help="邻居切片数，输入通道 = 2k+1（默认2→5层）")
    p.add_argument("--spacing", type=float, default=0.5,
                   help="重采样目标各向同性间距 mm（默认0.5）")
    p.add_argument("--hu-min", type=float, default=-200.0)
    p.add_argument("--hu-max", type=float, default=800.0)
    p.add_argument("--out-dir", default="debug_vis")
    return p.parse_args()


def load_case_record(split_json, case_index):
    with open(split_json) as f:
        split = json.load(f)
    rec = split["train"][case_index]
    # 兼容两种格式
    if isinstance(rec, dict):
        return rec.get("id", str(case_index)), rec["image"], rec["label"]
    else:
        raise ValueError("split.json 的 train 项不是 dict，请检查格式")


def preprocess(img_path, label_path, target_spacing, hu_min, hu_max):
    """
    用 MONAI transforms 做：读盘 -> 加通道 -> RAS 定向 -> 重采样各向同性
    -> HU 加窗归一化。返回 numpy: image (1,H,W,D), label (1,H,W,D)
    """
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRanged, Compose,
    )
    keys = ["image", "label"]
    transform = Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys,
                 pixdim=(target_spacing, target_spacing, target_spacing),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=hu_min, a_max=hu_max,
                             b_min=0.0, b_max=1.0, clip=True),
    ])
    data = transform({"image": img_path, "label": label_path})
    image = np.asarray(data["image"])   # (1, H, W, D)
    label = np.asarray(data["label"])   # (1, H, W, D)
    return image, label


def analyze_slices(label):
    """沿最后一个轴(z)统计每张切片的前景像素数。"""
    # label: (1, H, W, D)
    lab = label[0]                       # (H, W, D)
    fg_per_slice = (lab > 0).sum(axis=(0, 1))   # 长度 D
    vessel_slices = np.where(fg_per_slice > 0)[0]
    return fg_per_slice, vessel_slices


def make_2p5d_sample(image, center_z, k):
    """
    取中心层 center_z 及其上下各 k 层，堆叠成 (2k+1, H, W)。
    边界用 clip 处理（超出范围就重复边界层）。
    """
    img = image[0]                       # (H, W, D)
    H, W, D = img.shape
    zs = [np.clip(center_z + off, 0, D - 1) for off in range(-k, k + 1)]
    channels = [img[:, :, z] for z in zs]   # 每个 (H, W)
    stack = np.stack(channels, axis=0)      # (2k+1, H, W)
    return stack, zs


def save_visualization(stack, center_label_slice, zs, center_z, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = stack.shape[0]                   # 2k+1
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3.2))

    for i in range(n):
        axes[i].imshow(stack[i], cmap="gray", vmin=0, vmax=1)
        tag = "center" if zs[i] == center_z else f"z={zs[i]}"
        axes[i].set_title(f"ch{i} ({tag})", fontsize=9)
        axes[i].axis("off")

    # 最后一格：中心层标注叠加
    center_ch = stack[n // 2]
    axes[n].imshow(center_ch, cmap="gray", vmin=0, vmax=1)
    axes[n].imshow(np.ma.masked_where(center_label_slice == 0,
                                      center_label_slice),
                   cmap="autumn", alpha=0.7)
    axes[n].set_title("center + label", fontsize=9)
    axes[n].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("  Step 2: 预处理 + 2.5D 切片（单病例验证）")
    print("=" * 60)

    case_id, img_path, label_path = load_case_record(
        args.split_json, args.case_index)
    print(f"病例: {case_id}")
    print(f"  img:   {img_path}")
    print(f"  k = {args.k}  ->  输入通道 = {2 * args.k + 1}")
    print(f"  目标间距: {args.spacing}mm 各向同性")
    print(f"  HU 窗: [{args.hu_min}, {args.hu_max}] -> [0, 1]")

    print("\n[1/3] 预处理中（读盘+定向+重采样+加窗）...")
    image, label = preprocess(img_path, label_path, args.spacing,
                              args.hu_min, args.hu_max)
    print(f"  处理后 image shape: {image.shape}  (1, H, W, D)")
    print(f"  处理后 label shape: {label.shape}")
    print(f"  image 值域: [{image.min():.3f}, {image.max():.3f}]")

    print("\n[2/3] 统计切片...")
    fg_per_slice, vessel_slices = analyze_slices(label)
    D = image.shape[-1]
    print(f"  总切片数 (z): {D}")
    print(f"  含血管切片数: {len(vessel_slices)}  "
          f"({100 * len(vessel_slices) / D:.1f}%)")
    if len(vessel_slices) > 0:
        # 选前景最多的一张切片来可视化（血管最明显）
        center_z = int(fg_per_slice.argmax())
        print(f"  前景最多的切片 z={center_z}  "
              f"(该层前景像素 {int(fg_per_slice[center_z])})")
    else:
        print("  [!] 这个病例没有前景切片？请换 --case-index 试试。")
        return

    print("\n[3/3] 生成 2.5D 堆叠可视化...")
    stack, zs = make_2p5d_sample(image, center_z, args.k)
    center_label_slice = label[0][:, :, center_z]
    out_path = os.path.join(
        args.out_dir, f"case{case_id}_z{center_z}_k{args.k}.png")
    save_visualization(stack, center_label_slice, zs, center_z, out_path)
    print(f"  已保存: {out_path}")
    print(f"  堆叠 shape: {stack.shape}  (2k+1, H, W)")

    print("\n" + "=" * 60)
    print("完成。把终端输出贴给我，并下载 PNG 看一眼：")
    print("  - 5(或2k+1)个通道相邻层差异大不大")
    print("  - 最后一格标注(红色)是否准确压在血管上")
    print("  - 含血管切片占比是否合理")
    print("确认后写第三步（Dataset/DataLoader 封装）。")
    print("=" * 60)


if __name__ == "__main__":
    main()
