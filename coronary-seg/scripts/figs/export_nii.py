#!/usr/bin/env python3
"""
scripts/figs/export_nii.py — 把指定病例的各阶段分割结果导出成 .nii.gz
==================================================================
用途：论文 Fig. 1（原图 + 金标准）与 Fig. 5（GT / 单轴 / 三正交 / +Stage-2
四列定性对比）需要在 3D Slicer 里渲染，而现有推理脚本只写 csv 指标，
不落体积文件。本脚本补上这一步。

四列里有三列不需要跑模型 —— stage2_prepare.py 存的 npz 已经含
{image, prob, label}，单轴与三正交只是两个不同的 --out-dir：
  单轴   prob ← <cache>/stage2/test/<id>.npz
  三正交 prob ← <cache>/stage2_tri/test/<id>.npz
只有 Stage-2 那一列要加载 checkpoint 现算。

⚠️ 导出的是**预处理空间**（0.5mm 各向同性、已裁剪），不是原始 CTA 空间。
四个体积几何完全一致，因此可比 —— 这正是定性对比需要的。若要叠加原始
CTA，请改用 ImageCAS 原始 nii（见 --help 末尾说明）。

用法（在集群上跑）：
  PYTHONPATH=. python scripts/figs/export_nii.py \
      --tri-root  /path/to/cache/stage2_tri/test \
      --sa-root   /path/to/cache/stage2/test \
      --ckpt      runs/stage2_tri_nogate/best.pth --no-gate \
      --case-ids  293 748 \
      --out-dir   vis_nii

只要 Fig.1（原图 + GT）时，省掉 --sa-root 和 --ckpt 即可。
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict.predict_stage2 import load_npz, postprocess, infer_case


def save_nii(arr, path, spacing):
    """按各向同性 spacing 写出 nii.gz（预处理空间无原始 affine）。"""
    affine = np.diag([spacing, spacing, spacing, 1.0])
    nib.save(nib.Nifti1Image(np.asarray(arr), affine), path)
    print(f"  写出 {os.path.basename(path)}  shape={arr.shape}")


def prob_to_mask(prob, thr, min_voxels):
    return postprocess((prob >= thr).astype(np.uint8), min_voxels=min_voxels)


def parse_args():
    p = argparse.ArgumentParser(
        description="导出指定病例的 image/GT/各阶段预测为 nii.gz")
    p.add_argument("--tri-root", required=True,
                   help="三正交 stage2_prepare 输出的 test 子目录（提供 image/label/三正交 prob）")
    p.add_argument("--sa-root", default=None,
                   help="单轴 stage2_prepare 输出的 test 子目录（Fig.5 的单轴列，可省）")
    p.add_argument("--ckpt", default=None,
                   help="stage-2 权重；给了才导出 Stage-2 那一列")
    p.add_argument("--case-ids", nargs="+", required=True)
    p.add_argument("--out-dir", default="vis_nii")
    p.add_argument("--spacing", type=float, default=0.5,
                   help="预处理时的各向同性体素大小，须与 target_spacing 一致")
    p.add_argument("--thr", type=float, default=0.50)
    p.add_argument("--min-voxels", type=int, default=300)
    # 以下与 predict_stage2.py 保持同名同默认，避免两处口径漂移
    p.add_argument("--init-filters", type=int, default=16)
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--multi-view", action="store_true")
    p.add_argument("--roi", type=int, default=128)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--sw-batch", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def build_stage2(args, device):
    """cfg 的构造与 predict_stage2.py:220 保持逐字一致，防两处口径漂移。"""
    from src.stage2_model import build_stage2_model
    cfg = {"init_filters": args.init_filters, "use_gate": not args.no_gate,
           "multi_view": args.multi_view}
    model = build_stage2_model(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"已加载 {args.ckpt}"
          f"（epoch={ckpt.get('epoch')} val_dice={ckpt.get('val_dice')}）")
    return model


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_stage2(args, device) if args.ckpt else None
    if args.ckpt and device == "cpu":
        print("⚠️ 无 GPU，Stage-2 滑窗推理会很慢")

    for cid in args.case_ids:
        tri_npz = os.path.join(args.tri_root, f"{cid}.npz")
        if not os.path.exists(tri_npz):
            print(f"[跳过] 找不到 {tri_npz}")
            continue
        print(f"\n=== case {cid} ===")
        image, prob_tri, label, views = load_npz(tri_npz)

        save_nii(image.astype(np.float32),
                 os.path.join(args.out_dir, f"{cid}_image.nii.gz"), args.spacing)
        if label is not None:
            save_nii(label.astype(np.uint8),
                     os.path.join(args.out_dir, f"{cid}_gt.nii.gz"), args.spacing)
        else:
            print("  ⚠️ npz 里没有 label，跳过 GT")

        save_nii(prob_to_mask(prob_tri, args.thr, args.min_voxels),
                 os.path.join(args.out_dir, f"{cid}_triaxial.nii.gz"), args.spacing)

        if args.sa_root:
            sa_npz = os.path.join(args.sa_root, f"{cid}.npz")
            if os.path.exists(sa_npz):
                _, prob_sa, _, _ = load_npz(sa_npz)
                save_nii(prob_to_mask(prob_sa, args.thr, args.min_voxels),
                         os.path.join(args.out_dir, f"{cid}_singleaxis.nii.gz"),
                         args.spacing)
            else:
                print(f"  ⚠️ 找不到 {sa_npz}，跳过单轴列")

        if model is not None:
            prob_s2 = infer_case(model, image, prob_tri, device,
                                 args.roi, args.overlap,
                                 use_amp=(not args.no_amp and device == "cuda"),
                                 sw_batch=args.sw_batch,
                                 views=views if args.multi_view else None)
            save_nii(prob_to_mask(prob_s2, args.thr, args.min_voxels),
                     os.path.join(args.out_dir, f"{cid}_stage2.nii.gz"), args.spacing)

    print(f"\n完成。全部写入 {args.out_dir}/")


if __name__ == "__main__":
    main()
