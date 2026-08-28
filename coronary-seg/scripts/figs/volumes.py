"""定性图的体积数据层。

体积由 scripts/figs/export_nii.py 在集群上导出，默认放在项目外的 vis_nii/。
全部是**预处理空间**（0.5mm 各向同性、已裁剪、HU 窗已加），四个体积几何一致。
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.morphology import skeletonize

DEFAULT_DIR = os.path.expanduser("~/Downloads/Project/vis_nii")

COLUMNS = [
    ("gt",         "ground truth"),
    ("singleaxis", "single-axis"),
    ("triaxial",   "tri-axial"),
    ("stage2",     "final method"),
]


class MissingVolume(FileNotFoundError):
    pass


def load(case, name, vis_dir=DEFAULT_DIR):
    p = os.path.join(vis_dir, f"{case}_{name}.nii.gz")
    if not os.path.exists(p):
        raise MissingVolume(
            f"缺少 {p}\n"
            f"→ 在集群上跑：./scripts/figs/export_fig_assets.sh {case}\n"
            f"  再 scp 回 {vis_dir}/")
    arr = np.asanyarray(nib.load(p).dataobj)
    return arr if name == "image" else arr > 0


def n_components(mask):
    return int(ndimage.label(mask)[1])


def branch_partition(mask):
    """把血管体素按最近的骨架段划分，返回 (owner 标签图, 末梢段 id 集合)。

    用最近邻划分而不是「骨架段膨胀取邻域」—— 后者会把相邻分支的体素
    重复计入，算出来的分支体积偏大。
    """
    sk = skeletonize(mask)
    kern = np.ones((3, 3, 3), np.uint8)
    kern[1, 1, 1] = 0
    deg = np.where(sk, ndimage.convolve(sk.astype(np.uint8), kern,
                                        mode="constant"), 0)
    ends, junc = (deg == 1) & sk, (deg >= 3) & sk
    lab, n = ndimage.label(sk & ~junc, structure=np.ones((3, 3, 3)))
    _, idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    owner = np.where(mask, lab[tuple(idx)], 0)
    terminal = {i for i in range(1, n + 1) if ((lab == i) & ends).any()}
    return owner, terminal


def branch_stats(mask, owner, seg_id):
    """删掉某一分支后各指标的变化 —— Fig.1(c) 的旁注全部来自这里，不写死。"""
    branch = owner == seg_id
    kept = mask & ~branch
    sp, sg = skeletonize(kept), skeletonize(mask)
    tprec = (sp & mask).sum() / sp.sum()
    tsens = (sg & kept).sum() / sg.sum()
    return {
        "voxels": int(branch.sum()),
        "frac_vessel": branch.sum() / mask.sum(),
        "frac_volume": branch.sum() / mask.size,
        "dice": 2 * (kept & mask).sum() / (kept.sum() + mask.sum()),
        "cldice": 2 * tprec * tsens / (tprec + tsens),
        "n_pred": n_components(kept),
        "n_gt": n_components(mask),
        "mask": branch,
    }


def pick_terminal_branch(mask, owner, terminal, target_frac=0.02):
    """挑一条体积占比最接近 target_frac 的末梢分支，避免选到 2 体素的毛刺。"""
    best, best_d = None, None
    for i in terminal:
        f = (owner == i).sum() / mask.sum()
        d = abs(f - target_frac)
        if best_d is None or d < best_d:
            best, best_d = i, d
    return best
