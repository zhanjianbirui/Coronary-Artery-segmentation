"""Fig. 1 — 问题引入（论文 §1.1）

论证任务：让读者第一眼明白这个分割任务不是普通分割。

(a) 一张轴位切片 + 血管标注 —— 前景极小
(b) 整棵树的三维渲染 —— 细长、树状、左右两棵各自独立
(c) 一条远端分支的局部放大 + 删掉它之后各指标的变化

⚠️ (c) 的所有数字都由 volumes.branch_stats 实算，不沿用 figures.md 里
「占全体积 0.02%、Dice 降 <0.005」的估计 —— 实测该病例上一条真实的
远端分支占全体积 0.003%，删掉它 Dice 降 0.010、clDice 降 0.014。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from . import render3d as r3
from . import volumes as vol
from .style import (apply_style, despine, TEXTWIDTH_IN,
                    BLUE, ORANGE, GREY, INK, INK2, RED)

CASE = "702"
ELEV, AZIM = 35, -120


def panel_slice(ax, image, gt):
    """选血管体素最多的那张轴位切片，否则读者只看到一两个点。"""
    counts = gt.sum(axis=(0, 1))
    z = int(np.argmax(counts))
    ax.imshow(image[:, :, z].T, cmap="gray", vmin=0, vmax=1,
              interpolation="bilinear")
    overlay = np.ma.masked_where(~gt[:, :, z].T, np.ones_like(gt[:, :, z].T))
    ax.imshow(overlay, cmap=ListedColormap([RED]), vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(INK2); s.set_linewidth(0.6)
    ax.set_title("(a) one axial slice", loc="left", fontsize=7.5, color=INK)
    ax.text(0.03, 0.03, f"vessel: {100 * gt.mean():.2f}% of the volume",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.5, color="white")


def panel_tree(ax, gt):
    """左右两棵树分别上色 —— 「两棵独立的树」是这一格要说的事。"""
    main, _ = r3.split_components(gt, keep=2)
    from scipy import ndimage
    lab, _ = ndimage.label(main)
    sizes = [(lab == i).sum() for i in (1, 2)]
    order = np.argsort(sizes)[::-1] + 1
    r3.add_mask(ax, lab == order[0], BLUE, step=1)
    r3.add_mask(ax, lab == order[1], ORANGE, step=1)
    centre, half = r3.bbox_frame(gt)
    r3.frame(ax, gt.shape, ELEV, AZIM, zoom=1.32, centre=centre, half=half)
    ax.set_title("(b) the annotated tree", loc="left", fontsize=7.5, color=INK)
    ax.text2D(0.5, 0.02, "two disjoint trees", transform=ax.transAxes,
              ha="center", va="bottom", fontsize=6.5, color=INK2)


def panel_branch(ax, gt, stats):
    branch = stats["mask"]
    rest = gt & ~branch
    r3.add_mask(ax, rest, GREY, step=1)
    r3.add_mask(ax, branch, ORANGE, step=1)
    centre, half = r3.bbox_frame(branch, pad=26)
    r3.frame(ax, gt.shape, ELEV, AZIM, centre=centre, half=half)
    ax.set_title("(c) one distal branch", loc="left", fontsize=7.5, color=INK)
    txt = (f"{100 * stats['frac_vessel']:.1f}% of the vessel, "
           f"{100 * stats['frac_volume']:.3f}% of the volume\n"
           f"remove it: Dice {stats['dice']:.3f}, "
           f"clDice {stats['cldice']:.3f}")
    ax.text2D(0.5, 0.0, txt, transform=ax.transAxes, ha="center", va="bottom",
              fontsize=6.5, color=INK2, linespacing=1.35)


def build():
    apply_style()
    gt = vol.load(CASE, "gt")
    image = vol.load(CASE, "image")
    owner, terminal = vol.branch_partition(gt)
    seg = vol.pick_terminal_branch(gt, owner, terminal, target_frac=0.02)
    stats = vol.branch_stats(gt, owner, seg)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.5))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2, projection="3d")
    ax_c = fig.add_subplot(1, 3, 3, projection="3d")
    panel_slice(ax_a, image, gt)
    panel_tree(ax_b, gt)
    panel_branch(ax_c, gt, stats)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06,
                        wspace=0.05)
    return fig, stats
