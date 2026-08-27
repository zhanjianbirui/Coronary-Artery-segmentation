"""定性对比图（论文 §3.2）—— 把 Betti-0 从 8 降到 0 变成看得见的事实。

四列共用**同一个相机与同一个包围盒**（取自金标准），这是程序化渲染相对
手工渲染的关键优势：手工对齐视角/缩放最容易出错，而不同视角的四张图不可比。

配色只有两种，语义明确：
  蓝  = 最大的两个连通域 —— 金标准就是左右两棵冠脉树
  红  = 多出来的碎块 —— 正是 Betti-0 误差的来源
这样「碎成 11 块 → 2 块」不需要读数字就能看出来。
"""

import matplotlib.pyplot as plt

from . import render3d as r3
from . import volumes as vol
from .style import apply_style, TEXTWIDTH_IN, BLUE, RED, INK, INK2

CASE = "702"
ELEV, AZIM = 35, -120


def build():
    apply_style()
    gt = vol.load(CASE, "gt")
    centre, half = r3.bbox_frame(gt)

    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 2.15),
                             subplot_kw={"projection": "3d"})
    for ax, (key, name) in zip(axes, vol.COLUMNS):
        mask = gt if key == "gt" else vol.load(CASE, key)
        n = vol.n_components(mask)
        main, extra = r3.split_components(mask, keep=2)
        r3.add_mask(ax, main, BLUE, step=1)
        r3.add_mask(ax, extra, RED, step=1)
        r3.frame(ax, gt.shape, ELEV, AZIM, zoom=1.32, centre=centre, half=half)

        err = "" if key == "gt" else f"   Betti-0 err {abs(n - 2)}"
        ax.set_title(name, fontsize=7.5, color=INK, pad=-2)
        ax.text2D(0.5, 0.02, f"{n} components{err}", transform=ax.transAxes,
                  ha="center", va="bottom", fontsize=6.5, color=INK2)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.98, bottom=0.02,
                        wspace=0.0)
    return fig
