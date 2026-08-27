"""Fig. 7 — Stage-2 组件消融（论文 §3.3）

论证任务：证明拓扑改善来自两个设计选择本身，而非「多加了一个网络」。

**为什么不用分组柱状图**（figures.md 最初的提议）：四个指标里 Dice 与 clDice
的组间差只有千分之几，柱状图要么完全看不出差别，要么必须把 y 轴从非零处截断
—— 而截断的柱子是公认的误导性画法（柱长与数值不再成比例）。
点图没有面积语义，非零起点是正当的，因此改用**横向点图**：
每个 panel 一个指标，三行对应三个变体，灰色虚线是 Stage-1 基线。

每个点都标了数值 —— 既是给读者的精确值，也满足青色（#1baf7a）在白纸上
对比度 2.82 (<3:1) 所要求的 relief 规则（必须有可见的直接标签）。
"""

import numpy as np
import matplotlib.pyplot as plt

from . import data
from .style import (apply_style, despine, TEXTWIDTH_IN,
                    BLUE, ORANGE, AQUA, GREY, INK, INK2)

# (数据 key, 显示名) —— 分类槽位按固定顺序取用，不循环
VARIANTS = [
    ("abl_full",     "full (gate + clDice)", BLUE),
    ("final",        "no gate  [final]",     ORANGE),
    ("abl_nocldice", "no clDice",            AQUA),
]

# (metric, 面板标题, 越大越好?, 数值格式)
METRICS = [
    ("dice",       "Dice $\\uparrow$",     True,  "{:.4f}"),
    ("cldice",     "clDice $\\uparrow$",   True,  "{:.4f}"),
    ("betti0_err", "Betti-0 $\\downarrow$", False, "{:.2f}"),
    ("hd95",       "HD95 (mm) $\\downarrow$", False, "{:.2f}"),
]

# Stage-1 基线（三正交 v2）。取自同一批 csv，不写死数字。
BASELINE_KEY = "tri"


def _mean(key, metric):
    v = data.load(key, metric).values()
    return float(np.mean(list(v)))


def panel(ax, metric, title, higher_better, fmt, show_ylabels):
    ys = np.arange(len(VARIANTS))[::-1]          # 第一个变体画在最上面
    vals = [_mean(k, metric) for k, _, _ in VARIANTS]
    base = _mean(BASELINE_KEY, metric)

    ax.axvline(base, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)

    for y, val, (_, name, colour) in zip(ys, vals, VARIANTS):
        ax.plot([base, val], [y, y], color=colour, linewidth=1.2,
                alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot([val], [y], marker="o", markersize=5.5, color=colour,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)

    lo = min(vals + [base])
    hi = max(vals + [base])
    pad = (hi - lo) * 0.55 if hi > lo else abs(hi) * 0.05
    ax.set_xlim(lo - pad, hi + pad * 1.15)

    # 数值直接标在点旁，朝远离基线的一侧，避免压住虚线
    for y, val in zip(ys, vals):
        right = val >= base
        ax.annotate(fmt.format(val), (val, y),
                    xytext=(6 if right else -6, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center",
                    fontsize=6.5, color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n, _ in VARIANTS] if show_ylabels else [])
    ax.set_ylim(-0.6, len(VARIANTS) - 0.4)
    ax.set_title(title, loc="left", fontsize=7.5, color=INK)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", length=0)
    despine(ax, keep=("bottom",))
    ax.xaxis.grid(False)


def build():
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 1.85),
                             constrained_layout=True)
    for i, (metric, title, hb, fmt) in enumerate(METRICS):
        panel(axes[i], metric, title, hb, fmt, show_ylabels=(i == 0))

    base_line = plt.Line2D([], [], color=GREY, linewidth=0.8,
                           linestyle=(0, (3, 2)), label="Stage-1 baseline (tri-axial)")
    fig.legend(handles=[base_line], loc="lower center", ncol=1,
               fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    return fig
