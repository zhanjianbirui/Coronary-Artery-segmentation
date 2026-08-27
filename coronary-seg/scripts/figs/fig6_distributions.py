"""Fig. 6 — 逐病例分布（论文 §3.2）

论证任务：一张图说清「为什么均值不够、必须做配对检验」。
三个 panel 横向排列，整体按正文宽度 1:1 生成，LaTeX 里 width=\\linewidth 不缩放。

(a) 配对差值：三正交 − 单轴+TTA 的 ΔDice，按差值排序
(b) 最优方案 HD95 的排序曲线：均值被右尾拖离中位数
(c) 连通分量数分布：金标准 vs 三正交 vs 最优方案
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from . import data
from .style import (apply_style, despine, TEXTWIDTH_IN,
                    BLUE, ORANGE, RED, GREY, INK, INK2)

# 正文 tab:sig_fusion 报的是 Holm 校正后的 p（4 项校正）。
# 图上必须与表一致，否则读者会以为是两个不同的检验。
P_HOLM_DICE = r"$p_{\mathrm{Holm}} = 4.2 \times 10^{-8}$"


def panel_paired(ax):
    ids, tri, tta = data.paired("tri", "single_tta", "dice")
    d = np.sort(np.array(tri) - np.array(tta))
    x = np.arange(len(d))
    win, loss = int((d > 0).sum()), int((d < 0).sum())

    ax.bar(x[d <= 0], d[d <= 0], width=1.0, color=RED, linewidth=0)
    ax.bar(x[d > 0], d[d > 0], width=1.0, color=BLUE, linewidth=0)
    ax.axhline(0, color=INK2, linewidth=0.6)

    # 直接标注，不用图例 —— 两种颜色编码的是同一个量的正负，不是两个序列
    ax.text(0.96, 0.90, f"improved\n{win} cases", transform=ax.transAxes,
            ha="right", va="top", color=BLUE, fontsize=6.5, linespacing=1.25)
    # 放在零线以下、红色柱右侧的空白区，避免压在柱子上看不清
    ax.text(0.42, 0.19, f"worsened\n{loss} cases", transform=ax.transAxes,
            ha="left", va="top", color=RED, fontsize=6.5, linespacing=1.25)
    # 左上角：左半边是向下的红柱，这块是空的
    ax.text(0.03, 0.96, P_HOLM_DICE, transform=ax.transAxes,
            ha="left", va="top", fontsize=6.5, color=INK2)

    ax.set_xlabel("200 cases, sorted")
    ax.set_ylabel("$\\Delta$Dice")
    ax.set_xlim(-2, len(d) + 1)
    ax.set_ylim(d.min() * 1.35, d.max() * 1.25)
    ax.set_xticks([])
    ax.set_title("(a) tri-axial $-$ single-axis + TTA", loc="left",
                 color=INK, fontsize=7.5)
    despine(ax, keep=("left",))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def panel_hd95(ax):
    h = np.array(sorted(data.load("final", "hd95").values()))
    x = np.arange(1, len(h) + 1)
    mean, med = h.mean(), np.median(h)
    pct_of_mean = 100 * (h < mean).sum() / len(h)
    trimmed = h[:-5].mean()
    n_tail = int((h > 50).sum())

    ax.plot(x, h, color=BLUE, linewidth=1.4, solid_capstyle="round")
    # 右尾：>50mm 的病例
    tail = h > 50
    ax.fill_between(x, 0, h, where=tail, color=ORANGE, alpha=0.35, linewidth=0)

    ax.axhline(med, color=INK2, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.axhline(mean, color=RED, linewidth=0.8, linestyle=(0, (4, 2)))
    ax.text(4, med, f"median {med:.1f}", va="top", ha="left",
            fontsize=6.5, color=INK2)
    ax.text(4, mean, f"mean {mean:.1f} ({pct_of_mean:.0f}th pct)",
            va="bottom", ha="left", fontsize=6.5, color=RED)
    ax.annotate(f"{n_tail} cases $>$ 50 mm\ndrop worst 5: mean {trimmed:.1f}",
                xy=(len(h) - 10, h[-10]), xytext=(0.04, 0.80),
                textcoords="axes fraction",
                fontsize=6.5, color=INK2, linespacing=1.25, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6,
                                shrinkA=2, shrinkB=2))

    ax.set_xlabel("200 cases, sorted")
    ax.set_ylabel("HD95 (mm)")
    ax.set_xlim(0, len(h) + 2)
    ax.set_ylim(0, h.max() * 1.08)
    ax.set_title("(b) HD95, final method", loc="left", color=INK, fontsize=7.5)
    despine(ax, keep=("left", "bottom"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def panel_components(ax):
    gt = data.load("final_input", "n_gt")
    tri = data.load("final_input", "n_pred")
    fin = data.load("final", "n_pred")
    hi = int(max(max(gt.values()), max(tri.values()), max(fin.values())))
    bins = np.arange(1, hi + 2)

    def counts(d):
        return np.array([sum(1 for v in d.values() if int(v) == b) for b in bins])

    c_gt, c_tri, c_fin = counts(gt), counts(tri), counts(fin)
    w = 0.38
    # 金标准是参照，不是并列的第三个序列 —— 用中性色的阶梯轮廓画在底层
    ax.bar(bins, c_gt, width=0.86, color="none", edgecolor=GREY,
           linewidth=0.9, label="ground truth", zorder=1)
    ax.bar(bins - w / 2, c_tri, width=w, color=BLUE, linewidth=0,
           label="tri-axial", zorder=2)
    ax.bar(bins + w / 2, c_fin, width=w, color=ORANGE, linewidth=0,
           label="final method", zorder=2)

    ax.set_xlabel("connected components")
    ax.set_ylabel("number of cases")
    ax.set_xticks(bins[::2])
    ax.set_xlim(bins[0] - 0.8, bins[-1] + 0.8)
    ax.set_ylim(0, max(c_gt.max(), c_tri.max(), c_fin.max()) * 1.30)
    ax.set_title("(c) components per case", loc="left", color=INK, fontsize=7.5)
    ax.legend(loc="upper right", handlelength=1.0, borderaxespad=0.1,
              labelspacing=0.3, fontsize=6.5)
    despine(ax, keep=("left", "bottom"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def build():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.35),
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.06)
    panel_paired(axes[0])
    panel_hd95(axes[1])
    panel_components(axes[2])
    return fig
