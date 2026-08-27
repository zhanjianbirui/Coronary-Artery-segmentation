"""论文图表统一样式与配色。

配色取自经校验的分类调色板，并用 dataviz 的 validate_palette.js 在**白纸表面**
上重跑过（论文是印刷品，surface=#ffffff）：

  蓝 #2a78d6 / 橙 #eb6834 / 青 #1baf7a
    → 三色 all-pairs 全部 PASS，最差 CVD ΔE 9.2、常视觉 ΔE 24.0
    → 唯一 WARN：青对白纸对比度 2.82 (<3:1)，按「relief 规则」必须配可见的
      直接数值标签 —— Fig.7 的柱子上都标了数字，满足该要求
  发散两极 蓝 #2a78d6 ↔ 红 #e34948（Fig.6a 的「改善 / 变差」）
    → all-pairs PASS，CVD ΔE 21.6、常视觉 ΔE 32.3、对比度均 ≥3:1

⚠️ **不要用红绿表示改善/变差**（figures.md 原来的提议）。红绿是最典型的
色盲不可分对，发散配色必须是一冷一暖 + 中性灰中点。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 分类槽位，按固定顺序取用，不循环 ---
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"          # 仅用作发散的另一极
GREY = "#6f6e6a"         # 参考序列（金标准）用中性色，不占分类槽位

INK = "#0b0b0b"          # text-primary
INK2 = "#52514e"         # text-secondary
GRID = "#dcdbd6"

# 正文宽度 483.69687pt / 72.27 = 6.693 in，图按最终尺寸生成，
# LaTeX 里用 width=\linewidth 即 1:1 不缩放，字号才真是 8pt。
TEXTWIDTH_IN = 6.693


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,      # 字体嵌入为 TrueType，PDF/A 需要
        "ps.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")):
    """去掉多余边框，让网格与坐标轴退居次要。"""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
