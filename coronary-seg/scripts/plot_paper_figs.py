#!/usr/bin/env python3
"""
scripts/plot_paper_figs.py — 生成论文的 matplotlib 图（Fig. 6 / Fig. 7）
==================================================================
图按正文宽度 1:1 生成（\\textwidth = 483.7pt = 6.693 in），LaTeX 里用
width=\\linewidth 插入即不缩放，图上的 8pt 才真是 8pt。

数据源集中在 scripts/figs/data.py，只取 .kb/results.md 点名的那几个 csv。
缺 csv 时会明确告诉你跑 ./scripts/fetch_paper_csv.sh，不会画出半张图。

用法：
  PYTHONPATH=. python scripts/plot_paper_figs.py --out-dir <论文 images 目录>
  PYTHONPATH=. python scripts/plot_paper_figs.py --out-dir images --only fig6
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.figs import data


FIGURES = {
    "fig6": ("fig6_distributions", "Fig. 6 逐病例分布（§3.2）"),
    "fig7": ("fig7_ablation", "Fig. 7 Stage-2 消融（§3.3）"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True, help="PDF 输出目录（论文的 images/）")
    p.add_argument("--only", choices=sorted(FIGURES), action="append",
                   help="只生成指定的图，可重复；默认全部")
    p.add_argument("--png", action="store_true",
                   help="同时导出 PNG，方便快速预览")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    wanted = args.only or sorted(FIGURES)

    ok, skipped = [], []
    for key in wanted:
        modname, desc = FIGURES[key]
        try:
            mod = __import__(f"scripts.figs.{modname}", fromlist=["build"])
            fig = mod.build()
        except data.MissingCsv as e:
            skipped.append((key, str(e)))
            continue

        out = os.path.join(args.out_dir, f"{key}.pdf")
        fig.savefig(out)
        if args.png:
            fig.savefig(out[:-4] + ".png", dpi=200)
        size = os.path.getsize(out) / 1024
        print(f"[完成] {desc}\n         → {out}  ({size:.0f} KB)")
        ok.append(key)

    if skipped:
        print("\n以下图因缺数据未生成：")
        for key, msg in skipped:
            print(f"  {key}: {msg}")
        return 1 if not ok else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
