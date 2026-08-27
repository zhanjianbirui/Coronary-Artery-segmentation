#!/usr/bin/env bash
# 从 SLURM 拉回论文 Fig.6/7 所需的逐病例指标 csv
# 固化为脚本的理由见 .kb/decisions.md DEC-013（终端折行会吃掉命令参数）
set -euo pipefail

REMOTE=user@cluster.example.org
RROOT='~/scratch/Coronary-Artery-segmentation/coronary-seg'
LROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$LROOT/runs/exp_tri2p5d" "$LROOT/runs/stage2_tri_nogate"

# Fig.6 对照组：三正交 v2（注意是 _v2，不是本地那份已作废的 epoch36 版）
scp "$REMOTE:$RROOT/runs/exp_tri2p5d/test_metrics_tri_mean050_v2.csv" \
    "$LROOT/runs/exp_tri2p5d/"

# Fig.6/7 实验组：stage-2 无门控（不是含门控的旧 stage2/）
scp "$REMOTE:$RROOT/runs/stage2_tri_nogate/test_metrics.csv" \
    "$LROOT/runs/stage2_tri_nogate/"

echo "完成："
wc -l "$LROOT/runs/exp_tri2p5d/test_metrics_tri_mean050_v2.csv" \
      "$LROOT/runs/stage2_tri_nogate/test_metrics.csv"
