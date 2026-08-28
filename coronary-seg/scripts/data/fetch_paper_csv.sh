#!/usr/bin/env bash
# 从 SLURM 拉回论文图表所需的逐病例指标 csv
# 固化为脚本的理由见 .kb/decisions.md DEC-013（终端折行会吃掉命令参数）
set -euo pipefail

REMOTE=user@cluster.example.org
RROOT=/path/to/coronary-seg
LROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 每行一个待拉文件，路径相对于仓库根，本地按同样结构落盘。
# ⚠️ 改这里之前先读 .kb/results.md 的「复现所需的确切配置」—— 本项目
#    已因对照组用错 checkpoint 而发生过多次结论反转。
FILES=(
    # --- Fig. 6：三正交基线 vs 最优方案的逐病例分布 ---
    # 必须是 _v2；无后缀的那份是 epoch=36 的次优权重，results.md 明令不要用
    runs/exp_tri2p5d/test_metrics_tri_mean050_v2.csv
    runs/stage2_tri_nogate/test_metrics.csv

    # --- Fig. 7：Stage-2 三组消融（full / no-gate / no-clDice）---
    # no-gate 那组复用上面的 stage2_tri_nogate
    runs/stage2_tri/test_metrics_stage2_tri.csv
    runs/stage2_tri_nocldice/test_metrics.csv
)

for rel in "${FILES[@]}"; do
    dst="$LROOT/$rel"
    mkdir -p "$(dirname "$dst")"
    echo "拉取 $rel"
    scp -q "$REMOTE:$RROOT/$rel" "$dst"
done

echo
echo "完成。行数应均为 201（200 例 + 表头）："
for rel in "${FILES[@]}"; do
    printf '  %-52s %s\n' "$rel" "$(wc -l < "$LROOT/$rel")"
done
