#!/usr/bin/env bash
# 导出论文定性图（Fig.1/Fig.5）所需的 nii.gz —— 集群上跑
# 固化为脚本的理由见 .kb/decisions.md DEC-013：长命令行在终端折行会丢参数，
# 2026-08-27 一天内因此失败五次。
#
# 用法： ./scripts/figs/export_fig_assets.sh 702 [更多病例号...]
set -euo pipefail

CACHE=/path/to/cache
TRI_ROOT="$CACHE/stage2_tri/test"     # 三正交 prep：提供 image / label / 三正交 prob
SA_ROOT="$CACHE/stage2/test"          # 单轴 prep：Fig.5 的单轴列
CKPT=runs/stage2_tri_nogate/best.pth  # 最优方案，无门控（不是 stage2/ 那个旧的）
OUT_DIR=vis_nii

if [ $# -eq 0 ]; then
    echo "用法: $0 <case-id> [case-id ...]" >&2
    exit 1
fi

for d in "$TRI_ROOT" "$SA_ROOT"; do
    [ -d "$d" ] || { echo "缺目录: $d" >&2; exit 1; }
done
[ -f "$CKPT" ] || { echo "缺权重: $CKPT" >&2; exit 1; }

PYTHONPATH=. python scripts/figs/export_nii.py \
    --tri-root "$TRI_ROOT" \
    --sa-root  "$SA_ROOT" \
    --ckpt     "$CKPT" \
    --no-gate \
    --case-ids "$@" \
    --out-dir  "$OUT_DIR"

echo
echo "产物："
ls -lh "$OUT_DIR"
