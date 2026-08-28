#!/bin/bash --login
# ============================================================
# SLURM — 残差门控机制分析
#
# 检验 DEC-008 的设计假设：门控是否只在 stage-1 出错处激活。
# 这个问题消融答不了 —— 消融只比较指标输赢，不解释门控在做什么。
#
# 用 GPU 是因为要在 128³ patch 上做多次 3D 前向；CPU 也能跑，
# 但 30 例 × 8 patch 会很慢，建议走 GPU 作业。
#
# 提交: sbatch slurm/analysis/analyse_gate.sbatch
# 换模型: sbatch slurm/analysis/analyse_gate.sbatch --ckpt runs/stage2_tri/best.pth \
#            --data-root /path/to/cache/stage2_tri/test \
#            --out-csv runs/stage2_tri/gate_analysis.csv
# ============================================================
#SBATCH --job-name=gate_ana
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1                  # 资源名按站点而定；所报模型用的是 80GB 显存
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=runs/stage2/gate_ana_%j.out
#SBATCH --error=runs/stage2/gate_ana_%j.err

# 激活 Python 环境（按你的站点/个人配置修改这几行）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROJECT_DIR=/path/to/coronary-seg
cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
echo "透传参数: $*"

# 默认分析单轴起点的 stage-2（EXP-012 的权重，已训练完成）。
# 透传参数排最后，可覆盖 --ckpt / --data-root / --out-csv
python -u scripts/analysis/analyse_gate.py \
    --data-root /path/to/cache/stage2/test \
    --ckpt runs/stage2/best.pth \
    --out-csv runs/stage2/gate_analysis.csv \
    --max-cases 30 \
    --n-patch 8 \
    --device auto \
    "$@"

echo "作业 $SLURM_JOB_ID 结束"
