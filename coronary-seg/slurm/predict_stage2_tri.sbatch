#!/bin/bash --login
# ============================================================
# SLURM SLURM — stage-2（三正交起点）全量测试集评估
#
# 这一步才能回答「换三正交起点到底值不值」。对比对象有两个：
#   1. 同为 stage-2、但起点是单轴的 EXP-012：
#      Dice 0.8117 / clDice 0.8863 / Betti0 1.79 / HD95 22.31
#   2. 不做精修的三正交本身：
#      Dice 0.8012 / clDice 0.8733 / Betti0 2.24 / HD95 21.20
# 只有同时超过这两个，换起点才算成功。
#
# 注意：本作业打印的「stage1」列是 npz 里的概率（此处即三正交），
# 所以它的基线会是三正交的数字，而不是单轴的 24.98。
#
# !! 前置依赖 !!  先跑完 stage2_prep_tri（test split）与 train_stage2_tri
#   sbatch --dependency=afterok:<train_jobid> slurm/predict_stage2_tri.sbatch
# ============================================================
#SBATCH --job-name=pred_s2_tri
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/stage2_tri/pred_%j.out
#SBATCH --error=runs/stage2_tri/pred_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
DATA_ROOT=/path/to/cache/stage2_tri/test

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.
mkdir -p runs/stage2_tri

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

N_NPZ=$(ls "$DATA_ROOT"/*.npz 2>/dev/null | wc -l)
N_EXPECT=$(python -c "import json;print(len(json.load(open('splits/split.json'))['test']))")
echo "test npz: $N_NPZ / 期望 $N_EXPECT"
if [ "$N_NPZ" -ne "$N_EXPECT" ]; then
    echo "错误：npz 数量不符（$N_NPZ != $N_EXPECT）。"
    echo "  先跑 slurm/stage2_prep_tri.sbatch --splits test"
    echo "  并用 --dependency=afterok: 串好依赖"
    exit 1
fi

# 后处理参数与所有既有全量评估对齐，否则横向对比无效
python -u scripts/predict_stage2.py \
    --data-root "$DATA_ROOT" \
    --ckpt runs/stage2_tri/best.pth \
    --init-filters 16 \
    --thr 0.50 \
    --min-voxels 300 \
    --max-gap 0 \
    --roi 128 \
    --overlap 0.5 \
    --sw-batch 2 \
    --out-csv runs/stage2_tri/test_metrics_stage2_tri.csv \
    --max-cases 0

echo "作业 $SLURM_JOB_ID 结束"
