#!/bin/bash --login
# ============================================================
# SLURM SLURM — 空间先验后处理参数扫描（P1-a）
#
# 目标：找出能压掉 HD95 长尾（诊断出的 12 例 HD95>50）而又不伤
#       Dice/clDice 的 max_dist_mm × n_anchor 组合。
#
# 建议分两步跑：
#   1) 先只跑 5 个难例确认方向对不对（把 CASE_IDS 打开）
#   2) 再跑 40 例子集看有没有副作用
# 提交: sbatch slurm/sweep_spatial_prior.sbatch
# ============================================================
#SBATCH --job-name=sweep_sp
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_sweep_sp_%j.out
#SBATCH --error=runs/slurm_sweep_sp_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
CACHE_DIR=~/scratch/cache/preproc

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.

# 第一步用这行：只跑诊断出的 5 个难例，最快确认方向
# CASE_ARGS="--case-ids 931 630 595 741 728"
# 第二步用这行：40 例子集，看整体有没有副作用
CASE_ARGS="--max-cases 40"

python -u scripts/sweep_spatial_prior.py \
    --cache-dir "$CACHE_DIR" \
    --ckpt runs/exp_tri2p5d/best.pth \
    --tri --fuse mean --thr 0.50 \
    --min-voxels 300 \
    --dist-list 10,15,20,30,40,60 \
    --anchor-list 1,2,3 \
    --out-csv runs/exp_tri2p5d/sweep_spatial_prior.csv \
    --batch 1 --max-px-per-batch 500000 \
    --pad-multiple 32 \
    $CASE_ARGS
