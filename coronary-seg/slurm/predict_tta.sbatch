#!/bin/bash --login
# ============================================================
# CSF3 SLURM 推理作业 - 2.5D 全量测试集 + TTA
# 提交: sbatch slurm/predict_tta.sbatch
# ============================================================
#SBATCH --job-name=coro_predict
#SBATCH --account=gpu-fse-ugpgt01
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=runs/predict_%j.out

# ---- 环境 ----
module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 路径 ----
PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
CACHE_DIR=~/scratch/cache/preproc

cd "$PROJECT_DIR"
export PYTHONPATH=.

echo "推理作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 全量测试集推理：最优配置 + TTA ----
python scripts/predict.py \
    --cache-dir "$CACHE_DIR" \
    --ckpt runs/exp_2p5d/best.pth \
    --out-csv runs/exp_2p5d/test_final_tta.csv \
    --max-cases 0 \
    --thr 0.5 \
    --min-voxels 300 \
    --max-gap 0 \
    --pad-multiple 32 \
    --tta

echo "推理作业 $SLURM_JOB_ID 结束"
