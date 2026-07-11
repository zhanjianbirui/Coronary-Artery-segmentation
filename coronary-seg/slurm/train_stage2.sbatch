#!/bin/bash --login
# ============================================================
# CSF3 SLURM - stage-2 3D 残差门控精修训练
# 提交: sbatch slurm/train_stage2.sbatch
# 续训: sbatch slurm/train_stage2.sbatch --resume
# ============================================================
#SBATCH --job-name=coronary_s2
#SBATCH --account=gpu-fse-ugpgt01
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/stage2/slurm_%j.out

# ---- 环境 ----
module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 路径 ----
PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
DATA_DIR=~/scratch/cache/stage2        # stage2 的 train/ val/ npz

cd "$PROJECT_DIR"
export PYTHONPATH=.

mkdir -p runs/stage2

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 训练 ----
python scripts/train_stage2.py \
    --data-dir "$DATA_DIR" \
    --batch-size 4 \
    --epochs 80 \
    --lr 3e-4 \
    --w-cldice 0.5 \
    --cldice-warmup 500 \
    --cldice-k 5 \
    --samples-per-case 16 \
    --num-workers $SLURM_CPUS_PER_TASK \
    --out-dir runs/stage2 \
    "$@"

echo "作业 $SLURM_JOB_ID 结束"