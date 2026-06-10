#!/bin/bash --login
# ============================================================
# CSF3 SLURM 训练作业 - ImageCAS 冠脉分割
# 提交: sbatch slurm/train.sbatch
# 续训: sbatch slurm/train.sbatch --resume   (被4天上限杀掉后)
# ============================================================
#SBATCH --job-name=coronary
#SBATCH --account=gpu-fse-ugpgt01
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/slurm_%j.out

# ---- 环境 ----
module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

# 防止线程超订 (CPU 核数对齐 --cpus-per-task)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 路径 (按你的实际情况改这一行) ----
PROJECT_DIR=~/scratch/Coronary-Artery-segmentation
DATA_ROOT=~/scratch/ImageCAS          # prepare_data.py 下载/解压后的根目录
cd "$PROJECT_DIR"

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 训练 ----
# $@ 把命令行参数透传进去, 这样 sbatch slurm/train.sbatch --resume 能续训
python scripts/train.py \
    --config configs/default.yaml \
    "$@" \
    data.data_root="$DATA_ROOT" \
    data.num_workers=$SLURM_CPUS_PER_TASK \
    output.work_dir=runs/exp_a100

echo "作业 $SLURM_JOB_ID 结束"
