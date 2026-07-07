#!/bin/bash --login
# ============================================================
# CSF3 SLURM 训练作业 - ImageCAS 冠脉分割 (2.5D)
# 提交:     sbatch slurm/train_2p5d.sbatch
# 续训:     sbatch slurm/train_2p5d.sbatch --resume
#          (被4天上限杀掉后，从 last.pth 继续)
# ============================================================
#SBATCH --job-name=coronary25d
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

# 防止线程超订
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 路径 (按实际情况改) ----
PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
CACHE_DIR=~/scratch/cache/preproc      # 预处理磁盘缓存 (放 scratch)

cd "$PROJECT_DIR"

# 关键: 让 from src.xxx import 能找到 src/ 包
export PYTHONPATH=.

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 训练 ----
# "$@" 透传命令行参数 (如 --resume 等)
python scripts/train.py \
    --split-json splits/split.json \
    --cache-dir "$CACHE_DIR" \
    --k 2 \
    --crop-size 384 \
    --batch-size 32 \
    --epochs 70 \
    --lr 3e-4 \
    --grad-clip 1.0 \
    --backbone segresnet \
    --num-workers $SLURM_CPUS_PER_TASK \
    --pin-memory \
    --out-dir runs/exp_tri2p5d \
    "$@"

echo "作业 $SLURM_JOB_ID 结束"
