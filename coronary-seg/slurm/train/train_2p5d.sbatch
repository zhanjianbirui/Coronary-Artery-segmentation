#!/bin/bash --login
# ============================================================
# SLURM 训练作业 - ImageCAS 冠脉分割 (2.5D)
# 提交:     sbatch slurm/train/train_2p5d.sbatch
# 续训:     sbatch slurm/train/train_2p5d.sbatch --resume
#          (作业超时被中断后，从 last.pth 继续)
# ============================================================
#SBATCH --job-name=coronary25d
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1                  # 资源名按站点而定；所报模型用的是 80GB 显存
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/slurm_%j.out

# ---- 环境 ----
# 激活 Python 环境（按你的站点/个人配置修改这几行）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coronary

# 防止线程超订
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 路径 (按实际情况改) ----
PROJECT_DIR=/path/to/coronary-seg
CACHE_DIR=/path/to/cache/preproc      # 预处理磁盘缓存 (放大容量磁盘)

cd "$PROJECT_DIR"

# 关键: 让 from src.xxx import 能找到 src/ 包
export PYTHONPATH=.

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 训练 ----
# "$@" 透传命令行参数 (如 --resume 等)
python scripts/train/train.py \
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
