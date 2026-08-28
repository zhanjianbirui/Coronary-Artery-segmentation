#!/bin/bash --login
# ============================================================
# SLURM 训练作业 - ImageCAS 冠脉分割
# 提交: sbatch slurm/train/train.sbatch
# 续训: sbatch slurm/train/train.sbatch --resume   (作业超时被中断后)
# ============================================================
#SBATCH --job-name=coronary
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

# 防止线程超订 (CPU 核数对齐 --cpus-per-task)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 路径 (按你的实际情况改这一行) ----
PROJECT_DIR=/path/to/coronary-seg
DATA_ROOT=/path/to/ImageCAS          # prepare_data.py 下载/解压后的根目录
cd "$PROJECT_DIR"

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 训练 ----
# $@ 把命令行参数透传进去, 这样 sbatch slurm/train/train.sbatch --resume 能续训
python scripts/train/train.py \
    --config configs/default.yaml \
    "$@" \
    data.data_root="$DATA_ROOT" \
    data.num_workers=$SLURM_CPUS_PER_TASK \
    output.work_dir=runs/exp_segresnet

echo "作业 $SLURM_JOB_ID 结束"
