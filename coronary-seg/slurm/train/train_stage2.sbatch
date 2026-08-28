#!/bin/bash --login
# ============================================================
# SLURM - stage-2 3D 残差门控精修训练
# 提交: sbatch slurm/train/train_stage2.sbatch
# 续训: sbatch slurm/train/train_stage2.sbatch --resume
# ============================================================
#SBATCH --job-name=coronary_s2
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1                  # 资源名按站点而定；所报模型用的是 80GB 显存
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/stage2/slurm_%j.out

# ---- 环境 ----
# 激活 Python 环境（按你的站点/个人配置修改这几行）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coronary

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 路径 ----
PROJECT_DIR=/path/to/coronary-seg
DATA_DIR=/path/to/cache/stage2        # stage2 的 train/ val/ npz

cd "$PROJECT_DIR"
export PYTHONPATH=.

mkdir -p runs/stage2

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 训练 ----
python scripts/train/train_stage2.py \
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