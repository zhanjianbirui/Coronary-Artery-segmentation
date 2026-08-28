#!/bin/bash --login
# ============================================================
# SLURM 推理作业 - 2.5D 全量测试集 + TTA
# 提交: sbatch slurm/predict/predict_tta.sbatch
# ============================================================
#SBATCH --job-name=coro_predict
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1                  # 资源名按站点而定；所报模型用的是 80GB 显存
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=runs/predict_%j.out

# ---- 环境 ----
# 激活 Python 环境（按你的站点/个人配置修改这几行）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coronary
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- 路径 ----
PROJECT_DIR=/path/to/coronary-seg
CACHE_DIR=/path/to/cache/preproc

cd "$PROJECT_DIR"
export PYTHONPATH=.

echo "推理作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# ---- 全量测试集推理：最优配置 + TTA ----
python scripts/predict/predict.py \
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
