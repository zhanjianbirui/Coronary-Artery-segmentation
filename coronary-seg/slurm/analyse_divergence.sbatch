#!/bin/bash --login
# ============================================================
# SLURM SLURM — 三方向分歧评估
#
# 回答两个问题：
#   A. mean 融合丢弃的方向间分歧(std)，是否标记了 stage-1 的出错位置？
#   B. 用分歧做逐体素自适应阈值 thr(x)=0.5+alpha*std(x)，能否改善指标？
#
# 依据：analyse_gate.py 显示 stage-2 门控与预测熵的相关系数仅 -0.065，
# 网络无法自行从融合后的概率推断可靠性。若分歧有判别力，
# 把它显式提供给下游就有实验依据（论文 Future work 的具体提案）。
#
# 注意：本作业**不修改任何既有推理路径**，只是把 predict_tri_probs
# 已算出、随后被 mean 丢弃的信息捡回来分析。
#
# 提交: sbatch slurm/analyse_divergence.sbatch
# 加例: sbatch slurm/analyse_divergence.sbatch --max-cases 40
# ============================================================
#SBATCH --job-name=diverg
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3:00:00
#SBATCH --output=runs/exp_tri2p5d/diverg_%j.out
#SBATCH --error=runs/exp_tri2p5d/diverg_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
echo "透传参数: $*"
python -c "import torch;c=torch.load('runs/exp_tri2p5d/best.pth',map_location='cpu',weights_only=False);print('ckpt epoch=',c['epoch'],'val_dice=%.4f'%c['val_dice'])"

# 与主流程完全同口径：同一 ckpt、同 axes、同 thr、同 min_voxels
python -u scripts/analyse_divergence.py \
    --cache-dir /path/to/cache \
    --ckpt runs/exp_tri2p5d/best.pth \
    --axes 0 1 2 \
    --thr 0.50 \
    --min-voxels 300 \
    --max-cases 20 \
    --out-csv runs/exp_tri2p5d/divergence.csv \
    --device auto \
    "$@"

echo "作业 $SLURM_JOB_ID 结束"
