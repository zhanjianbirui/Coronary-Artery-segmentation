#!/bin/bash --login
# ============================================================
# SLURM SLURM — 三正交融合全量推理（用当前 best.pth 重跑）
#
# 为什么要重跑：旧的全量结果
#   runs/exp_tri2p5d/test_metrics_tri_mean050.csv
#   （Dice 0.8012 / clDice 0.8733 / Betti0 2.24 / HD95 21.20）
# 是用 epoch=36 val_dice=0.8106 那版权重跑的。之后训练继续，
# best.pth 已是 epoch=59 val_dice=0.8135。
# 所有横向对比都以三正交为基准，基准本身的口径必须是最新的。
#
# 【已于 2026-08-25 跑完，结论见 EXP-014】结果比预期重要得多：
#   v2: Dice 0.8098 / clDice 0.8762 / Betti0 2.46 / HD95 20.53
#   v2 显著优于 v1（Dice p=8e-20，167/200 例更优；HD95 p=4e-02）
# 光换这一个 checkpoint 就翻转了两条已写进文档的结论：
#   - Stage-2 vs 三正交的 Dice：p=5e-11「显著」→ p=0.051 不显著
#   - 三正交 vs 单轴+TTA 的 Dice：p=0.015「显著变差」→ p=4e-08「显著变好」
# 教训：baseline 必须用它自己的最优权重，且更新后要系统性重跑所有相关比较。
#
# !! 必须用新的 out-csv !!
#   predict_tri.py 的全量模式会读已有 csv 跳过已完成病例。
#   若沿用旧文件名，200 例会被全部跳过，白跑一趟还以为跑过了。
#   所以这里输出到 _v2.csv。
#
# 提交: sbatch slurm/predict_tri_mean050_v2.sbatch
# ============================================================
#SBATCH --job-name=pred_tri_v2
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/slurm_pred_tri_v2_%j.out
#SBATCH --error=runs/slurm_pred_tri_v2_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
CACHE_DIR=/path/to/cache/preproc
OUT_CSV=runs/exp_tri2p5d/test_metrics_tri_mean050_v2.csv

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "
import torch
ck = torch.load('runs/exp_tri2p5d/best.pth', map_location='cpu')
print('CUDA:', torch.cuda.is_available())
print('本次使用的 checkpoint: epoch=%s val_dice=%s' % (ck.get('epoch'), ck.get('val_dice')))
"

# 参数与首次全量跑完全一致，唯一变量是 checkpoint —— 保证可比
# 空间先验保持关闭（--sp-dist 默认 0），这一版仍是纯三正交基准
python -u scripts/predict_tri.py \
    --cache-dir "$CACHE_DIR" \
    --ckpt runs/exp_tri2p5d/best.pth \
    --fixed-fuse mean \
    --thr 0.50 \
    --min-voxels 300 \
    --max-gap 0 \
    --out-csv "$OUT_CSV" \
    --max-cases 0 \
    --batch 1 \
    --max-px-per-batch 500000 \
    --pad-multiple 32

echo "作业 $SLURM_JOB_ID 结束"
