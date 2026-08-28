#!/bin/bash --login
# ============================================================
# SLURM SLURM — Stage-2 3D 精修推理 + 全量测试集评估
#
# 这是当前最大的空白：Stage-2 已训完（best val_dice=0.8167 @ ep18），
# 但那是 3D patch 级指标，与 Stage-1 的 slice 级 0.8121 **不是一个口径**，
# 不能相比。只有跑完这个作业拿到 200 例的 Dice/clDice/Betti0/HD95，
# 才能判断 Stage-2 到底有没有增益。
#
# !! 前置依赖 !!
#   必须先跑 slurm/data/stage2_prep_test.sbatch 生成 test 集 npz。
#   检查方法（login 节点）：
#     ls /path/to/cache/stage2/test/*.npz | wc -l   # 应为 200
#   若为 0 或目录不存在，先 sbatch slurm/data/stage2_prep_test.sbatch
#
# 提交: sbatch slurm/predict/predict_stage2.sbatch
# ============================================================
#SBATCH --job-name=pred_s2
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/stage2/pred_s2_%j.out
#SBATCH --error=runs/stage2/pred_s2_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
# 注意是 <out-dir>/test —— predict_stage2.py 直接 glob <data_root>/*.npz，
# 不会自己往 split 子目录里走
DATA_ROOT=/path/to/cache/stage2/test

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.

mkdir -p runs/stage2

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 护栏：npz 数量必须等于 split 里的 test 病例数。
# 只判「非零」是不够的 —— 2026-08-26 就是因为 prep 还没写完，
# 本作业抢跑拿到 16/200 个 npz，白跑一遍还得出了不能用的结论（BUG-006）。
N_NPZ=$(ls "$DATA_ROOT"/*.npz 2>/dev/null | wc -l)
N_EXPECT=$(python -c "import json;print(len(json.load(open('splits/split.json'))['test']))")
echo "test npz: $N_NPZ / 期望 $N_EXPECT"
if [ "$N_NPZ" -ne "$N_EXPECT" ]; then
    echo "错误：npz 数量不符（$N_NPZ != $N_EXPECT）。"
    echo "  若为 0：先跑 slurm/data/stage2_prep_test.sbatch"
    echo "  若非 0：prep 尚未完成或中途失败。提交时请带依赖："
    echo "    sbatch --dependency=afterok:<prep_jobid> slurm/predict/predict_stage2.sbatch"
    exit 1
fi

# 后处理参数与 Stage-1 全量评估保持一致（min_voxels=300, max_gap=0），
# 否则两阶段的对比不同口径，结论无效
python -u scripts/predict/predict_stage2.py \
    --data-root "$DATA_ROOT" \
    --ckpt runs/stage2/best.pth \
    --init-filters 16 \
    --thr 0.50 \
    --min-voxels 300 \
    --max-gap 0 \
    --roi 128 \
    --overlap 0.5 \
    --sw-batch 2 \
    --out-csv runs/stage2/test_metrics_stage2.csv \
    --max-cases 0

echo "作业 $SLURM_JOB_ID 结束"
