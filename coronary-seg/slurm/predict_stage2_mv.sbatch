#!/bin/bash --login
# ============================================================
# SLURM SLURM — stage-2 多视角（EXP-019）全量测试集评估
#
# 与 predict_stage2_tri.sbatch 的三点差别，正是那个脚本对 mv 不适用的原因：
#   1. DATA_ROOT 指向 stage2_mv/test（tri 版写死 stage2_tri/test，
#      对 mv 是「假绿灯」—— 护栏查的数据和实际跑的数据不是同一份）
#   2. 传 --multi-view（输入 4 通道 [image,p0,p1,p2]，而非 2 通道）
#   3. 日志落在 runs/stage2_mv/ 而不是 runs/stage2_tri/
#
# --no-gate 必须带：predict_stage2.py 用 strict load_state_dict，
# 权重里没有门控参数，不带会直接 key mismatch 报错。
#
# 对照组（论文当前最优，results.md）：
#   Dice 0.8216 / clDice 0.8960 / Betti0 1.53 / HD95 19.90
#   runs/stage2_tri_nogate/test_metrics.csv
#
# 用法：
#   sbatch slurm/predict_stage2_mv.sbatch
# ============================================================

#SBATCH --job-name=pred_s2_mv
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/stage2_mv/pred_%j.out
#SBATCH --error=runs/stage2_mv/pred_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
DATA_ROOT=/path/to/cache/stage2_mv/test
CKPT=runs/stage2_mv/best.pth

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.
mkdir -p runs/stage2_mv

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 护栏一：数据。查的就是实际要跑的那份，不再假绿灯
N_NPZ=$(ls "$DATA_ROOT"/*.npz 2>/dev/null | wc -l)
N_EXPECT=$(python -c "import json;print(len(json.load(open('splits/split.json'))['test']))")
echo "test npz: $N_NPZ / 期望 $N_EXPECT  ($DATA_ROOT)"
if [ "$N_NPZ" -ne "$N_EXPECT" ]; then
    echo "错误：npz 数量不符（$N_NPZ != $N_EXPECT）"
    exit 1
fi

# 护栏二：权重。缺了就早死，别跑几小时才发现
if [ ! -f "$CKPT" ]; then
    echo "错误：找不到权重 $CKPT"
    exit 1
fi
python - <<'PY'
import torch
c = torch.load('runs/stage2_mv/best.pth', map_location='cpu', weights_only=False)
print('ckpt: epoch=%d val_dice=%.4f' % (c['epoch'], c['val_dice']))
PY

echo "透传参数: $*"

# 后处理参数与所有既有全量评估对齐（min_voxels=300 / max_gap=0 / thr=0.50），
# 否则与 results.md 里的数字口径不一致，横向对比无效
python -u scripts/predict_stage2.py \
    --data-root "$DATA_ROOT" \
    --ckpt "$CKPT" \
    --multi-view \
    --no-gate \
    --init-filters 16 \
    --thr 0.50 \
    --min-voxels 300 \
    --max-gap 0 \
    --roi 128 \
    --overlap 0.5 \
    --sw-batch 2 \
    --out-csv runs/stage2_mv/test_metrics.csv \
    --max-cases 0 \
    "$@"

echo "作业 $SLURM_JOB_ID 结束"
