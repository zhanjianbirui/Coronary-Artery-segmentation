#!/bin/bash --login
# ============================================================
# SLURM SLURM — 多视角 stage-2 训练（4 通道 [image, p0, p1, p2]）
#
# 唯一变量：输入是三个方向的概率，而非融合后的单一概率。
# 残差基准仍取三者平均，与 2 通道方案起点完全相同 —— 因此
# 「是否保留方向分歧」是本实验的唯一自变量。
#
# 对照组：runs/stage2_tri_nogate（当前最优，2 通道）
#   Dice 0.8216 / clDice 0.8960 / Betti-0 1.53 / HD95 19.90
# 成功判据：四项中至少两项显著改善，且无显著变差。
#
# 注意：本实验**不含门控**（DEC-008 已被 EXP-015/017 推翻）。
#
# !! 前置 !! 先跑完 stage2_prep_mv.sbatch 的 train 与 val
#   sbatch --dependency=afterok:<prep_jobid> slurm/train_stage2_mv.sbatch
# ============================================================
#SBATCH --job-name=s2_mv
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/stage2_mv/slurm_%j.out
#SBATCH --error=runs/stage2_mv/slurm_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
DATA_DIR=/path/to/cache/stage2_mv
cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.
mkdir -p runs/stage2_mv

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
echo "透传参数: $*"

# 护栏：npz 数量必须与 split.json 一致（BUG-006 的教训）
for sp in train val; do
    n=$(ls "$DATA_DIR/$sp"/*.npz 2>/dev/null | wc -l)
    want=$(python -c "import json;print(len(json.load(open('splits/split.json'))['$sp']))")
    echo "  $sp npz: $n / 期望 $want"
    if [ "$n" -ne "$want" ]; then
        echo "错误：$sp 数据不完整（$n != $want）"
        exit 1
    fi
done

# 护栏：npz 必须真的含 p0/p1/p2，否则训练会在第一个 batch 才报错
python - <<'PY'
import glob, numpy as np, sys
f = sorted(glob.glob("/path/to/cache/stage2_mv/train/*.npz"))[0]
keys = set(np.load(f).files)
need = {"p0", "p1", "p2", "image", "label", "prob"}
missing = need - keys
if missing:
    print(f"错误：{f} 缺少 {missing}，实际 keys={sorted(keys)}")
    print("  数据需用 stage2_prepare.py --save-views 生成")
    sys.exit(1)
print(f"  npz 键检查通过: {sorted(keys)}")
PY
[ $? -ne 0 ] && exit 1

# 超参与 train_stage2_tri.sbatch 完全一致，只加 --multi-view --no-gate
python -u scripts/train_stage2.py \
    --data-dir "$DATA_DIR" \
    --batch-size 4 \
    --epochs 30 \
    --lr 3e-4 \
    --w-cldice 0.5 \
    --cldice-warmup 500 \
    --cldice-k 5 \
    --samples-per-case 16 \
    --num-workers $SLURM_CPUS_PER_TASK \
    --multi-view \
    --no-gate \
    --out-dir runs/stage2_mv \
    "$@"

echo "作业 $SLURM_JOB_ID 结束"
