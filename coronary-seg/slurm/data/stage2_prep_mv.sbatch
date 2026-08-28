#!/bin/bash --login
# ============================================================
# SLURM SLURM — 多视角 stage-2 数据生成（保留三方向概率）
#
# 动机（EXP-018）：mean 融合丢弃了方向间分歧，而该分歧携带错误位置的
# 信息（AUC 0.64，最高/最低分歧桶错误率相差 3.6 倍）。但用人工阈值规则
# 利用它失败了 —— 最优参数符号随子集反转，因为细分支与大块假阳都是
# 高分歧，单凭标量无法区分。结论是：把三个概率体交给网络自己学融合。
#
# 本作业生成的 npz 同时含 prob(融合) 与 p0/p1/p2(三方向)，因此
# **同一份数据既能训练 4 通道模型，也能复现既有 2 通道结果**，口径一致。
#
# !! 磁盘 !! npz 体积约为单概率版的 3 倍，作业开头有空间检查。
#
# 提交: sbatch slurm/data/stage2_prep_mv.sbatch --splits val,test
#       sbatch slurm/data/stage2_prep_mv.sbatch --splits train
# ============================================================
#SBATCH --job-name=s2_prep_mv
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/stage2_mv/prep_%j.out
#SBATCH --error=runs/stage2_mv/prep_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
OUT_DIR=/path/to/cache/stage2_mv
cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.
mkdir -p runs/stage2_mv "$OUT_DIR"

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
echo "透传参数: $*"

# ---- 磁盘护栏 ----
# 现有 stage2_tri 的体积 × 3 是本次的粗略需求。空间不足时早点失败，
# 而不是跑到一半写盘出错（那会留下一堆半截数据）。
EXIST_DIR=/path/to/cache/stage2_tri
if [ -d "$EXIST_DIR" ]; then
    used_kb=$(du -sk "$EXIST_DIR" 2>/dev/null | cut -f1)
    need_kb=$((used_kb * 3))
    avail_kb=$(df -Pk /path/to/scratch | tail -1 | awk '{print $4}')
    echo "  现有 stage2_tri: $((used_kb/1024/1024)) GB"
    echo "  本次预计需要:    $((need_kb/1024/1024)) GB"
    echo "  可用空间:        $((avail_kb/1024/1024)) GB"
    if [ "$avail_kb" -lt "$need_kb" ]; then
        echo "错误：磁盘空间不足，需要约 $((need_kb/1024/1024)) GB"
        echo "  可先删除不再需要的中间结果，或改用只存 mean+std 的方案"
        exit 1
    fi
fi

python -c "import torch;c=torch.load('runs/exp_tri2p5d/best.pth',map_location='cpu',weights_only=False);print('stage-1 ckpt epoch=',c['epoch'],'val_dice=%.4f'%c['val_dice'])"

# 与 stage2_prep_tri.sbatch 完全相同的参数，唯一增加 --save-views
python -u scripts/data/stage2_prepare.py \
    --cache-dir /path/to/cache \
    --ckpt runs/exp_tri2p5d/best.pth \
    --out-dir "$OUT_DIR" \
    --tri --axes 0 1 2 --fuse mean \
    --save-views \
    --batch 16 \
    "$@"

echo "作业 $SLURM_JOB_ID 结束"
