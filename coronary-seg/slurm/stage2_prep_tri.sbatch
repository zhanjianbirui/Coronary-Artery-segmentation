#!/bin/bash --login
# ============================================================
# SLURM SLURM — 用【三正交】阶段1模型生成 stage-2 数据
#
# 动机（EXP-012，数字按 EXP-014 修正）：stage-2 从**单轴**起点（Dice 0.7955）
# 精修到 0.8117，在**拓扑指标**上超过了三正交融合本身
# （三正交 v2: Dice 0.8098 / clDice 0.8762 / Betti0 2.46 / HD95 20.53）——
# clDice p=2.2e-08、Betti-0 p=5.0e-07 显著优，Dice 则打平（p=0.051）。
# 若把起点从 0.7955 换成三正交的 0.8098（且拓扑更好），增益有望叠加，
# 正是把 Dice 也拉开的机会。
#
# 注意：旧文档里三正交是 0.8012（epoch=36 次优权重），用它当基准会高估收益。
#
# 与旧数据的区别只有 prob 那一路：image / label 完全相同。
#
# !! 输出到新目录 stage2_tri !!
#   本脚本靠「文件已存在就跳过」做续跑。若指向旧的 stage2/，
#   会把之前单轴生成的 npz 当成已完成，得到混了两种来源的数据集。
#
# 用法（建议分两批提交，先小后大）：
#   sbatch slurm/stage2_prep_tri.sbatch --splits val,test    # 300 例
#   sbatch slurm/stage2_prep_tri.sbatch --splits train       # 700 例
#
# 三正交推理约为单轴的 3 倍耗时，train 那批很可能一次跑不完 4 天上限。
# 脚本本身可续跑（且已改原子写，不会留下半个损坏文件），
# 被杀后原样再 sbatch 一次即可从断点继续。
#
# 磁盘：每例 npz 约 40~80MB，1000 例合计约 40~80GB。提交前先看配额：
#   df -h /path/to/scratch ; du -sh /path/to/cache/*
# ============================================================
#SBATCH --job-name=s2_prep_tri
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/stage2/prep_tri_%j.out
#SBATCH --error=runs/stage2/prep_tri_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
CACHE_DIR=/path/to/cache/preproc
OUT_DIR=/path/to/cache/stage2_tri     # 新目录，别用 stage2/

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.
mkdir -p runs/stage2

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "剩余磁盘:"; df -h "$(dirname "$OUT_DIR")" | tail -1

# --batch 1 / --max-px-per-batch 500000 与全量三正交推理保持一致：
# 冠状/矢状面切片比轴位大，默认 batch 会爆显存
python -u scripts/stage2_prepare.py \
    --split-json splits/split.json \
    --cache-dir "$CACHE_DIR" \
    --ckpt runs/exp_tri2p5d/best.pth \
    --out-dir "$OUT_DIR" \
    --tri --axes 0 1 2 --fuse mean \
    --k 2 --spacing 0.5 --hu-min -200 --hu-max 800 \
    --backbone segresnet --init-filters 32 --pad-multiple 32 \
    --batch 1 --max-px-per-batch 500000 \
    "$@"

echo "各 split 已生成数量:"
for sp in train val test; do
    n=$(ls "$OUT_DIR/$sp"/*.npz 2>/dev/null | wc -l)
    echo "  $sp: $n"
done
echo "作业 $SLURM_JOB_ID 结束"
