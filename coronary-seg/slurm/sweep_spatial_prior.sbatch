#!/bin/bash --login
# ============================================================
# SLURM SLURM — 空间先验后处理参数扫描（P1-a）
#
# 目标：找出能压掉 HD95 长尾（诊断出的 12 例 HD95>50）而又不伤
#       Dice/clDice 的 max_dist_mm × n_anchor 组合。
#
# 网格依据 EXP-010（2026-08-26 第一轮扫描）收窄：
#   - anchor 从 1 起步改为从 2 起步：a1 已证实灾难性
#     （recall 0.829->0.528，36/40 例 HD95>50，删掉整条右冠）
#   - anchor 上探到 6：第一轮 a1<a2<a3 单调递增而 a3 正是网格上界，
#     最优很可能在网格之外
#   - dist 收到 8~20mm：第一轮 d>=30mm 时一个分量都不删，
#     真实"远处孤块"距离集中在 10~30mm 内；补 8mm 探 10 以下
# 提交: sbatch slurm/sweep_spatial_prior.sbatch
# ============================================================
#SBATCH --job-name=sweep_sp2
#SBATCH --account=<your-account>
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm_sweep_sp_%j.out
#SBATCH --error=runs/slurm_sweep_sp_%j.err

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=~/scratch/Coronary-Artery-segmentation/coronary-seg
CACHE_DIR=~/scratch/cache/preproc

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.

# ---- 分层子集（不要再用 --max-cases N）----
# 2026-08-26 教训：--max-cases N 取的是**前 N 例**，而长尾病例全排在后面。
# 前 40 例只有 2 例 HD95>50（全量是 12/200）→ 长尾指标结论完全不可信。
#
# 这里用分层子集：全部 12 例长尾（HD95>50） + 12 例普通病例作对照。
# 前者看修复效果，后者看副作用。
#
# !! 注意 !! 该子集**富集了难例**，baseline 绝对值会明显差于全量 200 例，
# 不能拿来当最终指标报。它只用于**给配置排序**，选出最优 d/anchor 后
# 必须再跑全量验证。
TAIL_CASES="741 630 655 931 595 728 447 771 184 415 466 493"
CTRL_CASES="936 761 764 1000 916 5 69 908 790 941 727 203"
CASE_ARGS="--case-ids $TAIL_CASES $CTRL_CASES"

python -u scripts/sweep_spatial_prior.py \
    --cache-dir "$CACHE_DIR" \
    --ckpt runs/exp_tri2p5d/best.pth \
    --tri --fuse mean --thr 0.50 \
    --min-voxels 300 \
    --dist-list 8,10,12,15,20 \
    --anchor-list 2,3,4,5,6 \
    --out-csv runs/exp_tri2p5d/sweep_spatial_prior_r2.csv \
    --batch 1 --max-px-per-batch 500000 \
    --pad-multiple 32 \
    $CASE_ARGS
