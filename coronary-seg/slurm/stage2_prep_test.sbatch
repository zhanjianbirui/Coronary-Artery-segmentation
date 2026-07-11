#!/bin/bash --login
#SBATCH --job-name=s2_prep_test
#SBATCH --account=gpu-fse-ugpgt01
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=runs/stage2/prep_test_%j.out

module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd ~/scratch/Coronary-Artery-segmentation/coronary-seg
export PYTHONPATH=.

echo "生成 stage-2 test 数据 (单轴, ckpt=exp_2p5d)"
python scripts/stage2_prepare.py \
    --splits test \
    --split-json splits/split.json \
    --cache-dir /net/scratch/z67253xh/cache/preproc \
    --ckpt runs/exp_2p5d/best.pth \
    --out-dir /net/scratch/z67253xh/cache/stage2 \
    --k 2 --spacing 0.5 --hu-min -200 --hu-max 800 \
    --backbone segresnet --init-filters 32 --pad-multiple 32
echo "完成"