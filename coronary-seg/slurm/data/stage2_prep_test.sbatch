#!/bin/bash --login
#SBATCH --job-name=s2_prep_test
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1                  # 资源名按站点而定；所报模型用的是 80GB 显存
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=runs/stage2/prep_test_%j.out

# 激活 Python 环境（按你的站点/个人配置修改这几行）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coronary
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /path/to/coronary-seg
export PYTHONPATH=.

echo "生成 stage-2 test 数据 (单轴, ckpt=exp_2p5d)"
python scripts/data/stage2_prepare.py \
    --splits test \
    --split-json splits/split.json \
    --cache-dir /path/to/cache/preproc \
    --ckpt runs/exp_2p5d/best.pth \
    --out-dir /path/to/cache/stage2 \
    --k 2 --spacing 0.5 --hu-min -200 --hu-max 800 \
    --backbone segresnet --init-filters 32 --pad-multiple 32
echo "完成"