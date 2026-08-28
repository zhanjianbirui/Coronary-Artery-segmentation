#!/bin/bash --login
# ============================================================
# SLURM — stage-2 精修网络训练（【三正交】起点）
#
# 与 train_stage2.sbatch 的区别：
#   1. --data-dir 指向 stage2_tri（三正交概率），不是 stage2（单轴）
#   2. --epochs 80 -> 30：EXP-007 的训练日志显示 best 出现在 epoch 18，
#      之后 62 个 epoch 里 train_loss 从 0.181 降到 0.121 而 val_dice
#      在 0.79~0.813 震荡，是明确的过拟合。80 epoch 浪费约 3/4 算力
#   3. --out-dir runs/stage2_tri，不覆盖已有的单轴 stage-2 权重
#      （那个是当前最优方案 EXP-012 的产出，务必保留）
#
# !! 前置依赖 !!
#   必须先跑完 slurm/data/stage2_prep_tri.sbatch 的 train 和 val 两个 split。
#   提交时请带依赖，别重蹈 BUG-006（抢跑拿到不完整数据）：
#     sbatch --dependency=afterok:<prep_jobid> slurm/train/train_stage2_tri.sbatch
#
# 提交: sbatch slurm/train/train_stage2_tri.sbatch
# 续训: sbatch slurm/train/train_stage2_tri.sbatch --resume
# ============================================================
#SBATCH --job-name=s2_tri
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1                  # 资源名按站点而定；所报模型用的是 80GB 显存
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/stage2_tri/slurm_%j.out
#SBATCH --error=runs/stage2_tri/slurm_%j.err

# 激活 Python 环境（按你的站点/个人配置修改这几行）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coronary

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROJECT_DIR=/path/to/coronary-seg
DATA_DIR=/path/to/cache/stage2_tri

cd "$PROJECT_DIR" || exit 1
export PYTHONPATH=.
mkdir -p runs/stage2_tri

echo "作业 $SLURM_JOB_ID 在 $(hostname) 启动"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 护栏：train / val 的 npz 数量必须与 split.json 一致。
# BUG-006 的教训 —— 只判「非零」挡不住「部分完成」
for sp in train val; do
    n=$(ls "$DATA_DIR/$sp"/*.npz 2>/dev/null | wc -l)
    want=$(python -c "import json;print(len(json.load(open('splits/split.json'))['$sp']))")
    echo "  $sp npz: $n / 期望 $want"
    if [ "$n" -ne "$want" ]; then
        echo "错误：$sp 数据不完整（$n != $want）。"
        echo "  先跑完 slurm/data/stage2_prep_tri.sbatch --splits train,val"
        echo "  并用 --dependency=afterok:<prep_jobid> 提交本作业"
        exit 1
    fi
done

# 超参与 EXP-007 保持一致（只改 epochs），这样和单轴 stage-2 的对比
# 是干净的单变量实验：唯一变量就是阶段1概率的来源
python -u scripts/train/train_stage2.py \
    --data-dir "$DATA_DIR" \
    --batch-size 4 \
    --epochs 30 \
    --lr 3e-4 \
    --w-cldice 0.5 \
    --cldice-warmup 500 \
    --cldice-k 5 \
    --samples-per-case 16 \
    --num-workers $SLURM_CPUS_PER_TASK \
    --out-dir runs/stage2_tri \
    "$@"

echo "作业 $SLURM_JOB_ID 结束"
