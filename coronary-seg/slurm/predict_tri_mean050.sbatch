#!/bin/bash --login
#SBATCH --job-name=pred_tri050
#SBATCH --account=gpu-fse-ugpgt01
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=runs/slurm_pred_tri050_%j.out
#SBATCH --error=runs/slurm_pred_tri050_%j.err

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

python -u scripts/predict_tri.py \
    --cache-dir "$CACHE_DIR" \
    --ckpt runs/exp_tri2p5d/best.pth \
    --fixed-fuse mean \
    --thr 0.50 \
    --min-voxels 300 \
    --max-gap 0 \
    --out-csv runs/exp_tri2p5d/test_metrics_tri_mean050.csv \
    --max-cases 0 \
    --batch 1 \
    --max-px-per-batch 500000 \
    --pad-multiple 32