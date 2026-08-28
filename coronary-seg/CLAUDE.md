# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coronary artery segmentation from 3D CTA volumes (ImageCAS dataset: 1000 cases). Uses a **2.5D approach**: stacks 2k+1 adjacent slices as channels, feeds into a 2D network (SegResNet/UNet), predicts the center slice's vessel mask. At inference, slices are predicted individually then reassembled into 3D for evaluation. Designed for a SLURM cluster (SLURM + A100 80GB).

## Architecture

The codebase has evolved from a 3D U-Net pipeline (configs/default.yaml still references 3D settings) to a **2.5D tri-axial** approach. The active training path uses `scripts/train/train.py` which accepts CLI args directly (not the YAML config system).

**Two config systems coexist:**
- `src/config.py`: YAML-based dataclass config with dotted-key overrides — used by `prepare_data.py` and referenced in `configs/default.yaml`
- `scripts/train/train.py`, `scripts/predict/predict.py`: Use argparse with flat CLI flags — this is the active training/inference path

**Data pipeline (`src/data.py`):**
- Tri-axial 2.5D: slices along all 3 orthogonal axes (axis 0/1/2), one model learns all orientations
- `PersistentDataset` caches preprocessed 3D volumes to disk; `build_slice_index` creates per-slice indices with foreground/background balancing
- `CaseGroupedBatchSampler` groups slices from the same case in a batch for LRU cache efficiency
- `SliceDataset` uses an LRU cache over loaded volumes to minimize memory

**Model (`src/model.py`):**
- `spatial_dims=2`, `in_channels=2k+1`, `out_channels=1` (sigmoid binary)
- Backbones: `segresnet` (default) or `unet`

**Engine (`src/engine.py`):**
- Uses bfloat16 AMP on A100 (no GradScaler needed); falls back to float16+GradScaler
- Gradient-level nan/inf protection: skips optimizer step if gradients are non-finite

**Post-processing (`scripts/predict/predict.py` + `src/smart_reconnect.py`):**
- Small component removal → optional direction-aware endpoint reconnection
- Topology metrics: Dice, clDice (skeleton-based), Betti-0 error, HD95

## Common Commands

```bash
# Environment setup (SLURM login node)
module load apps/binapps/anaconda3/2024.10
conda activate ~/scratch/envs/coronary
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Download data + generate splits
python scripts/data/prepare_data.py --config configs/default.yaml

# Sanity check: overfit one batch (CPU, small scale)
CUDA_VISIBLE_DEVICES="" python scripts/train/train.py \
    --cache-dir /path/to/cache --overfit-one-batch \
    --crop-size 128 --max-cases 5 --steps 150 --num-workers 0

# Full training (GPU)
python scripts/train/train.py --cache-dir /path/to/cache \
    --epochs 100 --crop-size 512 --batch-size 8

# Resume training
python scripts/train/train.py --cache-dir /path/to/cache --resume

# Submit to SLURM
sbatch slurm/train/train.sbatch

# Inference + evaluation
PYTHONPATH=. python scripts/predict/predict.py \
    --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth \
    --out-csv runs/exp_2p5d/test_metrics.csv --pad-multiple 32

# Sweep post-processing params
PYTHONPATH=. python scripts/analysis/sweep_postproc.py --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth

# Sweep prediction threshold
PYTHONPATH=. python scripts/analysis/sweep_threshold.py --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth

# Self-test individual modules
python src/model.py --k 2 --backbone segresnet
python src/data.py --cache-dir /path/to/cache --max-cases 5
```

## Key Design Decisions

- **Class imbalance**: Coronary arteries are <1% of volume. Addressed via `neg_per_pos` ratio in slice indexing (default 0.25 negative slices per positive) and DiceCE loss
- **Checkpoint resilience**: Atomic writes (tmp file + rename) for `last.pth`; `--resume` restores model + optimizer + scheduler + epoch + best metric
- **Inference padding**: SegResNet requires H/W to be multiples of 32 for skip connections; `pad_to_multiple_2d` handles this transparently
- **Predict script has resume**: Reads existing CSV and skips already-evaluated cases

## Language

代码注释和日志输出使用中文。README 和文档也是中文。

---

## 知识库操作指南（必读）

本项目维护结构化知识库 `.kb/`，用于跨会话积累经验。**每次会话必须遵循以下流程。**

### 会话开始

1. **只读 `.kb/INDEX.md`**（约 80 行）。它是唯一的必读入口，含当前作业状态、
   按任务的加载表、最近活跃日志。**不要顺手多读几个文件"以防万一"。**
2. **按 INDEX 的「按任务加载」表**加载 1~2 个分类文件。禁止一次性读全部。
3. **需要历史上下文时**，读日志的 **TL;DR 一节即可**（长日志顶部都有）。
   只有在 TL;DR 里找到明确线索时，才去读对应编号的详细条目。

### 🔑 引用任何数字前，先读 `results.md`

`.kb/results.md` 是**结论的单一事实来源**，优先于任何其他文件。
做方案对比、写论文、往文档里写指标之前必须读它；日常改代码不需要。

这条规则是有代价换来的：本项目已发生过多次**结论反转**，原因都不是代码 bug，
而是**比较的口径不一致**（对照组用了次优 checkpoint、子集有偏）。
`results.md` 末尾有「引用数字前的自检清单」。

### 会话进行中

4. **实时记录**：完成一个有意义的操作后（修复 bug、完成实验、做出决策），
   立即写入对应分类文件，不要等到会话结束。
5. **追加而非覆盖**：在文件末尾按编号格式追加（`BUG-00N` / `EXP-0NN` / `DEC-0NN`）。
6. **结论变了就同步 `results.md`**，并给被推翻的旧条目加醒目标注
   （`> ⚠️ 本条已被 EXP-0NN 推翻`），**不要删除旧条目** —— 反转过程本身有价值。
7. **推翻一个 baseline 时，必须系统性排查所有以它为基准的比较**，
   不能只改当前关心的那一条。用 `grep -rn "<旧数字>" .kb/ README.md slurm/` 扫全库。

### 会话结束

8. **写日志**：`.kb/logs/YYYY-MM-DD.md`（同一天追加，不覆盖）。
   每条用 `### [{序号}] HH:MM {标题}` 格式。
9. **超过 150 行的日志必须加顶部 TL;DR**：本次最重要的 1~2 件事 +
   其余成果列表 + 编号导航。下次会话只读它。
10. **更新 `INDEX.md`**：当前状态、最近活跃表（保留 5 条）、标签速查（如有新标签）。

### 上下文预算（硬约束）

| 文件 | 上限 | 超了怎么办 |
|------|------|-----------|
| `INDEX.md` | **100 行** | 把稳定结论移到 `results.md`，把细节移到分类文件 |
| `results.md` | 150 行 | 把过期结论移到对应 experiments 条目 |
| 分类文件 | 250 行 | 拆 `-v2` 子文件，在 INDEX 更新链接（见 `experiments.md`/`experiments-v2.md`）|
| 单日日志 | 150 行 | 加 TL;DR；超 400 行时旧条目只经 TL;DR 索引 |

`INDEX.md` 是每次会话的**固定开销**，控制它的收益最大。

### 文件结构

```
.kb/
├── INDEX.md              # 唯一必读入口（≤100 行）：当前状态 + 加载表 + 导航
├── results.md            # 结论单一事实来源：最优结果、论文结论、已推翻的说法
├── experiments.md        # EXP-001~009（早期；三正交数字均为已作废的 v1）
├── experiments-v2.md     # EXP-010 起（含 EXP-014 —— 最重要）
├── bugs.md               # BUG-001~007
├── decisions.md          # DEC-001~012
├── architecture.md / pipeline.md / postprocessing.md / environment.md
└── logs/YYYY-MM-DD.md    # 会话日志（长文件顶部有 TL;DR）
```
