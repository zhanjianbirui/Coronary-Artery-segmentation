# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coronary artery segmentation from 3D CTA volumes (ImageCAS dataset: 1000 cases). Uses a **2.5D approach**: stacks 2k+1 adjacent slices as channels, feeds into a 2D network (SegResNet/UNet), predicts the center slice's vessel mask. At inference, slices are predicted individually then reassembled into 3D for evaluation. Designed for University of Manchester CSF3 cluster (SLURM + A100 80GB).

## Architecture

The codebase has evolved from a 3D U-Net pipeline (configs/default.yaml still references 3D settings) to a **2.5D tri-axial** approach. The active training path uses `scripts/train.py` which accepts CLI args directly (not the YAML config system).

**Two config systems coexist:**
- `src/config.py`: YAML-based dataclass config with dotted-key overrides — used by `prepare_data.py` and referenced in `configs/default.yaml`
- `scripts/train.py`, `scripts/predict.py`: Use argparse with flat CLI flags — this is the active training/inference path

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

**Post-processing (`scripts/predict.py` + `src/smart_reconnect.py`):**
- Small component removal → optional direction-aware endpoint reconnection
- Topology metrics: Dice, clDice (skeleton-based), Betti-0 error, HD95

## Common Commands

```bash
# Environment setup (CSF3 login node)
module load apps/binapps/anaconda3/2024.10
conda activate ~/scratch/envs/coronary
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Download data + generate splits
python scripts/prepare_data.py --config configs/default.yaml

# Sanity check: overfit one batch (CPU, small scale)
CUDA_VISIBLE_DEVICES="" python scripts/train.py \
    --cache-dir /path/to/cache --overfit-one-batch \
    --crop-size 128 --max-cases 5 --steps 150 --num-workers 0

# Full training (GPU)
python scripts/train.py --cache-dir /path/to/cache \
    --epochs 100 --crop-size 512 --batch-size 8

# Resume training
python scripts/train.py --cache-dir /path/to/cache --resume

# Submit to SLURM
sbatch slurm/train.sbatch

# Inference + evaluation
PYTHONPATH=. python scripts/predict.py \
    --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth \
    --out-csv runs/exp_2p5d/test_metrics.csv --pad-multiple 32

# Sweep post-processing params
PYTHONPATH=. python scripts/sweep_postproc.py --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth

# Sweep prediction threshold
PYTHONPATH=. python scripts/sweep_threshold.py --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth

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

本项目维护了一个结构化知识库 `.kb/`，用于跨会话积累经验。**每次会话必须遵循以下流程。**

### 会话开始

1. **读索引**：先读 `.kb/INDEX.md`，了解知识库当前状态和最近活跃会话
2. **按需加载**：根据当前任务，只读对应分类文件（**禁止一次性读所有文件**）
   - 改代码结构 → 读 `architecture.md`
   - 调参/训练 → 读 `experiments.md`
   - 遇到报错 → 读 `bugs.md`
   - 改数据流 → 读 `pipeline.md`
   - 改推理/后处理 → 读 `postprocessing.md`
   - 环境/部署 → 读 `environment.md`
   - 架构选择 → 读 `decisions.md`
3. **读最近日志**：如果需要恢复上下文，读 `INDEX.md` 中"最近活跃"表指向的日志文件

### 会话进行中

4. **实时记录**：完成一个有意义的操作后（修复 bug、完成实验、做出决策），立即写入对应分类文件，不要等到会话结束
5. **追加而非覆盖**：在分类文件末尾追加新条目，使用文件内定义的编号格式（如 `BUG-003`、`EXP-002`、`DEC-005`）

### 会话结束

6. **写日志**：在 `.kb/logs/YYYY-MM-DD.md` 记录本次操作摘要（同一天追加，不覆盖）
7. **更新索引**：更新 `INDEX.md` 的"最近活跃"表（只保留最近 5 条）和"标签速查"表（如有新标签）

### 知识库文件结构

```
.kb/
├── INDEX.md              # 总索引（必读入口）
├── architecture.md       # 代码架构演进、模块关系
├── experiments.md        # 实验记录：超参、指标、结论
├── bugs.md               # Bug 调试记录、踩坑经验
├── pipeline.md           # 数据流水线：预处理、切片、缓存
├── postprocessing.md     # 后处理策略、sweep 结果
├── environment.md        # 环境配置、SLURM、集群问题
├── decisions.md          # 设计决策及理由
└── logs/
    └── YYYY-MM-DD.md     # 按日期的会话操作日志
```

### 写入规范

- **experiments.md**：按 `EXP-{编号}` 模板追加，必须包含超参、结果、结论
- **bugs.md**：按 `BUG-{编号}` 追加，必须包含现象、原因、解决方案、相关文件
- **decisions.md**：按 `DEC-{编号}` 追加，必须说明选项对比和选择理由
- **标签**：为重要条目标注 `#标签`，并同步更新 `INDEX.md` 标签速查表
- **日志**：每条操作记录用 `### [{序号}] HH:MM {标题}` 格式，附简要说明。**每完成一个操作必须立即追加日志，带上当前时间**

### 上下文控制

- 单个分类文件超过 200 行时，拆分为子文件（如 `experiments-v2.md`），在 INDEX.md 中更新链接
- 日志文件超过 100 行时，旧日志不再加载，只通过 INDEX.md 的"最近活跃"表索引
- 优先从知识库获取信息，避免重复读取已记录过的源码
