# ImageCAS 冠状动脉分割（2.5D 三正交方向）

基于 MONAI 的 2.5D SegResNet 冠状动脉分割流水线。在 University of Manchester CSF3 集群（SLURM + A100 80GB）上开发和训练。

## 方法概述

采用 **2.5D 三正交方向**策略：沿三个正交轴（矢状面/冠状面/轴位面）切片，取中心层前后各 k 层堆叠成 `2k+1` 通道输入 2D SegResNet，只预测中心层的血管掩码。推理时逐切片预测再拼回 3D 体积。

```
预处理: RAS定向 → 0.5mm各向同性重采样 → HU加窗[-200,800]归一化
训练:   三方向切片(类别均衡采样) → 2D SegResNet → DiceCE loss → bfloat16 AMP
推理:   逐切片2.5D预测 → 拼回3D → 小连通域去除(min_voxels=300) → [可选] TTA
评估:   Dice / clDice / Betti-0 误差 / HD95
```

### 当前结果（200 例测试集）

| 配置                       | Dice   | clDice | Betti-0 err | HD95  |
| -------------------------- | ------ | ------ | ----------- | ----- |
| 优化后处理 (mv=300, gap=0) | 0.7955 | 0.8582 | 4.19        | 24.98 |

## 项目结构

```
coronary-seg/
├── configs/default.yaml          # 超参数配置（YAML，prepare_data.py 使用）
├── src/
│   ├── config.py                 # YAML 配置加载（dataclass + dotted-key 覆盖）
│   ├── utils.py                  # 种子 / 日志 / 统计
│   ├── data.py                   # 三方向2.5D数据流水线 / 切片索引 / DataLoader
│   ├── model.py                  # 模型工厂（SegResNet / UNet）+ 损失
│   ├── engine.py                 # 训练/验证循环（bfloat16 AMP + 梯度安全检查）
│   ├── checkpoint.py             # 断点续训（原子写）
│   └── smart_reconnect.py        # 方向感知端点重连（实验证明关闭更优）
├── scripts/
│   ├── prepare_data.py           # 下载 ImageCAS + 生成划分
│   ├── train.py                  # 训练入口（argparse CLI）
│   ├── predict.py                # 推理 + 后处理 + 拓扑评估
│   ├── scout_bbox.py             # 血管边界框侦察（确定裁剪尺寸）
│   ├── sweep_postproc.py         # 后处理参数扫描
│   ├── sweep_threshold.py        # 预测阈值扫描
│   ├── check_data.py             # 数据核对
│   ├── vis_slices.py             # 切片可视化
│   ├── vis_predict.py            # 预测结果可视化
│   └── analyze_cases.py          # 逐病例分析
├── slurm/
│   ├── train.sbatch              # 训练作业脚本
│   ├── train_2p5d.sbatch         # 2.5D 训练作业脚本
│   └── predict_tta.sbatch        # TTA 推理作业脚本
├── requirements.txt
└── README.md
```

## 使用流程

### 0. 环境配置（login 节点，一次性）

```bash
module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 1. 下载数据 + 生成划分（login 节点，需联网）

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

通过 kagglehub 下载 ImageCAS（~50GB），生成 `splits/split.json`（700/100/200 划分）。数据放 `~/scratch`，别放 home。

### 2. 训练

```bash
# Sanity check：过拟合单个 batch（CPU 可跑）
CUDA_VISIBLE_DEVICES="" python scripts/train.py \
    --cache-dir /path/to/cache --overfit-one-batch \
    --crop-size 128 --max-cases 5 --steps 150 --num-workers 0

# 正式训练（GPU）
python scripts/train.py --cache-dir /path/to/cache \
    --backbone segresnet --k 2 --crop-size 384 --batch-size 8 \
    --lr 3e-4 --epochs 100

# 提交 SLURM
sbatch slurm/train_2p5d.sbatch
```

### 3. 断点续训（被 4 天上限杀掉后）

```bash
python scripts/train.py --cache-dir /path/to/cache --resume
# 或
sbatch slurm/train_2p5d.sbatch --resume
```

从 `last.pth` 精确恢复模型 + 优化器 + scheduler + epoch + best_dice。

### 4. 推理 + 评估

```bash
PYTHONPATH=. python scripts/predict.py \
    --cache-dir /path/to/cache \
    --ckpt runs/exp_2p5d/best.pth \
    --out-csv runs/exp_2p5d/test_metrics.csv \
    --min-voxels 300 --max-gap 0 \
    --tta --pad-multiple 32
```

同时输出"带/不带后处理"两组指标（Dice / clDice / Betti-0 / HD95），支持断点续跑。

### 5. 参数扫描（可选）

```bash
# 后处理参数扫描：推理一次，缓存预测，零成本扫描 min_voxels × max_gap
PYTHONPATH=. python scripts/sweep_postproc.py \
    --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth

# 阈值扫描：缓存概率图，扫描不同二值化阈值
PYTHONPATH=. python scripts/sweep_threshold.py \
    --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth
```

## 调参速查

| 参数       | CLI flag       | 默认值    | 说明                        |
| ---------- | -------------- | --------- | --------------------------- |
| 骨干网络   | `--backbone`   | segresnet | segresnet / unet            |
| 上下文厚度 | `--k`          | 2         | 取中心层±k层，输入通道=2k+1 |
| 裁剪尺寸   | `--crop-size`  | 384       | 侦察脚本验证384零血管损失   |
| batch 大小 | `--batch-size` | 8         | A100 80G 可用               |
| 学习率     | `--lr`         | 3e-4      |                             |
| 梯度裁剪   | `--grad-clip`  | 1.0       |                             |
| 关闭 AMP   | `--no-amp`     | 开启      | CPU 测试时自动关            |
| 去碎片阈值 | `--min-voxels` | 300       | sweep 确定的最优值          |
| 端点重连   | `--max-gap`    | 0         | 实验证明关闭更优            |
| TTA        | `--tta`        | 关        | 4-way翻转，推理慢4倍但更准  |

## 设计要点

- **类别极不平衡**：冠脉 <1% 体积。含血管切片全保留 + 背景按 0.25 比例采样 + DiceCE loss
- **2.5D 三正交方向**：三个正交面切片混合训练，一个模型覆盖所有血管走向，显存友好可用大 FOV
- **bfloat16 AMP**：float16 在稀疏前景场景下会梯度溢出致 nan，bf16 动态范围与 fp32 相同，A100 原生支持
- **梯度安全**：loss 和梯度的双重 nan/inf 检查，非有限时跳过更新，参数永不被污染
- **断点续训**：原子写 `last.pth`（tmp + rename），4 天 SLURM 上限被杀后 `--resume` 无缝继续
- **推理断点续跑**：predict.py 读已有 CSV 跳过已完成病例
- **数据驱动决策**：裁剪尺寸用侦察脚本确定、后处理参数用 sweep 扫描、每步优化先诊断再对症
