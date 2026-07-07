# ImageCAS 冠脉分割 (CSF3)

基于 MONAI 的 3D U-Net 冠脉分割流水线，针对 University of Manchester CSF3 集群（SLURM + A100）设计。

## 项目结构

```
coronary-seg/
├── configs/default.yaml      # 所有超参数 (改实验只改这里)
├── src/
│   ├── config.py             # 配置加载 (YAML + 命令行覆盖)
│   ├── utils.py              # 种子 / 日志 / 统计
│   ├── data.py               # 数据发现 / 划分 / 预处理 / DataLoader
│   ├── model.py              # 模型工厂 / 损失 / 指标
│   ├── engine.py             # 训练 / 验证循环
│   └── checkpoint.py         # 断点续训
├── scripts/
│   ├── prepare_data.py       # 下载 + 划分 (login 节点跑)
│   ├── train.py              # 训练主入口
│   └── predict.py            # 推理
├── slurm/train.sbatch        # SLURM 作业脚本 (已填好你的账户/分区)
├── requirements.txt
└── README.md
```

## 数据流

```
原始 nii.gz → 窗位归一化 + 重采样 + 前景裁剪 → patch 采样(类别均衡)
           → 3D U-Net → DiceCE loss → (验证) 滑窗推理整图 → Dice
```

ImageCAS：1000 例 3D CTA，每例 `<case>/img.nii.gz` + `label.nii.gz`，冠脉二值掩码。

## 完整使用流程

### 0. 环境 (一次性，login 节点)

```bash
module load apps/binapps/anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/envs/coronary
pip install -r requirements.txt
# torch 单独装 (匹配 CSF3 的 CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 1. 下载数据 + 生成划分 (login 节点，需联网)

```bash
python scripts/prepare_data.py --config configs/default.yaml
# 它会用 kagglehub 下载 ImageCAS, 打印出真实 data_root, 并生成 splits/split.json
# 把打印出的 data_root 填进 slurm/train.sbatch 的 DATA_ROOT
```

> 数据较大 (~50GB)。下载到 `~/scratch`，别放 home（有配额）。

### 2. 提交训练 (gpuA / A100 80GB)

```bash
sbatch slurm/train.sbatch
squeue --me                    # 看排队/运行状态
tail -f runs/slurm_<jobid>.out # 实时看日志
```

### 3. 续训 (被 4 天上限杀掉后)

```bash
sbatch slurm/train.sbatch --resume
```

`--resume` 会从 `runs/exp_a100/last.pth` 精确恢复（模型 + 优化器 + scheduler + epoch + best），不会从头重来。

### 4. 推理

```bash
python scripts/predict.py --config configs/default.yaml \
    --ckpt runs/exp_a100/best.pth --split test
# 预测掩码保存到 runs/exp_a100/predictions/
```

## 调参速查 (命令行覆盖，不改代码)

```bash
# 临时改超参 (覆盖 yaml):
python scripts/train.py train.lr=0.0005 train.batch_size=4 preprocess.patch_size=96,96,96

# A100 80G 显存充裕, 可加大 batch / patch 提升效果:
sbatch slurm/train.sbatch train.batch_size=4 preprocess.patch_size=160,160,160
```

| 想做的事 | 改哪个 |
|---|---|
| 学习率 | `train.lr` |
| patch 大小 | `preprocess.patch_size` |
| batch 大小 | `train.batch_size` |
| 正样本比例(类别均衡) | `train.pos_ratio` |
| HU 窗位 | `preprocess.a_min` / `a_max` |
| 验证频率 | `train.val_interval` |
| 换模型 | `model.name` (需先在 `src/model.py` 登记) |

## 设计要点

- **类别极不平衡**：冠脉 <1% 体积。用 `RandCropByPosNegLabel`(pos_ratio=0.8) 强制 patch 含血管 + `DiceCELoss`。
- **整图放不进显存**：训练用 patch，推理用滑窗 `sliding_window_inference` 拼回整图。
- **4 天上限**：每 epoch 存 `last.pth`（原子写防损坏），`--resume` 续训。
- **可复现**：固定种子；每次训练把 resolved config 存进 work_dir 和 checkpoint。
- **混合精度**：A100 上开 AMP，提速省显存。

## 已验证

所有模块在 CPU 上以合成数据通过单元测试 + 端到端冒烟测试（含 resume）。真实训练需在 CSF3 gpuA 节点上跑。
