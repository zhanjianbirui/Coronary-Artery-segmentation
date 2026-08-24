# ImageCAS 冠状动脉分割（2.5D 三正交方向）

基于 MONAI 的 2.5D SegResNet 冠状动脉分割流水线。在 a SLURM 集群（SLURM + A100 80GB）上开发和训练。

## 方法概述

采用 **2.5D 三正交方向**策略：沿三个正交轴（矢状面/冠状面/轴位面）切片，取中心层前后各 k 层堆叠成 `2k+1` 通道输入 2D SegResNet，只预测中心层的血管掩码。推理时逐切片预测再拼回 3D 体积。

```
预处理: RAS定向 → 0.5mm各向同性重采样 → HU加窗[-200,800]归一化
训练:   三方向切片(类别均衡采样) → 2D SegResNet → DiceCE loss → bfloat16 AMP
推理:   三个正交方向各自逐切片预测 → 概率图 mean 融合 → 拼回3D
        → 小连通域去除(min_voxels=300) → [可选] TTA
评估:   Dice / clDice / Betti-0 误差 / HD95
```

在此之上还有一个可选的 **Stage-2 3D 残差门控精修网络**（已训练，测试集评估进行中），
详见下方"两阶段级联"章节。

### 当前结果（200 例测试集，全量同口径）

后处理统一为 `min_voxels=300, max_gap=0`。

| 配置 | Dice | clDice | Betti-0 err | HD95 |
|------|------|--------|-------------|------|
| 单轴，无后处理 | 0.7925 | 0.8457 | 23.69 | 27.77 |
| 单轴 + 后处理 | 0.7955 | 0.8582 | 4.19 | 24.98 |
| 单轴 + TTA + 后处理 | **0.8027** | 0.8670 | 3.68 | 23.66 |
| **三正交 mean(thr=0.5) + 后处理** | 0.8012 | **0.8733** | **2.24** | **21.20** |

**怎么读这张表**：Dice 已经在 0.80 附近触顶，四个方案差异很小。真正拉开差距的是拓扑与
边界指标 —— 三正交融合把 Betti-0 误差压到 2.24（连通分量数最接近真值），HD95 降到 21.20
（最远错误点距离最小）。对冠脉这种细长树状结构，连通性比体素重叠更能反映临床可用性。

值得注意的是，三正交融合**在后处理之前**的 Betti-0 就只有 14.51，而单轴是 23.69 ——
只在单个方向上出现的假阳碎片，在三方向取平均时自然被压到阈值以下，融合本身就在做去噪。

**消融证据**（子集验证）：方向越多越好，`{2}` 0.7361 < `{1,2}` 0.7497 < `{0,2}` 0.7547
< `{0,1,2}` 0.7602（pp 后 Dice）；`mean` 融合明显优于 `max`（0.7314 vs 0.6889）；
阈值 0.50 优于 0.55/0.60/0.65。

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
│   ├── smart_reconnect.py        # 方向感知端点重连（实验证明关闭更优）
│   ├── spatial_prior.py          # 空间先验分量过滤（删离血管树很远的大块假阳）
│   ├── stage2_model.py           # Stage-2 残差门控 3D SegResNet
│   ├── stage2_loss.py            # Stage-2 损失：DiceFocal + 3D soft-clDice
│   └── stage2_data.py            # Stage-2 3D patch 采样 Dataset（npz + LRU）
├── scripts/
│   ├── prepare_data.py           # 下载 ImageCAS + 生成划分
│   ├── train.py                  # 训练入口（argparse CLI）
│   ├── predict.py                # 单轴推理 + 后处理 + 拓扑评估
│   ├── predict_tri.py            # 三正交方向融合推理（sweep / full 两种模式）
│   ├── stage2_prepare.py         # 用 stage-1 生成 stage-2 训练数据（npz）
│   ├── train_stage2.py           # Stage-2 训练入口
│   ├── predict_stage2.py         # Stage-2 推理 + 评估（滑窗）
│   ├── scout_bbox.py             # 血管边界框侦察（确定裁剪尺寸）
│   ├── sweep_postproc.py         # 后处理参数扫描
│   ├── sweep_spatial_prior.py    # 空间先验参数扫描（max_dist_mm × n_anchor）
│   ├── sweep_threshold.py        # 预测阈值扫描
│   ├── check_data.py             # 数据核对
│   ├── vis_slices.py             # 切片可视化
│   ├── vis_predict.py            # 预测结果可视化
│   └── analyze_cases.py          # 逐病例分析
├── slurm/
│   ├── train.sbatch              # 训练作业脚本
│   ├── train_2p5d.sbatch         # 2.5D 训练作业脚本
│   ├── predict_tta.sbatch        # TTA 推理作业脚本
│   ├── predict_tri_mean050.sbatch# 三正交融合全量推理作业脚本
│   ├── sweep_spatial_prior.sbatch# 空间先验扫描作业脚本
│   ├── stage2_prep_test.sbatch   # Stage-2 数据生成作业脚本
│   └── train_stage2.sbatch       # Stage-2 训练作业脚本
├── .kb/                          # 跨会话知识库（实验/bug/决策记录）
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

### 4b. 三正交方向融合推理（当前最优方案）

```bash
# 全量模式：固定融合方式与阈值，跑完 200 例测试集
PYTHONPATH=. python scripts/predict_tri.py \
    --cache-dir /path/to/cache \
    --ckpt runs/exp_tri2p5d/best.pth \
    --fixed-fuse mean --thr 0.50 \
    --min-voxels 300 --max-gap 0 \
    --out-csv runs/exp_tri2p5d/test_metrics_tri_mean050.csv \
    --max-cases 0 --pad-multiple 32

# sweep 模式：在子集上扫 融合方式 x 阈值
PYTHONPATH=. python scripts/predict_tri.py \
    --cache-dir /path/to/cache --ckpt runs/exp_tri2p5d/best.pth --max-cases 8

# 提交 SLURM
sbatch slurm/predict_tri_mean050.sbatch
```

同一个模型分别沿 axis 0/1/2 逐切片预测，得到三张概率图后按 `mean` 融合再二值化。
推理成本约为单轴的 3 倍，同样支持读已有 CSV 断点续跑。

### 5. 参数扫描（可选）

```bash
# 后处理参数扫描：推理一次，缓存预测，零成本扫描 min_voxels × max_gap
PYTHONPATH=. python scripts/sweep_postproc.py \
    --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth

# 阈值扫描：缓存概率图，扫描不同二值化阈值
PYTHONPATH=. python scripts/sweep_threshold.py \
    --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth
```

### 6. 空间先验后处理（治 HD95 长尾）

**这是干什么的**：测试集 HD95 是重尾分布（中位数 14.9，均值 21.2，12 例 >50）。
逐例诊断发现长尾由**假阳**驱动而非漏检：

```
corr(HD95, precision) = -0.665      corr(HD95, recall) = -0.229
HD95>50  的 22 例: precision 0.694 / recall 0.789
HD95<=20 的 101 例: precision 0.837 / recall 0.816
```

差病例的 recall 和好病例几乎一样，precision 却低了 0.14 —— 模型没漏掉血管，
是**多画了东西**：把主动脉/静脉等结构认成冠脉，形成一块**体积很大但离真实
冠脉树很远**的假阳。这种假阳因为够大，`min_voxels` 删不掉。

`src/spatial_prior.py` 给后处理链补上缺失的空间约束：取 top-N 个分量作"主干"，
其余分量离主干超过 `max_dist_mm` 就删。

**选锚按骨架长度，不是体积** —— 这是实现时踩的坑：紧实的假阳团块体积可以
超过整棵冠脉树（合成测试里 4320 vs 904 体素），按体积选锚会把团块当成主干、
把真血管树删掉，是彻底的负优化。冠脉的判别特征不是"大"而是"**长**"，
所以按中心线长度排序。`n_anchor` 默认 2，因为 85/200 例的真值本身就是两个
分量 —— 左冠和右冠是两棵互不相连的树，只留最大分量会删掉整条右冠。

```bash
# 先扫参数（推理一次，缓存后零成本扫 max_dist_mm × n_anchor）
PYTHONPATH=. python scripts/sweep_spatial_prior.py \
    --cache-dir /path/to/cache --ckpt runs/exp_tri2p5d/best.pth \
    --tri --fuse mean --thr 0.50 --min-voxels 300 \
    --dist-list 10,15,20,30,40,60 --anchor-list 1,2,3 \
    --out-csv runs/exp_tri2p5d/sweep_spatial_prior.csv --max-cases 40

# 建议先只跑诊断出的 5 个难例确认方向
PYTHONPATH=. python scripts/sweep_spatial_prior.py ... \
    --case-ids 931 630 595 741 728

# 扫出 D 之后，在正式推理里启用
PYTHONPATH=. python scripts/predict_tri.py ... --sp-dist 20 --sp-anchor 2

# 提交 SLURM
sbatch slurm/sweep_spatial_prior.sbatch
```

扫描脚本除均值外还输出 `hd95_median` / `hd95_p90` / `n_hd95_gt50` —— 长尾问题
看均值会被少数极端值主导，必须看分位数才知道是真修好了还是被平均掉了。

**当前状态**：代码已实现并通过合成数据自测（`python src/spatial_prior.py`），
**真实数据上的 D 值尚未扫描**，所以 `--sp-dist` 默认 0（关闭），
行为与引入前完全一致。

**方法边界**：只能删**与血管树分离**的假阳。若假阳紧贴或连通冠脉树
（比如从冠脉开口连出去的主动脉），它到主干距离为 0，本方法无效 ——
Stage-2 也修不了，它是 128³ patch 级训练，看不到全局解剖。

## 调参速查

| 参数 | CLI flag | 默认值 | 说明 |
|------|----------|--------|------|
| 骨干网络 | `--backbone` | segresnet | segresnet / unet |
| 上下文厚度 | `--k` | 2 | 取中心层±k层，输入通道=2k+1 |
| 裁剪尺寸 | `--crop-size` | 384 | 侦察脚本验证384零血管损失 |
| batch 大小 | `--batch-size` | 8 | A100 80G 可用 |
| 学习率 | `--lr` | 3e-4 | |
| 梯度裁剪 | `--grad-clip` | 1.0 | |
| 关闭 AMP | `--no-amp` | 开启 | CPU 测试时自动关 |
| 去碎片阈值 | `--min-voxels` | 300 | sweep 确定的最优值 |
| 端点重连 | `--max-gap` | 0 | 实验证明关闭更优 |
| 空间先验距离 | `--sp-dist` | 0（关闭） | 分量到主干的最大距离(mm)，需先扫描 |
| 空间先验锚数 | `--sp-anchor` | 2 | 取前几"长"的分量作主干，2=左冠+右冠 |
| TTA | `--tta` | 关 | 4-way翻转，推理慢4倍但更准 |
| 融合方式 | `--fixed-fuse` | mean | mean / max，实验证明 mean 明显更优 |
| 融合阈值 | `--thr` | 0.50 | sweep 确认 0.50 最优 |
| 参与融合的轴 | `--axes` | 0,1,2 | 方向越多拓扑越好 |

## 两阶段级联（Stage-2 精修）

在 Stage-1 之上还有一个可选的 **3D 残差门控精修网络**，专门修复血管断裂：

```
Stage-1 (2.5D SegResNet)
  → stage2_prepare.py  逐病例推理，存 npz {image, prob, label}
Stage-2 (3D ResidualGatedSegResNet)
  → 输入: 2 通道 3D patch [原图, stage-1 概率]
  → 输出: final_logit = stage1_logit + sigmoid(gate) * delta
  → 损失: DiceFocal(γ=2) + soft-clDice(k=5, warmup 500 步)
```

核心设计是 **残差门控**：网络初始化为恒等映射（delta=0，gate bias=−2），
训练起步时输出就等于 stage-1，不会把已经对的地方改坏；gate 让网络自己决定
"哪里该改"—— 断裂/模糊处 gate→1，已确信处 gate→0。这才是 refinement 的本意。
损失里的 soft-clDice 用可微软骨架直接优化中心线连通性，是拓扑感知的。

```bash
# 1. 生成 stage-2 数据
python scripts/stage2_prepare.py --splits train val test \
    --cache-dir /path/to/cache --ckpt runs/exp_2p5d/best.pth \
    --out-dir /path/to/cache/stage2 --k 2 --pad-multiple 32

# 2. 训练（或 sbatch slurm/train_stage2.sbatch）
python scripts/train_stage2.py --data-dir /path/to/cache/stage2 \
    --batch-size 4 --epochs 30 --lr 3e-4 \
    --w-cldice 0.5 --cldice-warmup 500 --out-dir runs/stage2

# 3. 推理评估
PYTHONPATH=. python scripts/predict_stage2.py \
    --data-dir /path/to/cache/stage2 --ckpt runs/stage2/best.pth
```

**当前状态**：训练已跑满 80 epoch，best val_dice = 0.8167 @ epoch 18（3D patch 级）。
epoch 18 之后 train_loss 持续下降而 val_dice 不再提升，是明确的过拟合 —— 复现时
跑 25~30 epoch 加 early stopping 即可。**注意 patch 级 val_dice 与 Stage-1 的
slice 级 val_dice 口径不同，不能直接相比**；Stage-2 是否真有增益，要等全量测试集
的 Dice/clDice/Betti-0/HD95 评估出来才能下结论，目前这一步尚未完成，所以上面的
结果表里还没有 Stage-2。

## 设计要点

- **类别极不平衡**：冠脉 <1% 体积。含血管切片全保留 + 背景按 0.25 比例采样 + DiceCE loss
- **2.5D 三正交方向**：三个正交面切片混合训练，一个模型覆盖所有血管走向，显存友好可用大 FOV
- **bfloat16 AMP**：float16 在稀疏前景场景下会梯度溢出致 nan，bf16 动态范围与 fp32 相同，A100 原生支持
- **梯度安全**：loss 和梯度的双重 nan/inf 检查，非有限时跳过更新，参数永不被污染
- **断点续训**：原子写 `last.pth`（tmp + rename），4 天 SLURM 上限被杀后 `--resume` 无缝继续
- **推理断点续跑**：predict.py 读已有 CSV 跳过已完成病例
- **三正交融合优于单方向**：同一模型沿三个轴分别推理再取 mean，只在单一方向出现的假阳
  会被平均掉，Betti-0 误差在后处理前就从 23.69 降到 14.51
- **指标要选对**：冠脉是细长树状结构，Dice 对细分支不敏感（多丢一根远端分支，Dice 几乎不动，
  但 Betti-0 和 clDice 会立刻变差）。因此以 clDice / Betti-0 / HD95 作为主要优化目标
- **数据驱动决策**：裁剪尺寸用侦察脚本确定、后处理参数用 sweep 扫描、每步优化先诊断再对症

## 已知待办

1. 在集群上跑 `sweep_spatial_prior.py` 扫出 `--sp-dist`，然后全量验证空间先验的真实增益
2. 跑 `predict_stage2.py` 出 Stage-2 全量测试集指标（当前最大空白）
3. 用 val_dice=0.8135 的三正交 checkpoint 重跑全量推理（当前推理用的是 0.8106 那版）
4. 三正交 + TTA 叠加验证（两者收益方向不同，可能可加）
5. Stage-2 的训练数据是单轴 Stage-1 生成的，应改用三正交概率图重新 prepare
6. `stage2_prepare.py` 改原子写（曾出现 npz 写入截断导致训练崩溃）
