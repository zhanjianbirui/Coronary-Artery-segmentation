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
| 单轴 + TTA + 后处理 | 0.8027 | 0.8670 | 3.68 | 23.66 |
| **三正交 mean(thr=0.5) + 后处理** | **0.8098** | **0.8762** | 2.46 | **20.53** |
| **Stage-2 精修（单轴起点）+ pp** | 0.8117 | **0.8863** | **1.79** | 22.31 |

> ⚠️ 三正交一行用的是 **epoch=59 / val_dice=0.8135** 的 checkpoint
> （`test_metrics_tri_mean050_v2.csv`）。早期文档里的 0.8012 / 0.8733 / 21.20
> 出自 epoch=36 的**次优权重**，已作废 —— 换权重后 Dice 提升 0.0086
> （167/200 例更优，p=8e-20），足以改变结论的显著性，见下文 §7。

**怎么读这张表**：绝对数值差异看着很小，但配对检验下**每一步都站得住**：

- **三正交融合 vs 单轴+TTA：四项全部显著优**
  （Dice p=4.2e-08、clDice p=2.6e-07、Betti-0 p=1.5e-12、HD95 p=8.1e-04）
- **Stage-2 精修 vs 三正交：两个拓扑指标极显著优**
  （clDice p=2.2e-08、Betti-0 p=5.0e-07），Dice 与 HD95 无显著差异

对冠脉这种细长树状结构，连通性比体素重叠更能反映临床可用性 —— Dice 对细分支
不敏感（丢一根远端分支 Dice 几乎不动，Betti-0 和 clDice 会立刻变差），
所以拓扑指标上的提升比 Dice 上的同等提升更有价值。

值得注意的是，三正交融合**在后处理之前**的 Betti-0 就只有 14.51，而单轴是 23.69 ——
只在单个方向上出现的假阳碎片，在三方向取平均时自然被压到阈值以下，融合本身就在做去噪。
（14.51 出自 v1 权重；v2 未单独统计 raw 列，结论方向不受影响。）

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
│   ├── ckpt_init.py              # --resume / --init-from 的公共逻辑（见 BUG-007）
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
│   ├── compare_runs.py           # 多方案逐病例配对显著性比较（Wilcoxon+Holm）
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
│   ├── predict_tri_mean050_v2.sbatch # 同上，用当前 best.pth 重跑
│   ├── sweep_spatial_prior.sbatch# 空间先验扫描作业脚本
│   ├── stage2_prep_test.sbatch   # Stage-2 数据生成（单轴起点）
│   ├── stage2_prep_tri.sbatch    # Stage-2 数据生成（三正交起点）
│   ├── train_stage2_tri.sbatch   # Stage-2 训练（三正交起点）
│   ├── predict_stage2_tri.sbatch # Stage-2 评估（三正交起点）
│   └── train_stage2.sbatch       # Stage-2 训练作业脚本
├── tests/
│   └── test_init_from.py        # --init-from 校验 + BUG-007 机制复现（23 项）
├── .kb/                          # 跨会话知识库（不入版本控制）
│   ├── INDEX.md                 #   唯一必读入口（≤100 行）
│   ├── results.md               #   结论单一事实来源 —— 引用数字前先读这个
│   └── ...                      #   experiments / bugs / decisions / logs
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

**⚠ 这只适用于「训练被中途杀掉」。** 如果训练已经跑满 `--epochs`
（`last.pth` 的 epoch == epochs−1），`--resume` 不但没用，还会**毁掉权重**：
`CosineAnnealingLR` 的 LR 此时已退火到 0，而 scheduler 的 `state_dict()`
把 `T_max` 也存了进去，恢复时会把命令行新传的 `--epochs` 静默覆盖回旧值，
余弦进入下一个周期、LR 一路冲到初始值的约 98 倍（实测 3e-4 → 2.95e-2）。

想在跑满的权重上再训一段，用下面的 `--init-from`。

### 3b. 在已有权重上继续训练（`--init-from`）

```bash
# 只借模型权重，optimizer / scheduler / epoch / best_dice 全部重建
python scripts/train.py --cache-dir /path/to/cache \
    --init-from runs/exp_tri2p5d/best.pth \
    --out-dir runs/exp_tri2p5d_cont \
    --epochs 60 --lr 1e-4
```

两条硬护栏（`validate_init_from`）：

1. **不能与 `--resume` 同用** —— 语义冲突，直接报错
2. **源 checkpoint 不能位于 `--out-dir` 内** —— 否则训练第一次刷新 `best.pth`
   就会覆盖掉你拿来做初始化的那份权重。必须写到新目录。

`--lr` 建议比初始训练小（如 1e-4 对 3e-4），因为是在已收敛的权重上微调。

`--init-from` 对 stage-1（`train.py`）和 stage-2（`train_stage2.py`）都可用，
公共逻辑在 `src/ckpt_init.py`。

验证：`python tests/test_init_from.py`（23 项，不依赖 pytest；
其中第 4 组静态检查两个训练脚本都真的接上了，避免只改一个）

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
所以按中心线长度排序。`n_anchor` 默认 **2**，理由是分层子集实测（EXP-013），
**不是**"左冠+右冠所以取 2" —— 那个直觉解释已被第一轮扫描证伪：真值有 ≥3 个
连通分量的病例占 112/200。取 2 的真实机制见下文"方法论教训"：锚一多，大块
假阳会自己挤进 top-N 当上锚，从此免疫过滤。至于 `n_anchor=1` 则是灾难性的
（recall 0.829→0.528），只留最大分量确实会删掉整条右冠。

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

**第二轮扫描结果（24 例分层子集 = 12 长尾 + 12 对照，EXP-013）**：

| config | 删分量 | ΔDice | ΔclDice | ΔB0 | ΔHD95 | ΔP90 | >50例 |
|--------|-------|-------|---------|-----|-------|------|-------|
| d8_a2 | 41 | +0.0036 | +0.0094 | −1.46 | −8.07 | −5.78 | 8/12 |
| **d10_a2** | 32 | **+0.0041** | **+0.0091** | **−1.33** | **−10.58** | **−7.68** | **7/12** |
| d10_a3 | 27 | +0.0014 | +0.0053 | −1.12 | −6.41 | −7.68 | 9/12 |
| d10_a4 | 25 | +0.0033 | +0.0061 | −1.04 | −6.33 | −7.68 | 9/12 |
| d10_a6 | 21 | −0.0002 | +0.0004 | −0.88 | −4.22 | −1.02 | 10/12 |

最优 `--sp-dist 10 --sp-anchor 2`，四项指标全部改善。

**但这个 −10.58 不能直接引用**：子集里长尾占 50%，全量只有 6%。若增益全部
来自长尾病例，全量 ΔHD95 ≈ −10.58 × 24/200 ≈ **−1.3**。仍有意义，但只有
原数字的 1/8。全量验证尚未做，所以 `--sp-dist` 仍默认 0。

**一个值得记的方法论教训**：两轮扫描给出了**相反**的最优 anchor（第一轮 a3、
第二轮 a2），唯一差别是子集构成。机制是：没有大块假阳时主要风险是「误删合法
的第三段」，锚越多越安全；而长尾病例里假阳是主要矛盾，**锚一多，假阳团块自己
就挤进 top-N 当上锚**，从此免疫过滤（删分量数印证：a2 删 32 个，a6 只删 21 个）。
子集设计本身就是实验设计的一部分，比调参更重要。

<details>
<summary>第一轮扫描结果（前 40 例，有偏子集，结论已被推翻）</summary>


| config | 删分量 | ΔDice | ΔclDice | ΔB0 | ΔRecall | ΔP90 | >50例 |
|--------|-------|-------|---------|-----|---------|------|-------|
| d10_a1 | 101 | −0.1859 | −0.2003 | −0.53 | **−0.3007** | +120.78 | 36/2 |
| d10_a2 | 20 | −0.0013 | −0.0008 | −0.45 | −0.0056 | +3.98 | 2/2 |
| **d10_a3** | 17 | **+0.0018** | **+0.0022** | −0.375 | −0.0010 | **−4.01** | **1/2** |
| d≥30（任意 a） | 0 | 全 0 | | | | | |

**决定成败的是 `n_anchor`，不是距离阈值。** `n_anchor=1` 在真实数据上灾难性地
印证了设计时的担心：recall 从 0.829 崩到 **0.528**，36/40 例 HD95>50 ——
整条右冠被当成远处杂物删掉。`n_anchor=2` 也不够（HD95 反而 +2.42），因为真值
分量数分布中 **≥3 个分量的病例占 112/200**，锚数 2 会误删合法的第三段。
只有 `d10_a3` 全面正收益，且 `a1<a2<a3` 单调、a3 正是网格上界 ——
最优可能更大，第二轮已扩到 6。

</details>

**当前状态**：`--sp-dist` 仍默认 0（关闭）。真实增益比诊断预期小得多
（合成测试 Dice 0.347→0.970，真实只有 +0.0018，**高估两个数量级以上**），
且 40 例子集里只有 2 例长尾，"2→1" 实际只是一个病例，统计上脆弱。
等第二轮扩网格 + 全量验证后再决定是否进主线。

**⚠ 子集偏差**：`--max-cases N` 取的是**前 N 例**，而长尾病例都排在后面 ——
前 40 例只有 2 例 HD95>50（全量 12/200）。凡涉及 HD95/P90/B0 的结论，
必须用分层子集（`--case-ids`）或全量，不能用 `--max-cases`。

**方法边界**：只能删**与血管树分离**的假阳。若假阳紧贴或连通冠脉树
（比如从冠脉开口连出去的主动脉），它到主干距离为 0，本方法无效 ——
Stage-2 也修不了，它是 128³ patch 级训练，看不到全局解剖。

### 7. 方案间显著性比较

均值差不等于有提升，每个 Δ 都该配 p 值。两个来自本项目的实例：

- **看着有提升，其实不显著**：Stage-2 相对三正交 v2 的 Dice 是 +0.0019，
  但 p=0.051，逐例 116/84 —— 不能宣称 Dice 有改善。
- **看着差距很大，仍不显著**：空间先验在分层子集上 ΔHD95 高达 −10.58，
  但那是重尾均值被少数病例主导的结果，折算到全量只剩约 −1.3（§6）。

**配对检验也救不了错的对照组。** 早期版本这里举的例子是「三正交相对单轴+TTA
的 Dice 其实是统计显著的**下降**（p=0.015）」，并据此保守地写下"论文里只能
宣称 clDice 与 Betti-0"。后来发现那个"发现"本身就是**基准用了次优 checkpoint**
造成的假象 —— 换成 v2 后 Dice 变成**显著上升**（p=4.2e-08，逐例 132/68），
HD95 也从不显著变成显著（p=8.1e-04），**四项全部显著优**。

统计方法只保证"给定这两组数字，差异是否真实"；它不会告诉你**其中一组
本来可以更好**。对照组的质量是前置问题，见文末「方法论」一节。

`compare_runs.py` 只读结果 csv，不需要 GPU/torch/monai，**login 节点直接跑**：

```bash
PYTHONPATH=. python scripts/compare_runs.py \
    --baseline "三正交v2=runs/exp_tri2p5d/test_metrics_tri_mean050_v2.csv:pp" \
    --runs "单轴=runs/exp_2p5d/test_metrics_optimal.csv:pp" \
           "单轴+TTA=runs/exp_2p5d/test_final_tta.csv:pp" \
           "Stage2=runs/stage2/test_metrics_stage2.csv:s2" \
    --out-csv runs/significance.csv
```

`路径:前缀` 里的前缀对应 csv 列名：`predict.py`/`predict_tri.py` 用
`raw` / `pp`，`predict_stage2.py` 用 `s1` / `s2`。

检验用 Wilcoxon 符号秩（配对、非参数，HD95 重尾时比 t 检验稳），
多重比较做 Holm-Bonferroni 校正——一次比 4 个指标 × 若干方案，
不校正会虚增假阳性，这是论文里会被问的点。

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
| 空间先验锚数 | `--sp-anchor` | 2 | 取前几"长"的分量作主干（分层子集实测 2 最优） |
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

**全量 200 例结果（EXP-014，已用 v2 基准修正）**：

| 方案 | Dice | clDice | Betti0 | HD95 |
|------|------|--------|--------|------|
| 三正交融合 + pp（基准） | 0.8098 | 0.8762 | 2.46 | **20.53** |
| **Stage-2（单轴起点）+ pp** | **0.8117** | **0.8863** | **1.79** | 22.31 |

**配对显著性检验（Wilcoxon + Holm，200 例）**：

| 指标 | 三正交 v2 | Stage-2 | Δ | 优/劣 | p(Holm) | 结论 |
|------|-----------|---------|---|-------|---------|------|
| clDice | 0.8762 | 0.8863 | +0.0101 | 136/64 | 2.23e-08 | **显著提升** |
| Betti-0 | 2.46 | 1.79 | −0.67 | 101/51 | 5.01e-07 | **显著提升** |
| Dice | 0.8098 | 0.8117 | +0.0019 | 116/84 | 5.06e-02 | 不显著（边缘） |
| HD95 | 20.53 | 22.31 | +1.78 | 93/104 | 1.04e-01 | 不显著 |

准确的说法是：**Stage-2 在两个拓扑指标（clDice、Betti-0）上极显著优于三正交，
在体素重叠指标 Dice 上打平，HD95 无显著差异。**

这个结果与 Stage-2 的设计意图是**内部自洽**的：残差门控（见"设计取舍"）和
soft-clDice 损失针对的都是**拓扑**，而收益也正好出现在拓扑指标上。
Betti-0 降到 1.79 是全项目最好。

<details>
<summary>⚠️ 早期版本曾报告「四项赢三项、Dice 显著提升（p=5e-11）」—— 那是基准口径造成的假象</summary>

早期基准用的是三正交 **epoch=36** 的次优 checkpoint（Dice 0.8012）。
换成 epoch=59 的 v2（Dice 0.8098）后：

| 指标 | vs v1（旧，已作废） | **vs v2（正确）** |
|------|--------------------|-------------------|
| Dice | +0.0105, p=5.04e-11 显著 | +0.0019, **p=0.051 不显著** |
| clDice | +0.0130, p=1.88e-09 显著 | +0.0101, p=2.23e-08 显著 |
| Betti-0 | −0.45, p=2.80e-04 显著 | −0.67, p=5.01e-07 显著 |
| HD95 | +1.11, p=0.48 | +1.78, p=0.104 |

**光换一个 checkpoint（同模型、同参数、同后处理），就把 p=5e-11 的"极显著"
打成 p=0.051 的边缘不显著。** 说明此前 Dice 优势的大部分来自 baseline
没调到最好，而不是方法本身。拓扑指标的结论则不受影响。

这是本项目第二次因**对照组质量**导致结论反转（第一次是空间先验的子集选择，
见 §6）。教训写在文末"方法论"一节。

</details>

相对自己的起点（单轴 stage-1）效应更大且非常普遍：**178/200 例 Dice 上升、
183/200 例 clDice 上升**，四项 p 值都在 1e-5 以下。

训练侧注意：80 epoch 里 best 出现在 epoch 18，之后 62 个 epoch 是纯过拟合，
复现时跑 30 epoch 加 early stopping 即可。

### 换三正交起点（进行中）

既然精修能把 0.7955 抬到 0.8117，从更好的起点（三正交 v2 的 0.8098，
且拓扑更好）出发有望叠加增益。唯一变量是 npz 里 `prob` 的来源，
image/label 完全不变。

这一步现在**尤其关键**：换 v2 基准后 Stage-2 的 Dice 优势已降为不显著
（见上文 §Stage-2 结果），若三正交起点能把 Dice 也拉开，主结论会更强。

```bash
# ① 生成数据（三正交推理约为单轴 3 倍耗时，建议分两批）
sbatch slurm/stage2_prep_tri.sbatch --splits val,test   # 300 例
sbatch slurm/stage2_prep_tri.sbatch --splits train      # 700 例

# ② 训练（30 epoch）—— 务必串依赖
sbatch --dependency=afterok:<prep_jobid> slurm/train_stage2_tri.sbatch

# ③ 评估
sbatch --dependency=afterok:<train_jobid> slurm/predict_stage2_tri.sbatch
```

输出目录一律用 `stage2_tri` / `runs/stage2_tri`，不要覆盖单轴那套 ——
`stage2_prepare.py` 靠「文件已存在就跳过」续跑，指向旧目录会得到混了
两种来源的数据集；而 `runs/stage2/best.pth` 是当前最优方案的权重。

**成功判据**：必须同时超过「单轴起点的 Stage-2」和「不做精修的三正交 v2」。
只赢后者不算成功，那只说明精修有效，不说明换起点有用。
注意基准必须用 **v2**（epoch=59），用 v1 会高估收益。

### 两个消融（与主实验同批提交）

主结论既然是「拓扑指标显著改善」，就必须证明它来自设计本身：

```bash
sbatch slurm/train_stage2_tri.sbatch --no-gate    --out-dir runs/stage2_tri_nogate    # 验证 DEC-008
sbatch slurm/train_stage2_tri.sbatch --w-cldice 0 --out-dir runs/stage2_tri_nocldice  # 验证 DEC-009
```

两个判读要点：

- **门控只占 17 个参数**（4,701,618 vs 4,701,601）。两组容量几乎相同，
  所以这是干净的单变量消融，性能差异不能归因于参数量。
- **loss 不可横向比较**。去掉 clDice 项后 `train_loss` 从 0.178 掉到 0.133，
  纯粹是少加了一项，不代表训得更好。**只能比 `val_dice`。**
- 三个作业的 `#SBATCH --output` 都写死 `runs/stage2_tri/slurm_%j.out`，
  日志混在一起。靠日志开头的 `参数量=... gate=...` 与 `loss: ... w_cldice=...`
  两行反查 jobid 属于哪个实验（`scontrol` 的 Command 字段**不含**透传参数）。

## 设计要点

- **类别极不平衡**：冠脉 <1% 体积。含血管切片全保留 + 背景按 0.25 比例采样 + DiceCE loss
- **2.5D 三正交方向**：三个正交面切片混合训练，一个模型覆盖所有血管走向，显存友好可用大 FOV
- **bfloat16 AMP**：float16 在稀疏前景场景下会梯度溢出致 nan，bf16 动态范围与 fp32 相同，A100 原生支持
- **梯度安全**：loss 和梯度的双重 nan/inf 检查，非有限时跳过更新，参数永不被污染
- **断点续训**：原子写 `last.pth`（tmp + rename），4 天 SLURM 上限被杀后 `--resume` 无缝继续
- **`--resume` 与 `--init-from` 是两个不同场景，不能混用**：
  `--resume` 恢复完整训练状态，用于**被杀后接着跑**（此时 `last_epoch < T_max`，正确）；
  `--init-from` 只借模型权重、其余全部重建，用于**已跑满 `--epochs` 后再训一段**。
  对跑满的 checkpoint 用 `--resume` 会让 LR 冲到初始值的约 98 倍并摧毁权重 ——
  因为 PyTorch 的 scheduler `state_dict()` 连 `T_max` 一起存，
  命令行新传的 `--epochs` 会被静默覆盖回旧值，余弦进入下一周期。
  实测见 `python tests/test_init_from.py`
- **推理断点续跑**：predict.py 读已有 CSV 跳过已完成病例
- **三正交融合优于单方向**：同一模型沿三个轴分别推理再取 mean，只在单一方向出现的假阳
  会被平均掉，Betti-0 误差在后处理前就从 23.69 降到 14.51
- **指标要选对**：冠脉是细长树状结构，Dice 对细分支不敏感（多丢一根远端分支，Dice 几乎不动，
  但 Betti-0 和 clDice 会立刻变差）。因此以 clDice / Betti-0 / HD95 作为主要优化目标
- **数据驱动决策**：裁剪尺寸用侦察脚本确定、后处理参数用 sweep 扫描、每步优化先诊断再对症

## 已知待办

1. **两个消融的评估**（训练进行中）：`--no-gate` 与 `--w-cldice 0`。
   主结论现在就是拓扑指标，soft-clDice 消融是其**因果支柱**，优先级最高
2. 三正交起点 Stage-2 的全量评估，基准用 **v2**
3. 空间先验 `--sp-dist 10 --sp-anchor 2` 全量 200 例验证（目前只有分层子集证据，
   全量增益预计仅 ΔHD95 ≈ −1.3）
4. 三正交 + TTA 叠加验证（两者收益方向不同，可能可加）
5. `train_stage2_tri.sbatch` 的 `--output` 应跟随 `--out-dir`，
   否则多个消融的日志混在同一目录（本次已踩）

<details>
<summary>已完成</summary>

- ~~跑 `predict_stage2.py` 出 Stage-2 全量测试集指标~~（EXP-012）
- ~~用 val_dice=0.8135 的三正交 checkpoint 重跑全量推理~~（EXP-014，
  并因此推翻了 EXP-012 的部分结论）
- ~~Stage-2 改用三正交概率图重新 prepare~~（数据已生成，训练进行中）
- ~~`stage2_prepare.py` 改原子写~~（DEC-012）
- ~~跑 `sweep_spatial_prior.py` 扫出 `--sp-dist`~~（EXP-010 / EXP-013）

</details>

## 方法论：这个项目栽过的同一个跟头（两次）

**对照组的质量决定结论的可信度，而对照组极容易在不知不觉中被"配置得比它实际能力差"。**

| 次 | 变的是什么 | 后果 |
|----|-----------|------|
| 1 | **子集选择**：前 40 例 → 12 长尾 + 12 对照的分层子集 | 空间先验最优 `n_anchor` 从 3 反转为 2（§6） |
| 2 | **checkpoint 口径**：baseline 用 ep36 → ep59 | 同时翻转了**两条**结论 —— Stage-2 的 Dice 优势从 p=5e-11 变成 p=0.051 不显著；三正交相对单轴+TTA 的 Dice 从「显著变差 p=0.015」变成「显著变好 p=4e-08」 |

两次都不是代码 bug，跑出来的数字每一个都是对的 —— 错的是**比较的设置**。

第 2 次尤其值得记：同一个错误口径，**一边让人高估**了 Stage-2（Dice 虚假显著），
**一边让人低估**了三正交（把四项全显著压成只有两项）。所以不能指望
「保守一点就安全」—— 错误的对照组在两个方向上都会误导。而且它污染的是
**所有**以该 baseline 为基准的比较，不只是当时关心的那一条。

由此形成的规则：

1. 任何横向对比前，先确认 baseline 用的是**它自己的最优权重 / 最优配置**，
   并在论文里写明用的是哪个 checkpoint
1b. baseline 一旦更新，**系统性重跑全部以它为基准的比较**，
   不能只重跑当前关心的那一条
2. 涉及长尾指标（HD95 / P90 / Betti-0）的结论，**不能用 `--max-cases N`**
   （取的是前 N 例，长尾病例都排在后面），必须分层子集或全量
3. 子集只能用来**排序**候选参数，**不能报绝对增益** —— 富集会放大效应
   （分层子集 ΔHD95 −10.58，折算到全量只剩约 −1.3）
4. 每个 Δ 都配 p 值，用配对检验而非比较均值
