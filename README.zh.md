# CTA 冠状动脉分割

*[English version](README.md)*

基于 [ImageCAS](https://arxiv.org/abs/2211.01607) 数据集（1000 例）的两阶段冠脉树分割流水线。
曼彻斯特大学硕士项目，在 SLURM 集群（SLURM、A100 80GB）上训练。

冠脉管腔只占 CTA 体积的**不到 1%**，且是细长的分支树。丢掉一根远端分支，Dice 几乎不动，
但分割结果的临床价值恰恰就在这个连通性上。因此本项目**同时优化并报告拓扑指标与重叠指标**。

---

## 结果

以下均为**完整 200 例测试集**上的均值，经后处理。

| 配置 | Dice | clDice | Betti-0 误差 | HD95 (mm) |
|---|---|---|---|---|
| 单轴 2.5D | 0.7955 | 0.8581 | 4.19 | 24.98 |
| 单轴 2.5D + TTA | 0.8027 | 0.8670 | 3.68 | 23.66 |
| 三正交融合（阶段 1） | 0.8098 | 0.8762 | 2.46 | 20.53 |
| **+ 拓扑感知精修（最终）** | **0.8216** | **0.8960** | **1.53** | **19.90** |

两个阶段都在**四项指标上全部改善**。所有比较均**逐病例配对**，用 Wilcoxon 符号秩检验，
并对四个指标做 Holm–Bonferroni 校正：

- **三正交融合 vs 单轴 + TTA** —— 四项全部显著
  （Dice *p* = 4.2 × 10⁻⁸，Betti-0 *p* = 1.5 × 10⁻¹²）
- **最终方法 vs 阶段 1** —— 四项全部显著
  （clDice *p* = 5.1 × 10⁻²³，200 例中 165 例改善；HD95 *p* = 0.037）

相对增益最大的是 **Betti-0 误差，相对单轴基线下降 63%**（4.19 → 1.53）——
也就是说，输出更接近解剖上本该有的那两棵连通的血管树。

---

## 方法

**阶段 1 —— 2.5D 三正交分割。** 一个 2D 网络以**共享权重**处理三个正交方向的切片。
每个输入把相邻 `2k+1 = 5` 层堆成通道，网络只预测中心层；推理时对三个方向的概率体
做逐体素平均融合。这样能在显存预算内拿到很宽的面内视野：3D 网络即便在 80GB 上也
只能处理约 128³ 的 patch，而冠脉树跨度超过 200mm。

**阶段 2 —— 拓扑感知残差精修。** 一个 3D 网络接收原图与阶段 1 的概率，预测对阶段 1
logit 的**残差修正**；残差头权重零初始化，训练从恒等映射开始。损失在 DiceFocal 区域项
之上加了 **soft-clDice**（可微的中心线重叠项）。该模块**不依赖前面接的是什么架构** ——
它在两种不同的阶段 1 输出上都有效。

```
预处理    RAS 定向 → 0.5mm 各向同性 → HU 加窗 [-200, 800] → [0, 1]
阶段 1    三正交 2.5D 切片 → 2D SegResNet → DiceCE → bfloat16 AMP
融合      三方向逐体素平均 → 阈值 0.50
阶段 2    [原图, p_阶段1] → 3D SegResNet → 残差 → DiceFocal + 0.5·soft-clDice
后处理    去除小于 300 体素的连通分量
评估      Dice · clDice · Betti-0 误差 · HD95，逐病例配对检验
```

---

## 没走通的路

阴性结果连同产生它们的实验一并保留在仓库里。**跑不起来的阴性结果不算结果。**

- **残差门控。** 本意是让修正只作用在阶段 1 出错的地方。消融显示它让**三项指标变差**。
  进到训练好的模型里直接测量给出了原因：门控**饱和**了（均值 0.86，范围 0.78–0.98），
  而且它施加在**本不该改动**的区域上的修正，是该改动区域的 **2.6 倍**。已从最终方案中移除。
- **方向间分歧。** 三个方向不一致的地方确实标记了错误，但两条利用路径都没走通：
  显式的自适应阈值规则在不同子集上**符号翻转**；把分歧作为额外输入通道交给网络学，
  则在三项指标上**显著更差**。
- **端点重连。** 连接断开的端点，连错的比连对的多 —— Betti-0 误差随允许间隙**单调上升**。
- **空间先验**（删除远离血管树的假阳）在分层子集上有效，但参数**无法在子集间迁移**，
  因此代码保留、默认关闭。

---

## 仓库结构

`scripts/` 与 `slurm/` 用**同一套五组分类**，所以一个结果、产出它的脚本、跑它的作业，
路径是对齐的。

```
coronary-seg/
├── src/                       # 可复用模块
│   ├── data.py                #   三正交 2.5D 流水线、切片索引、类别均衡采样
│   ├── model.py               #   阶段 1 网络（2D SegResNet / UNet）
│   ├── engine.py              #   训练/验证循环、bfloat16 AMP、非有限梯度保护
│   ├── ckpt_init.py           #   --resume 与 --init-from（刻意分开）
│   ├── stage2_{data,model,loss}.py
│   ├── spatial_prior.py       #   探索过，未采用
│   └── smart_reconnect.py     #   实验否决，默认关闭
├── scripts/                   # 入口脚本，按流水线阶段分组
│   ├── data/  train/  predict/  analysis/  figs/
├── slurm/                     # 每个实验一个作业脚本，分组与 scripts/ 相同
│   ├── data/  train/  predict/  analysis/
├── runs/                      # 逐病例指标 csv，一个配置一个目录
├── splits/split.json          # 冻结的 700/100/200 划分，seed 42
├── tests/                     # 直接运行：python tests/xxx.py
└── configs/default.yaml
```

模型权重不入库，**但产出它们的逐病例 csv 入库** —— 上面每一条比较都是由
`scripts/analysis/compare_runs.py` 从这些 csv 重新算出来的。

---

## 复现

第 1–4 步需要 GPU，第 5 步只读 csv，在 login 节点即可。

```bash
cd coronary-seg
```

**0. 环境**

```bash
module load apps/binapps/anaconda3/2024.10        # SLURM 专用
conda activate ~/scratch/envs/coronary
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**1. 数据与划分** —— 通过 kagglehub 下载 ImageCAS（约 50GB），生成冻结的 700/100/200 划分。
数据放 scratch，别放 home。

```bash
python scripts/data/prepare_data.py --config configs/default.yaml
```

**2. 阶段 1** —— 三正交采样是默认行为，没有开关可以关掉。

```bash
python scripts/train/train.py --split-json splits/split.json --cache-dir <cache> \
    --k 2 --crop-size 384 --batch-size 32 --epochs 70 --lr 3e-4 \
    --backbone segresnet --out-dir runs/exp_tri2p5d

sbatch slurm/train/train_2p5d.sbatch          # 论文所报模型用的就是这个作业
```

**3. 阶段 1 推理** —— 跑三个方向、融合、二值化、去碎片，写出逐病例指标。

```bash
PYTHONPATH=. python scripts/predict/predict_tri.py \
    --cache-dir <cache> --ckpt runs/exp_tri2p5d/best.pth \
    --out-csv runs/exp_tri2p5d/test_metrics.csv
```

**4. 阶段 2**

```bash
PYTHONPATH=. python scripts/data/stage2_prepare.py   ...   # 缓存阶段 1 概率
PYTHONPATH=. python scripts/train/train_stage2.py    ...
PYTHONPATH=. python scripts/predict/predict_stage2.py ...
```

`predict_stage2.py` 把同一病例的阶段 1 与阶段 2 指标写进**同一行 csv**，
所以比较在结构上就不可能跨错病例集。

**5. 比较**

```bash
PYTHONPATH=. python scripts/analysis/compare_runs.py \
    --baseline "三正交=runs/exp_tri2p5d/test_metrics_tri_mean050_v2.csv:pp" \
    --runs "最终=runs/stage2_tri_nogate/test_metrics.csv:s2"
```

这条命令复现上面引用的全部配对 Wilcoxon 与 Holm 结果。

---

## 实现要点

- **类别极不平衡。** 含血管的切片全部保留，背景切片按 0.25 的比例采样，配合 DiceCE 损失。
  简单地对正样本病例过采样被否决了 —— 那会让网络反复看同一小批病例，抬高过拟合风险。
- **用 bfloat16 而不是 float16。** 前景这么稀疏时 fp16 会溢出成 NaN；bf16 的动态范围与
  fp32 相同，且 A100 原生支持。此外每步更新前都检查梯度是否有限，非有限则跳过该步。
- **断点续训。** `last.pth` 用原子写（临时文件 + rename），4 天 SLURM 上限被杀后
  `--resume` 可无缝继续。
- **`--resume` 与 `--init-from` 是两个独立的 flag。** 前者恢复模型、优化器、调度器、
  epoch 和最佳指标；后者只加载权重并重新开始调度。混用会在训练中途**静默重启学习率调度**。
  由 `tests/test_init_from.py` 覆盖。
- **推理可续跑。** 预测脚本读取已有 csv，跳过已评估的病例。

## 评估口径

四个指标，刻意选成互补而非冗余：**Dice**（体素重叠）、**clDice**（中心线一致性）、
**Betti-0 误差**（连通分量数之差）、**HD95**（边界距离的 95 分位，反映最坏情况）。

比较一律在完整 200 例测试集上**逐病例配对**，用 Wilcoxon 符号秩检验，
并对四个指标做 Holm–Bonferroni 校正。这一点很关键：**均值有差不等于有提升**，
而且在这份数据上，均值差与配对检验的结论**在两个方向上都出现过分歧**。

## 局限

- 只用了一个数据集（ImageCAS），没有跨中心 / 跨设备验证。
- 阶段 2 在 128³ patch 上工作，看不到全局解剖。
- HD95 有重尾，主因是**假阳**（把主动脉、静脉等远离冠脉树的管状结构认成冠脉），
  而不是漏掉血管。

## 数据

**ImageCAS** —— 1000 例心脏 CTA 及冠脉标注。
Zeng 等，*Computerized Medical Imaging and Graphics*，2023
（[arXiv:2211.01607](https://arxiv.org/abs/2211.01607)）。

本仓库**不转发数据集**。`scripts/data/prepare_data.py` 通过 `kagglehub` 从
`xiaoweixumedicalai/imagecas` 获取（约 50GB），请遵守数据集自身的许可条款。
放到 scratch 存储，不要放 home 目录。

更详细的文档（含完整实验历史）见
[`coronary-seg/README.md`](coronary-seg/README.md)。
