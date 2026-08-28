# Coronary Artery Segmentation from CTA

*[中文说明 / Chinese version](README.zh.md)*

A two-stage pipeline for segmenting the coronary artery tree from cardiac CTA volumes,
built on the [ImageCAS](https://arxiv.org/abs/2211.01607) dataset (1000 cases).
Developed as an MSc project at the a university and trained on the SLURM
cluster (SLURM, A100 80 GB).

The coronary lumen occupies **less than 1 %** of a CTA volume and forms a thin, branching
tree. Dice barely moves when a distal branch is lost, but the clinical value of a
segmentation rests on exactly that connectivity. This project therefore optimises and
reports **topology alongside overlap**.

---

## Results

All figures are means over the **full 200-case test set**, after post-processing.

| Configuration | Dice | clDice | Betti-0 error | HD95 (mm) |
|---|---|---|---|---|
| Single-axis 2.5D | 0.7955 | 0.8581 | 4.19 | 24.98 |
| Single-axis 2.5D + TTA | 0.8027 | 0.8670 | 3.68 | 23.66 |
| Tri-axial fusion (Stage 1) | 0.8098 | 0.8762 | 2.46 | 20.53 |
| **+ topology-aware refinement (final)** | **0.8216** | **0.8960** | **1.53** | **19.90** |

Both stages improve **all four metrics**. Every comparison is paired per case and tested
with a Wilcoxon signed-rank test with Holm–Bonferroni correction across the four metrics:

- **Tri-axial fusion vs. single-axis + TTA** — significant on all four
  (Dice *p* = 4.2 × 10⁻⁸, Betti-0 *p* = 1.5 × 10⁻¹²).
- **Final vs. Stage 1** — significant on all four
  (clDice *p* = 5.1 × 10⁻²³ with 165 of 200 cases improved; HD95 *p* = 0.037).

The largest relative gain is on **Betti-0 error, which falls by 63 %** against the
single-axis baseline (4.19 → 1.53) — that is, the output is much closer to the two
connected trees the anatomy actually has.

---

## Method

**Stage 1 — 2.5D tri-axial segmentation.** A single 2D network with shared weights
processes slices along all three orthogonal axes. Each input stacks `2k+1 = 5` adjacent
slices as channels and the network predicts the centre slice. At inference the three
directional probability volumes are fused by a voxel-wise mean. This buys wide
in-plane context under a memory budget that a full 3D network cannot match: a 3D model
on 80 GB handles patches of about 128³ voxels, while a coronary tree spans over 200 mm.

**Stage 2 — topology-aware residual refinement.** A 3D network takes the image and the
Stage-1 probability and predicts a **residual correction** to the Stage-1 logit, with a
zero-initialised head so training starts from the identity. Its loss adds
**soft-clDice** — a differentiable centreline-overlap term — to a DiceFocal region term.
The module is independent of the architecture it follows: it improves two different
Stage-1 outputs.

```
Preprocessing   RAS orientation → 0.5 mm isotropic → HU window [-200, 800] → [0, 1]
Stage 1         tri-axial 2.5D slices → 2D SegResNet → DiceCE → bfloat16 AMP
Fusion          voxel-wise mean over three axes → threshold 0.50
Stage 2         [image, p_stage1] → 3D SegResNet → residual → DiceFocal + 0.5·soft-clDice
Post-process    remove connected components below 300 voxels
Evaluation      Dice · clDice · Betti-0 error · HD95, per case, paired tests
```

---

## What did not work

Negative results are kept in the repository, with the experiments that produced them.
A negative result that cannot be re-run is not a result.

- **Residual gating.** A learnable gate was meant to restrict correction to places where
  Stage 1 had erred. Ablation shows it *harms* three metrics. Measuring inside the trained
  model explains why: the gate saturates (mean 0.86, range 0.78–0.98) and the correction
  it applies to regions that should be left alone is **2.6× larger** than the correction
  applied where it was needed. It was removed from the final pipeline.
- **Inter-direction disagreement.** Where the three directions disagree does mark errors,
  but neither exploitation route survived. An explicit adaptive-threshold rule reverses in
  sign between subsets; giving the disagreement to the network as extra input channels is
  significantly *worse* on three metrics.
- **Endpoint reconnection.** Joining broken endpoints creates more wrong connections than
  correct ones — Betti-0 error grows monotonically with the allowed gap.
- **A spatial prior** for removing distant false positives helps on a stratified subset but
  its parameters do not transfer between subsets, so it is implemented but disabled.

---

## Repository layout

`scripts/` and `slurm/` use the **same five groups**, so a result, the script that produced
it, and the job that ran it sit at matching paths.

```
coronary-seg/
├── src/                       # reusable modules
│   ├── data.py                #   tri-axial 2.5D pipeline, slice indexing, balanced sampling
│   ├── model.py               #   Stage-1 network (2D SegResNet / UNet)
│   ├── engine.py              #   train/val loop, bfloat16 AMP, non-finite gradient guard
│   ├── ckpt_init.py           #   --resume vs --init-from (deliberately separate)
│   ├── stage2_{data,model,loss}.py
│   ├── spatial_prior.py       #   explored, not adopted
│   └── smart_reconnect.py     #   evaluated and rejected, disabled by default
├── scripts/                   # entry points, grouped by pipeline stage
│   ├── data/  train/  predict/  analysis/  figs/
├── slurm/                     # one submission script per experiment, same grouping
│   ├── data/  train/  predict/  analysis/
├── runs/                      # per-case metric CSVs, one directory per configuration
├── splits/split.json          # frozen 700/100/200 split, seed 42
├── tests/                     # run directly: python tests/xxx.py
└── configs/default.yaml
```

Model weights are not in the repository. **The per-case CSVs they produced are** — every
comparison above is recomputed from those files by `scripts/analysis/compare_runs.py`.

---

## Reproduction

Steps 1–4 need a GPU. Step 5 reads CSVs only and runs on a login node.

```bash
cd coronary-seg
```

**0. Environment**

```bash
module load apps/binapps/anaconda3/2024.10        # SLURM-specific
conda activate ~/scratch/envs/coronary
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**1. Data and splits** — downloads ImageCAS (~50 GB) via kagglehub and writes the frozen
700/100/200 split. Put the data on scratch, not home.

```bash
python scripts/data/prepare_data.py --config configs/default.yaml
```

**2. Stage 1** — tri-axial sampling is the default; there is no flag to switch it off.

```bash
python scripts/train/train.py --split-json splits/split.json --cache-dir <cache> \
    --k 2 --crop-size 384 --batch-size 32 --epochs 70 --lr 3e-4 \
    --backbone segresnet --out-dir runs/exp_tri2p5d

sbatch slurm/train/train_2p5d.sbatch          # the exact job used for the reported model
```

**3. Stage-1 inference** — runs the three directions, fuses, thresholds, filters, and
writes per-case metrics.

```bash
PYTHONPATH=. python scripts/predict/predict_tri.py \
    --cache-dir <cache> --ckpt runs/exp_tri2p5d/best.pth \
    --out-csv runs/exp_tri2p5d/test_metrics.csv
```

**4. Stage 2**

```bash
PYTHONPATH=. python scripts/data/stage2_prepare.py   ...   # cache Stage-1 probabilities
PYTHONPATH=. python scripts/train/train_stage2.py    ...
PYTHONPATH=. python scripts/predict/predict_stage2.py ...
```

`predict_stage2.py` writes the Stage-1 and Stage-2 metrics for a case into the **same CSV
row**, so a comparison cannot silently be run across mismatched case sets.

**5. Compare**

```bash
PYTHONPATH=. python scripts/analysis/compare_runs.py \
    --baseline "tri-axial=runs/exp_tri2p5d/test_metrics_tri_mean050_v2.csv:pp" \
    --runs "final=runs/stage2_tri_nogate/test_metrics.csv:s2"
```

This reproduces the paired Wilcoxon and Holm output quoted above.

---

## Implementation notes

- **Class imbalance.** Slices containing vessel are all kept; background slices are sampled
  at a ratio of 0.25, combined with a DiceCE loss. Simple oversampling of positive cases was
  rejected — it revisits a small number of cases and raises the overfitting risk.
- **bfloat16, not float16.** With foreground this sparse, fp16 overflows to NaN. bf16 has the
  dynamic range of fp32 and is native on A100. Gradients are additionally checked for
  finiteness before every step, and non-finite steps are skipped.
- **Checkpoint resilience.** `last.pth` is written atomically (temp file + rename), so a job
  killed at the 4-day SLURM limit resumes cleanly with `--resume`.
- **`--resume` and `--init-from` are separate flags.** `--resume` restores model, optimiser,
  scheduler, epoch and best metric; `--init-from` loads weights only and starts a fresh
  schedule. Conflating them silently restarts the LR schedule mid-training. Covered by
  `tests/test_init_from.py`.
- **Inference is resumable.** The prediction scripts read an existing CSV and skip cases
  already evaluated.

## Evaluation protocol

Four metrics, chosen to be complementary rather than redundant: **Dice** (volumetric
overlap), **clDice** (centreline agreement), **Betti-0 error** (difference in the number of
connected components), and **HD95** (worst-case boundary distance, 95th percentile).

Comparisons are **per case and paired** on the full 200-case test set, tested with the
Wilcoxon signed-rank test and corrected with Holm–Bonferroni across the four metrics.
This matters: a difference in means is not evidence of an improvement, and on this data
mean differences and paired tests disagree in both directions.

## Limitations

- A single dataset (ImageCAS). No cross-centre or cross-scanner validation.
- Stage 2 operates on 128³ patches and therefore has no view of global anatomy.
- HD95 has a heavy tail driven by false positives — confusion with the aorta, veins and
  other tubular structures far from the coronary tree — rather than by missed vessel.

## Data

**ImageCAS** — 1000 cardiac CTA volumes with coronary artery annotations.
Zeng et al., *Computerized Medical Imaging and Graphics*, 2023
([arXiv:2211.01607](https://arxiv.org/abs/2211.01607)).

The dataset is **not redistributed here**. `scripts/data/prepare_data.py` fetches it via
`kagglehub` from `xiaoweixumedicalai/imagecas` (~50 GB); please follow the dataset's own
licence terms. Put it on scratch storage, not in your home directory.

Detailed documentation, including the full experiment history, is in
[`coronary-seg/README.md`](coronary-seg/README.md) (Chinese).
