# MBDG-RS

This repository contains the code needed to reproduce the classical
baselines, deep baselines, MBDG-RS ablations, unified Table-2 experiment,
paired primary statistics, and channel-gain robustness experiments on the
FaceEMG-11 dataset.

The physiological data are not stored in this repository. Download the data
from the official Zenodo record:

- Dataset: [FaceEMG-11](https://zenodo.org/records/22137268)
- DOI: [10.5281/zenodo.22137268](https://doi.org/10.5281/zenodo.22137268)
- Data license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Archive: `FaceEMG-11.zip`


## Evaluation protocol

All reported experiments use participant-level leave-one-subject-out (LOSO)
evaluation:

- 12 participants and 12 outer folds;
- 11 complete source participants for development;
- all 330 trials from the isolated target participant for testing;
- 30 blocks and 11 classes per participant;
- source-only fitting, normalization, and model selection;
- no target-participant sample or statistic used for fitting or selection.

When model or frequency-band selection is needed, selection uses inner LOSO
over the 11 source participants.

## Installation

Python 3.9 or newer is recommended.

```bash
conda create -n mbdg-rs python=3.10 -y
conda activate mbdg-rs
pip install -r requirements.txt
```

Deep models require a CUDA-capable PyTorch installation. Before launching a
GPU experiment, verify that CUDA is available

## Download and configure FaceEMG-11

Download and extract `FaceEMG-11.zip` outside the Git repository. The expected
layout is:

```text
FaceEMG-11/
└── trials/
    ├── sub-01_trials.npz
    ├── sub-02_trials.npz
    ├── ...
    └── sub-12_trials.npz
```

Each participant file must contain `data`, `labels`, `groups`, `fs`, and
`ch_names`. The expected data shape is `(330, 20, 1500)`.

Set the data and result locations before running an experiment:

```bash
export FACEEMG_DATA_ROOT=/absolute/path/to/FaceEMG-11
export FACEEMG_RESULT_ROOT=/absolute/path/to/faceemg11_results
```

`FACEEMG_DATA_ROOT` must be the directory that directly contains `trials/`.
If `FACEEMG_RESULT_ROOT` is omitted, outputs default to `results/` under the
repository root.


## Run commands

Run every command below from the repository root. Fold indices are zero-based:
fold `0` holds out `sub-01`, and fold `11` holds out `sub-12`.

Use new output directories when rerunning experiments. Some scripts refuse to
overwrite existing files, while others replace files with the same name.

### 1. Check the Table-2 installation

This command checks the Table-2 dependency chain without loading the dataset:

```bash
python run_table2_unified_ablation.py \
  --mode describe \
  --model all
```

Successful execution confirms that `run_frequency_band_ablation.py`,
`run_mbdg_rs_ablations.py`, `model_names.py`, and the core modules are
available.

### 2. Classical and geometric baselines

The released classical benchmark includes TD4, AIRM, tree baselines, and
MBDG-RS.

Run all 12 outer folds:

```bash
for fold in {0..11}; do
  python run_classical_baselines.py \
    --target-index "$fold" \
    --output-dir results/classical \
    --cache-dir results/classical_feature_cache
done
```

Aggregate the 12 participants:

```bash
python run_classical_baselines.py \
  --aggregate \
  --output-dir results/classical \
  --aggregate-output results/classical/aggregate.json
```

### 3. MBDG-RS component and sensitivity experiments

Each outer fold contains:

- the fixed 1.5-s representation ablation;
- 0.5/0.75/1.0/1.5-s duration sensitivity;
- 20/16/12/8-channel sensitivity;
- the original single-gain channel-scale stress test.

Run all folds:

```bash
for fold in {0..11}; do
  python run_mbdg_rs_ablations.py \
    --fold "$fold" \
    --output "results/mbdg_rs/fold_$(printf '%02d' "$fold").json"
done
```

Aggregate accuracy and macro-F1 across participants:

```bash
python aggregate_mbdg_rs_results.py \
  --input results/mbdg_rs \
  --output results/mbdg_rs/aggregate.json
```

Compute paired component-ablation statistics:

```bash
python analyze_mbdg_rs_ablations.py \
  --input-dir results/mbdg_rs \
  --output results/mbdg_rs/ablation_statistics.json
```

The statistical output contains participant-level differences, percentile
bootstrap 95% confidence intervals, exact two-sided sign-flip tests, and Holm
adjustment across the four component comparisons.

### 4. Unified Table-2 experiment

The unified Table-2 runner evaluates:

- five fixed single-band dual-geometry representations;
- source-selected Best-1 through Best-5 procedures;
- five-band channel log-power;
- five-band covariance without spectrum;
- broadband and five-band MBDG-RS component branches;
- full MBDG-RS.

Run every configuration for all 12 folds:

```bash
for fold in {0..11}; do
  python run_table2_unified_ablation.py \
    --mode run \
    --model all \
    --fold "$fold" \
    --output "results/table2/fold_$(printf '%02d' "$fold").json"
done
```

Aggregate Table 2:

```bash
python run_table2_unified_ablation.py \
  --mode aggregate \
  --model all \
  --input-dir results/table2 \
  --output results/table2/aggregate.json
```

The aggregate stores the recommended Table-2 rows in
`table2_rows_in_recommended_order`. Best-k selection frequencies and the
combination selected in every outer fold are stored under `model_summaries`.

To run only one fixed row, replace `--model all` with a model ID such as:

```bash
python run_table2_unified_ablation.py \
  --mode run \
  --model cov_lda \
  --fold 0 \
  --output results/table2_cov/fold_00.json
```

Valid fixed-row IDs include `bp_lda`, `cov_lda`,
`five_band_dual_geometry_no_spectrum`, and `full_mbdg_rs`.

### 5. Deep baselines

The deep benchmark contains four architectures:

```text
compactcnn
eegnet
facial1dcnn
cnntcn
```

It uses one source-only inner-LOSO epoch-selection run for every architecture
and outer fold, followed by five final training seeds (`11`, `23`, `37`, `53`,
`71`). CUDA is required.

Generate all 48 selection artifacts:

```bash
for architecture in compactcnn eegnet facial1dcnn cnntcn; do
  for fold in {0..11}; do
    CUDA_VISIBLE_DEVICES=0 python run_deep_baselines.py \
      --mode selection \
      --architecture "$architecture" \
      --target-index "$fold" \
      --selection-dir results/deep_selection
  done
done
```

Run all 240 final fits:

```bash
for architecture in compactcnn eegnet facial1dcnn cnntcn; do
  for fold in {0..11}; do
    for seed in 11 23 37 53 71; do
      CUDA_VISIBLE_DEVICES=0 python run_deep_baselines.py \
        --mode final \
        --architecture "$architecture" \
        --target-index "$fold" \
        --seed "$seed" \
        --selection-dir results/deep_selection \
        --final-dir results/deep_final
    done
  done
done
```

Aggregate the five seeds within each participant and then summarize the 12
participants:

```bash
python aggregate_deep_results.py \
  --final-dir results/deep_final \
  --output results/deep/aggregate.json
```

`aggregate_deep_results.py` expects all four architectures, 12 participants,
and five final seeds. The loops above run sequentially on GPU 0; jobs may be
distributed across multiple GPUs as long as every expected output is produced
exactly once.

### 6. Primary MBDG-RS versus Facial CNN statistics

This step requires the MBDG-RS aggregate, the 12 MBDG-RS fold files, and the
deep-model aggregate produced above.

```bash
python primary_stats.py \
  --mbdg-rs results/mbdg_rs/aggregate.json \
  --mbdg-rs-fold-dir results/mbdg_rs \
  --deep results/deep/aggregate.json \
  --output results/primary_statistics.json
```

The output reports MBDG-RS minus Facial CNN for accuracy and macro-F1 using
participant bootstrap confidence intervals and exact two-sided sign-flip
tests.

### 7. Channel-gain robustness

The full robustness experiment uses 20 positive channel-gain draws at each of
three severities (`0.25`, `0.5`, and `0.75`). The same gain draw is shared
across methods and reused for every trial of a held-out participant.

#### 7.1 Deterministic methods

Run all 12 folds:

```bash
for fold in {0..11}; do
  python scale_mechanism/run.py \
    --model deterministic \
    --fold "$fold" \
    --output "results/scale_mechanism/fold_$(printf '%02d' "$fold").json"
done
```

The deterministic comparison contains six pipelines.

#### 7.2 Facial CNN

Facial CNN robustness reuses the `facial1dcnn__sub-XX.json` selection files
created in the deep-baseline selection stage. It also reuses the exact gain
vectors stored in the deterministic fold outputs.

Run 12 folds and five training seeds:

```bash
for fold in {0..11}; do
  for seed in 11 23 37 53 71; do
    CUDA_VISIBLE_DEVICES=0 python scale_mechanism/run.py \
      --model facial_cnn \
      --fold "$fold" \
      --seed "$seed" \
      --selection-dir results/deep_selection \
      --reference-scale-dir results/scale_mechanism \
      --output "results/scale_mechanism_facialcnn/facial_cnn__fold_$(printf '%02d' "$fold")__seed${seed}.json"
  done
done
```

Aggregate deterministic and Facial CNN robustness:

```bash
python scale_mechanism/aggregate.py \
  --input-dir results/scale_mechanism \
  --facial-cnn-input-dir results/scale_mechanism_facialcnn \
  --output results/scale_mechanism/aggregate.json
```

Aggregation proceeds in this order:

1. average 20 perturbation draws within each Facial CNN training seed;
2. average five seeds within each held-out participant;
3. bootstrap the resulting 12 participant-level values.


```

## Dataset citation

If you use FaceEMG-11, cite:

> Dong, Mochu; Xu, Xiran; Yan, Yujie; Sun, Sinan; Chen, Jing (2026).
> *FaceEMG-11*. Zenodo. https://doi.org/10.5281/zenodo.22137268

```bibtex
@dataset{dong_2026_faceemg11,
  author    = {Dong, Mochu and Xu, Xiran and Yan, Yujie and
               Sun, Sinan and Chen, Jing},
  title     = {FaceEMG-11},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22137268},
  url       = {https://doi.org/10.5281/zenodo.22137268}
}
```
