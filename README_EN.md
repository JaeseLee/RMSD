# RMSD: Bidirectional SAR–Optical Translation

[한국어 README](README.md)

This repository contains models and architecture-ablation tools for
bidirectional translation between SAR (`VV`, `VH`, `RVI`) and optical indices
(`MNDWI`, `NDVI`, `NDWI`) using the SEN12-FLOOD dataset.

Raw data, processed NumPy arrays, model checkpoints, and generated figures are
not included in the repository.

## 1. Environment setup

Python 3.9 or newer is recommended.

```bash
pip install numpy pandas matplotlib rasterio scikit-learn torch tqdm
```

The evaluation scripts automatically select MPS on Apple Silicon, CUDA on a
supported NVIDIA system, and CPU otherwise. To select a device explicitly, use
`--device cpu`, `--device mps`, or `--device cuda`.

The expected dataset structure is:

```text
/path/to/SEN12FLOOD/
├── processed/
│   ├── trn_X_VV_VH_RVI_raw.npy
│   ├── trn_Y_MNDWI_NDVI_NDWI_raw.npy
│   ├── tst_X_VV_VH_RVI_raw.npy
│   └── tst_Y_MNDWI_NDVI_NDWI_raw.npy
└── output/
```

## 2. Workflow

```text
Data preprocessing
  → Model training
  → Architecture-ablation training
  → Test metrics and validation maps
  → Validation-loss and test-metric figures
```

### 2.1 Data preprocessing

Check the dataset paths and settings in `RMSD_preprocessing.py`, then run:

```bash
python3 RMSD_preprocessing.py
python3 RMSD_data_check.py
```

### 2.2 Train one bidirectional model

Train the LightAttention model:

```bash
python3 RMSD_model_bidirectional.py \
  --rootpath /path/to/SEN12FLOOD \
  --date_out example_light_attention \
  --model light_attention \
  --base_channels 32 \
  --num_epochs 100 \
  --bs 4 \
  --lr 0.0001 \
  --kl_weight 0.005 \
  --cycle_weight 0.0 \
  --gradient_weight 0.0 \
  --ssim_weight 0.0
```

To train the conventional U-Net control:

```bash
python3 RMSD_model_bidirectional.py \
  --rootpath /path/to/SEN12FLOOD \
  --date_out example_unet \
  --model unet \
  --num_epochs 100
```

Training outputs are stored in:

```text
/path/to/SEN12FLOOD/output/<date_out>/
├── config_<experiment>.json
├── loss_history.csv
├── best_model.pt
├── model_epoch_010.pt
├── ...
├── model_epoch_100.pt
└── validation_figures/
```

### 2.3 Train the architecture-ablation models

```bash
bash RMSD_architecture_ablation.sh /path/to/SEN12FLOOD 42 100
```

The positional arguments are `ROOTPATH`, `SEED`, and `EPOCHS`, respectively.
Outputs are stored under
`/path/to/SEN12FLOOD/output/architecture_ablation_seed_42/`.

| Directory | Condition |
|---|---|
| `arch_B0_unet` | U-Net control |
| `arch_B1_light_full` | Full LightAttention architecture |
| `arch_B2_no_depthwise` | Remove depthwise convolution |
| `arch_B3_no_residual` | Remove residual connections |
| `arch_B4_no_eca` | Remove ECA channel attention |
| `arch_B5_no_skip_attention` | Remove skip attention |
| `arch_B6_no_dilation` | Change bottleneck dilation to 1 |
| `arch_B7_batchnorm` | Replace GroupNorm with BatchNorm |
| `arch_B8_relu6` | Replace SiLU with ReLU6 |

`arch_B2_no_depthwise` produced NaN values in the current experiment and is
therefore excluded from evaluation by default. Add `--include_no_depthwise`
only when this condition should be included.

### 2.4 Evaluate the architecture ablation

The following command evaluates the epoch-100 checkpoints over the complete
test set:

```bash
python3 RMSD_evaluate_architecture_ablation.py \
  --study_dir /path/to/SEN12FLOOD/output/architecture_ablation_seed_42 \
  --rootpath /path/to/SEN12FLOOD \
  --epoch 100 \
  --seed 42 \
  --bs 4
```

The following metrics are calculated for both translation directions and every
output channel:

- Correlation (`corr`)
- Root mean squared error (`rmse`)
- Unbiased RMSD (`ubRMSD`)
- Bias (`bias`)
- Structural Similarity Index (`ssim`)

SSIM is calculated on normalized `[0, 1]` images using an 11×11 local window.
Only windows containing at least 80% valid pixels are included.

The main outputs are:

```text
epoch_100_evaluation/
├── all_test_metrics.csv
├── architecture_mean_test_metrics.csv
├── architecture_test_metrics_X_to_Y.png
├── architecture_test_metrics_Y_to_X.png
├── architecture_test_metrics_bidirectional_hue.png
├── architecture_validation_X_to_Y.png
├── architecture_validation_Y_to_X.png
├── <experiment>_test_metrics.csv
└── <experiment>_validation_maps.png
```

`architecture_test_metrics_bidirectional_hue.png` uses the architecture as the
x-axis and the X→Y and Y→X directions as the hue. RMSE, ubRMSD, and bias have
different physical units in the two directions. These metrics should therefore
be used to compare architectures within the same direction, rather than to
compare the two directions directly.

### 2.5 Generate validation maps only

Use `--maps_only` to evaluate validation sample 0 without recalculating the
complete test metrics:

```bash
python3 RMSD_evaluate_architecture_ablation.py \
  --study_dir /path/to/SEN12FLOOD/output/architecture_ablation_seed_42 \
  --rootpath /path/to/SEN12FLOOD \
  --epoch 100 \
  --maps_only
```

### 2.6 Rebuild metric figures from an existing CSV

The existing `all_test_metrics.csv` must contain an `ssim` column.

```bash
python3 RMSD_evaluate_architecture_ablation.py \
  --study_dir /path/to/SEN12FLOOD/output/architecture_ablation_seed_42 \
  --epoch 100 \
  --plot_only
```

Alternatively, use the standalone plotting script:

```bash
python3 plot_architecture_test_metrics.py \
  /path/to/SEN12FLOOD/output/architecture_ablation_seed_42/epoch_100_evaluation/all_test_metrics.csv \
  --output_dir architecture_ablation_evaluation \
  --epoch 100
```

### 2.7 Plot the validation-loss ablation

This script reads each experiment's `loss_history.csv` and does not run model
inference:

```bash
python3 plot_architecture_ablation.py \
  --study_dir /path/to/SEN12FLOOD/output/architecture_ablation_seed_42 \
  --output_dir architecture_ablation_evaluation
```

The resulting 2×2 figure contains:

- Complete validation-loss trajectories
- Best total validation loss
- X→Y validation loss at the best epoch
- Y→X validation loss at the best epoch

### 2.8 Evaluate one best checkpoint

```bash
python3 RMSD_evaluate_best_bidirectional.py \
  --rootpath /path/to/SEN12FLOOD \
  --date_out example_light_attention \
  --device auto
```

## 3. File descriptions

### Main workflow

| File | Description |
|---|---|
| `RMSD_preprocessing.py` | Converts the raw SEN12-FLOOD data into training and test NumPy arrays. |
| `RMSD_data_check.py` | Checks array shapes, value ranges, and missing values. |
| `rmsd_light_model.py` | Defines the LightAttention U-Net and its depthwise, residual, ECA, and skip-attention components. |
| `RMSD_model_bidirectional.py` | Trains the bidirectional models and saves checkpoints, losses, and validation maps. |
| `RMSD_architecture_ablation.sh` | Runs the enabled architecture conditions sequentially with a shared seed and training configuration. |
| `RMSD_evaluate_ablation_epoch100.py` | Provides common preprocessing, streaming pixel metrics, and masked local SSIM functions. |
| `RMSD_evaluate_architecture_ablation.py` | Generates architecture test metrics, summaries, directional/hue figures, and validation maps. |
| `plot_architecture_ablation.py` | Generates the 2×2 validation-ablation figure from `loss_history.csv`. |
| `plot_architecture_test_metrics.py` | Rebuilds directional and bidirectional-hue figures from an existing metric CSV. |

### Additional and earlier experiment code

| File | Description |
|---|---|
| `RMSD_model.py` | Trains the original one-way SAR→optical U-Net. |
| `RMSD_evaluate_best_bidirectional.py` | Evaluates `best_model.pt` from one bidirectional experiment. |
| `RMSD_ablation.sh` | Provides loss-component ablation examples. |

## 4. Files excluded from GitHub

The following generated or large files should be added to `.gitignore`:

```gitignore
.DS_Store
__pycache__/
*.pyc
*.pt
*.pth
*.npy
processed/
output/
architecture_ablation_evaluation/
```

Store only the source code and documentation in GitHub. Dataset and checkpoint
downloads can be provided through a release or a separate data repository.
