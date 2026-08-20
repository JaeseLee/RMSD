# RMSD: Bidirectional SAR–Optical Translation

[English version](README_EN.md)

SEN12-FLOOD 자료를 이용해 SAR(`VV`, `VH`, `RVI`)과 optical index
(`MNDWI`, `NDVI`, `NDWI`) 사이를 양방향으로 변환하는 모델과 architecture
ablation study 코드입니다.

원본 데이터, 전처리된 NumPy 배열, 모델 checkpoint 및 생성된 그림은 저장소에
포함하지 않습니다.

## 1. 환경 설정

Python 3.9 이상을 권장합니다.

```bash
pip install numpy pandas matplotlib rasterio scikit-learn torch tqdm
```

전처리에는 `pandas`와 `rasterio`, 데이터 점검 및 학습에는 `scikit-learn`도
사용됩니다.

평가 시 Apple Silicon에서는 MPS, NVIDIA 환경에서는 CUDA, 그 외 환경에서는
CPU가 자동 선택됩니다. 직접 지정하려면 `--device cpu`, `--device mps` 또는
`--device cuda`를 사용합니다.

데이터 폴더는 다음 구조를 사용합니다.

```text
/path/to/SEN12FLOOD/
├── processed/
│   ├── trn_X_VV_VH_RVI_raw.npy
│   ├── trn_Y_MNDWI_NDVI_NDWI_raw.npy
│   ├── tst_X_VV_VH_RVI_raw.npy
│   └── tst_Y_MNDWI_NDVI_NDWI_raw.npy
└── output/
```

## 2. 전체 실행 순서

```text
데이터 전처리
  → 모델 학습
  → architecture ablation 실행
  → test metric 및 지도 계산
  → validation-loss/test-metric 그림 생성
```

### 2.1 데이터 전처리

`RMSD_preprocessing.py`의 데이터 경로와 설정을 확인한 후 실행합니다.

```bash
python3 RMSD_preprocessing.py
python3 RMSD_data_check.py
```

### 2.2 단일 양방향 모델 학습

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

기존 U-Net을 사용하려면 다음처럼 지정합니다.

```bash
python3 RMSD_model_bidirectional.py \
  --rootpath /path/to/SEN12FLOOD \
  --date_out example_unet \
  --model unet \
  --num_epochs 100
```

학습 결과는 다음 위치에 저장됩니다.

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

### 2.3 Architecture ablation 학습

```bash
bash RMSD_architecture_ablation.sh /path/to/SEN12FLOOD 42 100
```

세 positional argument는 순서대로 `ROOTPATH`, `SEED`, `EPOCHS`입니다. 결과는
`/path/to/SEN12FLOOD/output/architecture_ablation_seed_42/`에 저장됩니다.

| 폴더 | 조건 |
|---|---|
| `arch_B0_unet` | U-Net control |
| `arch_B1_light_full` | 전체 LightAttention architecture |
| `arch_B2_no_depthwise` | depthwise convolution 제거 |
| `arch_B3_no_residual` | residual connection 제거 |
| `arch_B4_no_eca` | ECA channel attention 제거 |
| `arch_B5_no_skip_attention` | skip attention 제거 |
| `arch_B6_no_dilation` | bottleneck dilation을 1로 변경 |
| `arch_B7_batchnorm` | GroupNorm을 BatchNorm으로 변경 |
| `arch_B8_relu6` | SiLU를 ReLU6로 변경 |

`arch_B2_no_depthwise`는 현재 실험에서 NaN이 발생했으므로 평가에서 기본적으로
제외됩니다. 필요한 경우에만 `--include_no_depthwise`를 추가합니다.

### 2.4 Architecture ablation 평가

```bash
python3 RMSD_evaluate_architecture_ablation.py \
  --study_dir /path/to/SEN12FLOOD/output/architecture_ablation_seed_42 \
  --rootpath /path/to/SEN12FLOOD \
  --epoch 100 \
  --seed 42 \
  --bs 4
```

다음 metric을 X→Y, Y→X 및 각 채널별로 계산합니다.

- Correlation (`corr`)
- RMSE (`rmse`)
- Unbiased RMSD (`ubRMSD`)
- Bias (`bias`)
- Structural Similarity Index (`ssim`)

SSIM은 정규화된 `[0, 1]` 영상에서 11×11 local window로 계산하며 유효 픽셀이
80% 이상인 window만 포함합니다.

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

`architecture_test_metrics_bidirectional_hue.png`에서는 architecture를 x축으로
사용하고 X→Y와 Y→X를 hue로 표시합니다. 두 방향의 RMSE, ubRMSD 및 Bias는
물리 단위가 다르므로 동일 방향 내 architecture 비교용으로 해석해야 합니다.

### 2.5 지도만 생성

test metric을 다시 계산하지 않고 validation sample 0의 지도만 생성합니다.

```bash
python3 RMSD_evaluate_architecture_ablation.py \
  --study_dir /path/to/SEN12FLOOD/output/architecture_ablation_seed_42 \
  --rootpath /path/to/SEN12FLOOD \
  --epoch 100 \
  --maps_only
```

### 2.6 기존 CSV로 metric 그림만 다시 생성

`all_test_metrics.csv`에 `ssim` 열이 있어야 합니다.

```bash
python3 RMSD_evaluate_architecture_ablation.py \
  --study_dir /path/to/SEN12FLOOD/output/architecture_ablation_seed_42 \
  --epoch 100 \
  --plot_only
```

독립 plotting script를 사용할 수도 있습니다.

```bash
python3 plot_architecture_test_metrics.py \
  /path/to/SEN12FLOOD/output/architecture_ablation_seed_42/epoch_100_evaluation/all_test_metrics.csv \
  --output_dir architecture_ablation_evaluation \
  --epoch 100
```

### 2.7 Validation-loss ablation 그림

모델 추론 없이 각 실험의 `loss_history.csv`만 읽습니다.

```bash
python3 plot_architecture_ablation.py \
  --study_dir /path/to/SEN12FLOOD/output/architecture_ablation_seed_42 \
  --output_dir architecture_ablation_evaluation
```

생성되는 2×2 그림은 다음을 포함합니다.

- 전체 validation-loss trajectory
- Best total validation loss
- Best epoch에서의 X→Y validation loss
- Best epoch에서의 Y→X validation loss

### 2.8 단일 best checkpoint 평가

```bash
python3 RMSD_evaluate_best_bidirectional.py \
  --rootpath /path/to/SEN12FLOOD \
  --date_out example_light_attention \
  --device auto
```

## 3. 파일별 설명

### 주요 workflow

| 파일 | 설명 |
|---|---|
| `RMSD_preprocessing.py` | 원본 SEN12-FLOOD 자료를 학습/test NumPy 배열로 전처리합니다. |
| `RMSD_data_check.py` | 전처리된 배열의 shape, 값 범위 및 결측치를 점검합니다. |
| `rmsd_light_model.py` | LightAttention U-Net과 depthwise/residual/ECA/skip-attention 구성요소를 정의합니다. |
| `RMSD_model_bidirectional.py` | 양방향 모델을 학습하고 checkpoint, loss 및 validation 지도를 저장합니다. |
| `RMSD_architecture_ablation.sh` | 동일 seed와 설정으로 활성화된 architecture 조건들을 순차 실행합니다. |
| `RMSD_evaluate_ablation_epoch100.py` | 공통 전처리, streaming pixel metric 및 masked local SSIM 함수를 제공합니다. |
| `RMSD_evaluate_architecture_ablation.py` | Architecture test metric, summary, 방향별/hue 그림과 validation 지도를 생성합니다. |
| `plot_architecture_ablation.py` | `loss_history.csv`로 2×2 validation ablation 그림을 생성합니다. |
| `plot_architecture_test_metrics.py` | 기존 metric CSV로 방향별 및 bidirectional hue 그림을 다시 생성합니다. |

### 추가·이전 실험 코드

| 파일 | 설명 |
|---|---|
| `RMSD_model.py` | 단방향 SAR→optical U-Net 학습 코드입니다. |
| `RMSD_evaluate_best_bidirectional.py` | 단일 양방향 실험의 `best_model.pt`를 평가합니다. |
| `RMSD_ablation.sh` | Architecture가 아닌 loss-component ablation 실행 예시입니다. |

## 4. GitHub에 포함하지 않을 파일

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

GitHub에는 코드와 README를 저장하고, 데이터와 checkpoint는 release 또는 별도
데이터 저장소 링크로 제공하는 방식을 권장합니다.
