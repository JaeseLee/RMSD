#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash RMSD_architecture_ablation.sh [ROOTPATH] [SEED] [EPOCHS]
# Example:
#   bash RMSD_architecture_ablation.sh /Users/jslee/Downloads/SEN12FLOOD 42 100
ROOTPATH="${1:-/Users/jslee/Downloads/SEN12FLOOD}"
SEED="${2:-42}"
EPOCHS="${3:-100}"
OUTPUT_GROUP="architecture_ablation_seed_${SEED}"

# Architecture-only ablation: keep data, seed, optimizer, losses, and epochs fixed.
COMMON_ARGS=(
  --rootpath "$ROOTPATH"
  --seed "$SEED"
  --num_epochs "$EPOCHS"
  --bs 4
  --optimizer AdamW
  --lr 0.0001
  --do 0.0
  --shuffle true
  --loss L1
  --kl_weight 0.005
  --cycle_weight 0.0
  --gradient_weight 0.0
  --ssim_weight 0.00
  --vis_interval 10
)

# U-Net control: same full loss configuration.
# python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
#   --model unet \
#   --date_out "$OUTPUT_GROUP/arch_B0_unet" \
# && python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
#   --model light_attention \
#   --date_out "$OUTPUT_GROUP/arch_B1_light_full" \
# && python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
#   --model light_attention \
#   --date_out "$OUTPUT_GROUP/arch_B2_no_depthwise" \
#   --arch_use_depthwise false \
# && python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
# python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
#   --model light_attention \
#   --date_out "$OUTPUT_GROUP/arch_B3_no_residual" \
#   --arch_use_residual false \
# && python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
#   --model light_attention \
#   --date_out "$OUTPUT_GROUP/arch_B4_no_eca" \
#   --arch_use_eca false \
# && python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
# python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
#   --model light_attention \
#   --date_out "$OUTPUT_GROUP/arch_B5_no_skip_attention" \
#   --arch_use_skip_attention false \
# && python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
  --model light_attention \
  --date_out "$OUTPUT_GROUP/arch_B6_no_dilation" \
  --arch_bottleneck_dilation 1 \
&& python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
  --model light_attention \
  --date_out "$OUTPUT_GROUP/arch_B7_batchnorm" \
  --arch_norm batch \
&& python3 RMSD_model_bidirectional.py "${COMMON_ARGS[@]}" \
  --model light_attention \
  --date_out "$OUTPUT_GROUP/arch_B8_relu6" \
  --arch_activation relu6
