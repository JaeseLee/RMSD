#!/bin/bash
set -e
python3 RMSD_model_bidirectional.py \
  --model light_attention \
  --date_out ablation_A5_full \
  --seed 42 \
  --kl_weight 0.005 \
  --cycle_weight 0.1 \
  --gradient_weight 0.1 \
  --ssim_weight 0.05 \
  --vis_interval 5

# python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A2_gradient \
#   --seed 42 \
#   --kl_weight 0 \
#   --cycle_weight 0 \
#   --gradient_weight 0.1 \
#   --ssim_weight 0 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A3_ssim \
#   --seed 42 \
#   --kl_weight 0 \
#   --cycle_weight 0 \
#   --gradient_weight 0 \
#   --ssim_weight 0.05 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A4_kl \
#   --seed 42 \
#   --kl_weight 0.005 \
#   --cycle_weight 0 \
#   --gradient_weight 0 \
#   --ssim_weight 0 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A5_full \
#   --seed 42 \
#   --kl_weight 0.005 \
#   --cycle_weight 0.1 \
#   --gradient_weight 0.1 \
#   --ssim_weight 0.05 \
#   --vis_interval 5

# python3 RMSD_model_bidirectional.py \
#   --model light_attention \
#   --date_out ablation_A4_kl \
#   --seed 42 \
#   --kl_weight 0.005 \
#   --cycle_weight 0 \
#   --gradient_weight 0 \
#   --ssim_weight 0 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model light_attention \
#   --date_out ablation_A5_full \
#   --seed 42 \
#   --kl_weight 0.005 \
#   --cycle_weight 0.1 \
#   --gradient_weight 0.1 \
#   --ssim_weight 0.05 \
#   --vis_interval 5

# python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A0_direct \
#   --seed 42 \
#   --kl_weight 0 \
#   --cycle_weight 0 \
#   --gradient_weight 0 \
#   --ssim_weight 0 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A1_cycle \
#   --seed 42 \
#   --kl_weight 0 \
#   --cycle_weight 0.1 \
#   --gradient_weight 0 \
#   --ssim_weight 0 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A2_gradient \
#   --seed 42 \
#   --kl_weight 0 \
#   --cycle_weight 0 \
#   --gradient_weight 0.1 \
#   --ssim_weight 0 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A3_ssim \
#   --seed 42 \
#   --kl_weight 0 \
#   --cycle_weight 0 \
#   --gradient_weight 0 \
#   --ssim_weight 0.05 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A4_kl \
#   --seed 42 \
#   --kl_weight 0.005 \
#   --cycle_weight 0 \
#   --gradient_weight 0 \
#   --ssim_weight 0 \
#   --vis_interval 5 \
# && python3 RMSD_model_bidirectional.py \
#   --model unet \
#   --date_out unet_ablation_A5_full \
#   --seed 42 \
#   --kl_weight 0.005 \
#   --cycle_weight 0.1 \
#   --gradient_weight 0.1 \
#   --ssim_weight 0.05 \
#   --vis_interval 5