# RMSD

Preprocessing and U-Net modeling scripts for SEN12-FLOOD experiments.

The raw SEN12-FLOOD data, processed NumPy arrays, model checkpoints, and generated figures are intentionally not tracked in git.

You can prepare training, test data using RMSD_preprocessing

Then you can train models:
  SAR -> MS using RMSD_model.py
  MS <-> SAR using RMSD_model_bidirectional.py

Some codes are prepared to check data in RMSD_data_check.py
