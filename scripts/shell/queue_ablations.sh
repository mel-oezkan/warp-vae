#!/bin/bash
# Queue ablation runs (sequential training + eval)
# Ablations 1 (cosine) and 2 (combined) are running on another node.
#
# Usage:
#   tmux new -s ablations
#   source /visinf/home/lab_mozkan/miniconda3/etc/profile.d/conda.sh && conda activate cv
#   cd /visinf/home/lab_mozkan/computer-vision-proj-lab
#   bash scripts/queue_ablations.sh 2>&1 | tee outputs/ablation_queue.log

CD_DIR="/visinf/home/lab_mozkan/computer-vision-proj-lab"
GPUS="1,0"

cd "$CD_DIR"

# --- Ablation 4: Warp-VAE with L2 loss + warp reconstruction loss ---
echo "[$(date)] Starting: warp_vae L2 + warp reconstruction"
CUDA_VISIBLE_DEVICES=$GPUS python train.py --config-name=warp_vae_hydrant \
    trainer.params.consistency_loss_type=l2 \
    trainer.params.warp_reconstruction_weight=1.0 \
    training.batch_size=2 \
    training.output_dir=./outputs/warp_vae_hydrant_l2_warp_recon \
    training.note="ablation: l2 loss + warp reconstruction" \
    || echo "[$(date)] WARNING: L2 + warp recon ablation failed with exit code $?"
echo "[$(date)] Finished: warp_vae L2 + warp reconstruction"

# --- Ablation 6: Warp-VAE with cosine loss + warp reconstruction loss ---
echo "[$(date)] Starting: warp_vae cosine + warp reconstruction"
CUDA_VISIBLE_DEVICES=$GPUS python train.py --config-name=warp_vae_hydrant \
    trainer.params.consistency_loss_type=cosine \
    trainer.params.warp_reconstruction_weight=1.0 \
    training.batch_size=2 \
    training.output_dir=./outputs/warp_vae_hydrant_cosine_warp_recon \
    training.note="ablation: cosine loss + warp reconstruction" \
    || echo "[$(date)] WARNING: cosine + warp recon ablation failed with exit code $?"
echo "[$(date)] Finished: warp_vae cosine + warp reconstruction"

# # --- Ablation: Warp-VAE with bbox-cropped precomputed warps ---
# # Uses warps where images were bbox-cropped BEFORE RoMA matching
# # Filtered to 50seq subset (3745 warp pairs) for fair comparison with nocrop baseline
# echo "[$(date)] Starting: warp_vae bbox-cropped warps"
# CUDA_VISIBLE_DEVICES=$GPUS python train.py --config-name=warp_vae_hydrant \
#     data.params.dataset_config.params.warp_dir=/visinf/projects_students/dlcv2025_groupZ/precomputed_warps/hydrant_50seq_cropped \
#     data.params.dataset_config.params.crop_images=true \
#     training.batch_size=2 \
#     training.output_dir=./outputs/warp_vae_hydrant_bbox_cropped \
#     "training.note='ablation: bbox-cropped warps, crop before RoMA, 50seq subset'" \
#     || echo "[$(date)] WARNING: bbox-cropped ablation failed with exit code $?"
# echo "[$(date)] Finished: warp_vae bbox-cropped warps"

# echo "[$(date)] All ablation runs complete."

