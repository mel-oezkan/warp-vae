#!/bin/bash
# Queue ablation runs with warp reconstruction enabled (sequential training)
#
# Usage:
#   tmux new -s ablations-recon
#   source /visinf/home/lab_mozkan/miniconda3/etc/profile.d/conda.sh && conda activate cv
#   cd /visinf/home/lab_mozkan/computer-vision-proj-lab
#   bash scripts/queue_ablations_recon.sh 2>&1 | tee outputs/ablation_recon_queue.log

CD_DIR="/visinf/home/lab_mozkan/computer-vision-proj-lab"

cd "$CD_DIR"

# --- Ablation: Warp-VAE with L2 loss + reconstruction ---
echo "[$(date)] Starting: warp_vae L2 loss type + reconstruction"
CUDA_VISIBLE_DEVICES=1,0 python train.py --config-name=warp_vae_hydrant \
    trainer.params.consistency_loss_type=l2 \
    trainer.params.warp_reconstruction_weight=1.0 \
    training.batch_size=1 \
    training.output_dir=./outputs/warp_vae_hydrant_l2_recon \
    "training.note=l2+recon" \
    || echo "[$(date)] WARNING: L2 + reconstruction ablation failed with exit code $?"
echo "[$(date)] Finished: warp_vae L2 loss type + reconstruction"

# --- Ablation: Warp-VAE with cosine loss + reconstruction ---
echo "[$(date)] Starting: warp_vae cosine loss type + reconstruction"
CUDA_VISIBLE_DEVICES=1,0 python train.py --config-name=warp_vae_hydrant \
    trainer.params.consistency_loss_type=cosine \
    trainer.params.warp_reconstruction_weight=1.0 \
    training.batch_size=1 \
    training.output_dir=./outputs/warp_vae_hydrant_cosine_recon \
    "training.note=cosine+recon" \
    || echo "[$(date)] WARNING: cosine + reconstruction ablation failed with exit code $?"
echo "[$(date)] Finished: warp_vae cosine loss type + reconstruction"

echo "[$(date)] All ablation runs complete."
