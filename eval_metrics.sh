#!/bin/bash
# Evaluate g-FID, s-FID, LPIPS, MSE, PSNR, SSIM on ImageNet-256-10k subset
#
# Uses 10k-image subset of ImageNet-256 for faster evaluation

# python evaluation/evaluate_eqvae.py \
#     --checkpoints \
#         "checkpoints/accurate-courageous-agama-of-authority_EQ-VAE on CO3D hydrant 50seq, step-matched to Warp VAE e4ksa79v (~58K steps)/last.ckpt" \
#         "checkpoints/tested-fine-trout-of-authority_hydrant 50seq nocrop, from scratch, warp_w=1, warp_recon_w=1, disc_w=0.5, disc_start=15k/last.ckpt" \
#         "checkpoints/gentle-horned-grasshopper-of-serendipity_hydrant 50seq nocrop, toybus-matched hparams: kl=1e-5, warp_w=1, disc_w=0.5, disc_start=15k, grad_accum=4/last.ckpt" \
#         weights/f8/model.ckpt \
#     --configs \
#         config/eval_imagenet256_10k.yaml \
#         config/eval_imagenet256_10k.yaml \
#         config/eval_imagenet256_10k.yaml \
#         config/eval_imagenet256_10k.yaml \
#     --model_names "EQ-VAE" "Warp-VAE (recon)" "Warp-VAE (toybus-hp)" "SD-VAE" \
#     --data_config config/eval_imagenet256_10k.yaml \
#     --output_dir evaluation_outputs/imagenet256_10k_comparison \
#     --batch_size 8 \
#     --num_fid_samples 5000 \
#     --num_workers 4



python evaluation/evaluate_eqvae.py \
    --checkpoints \
        "checkpoints/energetic-gleaming-raven-from-camelot_Run A: 128x128, warp_w=0.1, warp_recon=0.1, disc_start=15k, kl=1e-6/last.ckpt" \
    --configs \
        config/warp_vae_hydrant_recon_small.yaml \
    --model_names "Warp-VAE (128x128, small)" \
    --data_config config/eval_imagenet128_10k.yaml \
    --output_dir evaluation_outputs/warp_vae_hydrant_recon_small \
    --batch_size 32 \
    --num_fid_samples 5000 \
    --num_workers 4
