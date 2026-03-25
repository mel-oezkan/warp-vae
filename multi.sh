# Analyze your custom model
# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoint checkpoints/massive-accurate-okapi-of-blizzard/last.ckpt \
#     --config config/warp_vae_co3d_precomputed.yaml \
#     --output_name massive-accurate-okapi \
#     --num_objects 50


# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoint checkpoints/pastel-chirpy-grebe-of-speed/last.ckpt \
#     --config config/warp_vae_co3d_small.yaml \
#     --output_name pastel-chirpy-grebe \
#     --num_objects 10 \
#     --image_size 256 \
#     --analyze_sequences \
#     --per_object_plots \
#     --n_detailed_objects 5 \
#     --compare_baseline


# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoint checkpoints/eq-vae/diffusion_pytorch_model.safetensors \
#     --config checkpoints/eq-vae/config.json \
#     --output_name eq-vae \
#     --num_objects 10 \
#     --image_size 256 \
#     --analyze_sequences \
#     --per_object_plots \
#     --n_detailed_objects 5 \
#     --compare_baseline


# # Also compare with f8 baseline
# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoint outputs/my_model/checkpoints/last.ckpt \
#     --config config/my_config.yaml \
#     --output_name my_model_analysis \
#     --num_objects 50 \


# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoint weights/f8/model.ckpt \
#     --config config/baseVAE.yaml \
#     --output_name multiview_sequence_test \
#     --num_objects 10 \
#     --image_size 256 \
#     --analyze_sequences \
#     --per_object_plots \
#     --n_detailed_objects 5

python scripts/analyze_multiview_latent_consistency.py \
    --dataset co3d_native \
    --checkpoints \
        "checkpoints/accurate-courageous-agama-of-authority_EQ-VAE on CO3D hydrant 50seq, step-matched to Warp VAE e4ksa79v (~58K steps)/last.ckpt" \
        "checkpoints/tested-fine-trout-of-authority_hydrant 50seq nocrop, from scratch, warp_w=1, warp_recon_w=1, disc_w=0.5, disc_start=15k/last.ckpt" \
        "checkpoints/gentle-horned-grasshopper-of-serendipity_hydrant 50seq nocrop, toybus-matched hparams: kl=1e-5, warp_w=1, disc_w=0.5, disc_start=15k, grad_accum=4/last.ckpt" \
        weights/f8/model.ckpt \
    --configs \
        config/eqvae_co3d_hydrant_50seq.yaml \
        config/warp_vae_hydrant_recon.yaml \
        config/warp_vae_hydrant.yaml \
        config/baseVAE.yaml \
    --model_names "EQ-VAE" "Warp-VAE (recon)" "Warp-VAE (toybus-hp)" "SD-VAE" \
    --mode roma \
    --roma_setting turbo \
    --roma_confidence_threshold 0.2 \
    --max_distance 3.0 \
    --min_distance 0.5 \
    --co3d_native_dir /visinf/projects_students/dlcv2025_groupZ/co3d_data \
    --output_name full_comparison_co3d_native

# Same eval but without distance filtering (all pairs)
python scripts/analyze_multiview_latent_consistency.py \
    --dataset co3d_native \
    --checkpoints \
        "checkpoints/accurate-courageous-agama-of-authority_EQ-VAE on CO3D hydrant 50seq, step-matched to Warp VAE e4ksa79v (~58K steps)/last.ckpt" \
        "checkpoints/tested-fine-trout-of-authority_hydrant 50seq nocrop, from scratch, warp_w=1, warp_recon_w=1, disc_w=0.5, disc_start=15k/last.ckpt" \
        "checkpoints/gentle-horned-grasshopper-of-serendipity_hydrant 50seq nocrop, toybus-matched hparams: kl=1e-5, warp_w=1, disc_w=0.5, disc_start=15k, grad_accum=4/last.ckpt" \
        weights/f8/model.ckpt \
    --configs \
        config/eqvae_co3d_hydrant_50seq.yaml \
        config/warp_vae_hydrant_recon.yaml \
        config/warp_vae_hydrant.yaml \
        config/baseVAE.yaml \
    --model_names "EQ-VAE" "Warp-VAE (recon)" "Warp-VAE (toybus-hp)" "SD-VAE" \
    --mode roma \
    --roma_setting turbo \
    --roma_confidence_threshold 0.2 \
    --co3d_native_dir /visinf/projects_students/dlcv2025_groupZ/co3d_data \
    --output_name full_comparison_co3d_native_alldist

# # RoMA mode only
# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoints weights/f8/model.ckpt \
#     --configs config/baseVAE.yaml \
#     --model_names "Baseline" \
#     --mode roma \
#     --roma_setting precise \
#     --roma_confidence_threshold 0.8 \
#     --output_name roma_analysis

# # Both global and RoMA analysis
# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoints model1.ckpt model2.ckpt \
#     --configs config1.yaml config2.yaml \
#     --model_names "Model A" "Model B" \
#     --mode both \
#     --output_name full_comparison