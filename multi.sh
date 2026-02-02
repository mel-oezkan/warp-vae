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
    --checkpoints \
        checkpoints/massive-accurate-okapi-of-blizzard/last.ckpt \
        checkpoints/eq-vae/diffusion_pytorch_model.safetensors \
        weights/f8/model.ckpt \
    --configs \
        config/warp_vae_co3d_precomputed.yaml \
        config/checkpoints/eq-vae/config.json \
        config/baseVAE.yaml \
    --model_names "WARP-VAE" "EQ-VAE" "SD-VAE" \
    --output_name multi_model_comparison