# Analyze your custom model
# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoint checkpoints/massive-accurate-okapi-of-blizzard/last.ckpt \
#     --config config/warp_vae_co3d_precomputed.yaml \
#     --output_name massive-accurate-okapi \
#     --num_objects 50


python scripts/analyze_multiview_latent_consistency.py \
    --checkpoint checkpoints/pastel-chirpy-grebe-of-speed/last.ckpt \
    --config config/warp_vae_co3d_small.yaml \
    --output_name pastel-chirpy-grebe \
    --num_objects 50 \
    --compare_baseline


# # Also compare with f8 baseline
# python scripts/analyze_multiview_latent_consistency.py \
#     --checkpoint outputs/my_model/checkpoints/last.ckpt \
#     --config config/my_config.yaml \
#     --output_name my_model_analysis \
#     --num_objects 50 \
