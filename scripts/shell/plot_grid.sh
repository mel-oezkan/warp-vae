# python scripts/visualization/visualize_latent_pca_grid.py \
#     --checkpoints \
#         weights/f8/model.ckpt \
#         # "checkpoints/accurate-courageous-agama-of-authority_EQ-VAE on CO3D hydrant 50seq, step-matched to Warp VAE e4ksa79v (~58K steps)/last.ckpt" \
#         "checkpoints/tested-fine-trout-of-authority_hydrant 50seq nocrop, from scratch, warp_w=1, warp_recon_w=1, disc_w=0.5, disc_start=15k/last.ckpt" \
#     --configs \
#         config/baseVAE.yaml \
#         # config/eqvae_co3d_hydrant_50seq.yaml \
#         config/warp_vae_hydrant_recon.yaml \
#     --model_names "SD-VAE" "EQ-VAE" "Warp-VAE"\
#     --co3d_native_dir /visinf/projects_students/dlcv2025_groupZ/co3d_data \
#     --categories toyplane\
#     --seed 101 \
#     --num_objects 1\
#     --views_per_object 5\
#     --output eval_outputs/comparison_view.png


    # "checkpoints/tested-fine-trout-of-authority_hydrant 50seq nocrop, from scratch, warp_w=1, warp_recon_w=1, disc_w=0.5, disc_start=15k/last.ckpt" \
CUDA_VISIBLE_DEVICES=1 python scripts/visualization/visualize_latent_pca_grid.py \
    --checkpoints \
        "checkpoints/beautiful-emu-of-fortunate-tempest_Vanilla SD-VAE baseline, hydrant 50seq cropped, same setup as warp variants/last.ckpt" \
        "checkpoints/natural-illegal-bullmastiff-from-tartarus_EQ-VAE on CO3D hydrant 50seq cropped, matched to warp_vae_hydrant_recon_crop runs/last.ckpt" \
        "checkpoints/stereotyped-tireless-starfish-of-fame_hydrant 50seq cropped, from scratch, warp_w=0.02, warp_recon_w=0.02, disc_w=0.5, l1 warp consistency/last.ckpt" \
        "checkpoints/gigantic-prophetic-mouse-of-anger_hydrant 50seq cropped, from scratch, warp_w=0.1, warp_recon_w=0.1, disc_w=0.5, disc_start=15k, cosine warp similarity/last.ckpt" \
    --configs \
        config/vanilla_vae_hydrant_crop.yaml \
        config/eqvae_co3d_hydrant_50seq.yaml \
        config/warp_vae_hydrant_recon_crop_l1.yaml \
        config/warp_vae_hydrant_recon_crop_cosine.yaml \
    --model_names "Vanilla SD-VAE" "EQ-VAE" "Warp-VAE (L1)" "Warp-VAE (Cosine)" \
    --co3d_native_dir /visinf/projects_students/dlcv2025_groupZ/co3d_data \
    --categories bench \
    --seed 101 \
    --num_objects 1 \
    --views_per_object 5 \
    --view_sampling consecutive \
    --view_stride 5 \
    --output eval_outputs/comparison_bench.png




# CUDA_VISIBLE_DEVICES=1 python scripts/visualization/visualize_latent_pca_grid.py \
#     --checkpoints \
#         weights/f8/model.ckpt \
#     --configs \
#         config/baseVAE.yaml \
#     --model_names "SD-VAE" "EQ-VAE" "Warp-VAE"\
#     --co3d_native_dir /visinf/projects_students/dlcv2025_groupZ/co3d_data \
#     --categories hydrant \
#     --seed 43 \
#     --num_objects 1 \
#     --views_per_object 5 \
#     --view_sampling consecutive \
#     --view_stride 5 \
#     --output eval_outputs/base_fig67.png


