# python scripts/visualize_latent_pca_grid.py \
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
python scripts/visualize_latent_pca_grid.py \
    --checkpoints \
        weights/f8/model.ckpt \
        "checkpoints/exotic-bald-chupacabra-of-artistry_256x256 50seq hydrant, disc_start=0, vanilla_prob=0.3, warp_w=0.1/last.ckpt" \
    --configs \
        config/baseVAE.yaml \
        config/warp_vae_hydrant_256_disc0_50seq.yaml \
    --model_names "SD-VAE" "Warp-VAE"\
    --co3d_native_dir /visinf/projects_students/dlcv2025_groupZ/co3d_data \
    --categories hydrant\
    --seed 101 \
    --num_objects 1 \
    --views_per_object 5\
    --output eval_outputs/comparison_view.png