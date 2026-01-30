# python compare_latents.py \
#     --checkpoint checkpoints/versatile-chimpanzee-of-sexy-politeness/last.ckpt \
#     --config config/warp_vae_co3d_small.yaml \
#     --output_name versatile-chimpanzee-ckpt_Latest 

# python compare_latents.py \
#     --checkpoint checkpoints/honest-copper-spider-of-aptitude/last.ckpt \
#     --config config/warp_vae_co3d_small.yaml \
#     --output_name honest-copper


# python compare_latents.py \
#     --checkpoint checkpoints/sd-vae2-1/sd-vae-2-1.safetensors \
#     --config checkpoints/sd-vae2-1/config.json \
#     --output_name vanilla2-1 

# python compare_latents.py \
#     --checkpoint weights/f8/model.ckpt \
#     --config config/baseVAE.yaml \
#     --output_name vanilla-1.5_f8


# python compare_latents.py \
#     --checkpoint checkpoints/pastel-chirpy-grebe-of-speed/last.ckpt \
#     --config config/warp_vae_co3d_small.yaml \
#     --output_name pastel-chirpy

# ------------------------------------------------
#             warp_vae_co3d_precomputed
# ------------------------------------------------

# python compare_latents.py \
#     --checkpoint checkpoints/slick-sidewinder-of-simple-justice/last.ckpt \
#     --config config/warp_vae_co3d_precomputed.yaml \
#     --output_name slick-sidewinder


python compare_latents.py \
    --checkpoint checkpoints/massive-accurate-okapi-of-blizzard/last.ckpt \
    --config config/warp_vae_co3d_precomputed.yaml \
    --output_name massive-accurate-okapi