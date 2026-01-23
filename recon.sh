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
#     --checkpoint weights/model.ckpt \
#     --config config/test.yml \
#     --output_name vanilla1-5


# python compare_latents.py \
#     --checkpoint checkpoints/pastel-chirpy-grebe-of-speed/last.ckpt \
#     --config config/warp_vae_co3d_small.yaml \
#     --output_name pastel-chirpy

python compare_latents.py \
    --checkpoint checkpoints/glossy-classic-limpet-of-mathematics/vae-epochepoch=014.ckpt \
    --config config/warp_vae_co3d.yaml \
    --output_name glossy-classic-limpet