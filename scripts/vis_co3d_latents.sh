# python evaluation/visualize_latents.py \
#     --checkpoint checkpoints/sd-vae2-1/sd-vae-2-1.safetensors \
#     --config checkpoints/sd-vae2-1/config.json \
#     --output_name vanilla2-1 \
#     --dataset_type co3d \
#     --data_dir /data/lab_moezkan/co3d_full \
#     --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz
#     # --checkpoint checkpoints/honest-copper-spider-of-aptitude/last.ckpt \
#     # --output_name my_experiment3 \

# python evaluation/visualize_latents.py \
#     --checkpoint checkpoints/honest-copper-spider-of-aptitude/vae-epochepoch=009.ckpt \
#     --config config/warp_vae_co3d_small.yaml \
#     --output_name honest-copper-spider-ckpt9 \
#     --dataset_type co3d \
#     --data_dir /data/lab_moezkan/co3d_full \
#     --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz
#     checkpoints/versatile-chimpanzee-of-sexy-politeness/last.ckpt

# python evaluation/visualize_latents.py \
#     --checkpoint checkpoints/versatile-chimpanzee-of-sexy-politeness/last.ckpt \
#     --config config/warp_vae_co3d_small.yaml \
#     --output_name versatile-chimpanzee-ckpt_Latest \
#     --dataset_type co3d \
#     --data_dir /data/lab_moezkan/co3d_full \
#     --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz

# python compare_latents.py \
#     --checkpoint checkpoints/pastel-chirpy-grebe-of-speed/last.ckpt \
#     --config config/warp_vae_co3d_small.yaml \
#     --output_name pastel-chirpy

python evaluation/visualize_latents.py \
    --checkpoint checkpoints/pastel-chirpy-grebe-of-speed/last.ckpt \
    --config config/warp_vae_co3d_small.yaml \
    --output_name pastel-chirpy \
    --dataset_type co3d \
    --data_dir /data/lab_moezkan/co3d_full \
    --bb_file /data/lab_moezkan/co3d_bboxes/toybus_test.jgz