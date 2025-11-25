import os 
import kagglehub

output_path = "/data/lab_moezkan/imagenet-256"
os.makedirs(output_path, exist_ok=True)

path = kagglehub.dataset_download("dimensi0n/imagenet-256", path=output_path)

print("Path to dataset files:", path)