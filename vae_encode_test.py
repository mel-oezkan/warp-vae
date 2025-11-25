import torch
import pytorch_lightning as pl
from tqdm import tqdm
from data_process.co3d_dataset import CO3D_Dataset , spherical_interpolation
from ldm.models.autoencoder import AutoencoderKL
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from matplotlib import pyplot as plt
import psutil

import os
#dataset dir
#ROOT_DIR=os.path.join(os.getcwd(),"data","co3d_data")
ROOT_DIR="../data/co3d_data"
BBOX_DIR="../data/co3d_v2_annotations"


# read config
config=OmegaConf.load("config/vae.yml")

#create the vae

device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")

transform=T.Compose([
    T.Resize((512,512),antialias=True),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
])

ckpt = torch.load("../weight/512-base-ema.ckpt", map_location="cpu")
vae_state_dict = {
    k.replace("first_stage_model.", ""): v
    for k, v in ckpt["state_dict"].items()
    if "first_stage_model." in k
}
vae=AutoencoderKL(
    ddconfig=config.model.params.ddconfig,
    lossconfig=config.model.params.lossconfig,
    embed_dim=config.model.params.embed_dim,

)
vae.load_state_dict(vae_state_dict,strict=False)

vae.eval().to(device)

bench_dataset=CO3D_Dataset(root_dir=ROOT_DIR,
                           bbox_dir=BBOX_DIR,
                           
                           category="cake",
                           subset_name="dev_0",
                           transform=transform
                           )


print("Memory usage after prepare info:", psutil.Process(os.getpid()).memory_info().rss / 1024**2, "MB")




#interpolate the latent space
first_img= bench_dataset[29]["image"].unsqueeze(0)
second_img=bench_dataset[39]["image"].unsqueeze(0)

first_img,second_img=first_img.to(device=device, dtype=torch.float32),second_img.to(device=device, dtype=torch.float32)

with torch.no_grad():
    z1=vae.encode(first_img).sample()
    z2=vae.encode(second_img).sample()

steps=10
alphas=np.linspace(0,1,steps)

z_spherical_inter=spherical_interpolation(z1=z1, z2=z2,alphas=alphas)

#decode
print("Memory usage before decode:", psutil.Process(os.getpid()).memory_info().rss / 1024**2, "MB")

decoded_imgs_spherical=[]

with torch.no_grad():
    for z in tqdm(z_spherical_inter,desc="spherical interpolation"):
        img_recover=vae.decode(z)
        decoded_imgs_spherical.append(img_recover)
        torch.mps.empty_cache()

print("Memory usage after decode before draw plt:", psutil.Process(os.getpid()).memory_info().rss / 1024**2, "MB")


fig2, axes2 = plt.subplots(1, steps, figsize=(steps * 3, 3))

for i,img in enumerate(decoded_imgs_spherical):
    img_np=img.detach().cpu().squeeze(0)

    img_np=(img_np *0.5 +0.5).clip(0,1)

    img_np=img_np.permute(1,2,0).numpy().astype("float32")
    axes2[i].imshow(img_np)
    axes2[i].axis("off")
plt.tight_layout(pad=0) 
plt.subplots_adjust(wspace=0, hspace=0)  


plt.savefig("base_results/interpolate_firset2end.png", bbox_inches='tight', pad_inches=0, dpi=300)

torch.mps.empty_cache()