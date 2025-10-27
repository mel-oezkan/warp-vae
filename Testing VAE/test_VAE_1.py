import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from ldm.models.autoencoder import AutoencoderKL

config_path = r"..\latent-diffusion\configs\autoencoder\autoencoder_kl_8x8x64.yaml"
config = OmegaConf.load(config_path)

autoencoder = AutoencoderKL(**config.model.params)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)
autoencoder.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# CIFAR-10
num_images = 4 
cifar_train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(cifar_train, batch_size=num_images, shuffle=True)

batch = next(iter(train_loader))
images = batch[0].to(device)

with torch.no_grad():
    z = autoencoder.encode(images)         
    z_sample = z.sample()                 
    recon = autoencoder.decode(z_sample)

images = images.cpu()
recon = recon.cpu()

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(4):
    axes[0, i].imshow(images[i].permute(1, 2, 0))
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')
    
    axes[1, i].imshow(recon[i].permute(1, 2, 0))
    axes[1, i].set_title("Reconstruction")
    axes[1, i].axis('off')

plt.show()