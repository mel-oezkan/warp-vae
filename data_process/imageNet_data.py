from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

class Imagenet_HF(IterableDataset):

    def __init__(self,split="train",img_size=512,class_filter=None,max_samples=None):
        super().__init__()
        self.dataset=load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
        self.class_filter=class_filter
        self.max_samples=max_samples

        self.trainsform=T.Compose([
            T.Resize((img_size,img_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __iter__(self):
        count=0
        for sample in self.dataset:
            image,label= sample["image"].convert("RGB") ,sample["label"]
            if self.class_filter is not None and label not in self.class_filter:
                continue 
            yield self.trainsform(image),label
            count+=1
            if self.max_samples is not None and count >= self.max_samples:
                break 
        try:
            if hasattr(self.dataset, "_exiter"):
                self.dataset._exiter.__exit__(None, None, None)
        except Exception:
            pass
    
    def save_class_data(self, class_id, output_path , max_samples=None):
        print(f"Streaming ImageNet-1k class: {class_id}")

        print("Counting total samples for class...")
        total_count = 0
        for sample in self.dataset:
            if sample["label"] == class_id:
                total_count += 1
                if max_samples and total_count >= max_samples:
                    total_count = max_samples
                    break
        if total_count == 0:
            print(f"No samples found for class_id={class_id}")
            return

        images =[]
        count=0

        for sample in tqdm(self.dataset, total=total_count , desc=f" Collecting class {class_id}"):
            img,label=sample["image"].convert("RGB"),sample["label"]
            if label == class_id:
                images.append(self.trainsform(img))
                count+=1
                if max_samples and count >= max_samples:
                    break
        
        if len(images)==0:
            print(f"no samples found in class {class_id}")
            return
        torch.save(images,output_path)
        print(f" Saved {len(images)} samples for class {class_id}")






    
