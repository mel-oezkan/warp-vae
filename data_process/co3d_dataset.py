from typing import Optional
from torch.utils.data import Dataset,DataLoader
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms as transforms
from tqdm import tqdm
from typing import List
from data_process.co3d.co3d.dataset.data_types import load_dataclass_jgzip, FrameAnnotation, SequenceAnnotation
import json
import warnings

class CO3D_Dataset(Dataset):
    def __init__(self,
                 root_dir:str,
                 category:str,
                 subset_name:str, #"test_0" / "dev_0" / "dev_1"
                 subset:str ="train", #"train"/"test"/"val"
                 transform=None,
                 ) :
        super().__init__()
        self.root_dir=root_dir
        self.category=category
        self.transform=transform

        #load the frame_annotation of the category
        frame_ann_path=os.path.join(root_dir,category,"frame_annotations.jgz")
        self.frames_ann=load_dataclass_jgzip(frame_ann_path,
                                             List[FrameAnnotation])

        #load the sequence_annotation of the category
        sequence_ann_path=os.path.join(root_dir,category,"sequence_annotations.jgz")
        self.seq_ann=load_dataclass_jgzip(sequence_ann_path,List[SequenceAnnotation])
        self.seq_dict={seq.sequence_name: seq for seq in self.seq_ann}


        #filter the subset
        set_files = [
            f for f in os.listdir(os.path.join(root_dir, category,"set_lists"))
            if f.startswith("set_lists_manyview_") and f.endswith(".json")
        ]
        if not set_files:
            raise RuntimeError(f"No set_lists_manyview_*.json files found in {category}.")


        subset_file=f"set_lists_manyview_{subset_name}.json"
        if subset_file not in set_files:
            warnings.warn(f"set_lists_manyview_{subset_name}.json doesn't exist,choose the test sequence(no pointcloud)")

            subset_file=f"set_lists_manyview_test_0.json"

        sub_type=str(subset_file).split('_')[3]
        if sub_type=="dev":
            self.is_dev=True
        else:
            self.is_dev=False

        set_list_path=os.path.join(root_dir,category,"set_lists",subset_file)

        with open(set_list_path,'r')as f:
             set_lists = json.load(f)
        subset_items = set_lists.get(subset, [])

        self.sequence_name=subset_items[0][0]

        #create the list of frames
        self.samples=[]
        for(_,frame_number,img_path) in subset_items:
            full_img_path=os.path.join(root_dir,img_path)
            self.samples.append((frame_number,full_img_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        frame_num,img_path=self.samples[index]

        #image
        image=Image.open(img_path).convert("RGB")
        if self.transform:
            image=self.transform(image)
        else:                        #(H,W,C).       (C,H,W)
            image=torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0 #value in [0,1]

        #mask
        mask_path=img_path.replace("/images/","/masks/").replace("jpg","png")
        mask=Image.open(mask_path) if os.path.exists(mask_path) else None
        mask= torch.from_numpy(np.array(mask)).unsqueeze(0).float() /255.0 if mask else None

        #depth
        depth_path=img_path.replace("/images/","/depths/")
        depth_path= str(depth_path) + ".geometric.png"
        depth=Image.open(depth_path) if os.path.exists(depth_path) else None
        depth=torch.from_numpy(np.array(depth)).unsqueeze(0).float() if depth else None

        #depth mask
        depth_mask_path=img_path.replace("/images/","/depth_masks/").replace(".jpg",".png")
        depth_mask=Image.open(depth_mask_path) if os.path.exists(depth_mask_path) else None
        depth_mask= torch.from_numpy(np.array(depth_mask)).unsqueeze(0).float() /255.0 if depth_mask else None

        #viewpoint
        frame= next(
            (f for f in self.frames_ann if f.sequence_name==self.sequence_name and f.frame_number==frame_num) ,None
            )

        if frame is not None:
            vp=frame.viewpoint
            R=torch.tensor(vp.R,dtype=torch.float32)
            T=torch.tensor(vp.T,dtype=torch.float32)
            focal_length=torch.tensor(vp.focal_length,dtype=torch.float32)
            principle_point=torch.tensor(vp.principal_point,dtype=torch.float32)
            scale_adjustment=frame.depth.scale_adjustment
        else:
            R=T=focal_length=principle_point=scale_adjustment=None

        #pointcloud

        if not self.is_dev:
            pointcloud=None
        else:
            seq_ann=self.seq_dict.get(self.sequence_name)
            if seq_ann and seq_ann.point_cloud:
                import open3d
                pc_path = seq_ann.point_cloud.path
                pc_score = seq_ann.point_cloud.quality_score
                pc_num_points = seq_ann.point_cloud.n_points

                pcd_full_path = os.path.join(self.root_dir,pc_path)
                if os.path.exists(pcd_full_path):
                    pcd=open3d.io.read_point_cloud(pcd_full_path)
                    points=np.asarray(pcd.points)
                    pointcloud=torch.from_numpy(points).float()
                else:
                    pointcloud=None

            else:
                pointcloud=None

        return {
            "image": image,
            "mask" : mask,
            "depth": depth,
            "scale_adjustment":scale_adjustment,
            "depth_mask": depth_mask,
            "R": R,
            "T": T,
            "focal_length": focal_length,
            "principle_point": principle_point,
            "pointcloud":pointcloud,
            "sequence_name": self.sequence_name,
            "frame_numer": frame_num,
        }
    
    def get_all_images(self):
        all_images=[]
        iterator=tqdm(range(len(self)),desc="Loading all images")

        for i in iterator:
            sample=self[i]
            img=sample['image']
            all_images.append(img)
        return all_images

    
    def get_all_R(self):
        all_R=[]
        iterator=tqdm(range(len(self)),desc="Loading all R")
        for i in iterator:
            sample=self[i]
            R=sample["R"]
            all_R.append(R)
        return all_R

    def get_all_T(self):
        all_T=[]
        iterator=tqdm(range(len(self)),desc="Loading all T")
        for i in iterator:
            sample=self[i]
            T=sample["T"]
            all_T.append(T)
        return all_T

    def get_all_RT(self):
        all_RT=[]
        iterator=tqdm(range(len(self)),desc="Loading all R,T")
        for i in iterator:
            sample=self[i]
            R=sample["R"]
            T=sample["T"]
            all_RT.append({"R":R , "T": T})
        return all_RT


def co3d_collate_func(batch):
    collated={}
    keys=batch[0].keys()
    for key in keys:
        values=[b[key] for b in batch]
        if isinstance(values[0],torch.Tensor):
            try:
                collated[key]=torch.stack(values)
            except:
                collated[key]=values

        else:
            collated[key]=values
    return collated


#calculate depth features of a frame
def calc_depth_features(depth:torch.Tensor,
                    scale:float,
                    depth_mask:torch.Tensor):
    depth=depth.squeeze()

    if depth_mask is not None :
        mask=depth_mask.squeeze()
    else:
        mask=torch.ones_like(depth)
    depth_meters= depth * scale

    valid_pixels=(mask > mask.mean() * 0.1) &(depth_meters>0)
    if valid_pixels.sum()==0:
        return float('nan')
    mean_depth=depth_meters[valid_pixels].mean().item()
    std_depth=depth_meters[valid_pixels].std().item()
    median_depth=depth_meters[valid_pixels].median().item()
    


    return [mean_depth, std_depth, median_depth]


def linear_interpolation(z1:torch.Tensor,z2:torch.Tensor,alphas):
    return [(1 - a) * z1 + a * z2 for a in alphas]


def spherical_interpolation(z1:torch.Tensor,z2:torch.Tensor,alphas):
    def calc(t,x1,x2):
        x1_norm=x1 / torch.norm(x1)
        x2_norm=x2 / torch.norm(x2)

        dot=torch.sum(x1_norm * x2_norm)

        theta= torch.acos(dot)
        sin_theta=torch.sin(theta)
        if sin_theta==0:
            return (1-t) * x1 + t * x2
        return (torch.sin((1- t) *theta)/ sin_theta) * x1 + (torch.sin(t * theta) / sin_theta) *x2

    return [ calc(a, z1, z2) for a in alphas]






