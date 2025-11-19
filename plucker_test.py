from data_process.plucker import plucker_encodeing
from data_process.co3d_dataset import CO3D_Dataset,co3d_collate_func
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch
import gzip
import json

import os

ROOT_DIR="../data/co3d_data"
BBOX_DIR="../data/co3d_v2_annotations"
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#prepare category "cake"

cake_dataset=CO3D_Dataset(root_dir=ROOT_DIR,
                          bbox_dir=BBOX_DIR,
                          category="cake",
                          subset_name="dev_0")

cake_dataloader=DataLoader(dataset=cake_dataset,batch_size=8,collate_fn=co3d_collate_func)

samples=next(iter(cake_dataloader))

cropped_images=samples["cropped_image"]
original_size=samples["original_size"][0]
cropped_size=samples["cropped_size"][0]
crop_params=samples["crop_params"]
Rs=samples["R"]
Ts=samples["T"]
pps=samples["principle_point"]
fls=samples["focal_length"]

# print(cropped_images.shape)
# print(original_size)
# print(cropped_size)
# print(crop_params)
# print(Rs.shape)
# print(Ts.shape)
# print(pps.shape)
# print(fls.shape)

#patch encoding
pluckers_p=[]
for R,T,pp,fl,crop_param in zip(Rs, Ts, pps, fls, crop_params ):
    plucker=plucker_encodeing(R,T,fl,pp,crop_param,
                            original_size=original_size,
                            cropped_size=cropped_size,
                            device=device,
                            patch_num=16
                            )
    pluckers_p.append(plucker)

print(len(pluckers_p))
print(pluckers_p[0].shape)

#visualize

# take the 0th and the 4th camera pose to test
ds_0=[]
ms_0=[]
ds_4=[]
ms_4=[]
pluckers_of_firstIm=pluckers_p[0]
pluckers_of_lastIm=pluckers_p[4]

for plucker_0, plucker_4 in zip(pluckers_of_firstIm,pluckers_of_lastIm):
    #print(plucker.shape)
    ds_0.append(plucker_0[:3])
    ms_0.append(plucker_0[3:])

    ds_4.append(plucker_4[:3])
    ms_4.append(plucker_4[3:])
# print(ds[1].shape)
# print(ms[1].shape)

R_sample_4=Rs[4]
T_sample_4=Ts[4]

R_sample_0=Rs[0]
T_sample_0=Ts[0]

print(f'4th R: {R_sample_4},T:{T_sample_4}')
print(f'0th R: {R_sample_0},T:{T_sample_0}')



C_4= - T_sample_4 @ torch.transpose(R_sample_4,0,1)
C_0= - T_sample_0 @ torch.transpose(R_sample_0,0,1)

print(f'C_4:{C_4}, C_0:{C_0}')


fig=plt.figure()

ax= fig.add_subplot(111,projection='3d')
for d in ds_4:
    P0= C_4
    P1= C_4 + 0.5 *d
    ax.plot([P0[0], P1[0]] , [P0[1] , P1[1]] , [P0[2] , P1[2]],color="g") #green

for d in ds_0:
    P0= C_0
    P1= C_0 + 0.5 *d
    ax.plot([P0[0], P1[0]] , [P0[1] , P1[1]] , [P0[2] , P1[2]],color="b") #blue



ax.scatter([C_0[0],C_4[0]], [C_0[1],C_4[1]] ,[C_0[2], C_4[2]] , c="red")

plt.savefig("base_results/plucer_visualize.png",bbox_inches='tight', pad_inches=0, dpi=300)