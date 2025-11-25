from torchvision.transforms.functional import crop
import torch

def update_intrinsics_after_crop(
    focal_length,
    principle_point,
    crop_params,
    original_image_size,
    cropped_image_size,
):
    """
    focal_length: (fx,fy)
    principle_point : (cx,cy)
    crop_params: [-cc[0], -cc[1], crop_width, s]
    original_image_size: tuple(orig_h, orig_w)
    cropped_image_size: tuple(cropped_h, cropped_w) final resized cropped window
    """
    orig_h, orig_w=original_image_size[0],original_image_size[1]

    cropped_h, cropped_w=cropped_image_size[0],cropped_image_size[1]

    tx, ty, crop_width, s= crop_params.tolist()

    length=max(orig_h,orig_w)

    pad_x= (length - orig_w) /2.0
    pad_y= (length - orig_h) /2.0

    fx=focal_length[0]
    fy=focal_length[1]

    cx=principle_point[0] + pad_x
    cy=principle_point[1] + pad_y

    # bbox (cropped window) in the padded pixel space
    bbox_w= crop_width * length
    bbox_h=bbox_w

    #cropped_center in the padded pixel space
    cropped_cx= (-tx *length / 2.0 + length / 2.0)
    cropped_cy= (-ty *length / 2.0 + length / 2.0)

    #upper-left corner of cropped window in the padded pixel space
    ul_cx= cropped_cx - bbox_w/2.0
    ul_cy= cropped_cy - bbox_h/2.0

    #find the pp in the cropped coordinate
    new_cx= cx -ul_cx
    new_cy= cy -ul_cy

    #scale intrinsics into the size of resized cropped window

    x_scale= cropped_w / bbox_w
    y_scale= cropped_h / bbox_h

    fx= fx * x_scale
    fy= fy * y_scale

    new_cx= new_cx * x_scale
    new_cy= new_cy * y_scale

    return torch.tensor([fx,fy]) , torch.tensor([new_cx,new_cy])


def create_grid(
    H,
    W,
    device,
    patch_num=16, #if None, no patch
   ):
    """Helper function to create the Image Coodinate grid. 


    Args:
        H (int): Height of image
        W (int): Width of image
        device (str): Pytorch device ("cpu" or "cuda")
        patch_num (int, optional): How large the grid should be. Defaults to 16.

    Returns:
        _type_: _description_
    """
    grids=[]
    if patch_num is not None:

        dh= H //patch_num
        dw= W //patch_num


        for i in range(patch_num):
            for j in range(patch_num):
                u0,u1= j* dw, (j+1) * dw
                v0,v1= i* dh, (i+1) * dh

                u=torch.arange(u0,u1,device=device)
                v=torch.arange(v0,v1,device=device)

                vv,uu=torch.meshgrid(v,u,indexing='ij')

                pixel_grid=torch.stack([uu,vv,torch.ones_like(uu)],dim=-1)

                grids.append(pixel_grid) #(dh,dw,3)

        return torch.stack(grids) #(patch_num ^2 , dh, dw, 3)

    else:
        u=torch.arange(H,device=device)
        v=torch.arange(W,device=device)

        vv,uu = torch.meshgrid(v,u,indexing='ij')

        pixel_grid=torch.stack([uu,vv,torch.ones_like(uu)],dim=-1)
        return pixel_grid #(H,W,3)


def plucker_from_all_pixels(R , T , pixel_grid, fl, pp ):
    """ Function to compute the Plücker coordinates given the full image.

    R: (3,3)
    T: (3,)
    fl: (2,)
    pp: (2,)
    pixel_grid : (H,W, 3)
    """
    u= pixel_grid[...,0] #(H,W)
    v= pixel_grid[...,1]

    H,W=u.shape
    N=H*W

    u=u.reshape(N)
    v=v.reshape(N)

    #convert pixel into camera cooridnate
    cx,cy=pp[0],pp[1]
    fx,fy=fl[0],fl[1]

    # normalize pixel coordinates [0, 1] -> [-1, 1]
    x=(u-cx) / fx #(N)
    y=(v-cy) / fy

    z=torch.ones_like(x)

    dir_cam = torch.stack([x,y,z],dim=-1) #(N,3)
    #dir_cam= torch.nn.functional.normalize(dir_cam,dim=-1)

    dir_world= dir_cam @ R #(N,3)
    C_world= -T @ R.T #(3)

    m_world= torch.cross(C_world.expand_as(dir_world), dir_world,dim=-1) #(N,3)

    return torch.cat([dir_world,m_world],dim=-1) #(N,6)


def plucker_from_single_pixels(R, T, fl, pp,u,v ):
    """
    R: (3,3)
    T: (3,)
    fl: (2,)
    pp: (2,)
    u, v: constant
    """
    #convert pixel into camera cooridnate
    cx,cy=pp[0],pp[1]
    fx,fy=fl[0],fl[1]

    x=(u-cx) / fx
    y=(v-cy) / fy

    z= 1.0

    dir_cam= torch.tensor([x,y,z])
    #dir_cam= dir_cam / torch.norm(dir_cam)

    dir_world= dir_cam @ R #(3,)

    C_world= -T @ R.T #(3,)

    m_world= torch.cross(C_world, dir_world,dim=-1) #(3,)

    return torch.cat([dir_world,m_world],dim=-1) #(6,)


def plucker_from_patches(R , T , pixel_grid, fl, pp):
    """
    R: (3,3)
    T: (3,)
    fl: (2,)
    pp: (2,)
    pixel_grid : (P=patch_num^2, dh, dw, 3)
    """

    # take the center pixel of each patch to calculate the plucker
    P, dh , dw,_ =pixel_grid.shape
    cen_x= dw //2
    cen_y= dh //2

    centers=pixel_grid[:,cen_x, cen_y , :] #(P,3)
    #print(f'centers shape:{centers.shape}')

    P= centers.shape[0]

    pluckers=[]

    for i in range(P):
        u,v,_ = centers[i]
        plucker=plucker_from_single_pixels(R=R,
                                           T=T,
                                           fl=fl,
                                           pp=pp,
                                           u=u,
                                           v=v)
        pluckers.append(plucker)

    #print(f'plucker shape: {pluckers[0].shape}')
    return torch.stack(pluckers,dim=0) #(P,6)



def plucker_encodeing(R,
                      T,
                      fl,
                      pp,
                      crop_params,
                      original_size,
                      cropped_size,
                      device,
                      patch_num=None):
    fl,pp=update_intrinsics_after_crop(focal_length=fl,
                                       principle_point=pp,
                                       crop_params=crop_params,
                                       original_image_size=original_size,
                                       cropped_image_size=cropped_size,
                                       )
    H,W=cropped_size


    pixel_grid=create_grid(H,W,device=device,patch_num=patch_num)

    if patch_num is None:
        return plucker_from_all_pixels(R=R,
                                       T=T,
                                       pixel_grid=pixel_grid,
                                       fl=fl,
                                       pp=pp

                                       )
    else:
        return plucker_from_patches(R=R,
                                    T=T,
                                    pixel_grid=pixel_grid,
                                    fl=fl,
                                    pp=pp)







