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
    orig_h, orig_w = original_image_size[0], original_image_size[1]

    cropped_h, cropped_w = cropped_image_size[0], cropped_image_size[1]

    tx, ty, crop_width, s = crop_params.tolist()

    length = max(orig_h, orig_w)

    pad_x = (length - orig_w) / 2.0
    pad_y = (length - orig_h) / 2.0

    fx = focal_length[0]
    fy = focal_length[1]

    cx = principle_point[0] + pad_x
    cy = principle_point[1] + pad_y

    # bbox (cropped window) in the padded pixel space
    bbox_w = crop_width * length
    bbox_h = bbox_w

    # cropped_center in the padded pixel space
    cropped_cx = -tx * length / 2.0 + length / 2.0
    cropped_cy = -ty * length / 2.0 + length / 2.0

    # upper-left corner of cropped window in the padded pixel space
    ul_cx = cropped_cx - bbox_w / 2.0
    ul_cy = cropped_cy - bbox_h / 2.0

    # find the pp in the cropped coordinate
    new_cx = cx - ul_cx
    new_cy = cy - ul_cy

    # scale intrinsics into the size of resized cropped window

    x_scale = cropped_w / bbox_w
    y_scale = cropped_h / bbox_h

    fx = fx * x_scale
    fy = fy * y_scale

    new_cx = new_cx * x_scale
    new_cy = new_cy * y_scale

    return torch.tensor([fx, fy]), torch.tensor([new_cx, new_cy])


def create_grid(
    H,
    W,
    device,
    patch_num=16,  # if None, no patch
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
    grids = []
    if patch_num is not None:
        dh = H // patch_num
        dw = W // patch_num

        for i in range(patch_num):
            for j in range(patch_num):
                u0, u1 = j * dw, (j + 1) * dw
                v0, v1 = i * dh, (i + 1) * dh

                u = torch.arange(u0, u1, device=device)
                v = torch.arange(v0, v1, device=device)

                vv, uu = torch.meshgrid(v, u, indexing="ij")

                pixel_grid = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)

                grids.append(pixel_grid)  # (dh,dw,3)

        return torch.stack(grids)  # (patch_num ^2 , dh, dw, 3)

    else:
        u = torch.arange(H, device=device)
        v = torch.arange(W, device=device)

        vv, uu = torch.meshgrid(v, u, indexing="ij")

        pixel_grid = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)
        return pixel_grid  # (H,W,3)


def plucker_from_all_pixels(R, T, pixel_grid, fl, pp):
    """Function to compute the Plücker coordinates given the full image.

    R: (3,3)
    T: (3,)
    fl: (2,)
    pp: (2,)
    pixel_grid : (H,W, 3)
    """
    u = pixel_grid[..., 0]  # (H,W)
    v = pixel_grid[..., 1]

    H, W = u.shape
    N = H * W

    u = u.reshape(N)
    v = v.reshape(N)

    # convert pixel into camera cooridnate
    cx, cy = pp[0], pp[1]
    fx, fy = fl[0], fl[1]

    # normalize pixel coordinates [0, 1] -> [-1, 1]
    x = (u - cx) / fx  # (N)
    y = (v - cy) / fy

    z = torch.ones_like(x)

    dir_cam = torch.stack([x, y, z], dim=-1)  # (N,3)
    # dir_cam= torch.nn.functional.normalize(dir_cam,dim=-1)

    dir_world = dir_cam @ R  # (N,3)
    C_world = -T @ R.T  # (3)

    m_world = torch.cross(C_world.expand_as(dir_world), dir_world, dim=-1)  # (N,3)

    return torch.cat([dir_world, m_world], dim=-1)  # (N,6)


def plucker_from_single_pixels(R, T, fl, pp, u: float, v: float):
    """ Compute the Plücler coordinate from a point to the camera origin.

    Arguments:
        R: Rotation Matrix of the cam with shape (3,3)
        T: Translation Matrix of cam with shape (3,)
        fl: (2,)
        pp: (2,)
        u, v: (float) constant
    Returns: 
        torch.Tensor: (6,) Plücker coordinates concatenated 
        as (d1, d2, d3, m1, m2, m3)
    """
    # convert pixel into camera cooridnate
    cx, cy = pp[0], pp[1]
    fx, fy = fl[0], fl[1]

    x = (u - cx) / fx
    y = (v - cy) / fy

    z = 1.0

    dir_cam = torch.tensor([x, y, z])
    # dir_cam= dir_cam / torch.norm(dir_cam)

    dir_world = dir_cam @ R  # (3,)
    C_world = -T @ R.T  # (3,) -> Camera center

    m_world = torch.cross(C_world, dir_world, dim=-1)  # (3,)
    return torch.cat([dir_world, m_world], dim=-1)  # (6,)


def plucker_from_patches(R, T, pixel_grid, fl, pp) -> torch.Tensor:
    """ Compute the plücker coordinates for the center pixel of each patch.

    Arguments:
        R: Rotation Matrix of the cam with shape (3,3)
        T: Translation Matrix of cam with shape (3,)
        fl: (2,)
        pp: (2,)
        pixel_grid : (P=patch_num^2, dh, dw, 3)
    Returns:
        torch.Tensor: (P,6) Plücker coordinates for each patch center pixel
    """

    # take the center pixel of each patch to calculate the plucker
    P, dh, dw, _ = pixel_grid.shape
    cen_x = dw // 2
    cen_y = dh // 2

    centers = pixel_grid[:, cen_x, cen_y, :]  # (P,3)
    # print(f'centers shape:{centers.shape}')

    P = centers.shape[0]

    pluckers = []
    for i in range(P):
        u, v, _ = centers[i]
        plucker = plucker_from_single_pixels(R=R, T=T, fl=fl, pp=pp, u=u, v=v)
        pluckers.append(plucker)

    # print(f'plucker shape: {pluckers[0].shape}')
    return torch.stack(pluckers, dim=0)  # (P,6)


def plucker_encodeing(
    R, T, fl, pp, crop_params, original_size, cropped_size, device, patch_num=None
):
    fl, pp = update_intrinsics_after_crop(
        focal_length=fl,
        principle_point=pp,
        crop_params=crop_params,
        original_image_size=original_size,
        cropped_image_size=cropped_size,
    )
    H, W = cropped_size

    pixel_grid = create_grid(H, W, device=device, patch_num=patch_num)

    if patch_num is None:
        return plucker_from_all_pixels(R=R, T=T, pixel_grid=pixel_grid, fl=fl, pp=pp)
    else:
        return plucker_from_patches(R=R, T=T, pixel_grid=pixel_grid, fl=fl, pp=pp)


def plucker_to_rays(pluck_feats: torch.Tensor, normalize_moment: bool = True):
    """
    Args:
        pluck_feats: (..., 6) tensor [direction(3), moment(3)]
    Returns:
        rays: (..., 6) tensor [origin(3), direction(3)]
    """
    direction = torch.nn.functional.normalize(pluck_feats[..., :3], dim=-1)

    moment = pluck_feats[..., 3:]

    if normalize_moment:
        c = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
        moment = moment / c

    # closest point to origin
    points = torch.cross(direction, moment, dim=-1)

    rays = torch.cat((points, direction), dim=-1)
    return rays

def simple_rays(directions: torch.Tensor, cam_pos: torch.Tensor):
    """
    Args:
        directions: (..., 3) tensor
        cam_pos: (..., 3) tensor
    Returns:
        rays: (..., 6) tensor [origin(3), direction(3)]
    """
    direction = torch.nn.functional.normalize(directions, dim=-1)
    points = cam_pos + direction 

    rays = torch.cat((points, direction), dim=-1)
    return rays

def ray_to_plucker(in_ray):
    """
    Convert to plucker representation <D, OxD>.
    """

    ray = in_ray.clone()
    ray_origins = ray[..., :3]
    ray_directions = ray[..., 3:]
    # Normalize ray directions to unit vectors
    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    plucker_ray = torch.cat([ray_directions, plucker_normal], dim=-1)
    
    return plucker_ray


def compute_ndc_coordinates(
    crop_parameters=None,
    use_half_pix=True,
    num_patches_x=16,
    num_patches_y=16,
    device=None,
):
    """
    Computes NDC Grid using crop_parameters. If crop_parameters is not provided,
    then it assumes that the crop is the entire image (corresponding to an NDC grid
    where top left corner is (1, 1) and bottom right corner is (-1, -1)).
    """
    if crop_parameters is None:
        cc_x, cc_y, width = 0, 0, 2
    else:
        device = crop_parameters.device
        cc_x, cc_y, width, _ = crop_parameters

    dx = 1 / num_patches_x
    dy = 1 / num_patches_y
    if use_half_pix:
        min_y = 1 - dy
        max_y = -min_y
        min_x = 1 - dx
        max_x = -min_x
    else:
        min_y = min_x = 1
        max_y = -1 + 2 * dy
        max_x = -1 + 2 * dx

    y, x = torch.meshgrid(
        torch.linspace(min_y, max_y, num_patches_y, dtype=torch.float32, device=device),
        torch.linspace(min_x, max_x, num_patches_x, dtype=torch.float32, device=device),
        indexing="ij",
    )
    x_prime = x * width / 2 - cc_x
    y_prime = y * width / 2 - cc_y
    xyd_grid = torch.stack([x_prime, y_prime, torch.ones_like(x)], dim=-1)
    return xyd_grid


def unproject_points(curr_sample, xyd_grid):
    xyz_flattened = xyd_grid.reshape(-1, 3)

    xy = xyz_flattened[..., :2]  # (N, 2)
    depth = xyz_flattened[..., 2:3]  # (N, 1)

    # Handle focal length
    if isinstance(curr_sample["focal_length"], (int, float)):
        fx = fy = curr_sample["focal_length"]
    else:
        fx, fy = curr_sample["focal_length"][0], curr_sample["focal_length"][1]

    # Handle principal point
    px = curr_sample["principal_point"][..., 0]
    py = curr_sample["principal_point"][..., 1]


    # Step 2: Convert camera parameters to NDC if needed
    fx_ndc = fx
    fy_ndc = fy
    px_ndc = px
    py_ndc = py


    X = (xy[..., 0:1] - px_ndc) * depth / fx_ndc
    Y = (xy[..., 1:2] - py_ndc) * depth / fy_ndc
    Z = depth

    # Points in camera view coordinates (shape: N, 3)
    points_view = torch.cat([X, Y, Z], dim=-1)


    # Step 4: Transform to world coordinates if needed
    # Camera view to world: X_world = R^T @ (X_cam - T)
    R = curr_sample["R"].unsqueeze(0)  # (1, 3, 3)
    T = curr_sample["T"].unsqueeze(0)  # (1, 3)
    
    # Compute camera center in world coordinates
    camera_center = -(R.transpose(-2, -1) @ T.unsqueeze(-1)).squeeze(-1)  # (N_cams, 3)
    
    # Single camera
    points_world = (R[0].T @ points_view.T).T + camera_center[0]  # (N, 3)
    return points_world, camera_center[0]
    

def compute_directions_from_sample(sample, patch_size: int) -> torch.Tensor:
    xyd_grid = compute_ndc_coordinates(
        crop_parameters=sample["crop_params"],
        use_half_pix=True,
        num_patches_x=patch_size,
        num_patches_y=patch_size,
    )

    unprojected, origins = unproject_points(sample, xyd_grid)
    # unprojected =  unprojected.unsqueeze(0) # (N, P, 3)

    origins = origins.repeat(patch_size * patch_size, 1)  # (N, P, 3)
    directions = unprojected - origins

    return torch.cat((origins, directions), dim=-1)
