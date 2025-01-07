import os

import numpy as np
import torch
import torchvision.transforms.functional as torchvision_F
from PIL import Image
from transparent_background import Remover

import spar3d.models.utils as spar3d_utils


def get_device():
    if os.environ.get("SPAR3D_USE_CPU", "0") == "1":
        return "cpu"

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    return device


def create_intrinsic_from_fov_rad(fov_rad: float, cond_height: int, cond_width: int):
    intrinsic = spar3d_utils.get_intrinsic_from_fov(
        fov_rad,
        H=cond_height,
        W=cond_width,
    )
    intrinsic_normed_cond = intrinsic.clone()
    intrinsic_normed_cond[..., 0, 2] /= cond_width
    intrinsic_normed_cond[..., 1, 2] /= cond_height
    intrinsic_normed_cond[..., 0, 0] /= cond_width
    intrinsic_normed_cond[..., 1, 1] /= cond_height

    return intrinsic, intrinsic_normed_cond


def create_intrinsic_from_fov_deg(fov_deg: float, cond_height: int, cond_width: int):
    return create_intrinsic_from_fov_rad(np.deg2rad(fov_deg), cond_height, cond_width)


def default_cond_c2w(distance: float):
    c2w_cond = torch.as_tensor(
        [
            [0, 0, 1, distance],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ]
    ).float()
    return c2w_cond


def normalize_pc_bbox(pc, scale=1.0):
    # get the bounding box of the mesh
    assert len(pc.shape) in [2, 3] and pc.shape[-1] in [3, 6, 9]
    n_dim = len(pc.shape)
    device = pc.device
    pc = pc.cpu()
    if n_dim == 2:
        pc = pc.unsqueeze(0)
    normalize_pc = []
    for b in range(pc.shape[0]):
        xyz = pc[b, :, :3]  # [N, 3]
        bound_x = (xyz[:, 0].max(), xyz[:, 0].min())
        bound_y = (xyz[:, 1].max(), xyz[:, 1].min())
        bound_z = (xyz[:, 2].max(), xyz[:, 2].min())
        # get the center of the bounding box
        center = np.array(
            [
                (bound_x[0] + bound_x[1]) / 2,
                (bound_y[0] + bound_y[1]) / 2,
                (bound_z[0] + bound_z[1]) / 2,
            ]
        )
        # get the largest dimension of the bounding box
        scale = max(
            bound_x[0] - bound_x[1], bound_y[0] - bound_y[1], bound_z[0] - bound_z[1]
        )
        xyz = (xyz - center) / scale
        extra = pc[b, :, 3:]
        normalize_pc.append(torch.cat([xyz, extra], dim=-1))
    return (
        torch.stack(normalize_pc, dim=0).to(device)
        if n_dim == 3
        else normalize_pc[0].to(device)
    )


def remove_background(
    image: Image,
    bg_remover: Remover = None,
    force: bool = False,
    **transparent_background_kwargs,
) -> Image:
    do_remove = True
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        do_remove = False
    do_remove = do_remove or force
    if do_remove:
        image = bg_remover.process(
            image.convert("RGB"), **transparent_background_kwargs
        )
    return image


def get_1d_bounds(arr):
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1]


def get_bbox_from_mask(mask, thr=0.5):
    masks_for_box = (mask > thr).astype(np.float32)
    assert masks_for_box.sum() > 0, "Empty mask!"
    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))
    return x0, y0, x1, y1


def foreground_crop(image_rgba, crop_ratio=1.3, newsize=None, no_crop=False):
    # make sure the image is a PIL image in RGBA mode
    assert image_rgba.mode == "RGBA", "Image must be in RGBA mode!"
    if not no_crop:
        mask_np = np.array(image_rgba)[:, :, -1]
        mask_np = (mask_np >= 1).astype(np.float32)
        x1, y1, x2, y2 = get_bbox_from_mask(mask_np, thr=0.5)
        h, w = y2 - y1, x2 - x1
        yc, xc = (y1 + y2) / 2, (x1 + x2) / 2
        scale = max(h, w) * crop_ratio
        image = torchvision_F.crop(
            image_rgba,
            top=int(yc - scale / 2),
            left=int(xc - scale / 2),
            height=int(scale),
            width=int(scale),
        )
    else:
        image = image_rgba
    # resize if needed
    if newsize is not None:
        image = image.resize(newsize)
    return image
