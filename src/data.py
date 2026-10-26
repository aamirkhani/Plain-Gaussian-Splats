"""NeRF-synthetic (Blender) dataset loader.

Two conventions have to be reconciled:

  Blender/OpenGL cameras look down -z, with +y up.
  Our rasterizer treats +z as depth (the camera looks down +z), +y down.

so  c2w_ours = c2w_blender @ diag(1, -1, -1, 1)  and  viewmat = inverse(that).

Images are RGBA; the standard protocol composites them over WHITE.
Everything is downsampled on load - at 12 GB total we cannot afford 800x800.
"""

import json
import math
import os

import numpy as np
import torch
from PIL import Image

# flips y and z: Blender camera axes -> ours
_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


class Camera:
    __slots__ = ("viewmat", "cam_pos", "image", "fx", "fy", "cx", "cy", "H", "W", "name")


def load_blender(root, split="train", resolution=400, device="cuda", limit=None):
    """Load a NeRF-synthetic split into a list of Camera objects."""
    with open(os.path.join(root, f"transforms_{split}.json")) as f:
        meta = json.load(f)

    frames = meta["frames"]
    if limit is not None:
        frames = frames[:limit]

    cams = []
    for fr in frames:
        path = os.path.join(root, fr["file_path"].lstrip("./") + ".png")
        im = Image.open(path)
        W0 = im.size[0]
        if resolution != W0:
            im = im.resize((resolution, resolution), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0      # [H,W,4]

        if arr.shape[2] == 4:
            rgb, a = arr[..., :3], arr[..., 3:4]
            arr = rgb * a + (1.0 - a)                        # over white
        else:
            arr = arr[..., :3]

        H, W = arr.shape[:2]
        # focal length from the horizontal field of view
        fx = fy = 0.5 * W / math.tan(0.5 * meta["camera_angle_x"])

        c2w = np.array(fr["transform_matrix"], dtype=np.float32) @ _FLIP
        w2c = np.linalg.inv(c2w)

        c = Camera()
        c.viewmat = torch.from_numpy(w2c).to(device)
        c.cam_pos = torch.from_numpy(np.ascontiguousarray(c2w[:3, 3])).to(device)
        c.image = torch.from_numpy(arr).to(device)
        c.fx, c.fy = fx, fy
        c.cx, c.cy = W * 0.5, H * 0.5
        c.H, c.W = H, W
        c.name = os.path.basename(path)
        cams.append(c)
    return cams


def scene_extent(cams):
    """Radius of the camera sphere - sets the length scale for lr and splitting."""
    pos = torch.stack([c.cam_pos for c in cams])
    center = pos.mean(0)
    return float((pos - center).norm(dim=1).max().item() * 1.1)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1e-12)).item())
