"""Image preprocessing mirroring HF Qwen2VLImageProcessor._preprocess.

Output row order matches the HF layout exactly so `grid_thw` and the patch
sequence line up with what the vision tower and the placeholder tokens expect;
this function is a faithful re-implementation for the no-HF path in this node.
"""
import math

import torch


def smart_resize(height, width, factor=32, min_pixels=65536,
                 max_pixels=16777216):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def preprocess_images(images, patch_size=16, temporal_patch_size=2,
                      merge_size=2, min_pixels=65536, max_pixels=16777216,
                      device="cpu"):
    """HF-equivalent preprocessing for single (unbatched) PIL images.

    Returns `(pixel_values [sum_tokens, 3*temporal_patch*patch*patch],
    grid_thw)` — pixel_values in the patch-grid order HF uses.
    """
    import numpy as np
    from PIL import Image

    factor = patch_size * merge_size
    patches, grids = [], []
    for img in images:
        img = img.convert("RGB")
        w, h = img.size
        h_r, w_r = smart_resize(h, w, factor=factor, min_pixels=min_pixels,
                                max_pixels=max_pixels)
        img = img.resize((w_r, h_r), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img).copy()).to(
            device=device, dtype=torch.float32).div_(255.0)
        x = (x - 0.5) / 0.5                           # mean/std = 0.5
        x = x.permute(2, 0, 1)                         # [C, H, W]
        grid_h, grid_w = h_r // patch_size, w_r // patch_size
        # [C, gh//m, m, p, gw//m, m, p] -> [gh//m, gw//m, m, m, C, p, p]
        x = x.reshape(3, grid_h // merge_size, merge_size, patch_size,
                      grid_w // merge_size, merge_size, patch_size)
        x = x.permute(1, 4, 2, 5, 0, 3, 6)
        # temporal replication: [gh//m, gw//m, m, m, C, t, p, p]
        x = x.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal_patch_size, -1, -1)
        flat = x.reshape(grid_h * grid_w,
                         3 * temporal_patch_size * patch_size * patch_size)
        patches.append(flat)
        grids.append([1, grid_h, grid_w])
    pixel_values = torch.cat(patches, dim=0).contiguous()
    grid_thw = torch.tensor(grids, dtype=torch.long)
    return pixel_values, grid_thw
