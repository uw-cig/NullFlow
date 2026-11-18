"""Datasets: fastMRI brain slices (FastMRIBrain) and FFHQ 256 x 256 images (FFHQ, at the end).

fastMRI brain (AXT2, multicoil) slices from the 320 x 320 cache built by scripts/build_cache.py.

Split membership comes from data_splits/brain/index_AXT2_320.json (615 train / 77 val / 77 test
volumes, all from fastMRI's brain multicoil_val release). Within a volume the first 4 and last 5
slices are dropped, so the train split has 4,305 slices; val20.txt / test100.txt pick fixed
evaluation slices by their position in this enumeration.

Each cache file <volume>_sNN.npz holds
    x0    complex64 (320, 320)      coil-combined image  sum_c conj(S_c) crop(F^-1 k_c)
    maps  complex64 (C, 320, 320)   ESPIRiT sensitivities, C = the volume's coil count
Every slice is divided by max|x0| and its maps are zero-padded to `max_coils`.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def cache_name(fname: str, s: int) -> str:
    return f"{os.path.splitext(fname)[0]}_s{s:02d}.npz"


def enumerate_split(index_json: str, split: str, slice_start: int = 4, slice_end: int = -5):
    """[(volume file, slice)] of one split, in the order the id files refer to."""
    with open(index_json) as f:
        records = json.load(f)[split]
    out = []
    for rec in records:
        n = int(rec["slices"])
        lo = slice_start if slice_start >= 0 else n + slice_start
        hi = slice_end if slice_end >= 0 else n + slice_end
        out += [(rec["file"], s) for s in range(max(0, lo), min(n, hi))]
    return out


def read_ids(ids) -> list:
    raw = open(ids).read() if isinstance(ids, str) and os.path.exists(ids) else str(ids)
    return [int(t) for t in raw.replace("\n", ",").split(",") if t.strip()]


class FastMRIBrain(Dataset):
    def __init__(self, cache_dir: str, index_json: str, split: str = "train",
                 ids: Optional[str] = None, max_coils: int = 20):
        self.cache_dir = cache_dir
        self.max_coils = int(max_coils)
        self.items = enumerate_split(index_json, split)
        if ids is not None:
            self.items = [self.items[i] for i in read_ids(ids)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fname, s = self.items[i]
        path = os.path.join(self.cache_dir, cache_name(fname, s))
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing; build it with scripts/build_cache.py")
        with np.load(path) as z:
            x0, maps = z["x0"], z["maps"]
        x0 = x0 / max(float(np.abs(x0).max()), 1e-12)
        c = maps.shape[0]
        if c > self.max_coils:
            raise ValueError(f"{c} coils exceeds max_coils={self.max_coils}")
        padded = np.zeros((self.max_coils,) + maps.shape[1:], dtype=maps.dtype)
        padded[:c] = maps
        target = torch.from_numpy(np.stack([x0.real, x0.imag], axis=0).astype(np.float32))
        return {"target": target, "maps": torch.from_numpy(padded)}


class FFHQ(Dataset):
    """FFHQ at 256 x 256: one flat folder of 00000.png ... 69999.png. A split is the index range
    [start, end) of the config (train 1000-69999, val 500-999, test 0-499); `ids` keeps those
    positions within it. Images are mapped to [-1, 1]; `hflip` adds a random horizontal flip."""

    def __init__(self, root: str, index_range, ids: Optional[str] = None, size: int = 256,
                 hflip: bool = False):
        start, end = (int(v) for v in index_range)
        idx = list(range(start, end))
        if ids is not None:
            idx = [idx[i] for i in read_ids(ids)]
        names = set(os.listdir(root))
        missing = [i for i in idx if f"{i:05d}.png" not in names]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing from {root}, e.g. {missing[0]:05d}.png")
        self.files = [os.path.join(root, f"{i:05d}.png") for i in idx]
        self.size = int(size)
        self.hflip = bool(hflip)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        with Image.open(self.files[i]) as im:
            im = im.convert("RGB")
            if im.size != (self.size, self.size):
                im = im.resize((self.size, self.size), Image.BILINEAR)
            x = to_tensor(im) * 2.0 - 1.0
        if self.hflip and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[-1])
        return {"target": x}
