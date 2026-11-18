"""Build the 320 x 320 fastMRI-brain cache that NullFlow trains and evaluates on.

    python scripts/build_cache.py --fastmri_dir /data/fastMRI/brain/multicoil_val --out data/cache320 --workers 8

Input: the fastMRI brain multicoil_val release (file_brain_*.h5). Every volume of
data_splits/brain/index_AXT2_320.json is in it: the train split (615 volumes, slices 4 .. n-6, 4,305
slices) and the val20 / test100 evaluation slices (data_splits/brain/{val20,test100}_provenance.json).
Add --all_eval_slices to also cache every slice of the 77 + 77 val / test volumes.

Per slice, with k the fully sampled multi-coil k-space (C, H, W):
    S    = ESPIRiT maps of k (sigpy EspiritCalib defaults: calib 24, kernel 6, thresh 0.02, crop 0.95),
           computed at the acquired FOV, then centre-cropped to 320 x 320
    x0   = sum_c conj(S_c) crop(F^-1 k_c)          (coil-combined image on the 320 x 320 grid)
saved as <out>/<volume>_sNN.npz with x0 and maps, both complex64 (maps keep the true coil count).
Re-running skips slices that already exist.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nullflow.data import cache_name, enumerate_split  # noqa: E402

CROP = 320


def ifft2c(x):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))


def center_crop(x, th, tw):
    h, w = x.shape[-2:]
    t, l = (h - th) // 2, (w - tw) // 2
    return x[..., t:t + th, l:l + tw]


def build_volume(job):
    path, fname, slices, out = job
    from sigpy.mri.app import EspiritCalib
    t0 = time.time()
    todo = [s for s in slices if not os.path.exists(os.path.join(out, cache_name(fname, s)))]
    with h5py.File(path, "r") as f:
        for s in todo:
            k = f["kspace"][s]                                   # (C, H, W) complex64
            maps = center_crop(np.asarray(EspiritCalib(k, show_pbar=False).run()), CROP, CROP)
            coil_img = center_crop(ifft2c(k), CROP, CROP)
            x0 = (coil_img * np.conj(maps)).sum(0)
            final = os.path.join(out, cache_name(fname, s))
            tmp = final[:-4] + ".tmp"
            np.savez(tmp, x0=x0.astype(np.complex64), maps=maps.astype(np.complex64))
            os.replace(tmp + ".npz", final)
    return fname, len(todo), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastmri_dir", required=True, help="folder with fastMRI brain multicoil_val .h5 files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits_dir", default=os.path.join(ROOT, "data_splits", "brain"))
    ap.add_argument("--all_eval_slices", action="store_true")
    ap.add_argument("--volumes", nargs="*", default=None, help="only these volume files (for a quick check)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    index = os.path.join(a.splits_dir, "index_AXT2_320.json")
    wanted = {}
    for f, s in enumerate_split(index, "train"):
        wanted.setdefault(f, set()).add(s)
    if a.all_eval_slices:
        for split in ("val", "test"):
            for f, s in enumerate_split(index, split):
                wanted.setdefault(f, set()).add(s)
    else:
        for name in ("val20_provenance.json", "test100_provenance.json"):
            for rec in json.load(open(os.path.join(a.splits_dir, name)))["slices"]:
                wanted.setdefault(rec["file"], set()).add(int(rec["slice"]))
    if a.volumes:
        wanted = {f: s for f, s in wanted.items() if f in set(a.volumes)}

    missing = [f for f in wanted if not os.path.exists(os.path.join(a.fastmri_dir, f))]
    if missing:
        sys.exit(f"{len(missing)} volumes missing from {a.fastmri_dir}, e.g. {missing[:3]}")
    os.makedirs(a.out, exist_ok=True)
    jobs = [(os.path.join(a.fastmri_dir, f), f, sorted(s), a.out) for f, s in sorted(wanted.items())]
    print(f"{len(jobs)} volumes, {sum(len(j[2]) for j in jobs)} slices -> {a.out}", flush=True)
    t0, done = time.time(), 0
    with Pool(a.workers) as pool:
        for fname, n, dt in pool.imap_unordered(build_volume, jobs):
            done += 1
            print(f"[{time.time() - t0:6.0f} s] {done}/{len(jobs)} {fname}: {n} slices in {dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
