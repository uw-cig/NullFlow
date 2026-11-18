"""Evaluate NullFlow checkpoints: M one-step draws per item, averaged (posterior-mean estimate).

    # MRI test: one draw and the mean of 16, with LPIPS
    python evaluate.py --config configs/brain.yaml --ckpt nullflow_brain_R4.pt \
        --split test --ids data_splits/brain/test100.txt --M 1 16 --lpips
    # FFHQ box inpainting test: 1, 4, 16 and 100 draws
    python evaluate.py --config configs/ffhq_box.yaml --ckpt nullflow_ffhq_box.pt \
        --split test --M 1 4 16 100 --lpips

    # checkpoint selection on the validation split (one draw), e.g. over every saved epoch
    python evaluate.py --config runs/brain_R4/config.yaml --ckpt runs/brain_R4/checkpoints/*.pt \
        --split val --ids data_splits/brain/val20.txt --M 1

Item i's measurement is drawn once, with the global RNG seeded seed * 100003 + i + 777767 (this
fixes a random inpainting mask; the MRI mask is fixed by the config). Only the source draws are
resampled, from a generator seeded seed * 100003 + i, `--chunk` draws per generator call, and the
M values are prefixes of one pool of draws.

Metrics (nullflow/metrics.py): MRI on the magnitude image; inpainting on the RGB network output in
[0, 1], or with the observed pixels pasted back (--restore).
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.utils import save_image

from nullflow.config import build_dataset, build_model, describe, load_config
from nullflow.metrics import LPIPS, psnr_ssim, rgb_psnr_ssim


def observation_seed(seed: int, i: int) -> int:
    return seed * 100003 + i + 777767


@torch.no_grad()
def evaluate(model, ds, M_list, seed=0, chunk=8, net_batch=None, lpips_fn=None, restore=False,
             save_dir=None, progress=False):
    dev = next(model.parameters()).device
    mri = model.op.name == "mri"
    if restore and not hasattr(model.op, "restore"):
        raise ValueError("--restore applies to inpainting only")
    quality = psnr_ssim if mri else rgb_psnr_ssim
    perceptual = None if lpips_fn is None else (lpips_fn if mri else lpips_fn.rgb)
    rows, times = [], []
    for i in range(len(ds)):
        b = {k: v[None] for k, v in ds[i].items()}
        x0 = b["target"].to(dev)
        torch.manual_seed(observation_seed(seed, i))     # the measurement noise and a random mask
        ctx = model.op.context(b, dev)
        y = model.measure(x0, ctx)
        gen = torch.Generator(device=dev).manual_seed(seed * 100003 + i)
        torch.cuda.synchronize(); t0 = time.time()
        means = model.sample_mean(y, ctx, M_list, generator=gen, chunk=chunk, net_batch=net_batch)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        if restore:
            means = {M: model.op.restore(xh, y, ctx) for M, xh in means.items()}
        x_in = model.op.AH(y, ctx)
        row = {"index": i}
        row["psnr_in"], row["ssim_in"] = quality(x_in[0], x0[0])
        for M, xh in means.items():
            row[f"psnr_M{M}"], row[f"ssim_M{M}"] = quality(xh[0], x0[0])
            if perceptual is not None:
                row[f"lpips_M{M}"] = perceptual(xh[0], x0[0])
        rows.append(row)
        if save_dir and mri:
            for M, xh in means.items():
                os.makedirs(os.path.join(save_dir, f"M{M}"), exist_ok=True)
                torch.save({"recon": xh[0].cpu(), "target": x0[0].cpu()},
                           os.path.join(save_dir, f"M{M}", f"{i:04d}.pt"))
        elif save_dir:
            for name, img in [("target", x0), ("input", x_in)] + [(f"M{M}", xh) for M, xh in means.items()]:
                os.makedirs(os.path.join(save_dir, name), exist_ok=True)
                save_image(((img + 1) / 2).clamp(0, 1), os.path.join(save_dir, name, f"{i:04d}.png"))
        if progress and (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}", flush=True)
    return rows, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ids", default=None, help="positions within the split: id file or comma list, "
                                                "e.g. data_splits/brain/test100.txt")
    ap.add_argument("--n", type=int, default=None, help="only the first n items")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=None,
                    help="source draws per generator call (default: the config's eval.chunk, else 8)")
    ap.add_argument("--net_batch", type=int, default=None,
                    help="draws per network call (default: the config's eval.net_batch, else chunk)")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--restore", action="store_true",
                    help="inpainting: score with the observed pixels pasted back")
    ap.add_argument("--out", default=None, help="directory for per-item CSVs and reconstructions")
    ap.add_argument("--set", nargs="+", action="extend", default=[], metavar="key=value")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    dev = torch.device("cuda")
    M_list = sorted(set(a.M))
    ecfg = cfg.get("eval") or {}
    chunk = int(a.chunk or ecfg.get("chunk") or 8)
    net_batch = int(a.net_batch or ecfg.get("net_batch") or chunk)

    ds = build_dataset(cfg, a.split, a.ids)
    if a.n is not None:
        ds = Subset(ds, range(min(a.n, len(ds))))
    model = build_model(cfg, dev).eval()
    lpips_fn = LPIPS(dev) if a.lpips else None
    print(f"{describe(cfg)} | {len(ds)} {a.split} items | M = {M_list} | seed {a.seed}"
          + (" | observed pixels restored" if a.restore else ""))

    summary = []
    for ck_path in a.ckpt:
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        model.net.load_state_dict(ck["model"] if "model" in ck else ck)
        name = os.path.splitext(os.path.basename(ck_path))[0]
        save_dir = os.path.join(a.out, name) if a.out else None
        rows, times = evaluate(model, ds, M_list, a.seed, chunk, net_batch, lpips_fn, a.restore, save_dir,
                               progress=len(ds) >= 200)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, f"{name}_{a.split}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
        print(f"\n{ck_path} (epoch {ck.get('epoch', '?') if isinstance(ck, dict) else '?'})")
        print(f"{'M':>5} {'PSNR':>8} {'sd':>6} {'SSIM':>8}" + (f" {'LPIPS':>8}" if a.lpips else ""))
        for M in M_list:
            p = np.array([r[f"psnr_M{M}"] for r in rows])
            s = np.mean([r[f"ssim_M{M}"] for r in rows])
            lp = f" {np.mean([r[f'lpips_M{M}'] for r in rows]):8.4f}" if a.lpips else ""
            print(f"{M:>5} {p.mean():8.3f} {p.std():6.3f} {s:8.4f}{lp}")
        print(f"input A^H y: {np.mean([r['psnr_in'] for r in rows]):.3f} dB / "
              f"{np.mean([r['ssim_in'] for r in rows]):.4f}")
        if len(times) > 1:
            t = np.mean(times[1:])
            print(f"time per item ({M_list[-1]} draws, after the first item): {t:.3f} s "
                  f"= {t / M_list[-1] * 1e3:.1f} ms per draw")
        summary.append((ck_path, float(np.mean([r[f"psnr_M{M_list[0]}"] for r in rows]))))

    if len(summary) > 1:
        print(f"\nPSNR at M={M_list[0]} per checkpoint:")
        for path, p in summary:
            print(f"  {p:8.3f}  {path}")
        best = max(summary, key=lambda t: t[1])
        print(f"best: {best[0]} ({best[1]:.3f} dB)")


if __name__ == "__main__":
    main()
