"""Train NullFlow on any task (configs/brain.yaml, configs/ffhq_box.yaml, configs/ffhq_random.yaml).
run.sh wraps these commands for every task.

    python train.py --config configs/brain.yaml --out runs/brain_R4                        # 1 GPU
    torchrun --nproc_per_node=4 train.py --config configs/brain.yaml --out runs/brain_R4   # 4 GPUs
    torchrun --nproc_per_node=4 train.py --config configs/ffhq_box.yaml --out runs/ffhq_box

Batch. training.batch_size is the global batch. Each GPU runs training.micro_batch samples per forward
pass (default batch_size / #GPUs) and accumulates gradients over batch_size / (micro_batch x #GPUs)
of them, so the schedule does not depend on the GPU count. An epoch is training.steps_per_epoch
optimizer steps, or one pass over the training split when that is null.

Learning rate. lr_schedule "cosine" decays lr to lr_min once per epoch; "constant" keeps it, with
    lr_drop      {threshold, patience, scale}: multiply the lr by `scale` once, the first time the
                 validation PSNR is >= threshold on `patience` consecutive validations
    lr_at_epoch  {e: lr}: set the lr at the start of 0-based epoch e, i.e. from epoch e + 1 on
Validation (training.val_every > 0, on rank 0): one-draw PSNR on the validation split, as
evaluate.py --split val --M 1.

Checkpoints: <out>/checkpoints/epoch_NNNN.pt every training.save_every epochs.
Resume with --resume <out>/checkpoints/epoch_NNNN.pt.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from evaluate import evaluate
from nullflow.config import build_dataset, build_model, describe, load_config


def validate(model, ds) -> float:
    """Mean one-draw PSNR over `ds`, leaving the training RNG streams untouched."""
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    model.net.eval()
    try:
        rows, _ = evaluate(model, ds, [1], seed=0, chunk=1)
    finally:
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        model.net.train()
    return float(np.mean([r["psnr_M1"] for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--set", nargs="+", action="extend", default=[], metavar="key=value")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    tc = cfg["training"]

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group(tc.get("backend", "nccl"))
    main_rank = rank == 0

    seed = int(tc.get("seed", 42)) + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    batch = int(tc["batch_size"])
    micro = int(tc.get("micro_batch") or batch // world)
    if batch % (micro * world):
        raise ValueError(f"training.batch_size {batch} is not a multiple of micro_batch {micro} x {world} GPUs")
    accum = batch // (micro * world)

    ds = build_dataset(cfg, "train", cfg["data"].get("train_ids"))
    steps = tc.get("steps_per_epoch")
    if steps:
        sampler = RandomSampler(ds, replacement=True, num_samples=int(steps) * accum * micro)
    else:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True,
                                     seed=int(tc.get("seed", 42)), drop_last=True)
    nw = int(tc.get("num_workers", 8))
    loader = DataLoader(ds, batch_size=micro, sampler=sampler, drop_last=True, num_workers=nw,
                        persistent_workers=nw > 0)

    model = build_model(cfg, device)
    net = model.net
    model.weight_norm_decay **= 1.0 / accum      # loss() runs once per micro-batch: keep the per-step decay
    opt = torch.optim.AdamW(net.parameters(), lr=float(tc["lr"]),
                            betas=tuple(float(b) for b in tc.get("betas", (0.9, 0.999))),
                            weight_decay=float(tc["weight_decay"]))
    sched = None
    if tc.get("lr_schedule", "cosine") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tc["epochs"]),
                                                           eta_min=float(tc["lr_min"]))
    lr_drop = {"hits": 0, "triggered": False, "epoch": None}
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if sched is not None and "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        lr_drop.update(ck.get("lr_drop") or ck.get("val_lr_drop") or {})
        start = int(ck["epoch"])

    sync = {"on": True}
    if world > 1:
        # Data parallel by hand: the MeanFlow loss takes a forward-mode JVP through the network,
        # so gradients are averaged with per-parameter all-reduce hooks rather than a DDP wrapper.
        # They fire on the last micro-batch of each accumulation window only.
        with torch.no_grad():
            for p in net.parameters():
                dist.broadcast(p.data, src=0)

        def average_grad(p):
            if sync["on"] and p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(world)

        for p in net.parameters():
            p.register_post_accumulate_grad_hook(average_grad)

    val_every = int(tc.get("val_every", 0))
    val_ds = build_dataset(cfg, "val", tc.get("val_ids")) if val_every > 0 and main_rank else None
    drop_cfg = tc.get("lr_drop")
    lr_at = {int(k): float(v) for k, v in (tc.get("lr_at_epoch") or {}).items()}

    log = None
    if main_rank:
        os.makedirs(os.path.join(a.out, "checkpoints"), exist_ok=True)
        with open(os.path.join(a.out, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        log = open(os.path.join(a.out, "train.log"), "a")
        n_par = sum(p.numel() for p in net.parameters())
        msg = (f"{describe(cfg)} | {len(ds)} train items | {world} GPU(s) x micro-batch {micro} x "
               f"{accum} accumulation = batch {batch} | {len(loader)} micro-batches/epoch per GPU | "
               f"{n_par / 1e6:.1f}M parameters | start epoch {start}")
        print(msg); log.write(msg + "\n"); log.flush()

    for epoch in range(start, int(tc["epochs"])):
        if epoch in lr_at:
            for g in opt.param_groups:
                g["lr"] = lr_at[epoch]
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        net.train()
        t0, losses, ms, n_steps, k = time.time(), [], [], 0, 0
        n_micro = len(loader)
        for step, b in enumerate(loader):
            x0 = b["target"].to(device, non_blocking=True)
            ctx = model.op.context(b, device)
            if k == 0:
                opt.zero_grad(set_to_none=True)
            loss, m = model.loss(x0, ctx)
            bad = (~torch.isfinite(loss.detach())).float()
            if world > 1:
                dist.all_reduce(bad)          # skip on every rank, or the gradient all-reduce hangs
            if bad.item() > 0:
                opt.zero_grad(set_to_none=True)   # drop the whole accumulation window
                k = 0
                losses.append(float("nan"))
                continue
            sync["on"] = k + 1 == accum or step + 1 == n_micro
            (loss / accum).backward()
            k += 1
            if sync["on"]:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(tc["grad_clip"]))
                opt.step()
                n_steps += 1
                k = 0
            losses.append(float(loss.detach()))
            ms.append(float(m.mean()))
        lr = opt.param_groups[0]["lr"]
        if sched is not None:
            sched.step()

        val, note = None, ""
        if val_every > 0 and (epoch + 1) % val_every == 0:
            if main_rank:
                val = validate(model, val_ds)
            if drop_cfg and not lr_drop["triggered"]:
                flag = torch.zeros(1, device=device)
                if main_rank:
                    lr_drop["hits"] = lr_drop["hits"] + 1 if val >= float(drop_cfg["threshold"]) else 0
                    flag[0] = float(lr_drop["hits"] >= int(drop_cfg.get("patience", 1)))
                if world > 1:
                    dist.broadcast(flag, src=0)   # every rank must step with the same lr
                if flag.item() > 0:
                    for g in opt.param_groups:
                        g["lr"] *= float(drop_cfg.get("scale", 0.1))
                    lr_drop.update(triggered=True, epoch=epoch + 1)
                    note = f"  (lr_drop: lr -> {opt.param_groups[0]['lr']:.2e})"
        if main_rank:
            msg = (f"epoch {epoch + 1}/{tc['epochs']}  loss {np.nanmean(losses):.6f}  "
                   f"residual {np.mean(ms) if ms else math.nan:.6f}  lr {lr:.2e}  {n_steps} steps"
                   + (f"  val_psnr {val:.4f}" if val is not None else "")
                   + f"  {time.time() - t0:.0f} s{note}")
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if (epoch + 1) % int(tc["save_every"]) == 0 or epoch + 1 == int(tc["epochs"]):
                state = {"epoch": epoch + 1, "model": net.state_dict(), "optimizer": opt.state_dict(),
                         "lr_drop": dict(lr_drop), "config": cfg}
                if sched is not None:
                    state["scheduler"] = sched.state_dict()
                torch.save(state, os.path.join(a.out, "checkpoints", f"epoch_{epoch + 1:04d}.pt"))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
