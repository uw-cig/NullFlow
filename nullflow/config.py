from __future__ import annotations

import yaml
from torch.utils.data import Dataset

from .data import FFHQ, FastMRIBrain
from .inpainting import Inpainting
from .mri import MultiCoilMRI
from .nullflow import NullFlow
from .unet import build_net


def load_config(path: str, overrides=()) -> dict:
    """YAML config plus `key.sub=value` overrides (values parsed as YAML)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or ():
        key, val = item.split("=", 1)
        d = cfg
        *parents, last = key.strip().split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[last] = yaml.safe_load(val)
    return cfg


def task_of(cfg: dict) -> str:
    return cfg.get("task", "mri")


def describe(cfg: dict) -> str:
    if task_of(cfg) == "mri":
        return f"MRI R={cfg['mri']['acceleration']}"
    return f"{cfg['inpainting']['mask']} inpainting"


def build_operator(cfg: dict):
    task = task_of(cfg)
    if task == "mri":
        m = cfg["mri"]
        return MultiCoilMRI(acceleration=m["acceleration"], total_lines=m.get("total_lines", 320),
                            mask_seed=m.get("mask_seed", 0), pattern=m.get("pattern", "random"),
                            cg_iters=m.get("cg_iters", 200), cg_tol=m.get("cg_tol", 1e-6))
    if task == "inpainting":
        p = cfg["inpainting"]
        return Inpainting(mask=p.get("mask", "box"), box_size=p.get("box_size", 128), drop=p.get("drop", 0.7))
    raise ValueError(f"unknown task {task!r} (expected 'mri' or 'inpainting')")


def build_model(cfg: dict, device) -> NullFlow:
    op = build_operator(cfg)
    nf = dict(cfg.get("nullflow") or {})
    sigma = float(nf.pop("sigma", 0.0))
    # Older configs list the fixed settings too: accept them only with the fixed values.
    fixed = {k.lower(): getattr(NullFlow, k) for k in ("TAU", "P_RATIO", "CORNER_FRAC", "LOGIT_MU", "LOGIT_SIGMA",
                                                      "WEIGHT_C", "WEIGHT_P", "WEIGHT_NORM_DECAY", "T_EPS")}
    fixed["weight_norm"] = op.weight_norm
    for k, v in nf.items():
        if k not in fixed:
            raise ValueError(f"unknown setting nullflow.{k}: only nullflow.sigma can be set")
        if v != fixed[k]:
            raise ValueError(f"nullflow.{k}={v!r} cannot be changed: NullFlow fixes {k} = {fixed[k]!r}")
    return NullFlow(build_net(cfg["model"], op.extra_cond), op, sigma=sigma).to(device)


def build_dataset(cfg: dict, split: str, ids=None) -> Dataset:
    """The `split` of the config's dataset; `ids` (file or comma list) keeps those positions."""
    d = cfg["data"]
    if task_of(cfg) == "mri":
        return FastMRIBrain(d["cache_dir"], d["index_json"], split, ids, d.get("max_coils", 20))
    return FFHQ(d["root"], d["splits"][split], ids=ids, size=d.get("size", 256),
                hflip=bool(d.get("hflip", False)) and split == "train")
