"""Strip a training checkpoint down to the network weights (about 163 MB instead of 490 MB).

    python scripts/export_weights.py runs/brain_R4/checkpoints/epoch_0060.pt nullflow_brain_R4.pt
"""
import sys

import torch

src, dst = sys.argv[1:3]
ck = torch.load(src, map_location="cpu", weights_only=False)
out = {"model": ck["model"], "epoch": ck.get("epoch")}
if "config" in ck:
    out["config"] = ck["config"]
torch.save(out, dst)
print(f"{dst}: epoch {out['epoch']}, {sum(v.numel() for v in out['model'].values()) / 1e6:.1f}M parameters")
