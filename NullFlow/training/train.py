"""Command-line entry point for end-to-end training."""

from __future__ import annotations

import argparse
from typing import Sequence

import torch

from NullFlow.training.engine import EngineResult, run_training
from NullFlow.utils.config import load_experiment_config


def _default_device() -> str:
    """Returns the default runtime device for the current machine."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line parser for training."""
    parser = argparse.ArgumentParser(
        description="Run one-step PMF training for FFHQ inpainting.",
    )
    parser.add_argument(
        "--config",
        default="configs/ffhq256_inpainting.yaml",
        help="Path to the experiment YAML file.",
    )
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Torch device passed to the training engine, such as cpu or cuda:0.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> EngineResult:
    """Loads config and starts training from the command line."""
    args = build_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    return run_training(
        config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
