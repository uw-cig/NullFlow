"""Render a compact figure from a NullFlow inference output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


def _load_metrics(metrics_path: Path) -> dict[str, float]:
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): float(value) for key, value in data.items()}


def _format_metrics(metrics: dict[str, float]) -> list[str]:
    if not metrics:
        return ["metrics.json not found"]
    return [f"{name}: {value:.4f}" for name, value in sorted(metrics.items())]


def _measure_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    font: ImageFont.ImageFont,
) -> tuple[int, int]:
    widths = []
    heights = []
    for line in lines:
        left, top, right, bottom = draw.textbbox((0, 0), line, font=font)
        widths.append(right - left)
        heights.append(bottom - top)
    return max(widths, default=0), sum(heights) + max(0, len(lines) - 1) * 8


def render_summary(solve_dir: Path, output_path: Path) -> Path:
    """Creates a side-by-side summary image from inference artifacts."""
    grid_path = solve_dir / "grid.png"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing inference grid: {grid_path}")

    grid = Image.open(grid_path).convert("RGB")
    metrics = _load_metrics(solve_dir / "metrics.json")
    metric_lines = ["NullFlow inference"] + _format_metrics(metrics)

    font = ImageFont.load_default()
    scratch = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(scratch)
    text_width, text_height = _measure_lines(draw, metric_lines, font)

    padding = 24
    panel_width = max(240, text_width + 2 * padding)
    height = max(grid.height, text_height + 2 * padding)
    canvas = Image.new("RGB", (grid.width + panel_width, height), "white")
    canvas.paste(grid, (0, 0))

    draw = ImageDraw.Draw(canvas)
    x = grid.width + padding
    y = padding
    for line in metric_lines:
        draw.text((x, y), line, fill=(20, 20, 20), font=font)
        _, top, _, bottom = draw.textbbox((x, y), line, font=font)
        y += bottom - top + 8

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a compact visualization from NullFlow inference outputs.",
    )
    parser.add_argument(
        "--solve-dir",
        required=True,
        type=Path,
        help="Directory containing grid.png and optional metrics.json.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to the rendered PNG summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    output_path = render_summary(args.solve_dir, args.output)
    print(f"saved: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
