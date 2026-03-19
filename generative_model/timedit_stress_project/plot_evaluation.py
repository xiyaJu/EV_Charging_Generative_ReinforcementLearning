from __future__ import annotations

import argparse
from pathlib import Path

from timedit_stress.evaluation_plots import create_evaluation_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create PNG plots from an existing evaluation directory.")
    parser.add_argument("--eval-dir", type=str, required=True, help="Directory created by evaluate.py.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save plots. Defaults to <eval-dir>/plots.")
    parser.add_argument("--format", type=str, default="png", help="Image format, e.g. png or pdf.")
    parser.add_argument("--dpi", type=int, default=160, help="Image DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = create_evaluation_plots(
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
        image_format=args.format,
        dpi=args.dpi,
    )
    print("[DONE] Plot generation finished.")
    print(f"[DONE] Plot directory: {Path(args.output_dir) if args.output_dir else Path(args.eval_dir) / 'plots'}")
    for name, path in sorted(paths.items()):
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
