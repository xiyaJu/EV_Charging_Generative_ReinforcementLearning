from __future__ import annotations

import argparse
from pathlib import Path

from timedit_stress.evaluation import (
    build_evaluation_context,
    infer_steps_per_day,
    parse_named_paths,
    run_full_evaluation,
)
from timedit_stress.evaluation_plots import create_evaluation_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation and generate plots in one command.")
    parser.add_argument("--real-csv", type=str, required=True, help="Path to the original real CSV.")
    parser.add_argument(
        "--synthetic",
        type=str,
        action="append",
        required=True,
        help="Synthetic dataset in NAME=PATH format. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        action="append",
        default=None,
        help="Optional metadata CSV in NAME=PATH format, typically generated_*_meta.csv.",
    )
    parser.add_argument("--bundle", type=str, default=None, help="Optional model_bundle.pt.")
    parser.add_argument("--steps-per-day", type=int, default=None, help="Override steps_per_day.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for evaluation tables and plots.")
    parser.add_argument("--acf-max-lag", type=int, default=24, help="Max daily ACF lag to compare.")
    parser.add_argument("--cross-max-lag", type=int, default=8, help="Max lead/lag for daily cross-correlation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation helpers.")
    parser.add_argument("--plot-dir", type=str, default=None, help="Optional separate plot directory.")
    parser.add_argument("--plot-format", type=str, default="png", help="Plot format, e.g. png or pdf.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Plot DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic_paths = parse_named_paths(args.synthetic)
    metadata_paths = parse_named_paths(args.metadata)
    steps_per_day = infer_steps_per_day(args.bundle, args.steps_per_day)

    context = build_evaluation_context(
        real_csv=args.real_csv,
        synthetic_paths=synthetic_paths,
        metadata_paths=metadata_paths,
        bundle_path=args.bundle,
        steps_per_day=steps_per_day,
    )
    outputs = run_full_evaluation(
        context=context,
        output_dir=args.output_dir,
        acf_max_lag=args.acf_max_lag,
        cross_max_lag=args.cross_max_lag,
        seed=args.seed,
    )
    plot_paths = create_evaluation_plots(
        eval_dir=args.output_dir,
        output_dir=args.plot_dir,
        image_format=args.plot_format,
        dpi=args.plot_dpi,
    )

    print("[DONE] Evaluation + plotting finished.")
    print(f"[DONE] Summary JSON: {Path(args.output_dir) / 'evaluation_summary.json'}")
    print(f"[DONE] Tables written: {len(outputs.paths)}")
    print(f"[DONE] Plots written: {len(plot_paths)}")


if __name__ == "__main__":
    main()
