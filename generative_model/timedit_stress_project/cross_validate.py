from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from timedit_stress.cross_validation import make_fold_specs, summarize_cv_results, train_one_fold
from timedit_stress.preprocessing import prepare_dataframe
from timedit_stress.utils import ensure_dir, pick_device, set_seed
from train import load_config


class _ConfigArgs:
    def __init__(self, namespace: argparse.Namespace) -> None:
        self.csv_path = namespace.csv_path
        self.output_dir = namespace.output_dir
        self.epochs = namespace.epochs
        self.batch_size = namespace.batch_size
        self.learning_rate = namespace.learning_rate
        self.device = namespace.device
        self.seed = namespace.seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-fold cross-validation for the TimeDiT-inspired generator.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--csv-path", type=str, default=None, help="Override data.csv path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override CV output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for every fold.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds, e.g. 5.")
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["blocked", "shuffled"],
        default="blocked",
        help="blocked keeps contiguous day blocks in each fold; shuffled randomly assigns days to folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, _ConfigArgs(args))

    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    device = pick_device(config["experiment"].get("device"))
    if device.type == "cpu":
        torch.set_num_threads(int(config["experiment"].get("cpu_threads", 1)))
        torch.set_num_interop_threads(1)

    output_dir = ensure_dir(config["experiment"]["output_dir"])
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] CV output directory: {output_dir}")

    steps_per_day = int(config["data"]["steps_per_day"])
    full_df = prepare_dataframe(
        csv_path=config["data"]["csv_path"],
        steps_per_day=steps_per_day,
        strict_complete_days=bool(config["data"].get("strict_complete_days", True)),
    )
    unique_day_ids = full_df["day_id"].drop_duplicates().sort_values().to_numpy(dtype=int)
    print(f"[INFO] Loaded {len(full_df):,} rows across {len(unique_day_ids):,} complete days.")

    fold_specs = make_fold_specs(
        day_ids=unique_day_ids,
        n_folds=int(args.n_folds),
        mode=args.split_mode,
        seed=seed,
    )

    assignment_rows = []
    for spec in fold_specs:
        for day_id in spec.train_day_ids.tolist():
            assignment_rows.append({"fold": int(spec.fold_index), "day_id": int(day_id), "split": "train"})
        for day_id in spec.val_day_ids.tolist():
            assignment_rows.append({"fold": int(spec.fold_index), "day_id": int(day_id), "split": "val"})
    pd.DataFrame(assignment_rows).sort_values(["fold", "split", "day_id"]).to_csv(
        output_dir / "cv_fold_assignments.csv", index=False
    )

    results = []
    for spec in fold_specs:
        print(
            f"[INFO] Starting fold {spec.fold_index}/{len(fold_specs)} | "
            f"train_days={len(spec.train_day_ids)} | val_days={len(spec.val_day_ids)}"
        )
        fold_dir = output_dir / f"fold_{spec.fold_index:02d}"
        result = train_one_fold(
            config=config,
            full_df=full_df,
            fold_spec=spec,
            output_dir=fold_dir,
            device=device,
            seed=seed + spec.fold_index,
            n_folds=int(args.n_folds),
            split_mode=args.split_mode,
        )
        results.append(result)
        print(
            f"[DONE] Fold {spec.fold_index:02d} | best_epoch={result.best_epoch} | "
            f"best_val_loss={result.best_val_loss:.6f}"
        )

    summary = summarize_cv_results(
        results=results,
        output_dir=output_dir,
        n_folds=int(args.n_folds),
        split_mode=args.split_mode,
        config_path=args.config,
        csv_path=str(config["data"]["csv_path"]),
    )

    print("[DONE] Cross-validation finished.")
    print(f"[DONE] Summary JSON: {Path(output_dir) / 'cv_summary.json'}")
    print(f"[DONE] Mean best val loss: {summary['mean_best_val_loss']:.6f} +- {summary['std_best_val_loss']:.6f}")
    print(f"[DONE] Best fold: {summary['best_fold']}")


if __name__ == "__main__":
    main()
