from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DailyWindowDataset
from .factory import build_diffusion, build_model
from .masks import MaskSampler
from .preprocessing import MinMaxScaler3D, build_daily_windows
from .stress import StressConfig, StressScorer
from .utils import DEFAULT_REQUIRED_COLUMNS, ensure_dir, save_json


@dataclass
class FoldSpec:
    fold_index: int
    train_day_ids: np.ndarray
    val_day_ids: np.ndarray


@dataclass
class FoldResult:
    fold_index: int
    output_dir: Path
    model_bundle_path: Path
    best_epoch: int
    best_val_loss: float
    n_train_days: int
    n_val_days: int
    holdout_real_csv: Path
    training_history_csv: Path


def make_fold_specs(
    day_ids: Sequence[int],
    n_folds: int,
    mode: str = "blocked",
    seed: int = 42,
) -> List[FoldSpec]:
    unique_day_ids = np.asarray(list(day_ids), dtype=np.int64)
    if unique_day_ids.ndim != 1 or unique_day_ids.size == 0:
        raise ValueError("day_ids must contain at least one day.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_folds > unique_day_ids.size:
        raise ValueError("n_folds cannot exceed the number of complete days.")

    mode = str(mode).lower()
    if mode not in {"blocked", "shuffled"}:
        raise ValueError("mode must be one of: blocked, shuffled")

    if mode == "blocked":
        ordered_ids = unique_day_ids.copy()
        split_arrays = np.array_split(np.arange(ordered_ids.size), n_folds)
        val_id_splits = [ordered_ids[idxs] for idxs in split_arrays]
    else:
        rng = np.random.default_rng(seed)
        shuffled = unique_day_ids.copy()
        rng.shuffle(shuffled)
        split_arrays = np.array_split(np.arange(shuffled.size), n_folds)
        val_id_splits = [np.sort(shuffled[idxs]) for idxs in split_arrays]

    specs: List[FoldSpec] = []
    for fold_index, val_ids in enumerate(val_id_splits, start=1):
        if val_ids.size == 0:
            raise RuntimeError(f"Fold {fold_index} has no validation days.")
        val_set = set(int(x) for x in val_ids.tolist())
        train_ids = np.asarray([int(x) for x in unique_day_ids.tolist() if int(x) not in val_set], dtype=np.int64)
        if train_ids.size == 0:
            raise RuntimeError(f"Fold {fold_index} has no training days.")
        specs.append(FoldSpec(fold_index=fold_index, train_day_ids=np.sort(train_ids), val_day_ids=np.sort(val_ids)))
    return specs


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _evaluate(model, diffusion, loader, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            loss, _ = diffusion.p_losses(model, batch)
            losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(1, len(losses)))


def _merge_daily_scores(df: pd.DataFrame, daily_scores: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(daily_scores[["day_id", "stress_score"]], on="day_id", how="left")
    if out["stress_score"].isna().any():
        raise RuntimeError("Failed to merge daily stress scores onto all rows.")
    return out


def _subset_by_days(df: pd.DataFrame, day_ids: np.ndarray) -> pd.DataFrame:
    subset = df[df["day_id"].isin(day_ids.tolist())].copy()
    sort_cols = [col for col in ["day_id", "t"] if col in subset.columns]
    if sort_cols:
        subset = subset.sort_values(sort_cols).reset_index(drop=True)
    else:
        subset = subset.reset_index(drop=True)
    return subset


def _build_stress_config(config: Dict[str, object], steps_per_day: int) -> StressConfig:
    stress_cfg = config["stress"]
    return StressConfig(
        steps_per_day=steps_per_day,
        min_group_size=int(stress_cfg["min_group_size"]),
        active_threshold=float(stress_cfg["active_threshold"]),
        deviation_clip=float(stress_cfg["deviation_clip"]),
        variable_weights=stress_cfg["variable_weights"],
        variable_component_weights=stress_cfg["variable_component_weights"],
        joint_weights=stress_cfg["joint_weights"],
    )


def train_one_fold(
    config: Dict[str, object],
    full_df: pd.DataFrame,
    fold_spec: FoldSpec,
    output_dir: str | Path,
    device: torch.device,
    seed: int,
    n_folds: int | None = None,
    split_mode: str | None = None,
) -> FoldResult:
    steps_per_day = int(config["data"]["steps_per_day"])
    fold_dir = ensure_dir(output_dir)
    (fold_dir / "checkpoints").mkdir(exist_ok=True)

    raw_train_df = _subset_by_days(full_df, fold_spec.train_day_ids)
    raw_val_df = _subset_by_days(full_df, fold_spec.val_day_ids)
    raw_train_df[DEFAULT_REQUIRED_COLUMNS].to_csv(fold_dir / "train_real.csv", index=False)
    raw_val_df[DEFAULT_REQUIRED_COLUMNS].to_csv(fold_dir / "holdout_real.csv", index=False)

    split_rows = []
    for day_id in fold_spec.train_day_ids.tolist():
        split_rows.append({"day_id": int(day_id), "split": "train", "fold": int(fold_spec.fold_index)})
    for day_id in fold_spec.val_day_ids.tolist():
        split_rows.append({"day_id": int(day_id), "split": "val", "fold": int(fold_spec.fold_index)})
    pd.DataFrame(split_rows).sort_values(["split", "day_id"]).to_csv(fold_dir / "fold_day_split.csv", index=False)

    scorer = StressScorer(_build_stress_config(config, steps_per_day=steps_per_day))
    scorer.fit(raw_train_df)
    train_daily_scores = scorer.training_daily_scores_.copy()
    val_daily_scores = scorer.score_dataframe(raw_val_df)
    train_daily_scores.to_csv(fold_dir / "daily_stress_scores_train.csv", index=False)
    val_daily_scores.to_csv(fold_dir / "daily_stress_scores_holdout.csv", index=False)

    train_df = _merge_daily_scores(raw_train_df, train_daily_scores)
    val_df = _merge_daily_scores(raw_val_df, val_daily_scores)
    train_df.to_csv(fold_dir / "training_data_with_stress.csv", index=False)
    val_df.to_csv(fold_dir / "holdout_real_with_stress.csv", index=False)

    prepared_train = build_daily_windows(train_df, steps_per_day=steps_per_day)
    prepared_val = build_daily_windows(val_df, steps_per_day=steps_per_day)

    scaler = MinMaxScaler3D().fit(prepared_train.target_windows)
    train_targets = scaler.transform(prepared_train.target_windows)
    val_targets = scaler.transform(prepared_val.target_windows)
    train_token_cond = prepared_train.token_cond_windows
    val_token_cond = prepared_val.token_cond_windows
    train_global_cond = prepared_train.global_cond
    val_global_cond = prepared_val.global_cond

    data_cfg = dict(config["data"])
    data_cfg["target_dim"] = int(train_targets.shape[-1])
    data_cfg["token_cond_dim"] = int(train_token_cond.shape[-1])
    data_cfg["global_cond_dim"] = int(train_global_cond.shape[-1])

    train_mask_sampler = MaskSampler(
        seq_len=steps_per_day,
        n_channels=train_targets.shape[-1],
        probs=config["training"]["mask_probs"],
        seed=seed,
    )
    val_mask_sampler = MaskSampler(
        seq_len=steps_per_day,
        n_channels=val_targets.shape[-1],
        probs={"reconstruction": 1.0},
        seed=seed + 1,
    )

    train_dataset = DailyWindowDataset(train_targets, train_token_cond, train_global_cond, train_mask_sampler, mode="train")
    val_dataset = DailyWindowDataset(val_targets, val_token_cond, val_global_cond, val_mask_sampler, mode="val")

    batch_size = int(config["training"]["batch_size"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model = build_model(config["model"], data_cfg).to(device)
    diffusion = build_diffusion(config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    history: List[Dict[str, float | int]] = []
    best_val = float("inf")
    best_epoch = -1
    best_bundle: Dict[str, object] | None = None

    epochs = int(config["training"]["epochs"])
    grad_clip = float(config["training"]["grad_clip"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        progress = tqdm(train_loader, desc=f"Fold {fold_spec.fold_index} | epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = _move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = diffusion.p_losses(model, batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(metrics["loss"]))
            progress.set_postfix(loss=f"{float(metrics['loss']):.4f}")

        train_loss = float(sum(train_losses) / max(1, len(train_losses)))
        val_loss = _evaluate(model, diffusion, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"[INFO] Fold {fold_spec.fold_index:02d} | epoch={epoch:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_bundle = {
                "model_state": copy.deepcopy(model.state_dict()),
                "model_config": dict(config["model"]),
                "diffusion_config": dict(config["diffusion"]),
                "data_config": data_cfg,
                "stress_config": dict(config["stress"]),
                "training_config": dict(config["training"]),
                "experiment_config": dict(config["experiment"]),
                "generation_config": dict(config.get("generation", {})),
                "scaler_state": scaler.state_dict(),
                "stress_scorer_state": scorer.state_dict(),
                "history": history,
                "daily_stress_scores": scorer.training_daily_scores_.to_dict(orient="list"),
                "day_meta": prepared_train.day_meta.to_dict(orient="list"),
                "cv_info": {
                    "fold_index": int(fold_spec.fold_index),
                    "n_folds": int(n_folds) if n_folds is not None else None,
                    "train_day_ids": fold_spec.train_day_ids.tolist(),
                    "val_day_ids": fold_spec.val_day_ids.tolist(),
                    "split_mode": str(split_mode) if split_mode is not None else None,
                },
            }
            torch.save(best_bundle, fold_dir / "model_bundle.pt")

    if best_bundle is None:
        raise RuntimeError(f"Fold {fold_spec.fold_index} training failed: no bundle was saved.")

    history_path = fold_dir / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    save_json(
        {
            "fold_index": int(fold_spec.fold_index),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "n_train_days": int(train_targets.shape[0]),
            "n_val_days": int(val_targets.shape[0]),
            "device": str(device),
            "train_day_ids": fold_spec.train_day_ids.tolist(),
            "val_day_ids": fold_spec.val_day_ids.tolist(),
            "split_mode": str(split_mode) if split_mode is not None else None,
        },
        fold_dir / "training_summary.json",
    )

    return FoldResult(
        fold_index=int(fold_spec.fold_index),
        output_dir=fold_dir,
        model_bundle_path=fold_dir / "model_bundle.pt",
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val),
        n_train_days=int(train_targets.shape[0]),
        n_val_days=int(val_targets.shape[0]),
        holdout_real_csv=fold_dir / "holdout_real.csv",
        training_history_csv=history_path,
    )


def summarize_cv_results(
    results: Sequence[FoldResult],
    output_dir: str | Path,
    n_folds: int,
    split_mode: str,
    config_path: str,
    csv_path: str,
) -> Dict[str, object]:
    out_dir = ensure_dir(output_dir)
    fold_rows = []
    for result in results:
        fold_rows.append(
            {
                "fold": int(result.fold_index),
                "best_epoch": int(result.best_epoch),
                "best_val_loss": float(result.best_val_loss),
                "n_train_days": int(result.n_train_days),
                "n_val_days": int(result.n_val_days),
                "model_bundle": str(result.model_bundle_path),
                "holdout_real_csv": str(result.holdout_real_csv),
                "training_history_csv": str(result.training_history_csv),
            }
        )
    fold_df = pd.DataFrame(fold_rows).sort_values("fold")
    fold_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)

    losses = fold_df["best_val_loss"].to_numpy(dtype=np.float64)
    summary = {
        "n_folds": int(n_folds),
        "split_mode": str(split_mode),
        "config_path": str(config_path),
        "csv_path": str(csv_path),
        "mean_best_val_loss": float(losses.mean()),
        "std_best_val_loss": float(losses.std(ddof=0)),
        "min_best_val_loss": float(losses.min()),
        "max_best_val_loss": float(losses.max()),
        "best_fold": int(fold_df.loc[fold_df["best_val_loss"].idxmin(), "fold"]),
        "fold_metrics_csv": str(out_dir / "cv_fold_metrics.csv"),
    }
    save_json(summary, out_dir / "cv_summary.json")
    return summary
