from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from timedit_stress.dataset import DailyWindowDataset
from timedit_stress.factory import build_diffusion, build_model
from timedit_stress.masks import MaskSampler
from timedit_stress.preprocessing import MinMaxScaler3D, build_daily_windows, prepare_dataframe
from timedit_stress.stress import StressConfig, StressScorer
from timedit_stress.utils import ensure_dir, merge_nested_dict, pick_device, save_json, set_seed



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TimeDiT-inspired generator on data.csv")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--csv-path", type=str, default=None, help="Override data.csv path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cpu or cuda.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()



def load_config(config_path: str, args: argparse.Namespace) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    overrides: Dict[str, object] = {}
    if args.csv_path is not None:
        overrides.setdefault("data", {})["csv_path"] = args.csv_path
    if args.output_dir is not None:
        overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.device is not None:
        overrides.setdefault("experiment", {})["device"] = args.device
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = args.seed
    return merge_nested_dict(config, overrides)



def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}



def evaluate(model, diffusion, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss, _ = diffusion.p_losses(model, batch)
            losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(1, len(losses)))



def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)

    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    device = pick_device(config["experiment"].get("device"))
    if device.type == "cpu":
        torch.set_num_threads(int(config["experiment"].get("cpu_threads", 1)))
        torch.set_num_interop_threads(1)
    output_dir = ensure_dir(config["experiment"]["output_dir"])
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Output directory: {output_dir}")

    steps_per_day = int(config["data"]["steps_per_day"])
    df = prepare_dataframe(
        csv_path=config["data"]["csv_path"],
        steps_per_day=steps_per_day,
        strict_complete_days=bool(config["data"].get("strict_complete_days", True)),
    )
    print(f"[INFO] Loaded {len(df):,} rows across {df['day_id'].nunique():,} complete days.")

    stress_cfg = StressConfig(
        steps_per_day=steps_per_day,
        min_group_size=int(config["stress"]["min_group_size"]),
        active_threshold=float(config["stress"]["active_threshold"]),
        deviation_clip=float(config["stress"]["deviation_clip"]),
        variable_weights=config["stress"]["variable_weights"],
        variable_component_weights=config["stress"]["variable_component_weights"],
        joint_weights=config["stress"]["joint_weights"],
    )
    scorer = StressScorer(stress_cfg)
    df_scored = scorer.fit_transform(df)
    scorer.training_daily_scores_.to_csv(output_dir / "daily_stress_scores.csv", index=False)
    df_scored.to_csv(output_dir / "training_data_with_stress.csv", index=False)

    prepared = build_daily_windows(df_scored, steps_per_day=steps_per_day)
    scaler = MinMaxScaler3D().fit(prepared.target_windows)
    target_scaled = scaler.transform(prepared.target_windows)

    n_days = target_scaled.shape[0]
    val_fraction = float(config["training"]["val_fraction"])
    n_val = max(1, int(round(n_days * val_fraction))) if n_days > 1 else 0
    n_train = max(1, n_days - n_val)
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_days)
    if n_val == 0:
        val_slice = slice(0, min(1, n_days))

    train_targets = target_scaled[train_slice]
    val_targets = target_scaled[val_slice]
    train_token_cond = prepared.token_cond_windows[train_slice]
    val_token_cond = prepared.token_cond_windows[val_slice]
    train_global_cond = prepared.global_cond[train_slice]
    val_global_cond = prepared.global_cond[val_slice]

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    model = build_model(config["model"], data_cfg).to(device)
    diffusion = build_diffusion(config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    history = []
    best_val = float("inf")
    best_epoch = -1
    best_bundle: Dict[str, object] | None = None

    epochs = int(config["training"]["epochs"])
    grad_clip = float(config["training"]["grad_clip"])

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = diffusion.p_losses(model, batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(metrics["loss"])
            progress.set_postfix(loss=f"{metrics['loss']:.4f}")

        train_loss = float(sum(train_losses) / max(1, len(train_losses)))
        val_loss = evaluate(model, diffusion, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[INFO] Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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
                "day_meta": prepared.day_meta.to_dict(orient="list"),
            }
            torch.save(best_bundle, output_dir / "model_bundle.pt")

    if best_bundle is None:
        raise RuntimeError("Training failed: no best model bundle was created.")

    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "n_days": int(n_days),
            "n_train_days": int(train_targets.shape[0]),
            "n_val_days": int(val_targets.shape[0]),
            "device": str(device),
        },
        output_dir / "training_summary.json",
    )
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    print(f"[DONE] Best epoch: {best_epoch}, best val loss: {best_val:.6f}")
    print(f"[DONE] Saved bundle to: {output_dir / 'model_bundle.pt'}")


if __name__ == "__main__":
    main()
