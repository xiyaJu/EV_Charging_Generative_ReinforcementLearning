from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .factory import build_diffusion, build_model
from .preprocessing import MinMaxScaler3D
from .stress import StressScorer
from .utils import cycle_encode, pick_device, set_seed


@dataclass
class LoadedBundle:
    model: torch.nn.Module
    diffusion: torch.nn.Module
    scaler: MinMaxScaler3D
    scorer: StressScorer
    bundle: Dict[str, object]
    device: torch.device



def load_bundle(bundle_path: str | Path, device: str | None = None) -> LoadedBundle:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    data_cfg = dict(bundle["data_config"])
    model = build_model(bundle["model_config"], data_cfg)
    model.load_state_dict(bundle["model_state"])
    diffusion = build_diffusion(bundle["diffusion_config"])
    scaler = MinMaxScaler3D.from_state_dict(bundle["scaler_state"])
    scorer = StressScorer.from_state_dict(bundle["stress_scorer_state"])
    torch_device = pick_device(device)
    if torch_device.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    model.to(torch_device)
    diffusion.to(torch_device)
    model.eval()
    return LoadedBundle(
        model=model,
        diffusion=diffusion,
        scaler=scaler,
        scorer=scorer,
        bundle=bundle,
        device=torch_device,
    )



def build_calendar_windows(
    n_days: int,
    steps_per_day: int,
    start_day_of_week: int,
    start_t: int = 0,
    start_day_id: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    token_cond = []
    global_meta = []
    for local_day in range(n_days):
        day_id = start_day_id + local_day
        dow = int((start_day_of_week + local_day) % 7)
        is_weekend = int(dow in (5, 6))
        dow_sin, dow_cos = cycle_encode([dow], period=7)
        dow_sin_val = float(dow_sin[0])
        dow_cos_val = float(dow_cos[0])
        day_token_cond = []
        for step in range(steps_per_day):
            angle = 2.0 * np.pi * (step / steps_per_day)
            sin_hour = float(np.sin(angle))
            cos_hour = float(np.cos(angle))
            absolute_t = start_t + local_day * steps_per_day + step
            rows.append(
                {
                    "sin_hour": sin_hour,
                    "cos_hour": cos_hour,
                    "day_of_week": dow,
                    "is_weekend": is_weekend,
                    "t": absolute_t,
                    "day_id": day_id,
                    "step_in_day": step,
                }
            )
            day_token_cond.append([sin_hour, cos_hour, dow_sin_val, dow_cos_val, float(is_weekend)])
        token_cond.append(day_token_cond)
        global_meta.append([dow, is_weekend, dow_sin_val, dow_cos_val])
    calendar_df = pd.DataFrame(rows)
    token_cond_arr = np.asarray(token_cond, dtype=np.float32)
    day_of_week_arr = np.asarray([m[0] for m in global_meta], dtype=np.int64)
    is_weekend_arr = np.asarray([m[1] for m in global_meta], dtype=np.int64)
    global_meta_arr = np.asarray([[m[2], m[3], m[1]] for m in global_meta], dtype=np.float32)
    return calendar_df, token_cond_arr, day_of_week_arr, is_weekend_arr, global_meta_arr



def sample_target_stress(
    scores: np.ndarray,
    scenario: str,
    n_days: int,
    rng: np.random.Generator,
    generation_cfg: Dict[str, object],
    fixed_stress: float | None = None,
) -> np.ndarray:
    if fixed_stress is not None:
        return np.full(n_days, float(fixed_stress), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        raise ValueError("No historical stress scores found in the bundle.")
    if scenario == "mainB":
        q_low, q_high = generation_cfg["mainB_quantile_range"]
    elif scenario == "stressA":
        q_low, q_high = generation_cfg["stressA_quantile_range"]
    else:
        raise ValueError("scenario must be 'mainB' or 'stressA' unless fixed_stress is provided.")
    lo = float(np.quantile(scores, q_low))
    hi = float(np.quantile(scores, q_high))
    pool = scores[(scores >= lo) & (scores <= hi)]
    if pool.size == 0:
        pool = scores
    return rng.choice(pool, size=n_days, replace=True).astype(np.float32)


def compute_proxy_utility_score(window: np.ndarray) -> float:
    price = window[:, 0].astype(np.float32)
    load = window[:, 1].astype(np.float32)
    lam = window[:, 2].astype(np.float32)

    price_spread = float(np.quantile(price, 0.95) - np.quantile(price, 0.05))
    load_ramp = float(np.mean(np.abs(np.diff(load)))) if len(load) > 1 else 0.0
    lambda_ramp = float(np.mean(np.abs(np.diff(lam)))) if len(lam) > 1 else 0.0
    lambda_peak_ratio = float(np.max(lam) / (np.mean(lam) + 1e-6))

    # Larger price spread creates more flexibility value; smoother load/lambda reduce control difficulty.
    return price_spread - 0.5 * load_ramp - 0.25 * lambda_ramp - 0.1 * lambda_peak_ratio


@torch.no_grad()

def generate_scenario(
    loaded: LoadedBundle,
    scenario: str,
    n_days: int,
    start_day_of_week: int,
    num_candidates: int,
    seed: int,
    fixed_stress: float | None = None,
    start_t: int = 0,
    start_day_id: int = 0,
    utility_mode: str = "none",
    utility_weight: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    bundle = loaded.bundle
    data_cfg = bundle["data_config"]
    generation_cfg = dict(bundle.get("generation_config", {}))
    if not generation_cfg:
        generation_cfg = {
            "mainB_quantile_range": [0.35, 0.70],
            "stressA_quantile_range": [0.88, 0.98],
            "num_candidates": 4,
        }
    steps_per_day = int(data_cfg["steps_per_day"])
    target_dim = int(data_cfg["target_dim"])

    daily_scores = pd.DataFrame(bundle["daily_stress_scores"])
    target_stress = sample_target_stress(
        scores=daily_scores["stress_score"].to_numpy(dtype=np.float32),
        scenario=scenario,
        n_days=n_days,
        rng=rng,
        generation_cfg=generation_cfg,
        fixed_stress=fixed_stress,
    )

    calendar_df, token_cond_np, day_of_week_arr, is_weekend_arr, global_meta_np = build_calendar_windows(
        n_days=n_days,
        steps_per_day=steps_per_day,
        start_day_of_week=start_day_of_week,
        start_t=start_t,
        start_day_id=start_day_id,
    )

    generated_days = []
    metadata_rows = []

    for day_idx in range(n_days):
        cond_token = np.repeat(token_cond_np[day_idx : day_idx + 1], num_candidates, axis=0)
        global_cond = np.zeros((num_candidates, int(data_cfg["global_cond_dim"])), dtype=np.float32)
        global_cond[:, 0] = target_stress[day_idx]
        global_cond[:, 1:] = global_meta_np[day_idx]

        cond_token_t = torch.from_numpy(cond_token).to(loaded.device)
        global_cond_t = torch.from_numpy(global_cond).to(loaded.device)
        obs_mask_t = torch.zeros((num_candidates, steps_per_day, target_dim), dtype=torch.float32, device=loaded.device)
        obs_values_t = torch.zeros_like(obs_mask_t)

        samples_scaled = loaded.diffusion.sample(
            loaded.model,
            shape=(num_candidates, steps_per_day, target_dim),
            obs_values=obs_values_t,
            obs_mask=obs_mask_t,
            token_cond=cond_token_t,
            global_cond=global_cond_t,
            device=loaded.device,
        )
        samples_np = samples_scaled.detach().cpu().numpy().astype(np.float32)
        samples_raw = loaded.scaler.inverse_transform(samples_np)
        scored = loaded.scorer.score_generated_windows(
            samples_raw,
            day_of_week=[int(day_of_week_arr[day_idx])] * num_candidates,
            is_weekend=[int(is_weekend_arr[day_idx])] * num_candidates,
        )
        scored["target_stress"] = float(target_stress[day_idx])
        scored["candidate_idx"] = np.arange(num_candidates)
        scored["distance_to_target"] = np.abs(scored["stress_score"] - scored["target_stress"])
        if utility_mode == "proxy":
            scored["utility_score"] = np.asarray(
                [compute_proxy_utility_score(samples_raw[candidate_idx]) for candidate_idx in range(num_candidates)],
                dtype=np.float32,
            )
            utility_values = scored["utility_score"].to_numpy(dtype=np.float32)
            utility_values = (utility_values - utility_values.mean()) / (utility_values.std() + 1e-6)
            scored["selection_score"] = -scored["distance_to_target"] + utility_weight * utility_values
        else:
            scored["utility_score"] = 0.0
            scored["selection_score"] = -scored["distance_to_target"]
        best_idx = int(scored["selection_score"].to_numpy().argmax())
        best_window = samples_raw[best_idx]
        best_meta = scored.iloc[best_idx].to_dict()
        best_meta["scenario"] = scenario
        best_meta["day_id"] = int(start_day_id + day_idx)
        metadata_rows.append(best_meta)
        generated_days.append(best_window)

    generated_arr = np.stack(generated_days, axis=0)
    flat_targets = generated_arr.reshape(-1, target_dim)
    out_df = calendar_df.copy()
    out_df["price"] = flat_targets[:, 0]
    out_df["load"] = flat_targets[:, 1]
    out_df["lambda"] = flat_targets[:, 2]
    out_df = out_df[["price", "load", "lambda", "sin_hour", "cos_hour", "day_of_week", "is_weekend", "t", "day_id"]]
    meta_df = pd.DataFrame(metadata_rows)
    return out_df, meta_df
