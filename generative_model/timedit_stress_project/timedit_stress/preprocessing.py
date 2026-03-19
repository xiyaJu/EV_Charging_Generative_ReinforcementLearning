from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .utils import DEFAULT_REQUIRED_COLUMNS, cycle_encode


TARGET_COLS = ["price", "load", "lambda"]
RAW_COND_COLS = ["sin_hour", "cos_hour", "day_of_week", "is_weekend"]
TOKEN_COND_COLS = ["sin_hour", "cos_hour", "dow_sin", "dow_cos", "is_weekend"]
GLOBAL_COND_COLS = ["stress_score", "dow_sin", "dow_cos", "is_weekend"]


@dataclass
class PreparedWindows:
    df: pd.DataFrame
    target_windows: np.ndarray
    token_cond_windows: np.ndarray
    global_cond: np.ndarray
    day_meta: pd.DataFrame


class MinMaxScaler3D:
    """Feature-wise min-max scaler to [-1, 1] for (N, L, C) arrays."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.feature_min_: np.ndarray | None = None
        self.feature_max_: np.ndarray | None = None

    def fit(self, array_3d: np.ndarray) -> "MinMaxScaler3D":
        if array_3d.ndim != 3:
            raise ValueError("Expected a 3D array [N, L, C].")
        flat = array_3d.reshape(-1, array_3d.shape[-1])
        self.feature_min_ = flat.min(axis=0)
        self.feature_max_ = flat.max(axis=0)
        return self

    def transform(self, array_3d: np.ndarray) -> np.ndarray:
        self._check_fitted()
        denom = np.maximum(self.feature_max_ - self.feature_min_, self.eps)
        scaled = (array_3d - self.feature_min_) / denom
        return (scaled * 2.0 - 1.0).astype(np.float32)

    def inverse_transform(self, array_3d: np.ndarray) -> np.ndarray:
        self._check_fitted()
        denom = np.maximum(self.feature_max_ - self.feature_min_, self.eps)
        base = (array_3d + 1.0) / 2.0
        return (base * denom + self.feature_min_).astype(np.float32)

    def _check_fitted(self) -> None:
        if self.feature_min_ is None or self.feature_max_ is None:
            raise RuntimeError("Scaler has not been fitted.")

    def state_dict(self) -> Dict[str, np.ndarray | float]:
        self._check_fitted()
        return {
            "feature_min": self.feature_min_.copy(),
            "feature_max": self.feature_max_.copy(),
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, np.ndarray | float]) -> "MinMaxScaler3D":
        scaler = cls(eps=float(state["eps"]))
        scaler.feature_min_ = np.asarray(state["feature_min"], dtype=np.float32)
        scaler.feature_max_ = np.asarray(state["feature_max"], dtype=np.float32)
        return scaler



def read_and_validate_csv(csv_path: str | Path, required_columns: Sequence[str] | None = None) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    required = list(required_columns or DEFAULT_REQUIRED_COLUMNS)
    df = pd.read_csv(path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df



def prepare_dataframe(
    csv_path: str | Path,
    steps_per_day: int = 96,
    required_columns: Sequence[str] | None = None,
    strict_complete_days: bool = True,
) -> pd.DataFrame:
    df = read_and_validate_csv(csv_path, required_columns=required_columns)
    sort_cols: List[str] = []
    if "day_id" in df.columns:
        sort_cols.append("day_id")
    if "t" in df.columns:
        sort_cols.append("t")
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df["day_id"] = df["day_id"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["step_in_day"] = df.groupby("day_id").cumcount().astype(int)

    counts = df.groupby("day_id").size().rename("n_steps")
    complete_days = counts[counts == steps_per_day].index
    incomplete_days = counts[counts != steps_per_day].index.tolist()
    if incomplete_days and strict_complete_days:
        df = df[df["day_id"].isin(complete_days)].copy()
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError(
                "No complete days remain after dropping incomplete day_id groups. "
                f"Expected {steps_per_day} rows per day."
            )

    if (df.groupby("day_id")["step_in_day"].max() + 1 > steps_per_day).any():
        raise ValueError("Found day groups longer than the configured steps_per_day.")

    dow_sin, dow_cos = cycle_encode(df["day_of_week"].to_numpy(), period=7)
    df["dow_sin"] = dow_sin
    df["dow_cos"] = dow_cos

    for col in TARGET_COLS + ["sin_hour", "cos_hour", "dow_sin", "dow_cos"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_weekend"] = df["is_weekend"].astype(np.float32)

    if df[TARGET_COLS + ["sin_hour", "cos_hour"]].isna().any().any():
        bad_cols = df[TARGET_COLS + ["sin_hour", "cos_hour"]].columns[
            df[TARGET_COLS + ["sin_hour", "cos_hour"]].isna().any()
        ].tolist()
        raise ValueError(f"Found NaN values in required numeric columns: {bad_cols}")

    return df



def build_daily_windows(df: pd.DataFrame, steps_per_day: int = 96) -> PreparedWindows:
    day_frames = []
    day_meta = []
    for day_id, g in df.groupby("day_id", sort=True):
        g = g.sort_values("step_in_day")
        if len(g) != steps_per_day:
            continue
        day_frames.append(g)
        day_meta.append(
            {
                "day_id": int(day_id),
                "day_of_week": int(g["day_of_week"].iloc[0]),
                "is_weekend": int(g["is_weekend"].iloc[0]),
                "dow_sin": float(g["dow_sin"].iloc[0]),
                "dow_cos": float(g["dow_cos"].iloc[0]),
                "n_steps": int(len(g)),
            }
        )

    if not day_frames:
        raise ValueError("No complete daily windows were found.")

    target_windows = np.stack([g[TARGET_COLS].to_numpy(dtype=np.float32) for g in day_frames], axis=0)
    token_cond_windows = np.stack([g[TOKEN_COND_COLS].to_numpy(dtype=np.float32) for g in day_frames], axis=0)
    day_meta_df = pd.DataFrame(day_meta)

    if "stress_score" not in df.columns:
        raise ValueError("The prepared DataFrame must already contain a 'stress_score' column.")

    stress_scores = []
    for g in day_frames:
        unique_scores = g["stress_score"].unique()
        if unique_scores.size != 1:
            raise ValueError("Each day must have exactly one stress_score value.")
        stress_scores.append(float(unique_scores[0]))
    day_meta_df["stress_score"] = stress_scores

    global_cond = day_meta_df[["stress_score", "dow_sin", "dow_cos", "is_weekend"]].to_numpy(dtype=np.float32)

    return PreparedWindows(
        df=df.copy(),
        target_windows=target_windows.astype(np.float32),
        token_cond_windows=token_cond_windows.astype(np.float32),
        global_cond=global_cond.astype(np.float32),
        day_meta=day_meta_df,
    )
