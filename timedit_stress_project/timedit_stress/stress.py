from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .preprocessing import TARGET_COLS
from .utils import percentile_from_sorted, topk_mean


@dataclass
class StressConfig:
    steps_per_day: int = 96
    min_group_size: int = 6
    active_threshold: float = 1.5
    deviation_clip: float = 8.0
    variable_weights: Dict[str, float] | None = None
    variable_component_weights: Dict[str, float] | None = None
    joint_weights: Dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.variable_weights is None:
            self.variable_weights = {"price": 0.35, "load": 0.35, "lambda": 0.15, "joint": 0.15}
        if self.variable_component_weights is None:
            self.variable_component_weights = {"level": 0.5, "ramp": 0.25, "duration": 0.25}
        if self.joint_weights is None:
            self.joint_weights = {"two_or_more": 0.7, "three": 0.3}


class StressScorer:
    """Automatically derive a daily stress_score from historical data.

    The score is computed in three stages:
    1. Row-level robust deviations from a hierarchical baseline.
    2. Day-level raw metrics (level / ramp / duration / joint activity).
    3. Empirical percentile calibration into a final stress_score in [0, 1].
    """

    def __init__(self, config: StressConfig | None = None) -> None:
        self.config = config or StressConfig()
        self.target_cols = TARGET_COLS.copy()
        self.group_strategies: List[Tuple[str, ...]] = [
            ("day_of_week", "step_in_day"),
            ("is_weekend", "step_in_day"),
            ("step_in_day",),
        ]
        self.baseline_tables: Dict[str, Dict[Tuple[str, ...], pd.DataFrame]] = {}
        self.diff_scales: Dict[str, float] = {}
        self.sorted_feature_values: Dict[str, np.ndarray] = {}
        self.sorted_total_values: np.ndarray | None = None
        self.training_daily_scores_: pd.DataFrame | None = None
        self.fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "StressScorer":
        self._validate_input(df)
        self.baseline_tables = {var: {} for var in self.target_cols}

        for var in self.target_cols:
            for strategy in self.group_strategies:
                agg = (
                    df.groupby(list(strategy))[var]
                    .agg(count="count", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
                    .reset_index()
                )
                self.baseline_tables[var][strategy] = agg

            diffs = df.groupby("day_id")[var].diff().abs().dropna().to_numpy(dtype=np.float32)
            if diffs.size == 0:
                scale = 1.0
            else:
                scale = float(np.quantile(diffs, 0.75))
                if not np.isfinite(scale) or scale <= 1e-6:
                    scale = float(np.mean(diffs) + 1e-6)
                scale = max(scale, 1e-6)
            self.diff_scales[var] = scale

        enriched = self._compute_row_deviations(df.copy())
        daily_raw = self._compute_daily_raw_features(enriched)
        calibrated = self._calibrate_daily_features(daily_raw)
        self.training_daily_scores_ = calibrated.copy()
        self.fitted_ = True
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        if self.training_daily_scores_ is None:
            raise RuntimeError("StressScorer fit failed to produce daily scores.")
        merged = df.merge(self.training_daily_scores_[["day_id", "stress_score"]], on="day_id", how="left")
        if merged["stress_score"].isna().any():
            raise RuntimeError("Failed to assign stress_score to every row.")
        return merged

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        self._validate_input(df)
        enriched = self._compute_row_deviations(df.copy())
        daily_raw = self._compute_daily_raw_features(enriched)
        return self._apply_calibration_to_raw(daily_raw)

    def score_generated_windows(
        self,
        windows: np.ndarray,
        day_of_week: Sequence[int],
        is_weekend: Sequence[int],
    ) -> pd.DataFrame:
        self._require_fitted()
        if windows.ndim != 3 or windows.shape[-1] != len(self.target_cols):
            raise ValueError("windows must have shape [N, steps_per_day, 3].")
        if len(day_of_week) != windows.shape[0] or len(is_weekend) != windows.shape[0]:
            raise ValueError("day_of_week and is_weekend must match the number of windows.")

        rows: List[Dict[str, float | int]] = []
        for day_idx in range(windows.shape[0]):
            for step in range(windows.shape[1]):
                record: Dict[str, float | int] = {
                    "day_id": int(day_idx),
                    "step_in_day": int(step),
                    "day_of_week": int(day_of_week[day_idx]),
                    "is_weekend": int(is_weekend[day_idx]),
                }
                for ch_idx, col in enumerate(self.target_cols):
                    record[col] = float(windows[day_idx, step, ch_idx])
                rows.append(record)
        temp_df = pd.DataFrame(rows)
        return self.score_dataframe(temp_df)

    def state_dict(self) -> Dict[str, object]:
        self._require_fitted()
        baseline_state: Dict[str, Dict[Tuple[str, ...], Dict[str, list]]] = {}
        for var, table_dict in self.baseline_tables.items():
            baseline_state[var] = {}
            for strategy, table in table_dict.items():
                baseline_state[var][strategy] = table.to_dict(orient="list")
        return {
            "config": {
                "steps_per_day": self.config.steps_per_day,
                "min_group_size": self.config.min_group_size,
                "active_threshold": self.config.active_threshold,
                "deviation_clip": self.config.deviation_clip,
                "variable_weights": self.config.variable_weights,
                "variable_component_weights": self.config.variable_component_weights,
                "joint_weights": self.config.joint_weights,
            },
            "baseline_tables": baseline_state,
            "diff_scales": self.diff_scales,
            "sorted_feature_values": {k: v.tolist() for k, v in self.sorted_feature_values.items()},
            "sorted_total_values": self.sorted_total_values.tolist() if self.sorted_total_values is not None else [],
            "training_daily_scores": self.training_daily_scores_.to_dict(orient="list") if self.training_daily_scores_ is not None else {},
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "StressScorer":
        config = StressConfig(**state["config"])
        obj = cls(config=config)
        baseline_tables: Dict[str, Dict[Tuple[str, ...], pd.DataFrame]] = {}
        for var, table_dict in state["baseline_tables"].items():
            baseline_tables[var] = {}
            for strategy, table_dict_values in table_dict.items():
                baseline_tables[var][tuple(strategy)] = pd.DataFrame(table_dict_values)
        obj.baseline_tables = baseline_tables
        obj.diff_scales = {str(k): float(v) for k, v in state["diff_scales"].items()}
        obj.sorted_feature_values = {
            str(k): np.asarray(v, dtype=np.float32) for k, v in state["sorted_feature_values"].items()
        }
        obj.sorted_total_values = np.asarray(state["sorted_total_values"], dtype=np.float32)
        training_daily_scores = state.get("training_daily_scores", {})
        obj.training_daily_scores_ = pd.DataFrame(training_daily_scores) if training_daily_scores else None
        obj.fitted_ = True
        return obj

    def _validate_input(self, df: pd.DataFrame) -> None:
        required = {"day_id", "step_in_day", "day_of_week", "is_weekend", *self.target_cols}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for stress scoring: {sorted(missing)}")

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError("StressScorer has not been fitted.")

    def _lookup_baseline(self, row: pd.Series, var: str) -> Tuple[float, float]:
        for strategy in self.group_strategies:
            table = self.baseline_tables[var][strategy]
            mask = np.ones(len(table), dtype=bool)
            for key in strategy:
                mask &= table[key].to_numpy() == row[key]
            matched = table.loc[mask]
            if not matched.empty and int(matched["count"].iloc[0]) >= self.config.min_group_size:
                median = float(matched["median"].iloc[0])
                scale = float((matched["q75"].iloc[0] - matched["q25"].iloc[0]) / 1.349)
                return median, max(scale, 1e-6)
        global_series = pd.concat(
            [self.baseline_tables[var][strategy]["median"] for strategy in self.group_strategies],
            axis=0,
            ignore_index=True,
        )
        fallback_median = float(global_series.median()) if not global_series.empty else 0.0
        return fallback_median, 1.0

    def _compute_row_deviations(self, df: pd.DataFrame) -> pd.DataFrame:
        for var in self.target_cols:
            medians = []
            scales = []
            for _, row in df.iterrows():
                median, scale = self._lookup_baseline(row, var)
                medians.append(median)
                scales.append(scale)
            medians_arr = np.asarray(medians, dtype=np.float32)
            scales_arr = np.asarray(scales, dtype=np.float32)
            raw = df[var].to_numpy(dtype=np.float32)
            z = (raw - medians_arr) / np.maximum(scales_arr, 1e-6)
            z_pos = np.clip(z, 0.0, self.config.deviation_clip)
            df[f"{var}_baseline"] = medians_arr
            df[f"{var}_scale"] = scales_arr
            df[f"{var}_dev_pos"] = z_pos
            diffs = df.groupby("day_id")[var].diff().fillna(0.0).abs().to_numpy(dtype=np.float32)
            df[f"{var}_abs_diff_norm"] = diffs / max(self.diff_scales[var], 1e-6)
        return df

    def _compute_daily_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict[str, float | int]] = []
        for day_id, g in df.groupby("day_id", sort=True):
            row: Dict[str, float | int] = {
                "day_id": int(day_id),
                "day_of_week": int(g["day_of_week"].iloc[0]),
                "is_weekend": int(g["is_weekend"].iloc[0]),
            }
            active_cols = []
            for var in self.target_cols:
                dev = g[f"{var}_dev_pos"].to_numpy(dtype=np.float32)
                diffs = g[f"{var}_abs_diff_norm"].to_numpy(dtype=np.float32)
                level_raw = 0.6 * float(np.quantile(dev, 0.95)) + 0.4 * topk_mean(dev, frac=0.10)
                duration_raw = float(np.mean(dev > self.config.active_threshold))
                ramp_raw = float(np.quantile(diffs, 0.95)) if diffs.size > 0 else 0.0
                row[f"{var}_level_raw"] = level_raw
                row[f"{var}_duration_raw"] = duration_raw
                row[f"{var}_ramp_raw"] = ramp_raw
                active_cols.append(dev > self.config.active_threshold)

            active_mat = np.column_stack(active_cols)
            num_active = active_mat.sum(axis=1)
            row["joint_raw"] = (
                self.config.joint_weights["two_or_more"] * float(np.mean(num_active >= 2))
                + self.config.joint_weights["three"] * float(np.mean(num_active == 3))
            )
            records.append(row)
        return pd.DataFrame(records)

    def _calibrate_daily_features(self, daily_raw: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            *(f"{var}_{metric}_raw" for var in self.target_cols for metric in ["level", "ramp", "duration"]),
            "joint_raw",
        ]
        for col in feature_cols:
            self.sorted_feature_values[col] = np.sort(daily_raw[col].to_numpy(dtype=np.float32))
        calibrated = self._apply_calibration_to_raw(daily_raw)
        self.sorted_total_values = np.sort(calibrated["stress_unscaled"].to_numpy(dtype=np.float32))
        calibrated["stress_score"] = calibrated["stress_unscaled"].apply(
            lambda v: percentile_from_sorted(self.sorted_total_values, float(v))
        )
        return calibrated

    def _apply_calibration_to_raw(self, daily_raw: pd.DataFrame) -> pd.DataFrame:
        out = daily_raw.copy()
        for var in self.target_cols:
            for metric in ["level", "ramp", "duration"]:
                raw_col = f"{var}_{metric}_raw"
                pct_col = f"{var}_{metric}_pct"
                sorted_vals = self.sorted_feature_values.get(raw_col)
                if sorted_vals is None:
                    raise RuntimeError(f"Missing calibration values for {raw_col}")
                out[pct_col] = out[raw_col].apply(lambda v: percentile_from_sorted(sorted_vals, float(v)))
        joint_sorted = self.sorted_feature_values.get("joint_raw")
        if joint_sorted is None:
            raise RuntimeError("Missing calibration values for joint_raw")
        out["joint_pct"] = out["joint_raw"].apply(lambda v: percentile_from_sorted(joint_sorted, float(v)))

        weights = self.config.variable_component_weights
        for var in self.target_cols:
            out[f"{var}_stress"] = (
                weights["level"] * out[f"{var}_level_pct"]
                + weights["ramp"] * out[f"{var}_ramp_pct"]
                + weights["duration"] * out[f"{var}_duration_pct"]
            )

        var_weights = self.config.variable_weights
        out["stress_unscaled"] = (
            var_weights["price"] * out["price_stress"]
            + var_weights["load"] * out["load_stress"]
            + var_weights["lambda"] * out["lambda_stress"]
            + var_weights["joint"] * out["joint_pct"]
        )

        if self.sorted_total_values is not None and len(self.sorted_total_values) > 0:
            out["stress_score"] = out["stress_unscaled"].apply(
                lambda v: percentile_from_sorted(self.sorted_total_values, float(v))
            )
        return out
