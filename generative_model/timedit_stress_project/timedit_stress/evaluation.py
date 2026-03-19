from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import TARGET_COLS, prepare_dataframe
from .stress import StressConfig, StressScorer
from .utils import ensure_dir


TARGET_PAIRS = [("price", "load"), ("price", "lambda"), ("load", "lambda")]


@dataclass
class PreparedDataset:
    name: str
    csv_path: Path
    df: pd.DataFrame
    windows: np.ndarray
    day_meta: pd.DataFrame
    daily_scores: pd.DataFrame
    metadata: pd.DataFrame | None = None


@dataclass
class EvaluationContext:
    steps_per_day: int
    scorer: StressScorer
    real: PreparedDataset
    synthetics: Dict[str, PreparedDataset]
    source_bundle: str | None = None


@dataclass
class EvaluationOutputs:
    summary: Dict[str, object]
    paths: Dict[str, str]


def save_json(data: Mapping[str, object], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(data), f, indent=2, ensure_ascii=False)


def parse_named_paths(items: Sequence[str] | None) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH format, got: {item}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        raw_path = raw_path.strip()
        if not name:
            raise ValueError(f"Missing dataset name in argument: {item}")
        out[name] = Path(raw_path)
    return out


def infer_steps_per_day(bundle_path: str | Path | None, explicit_steps_per_day: int | None) -> int:
    if explicit_steps_per_day is not None:
        return int(explicit_steps_per_day)
    if bundle_path is None:
        return 96
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return int(bundle["data_config"]["steps_per_day"])


def build_or_load_scorer(real_df: pd.DataFrame, steps_per_day: int, bundle_path: str | Path | None) -> tuple[StressScorer, str | None]:
    if bundle_path is not None:
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        scorer = StressScorer.from_state_dict(bundle["stress_scorer_state"])
        return scorer, str(bundle_path)
    scorer = StressScorer(StressConfig(steps_per_day=steps_per_day))
    scorer.fit(real_df)
    return scorer, None


def windows_from_prepared_df(df: pd.DataFrame, steps_per_day: int) -> tuple[np.ndarray, pd.DataFrame]:
    windows: List[np.ndarray] = []
    rows: List[Dict[str, int | float]] = []
    for day_id, g in df.groupby("day_id", sort=True):
        g = g.sort_values("step_in_day")
        if len(g) != steps_per_day:
            continue
        windows.append(g[TARGET_COLS].to_numpy(dtype=np.float32))
        rows.append(
            {
                "day_id": int(day_id),
                "day_of_week": int(g["day_of_week"].iloc[0]),
                "is_weekend": int(g["is_weekend"].iloc[0]),
                "n_steps": int(len(g)),
            }
        )
    if not windows:
        raise ValueError("No complete daily windows were found during evaluation.")
    return np.stack(windows, axis=0), pd.DataFrame(rows)


def merge_daily_stress_to_rows(df: pd.DataFrame, daily_scores: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(daily_scores[["day_id", "stress_score"]], on="day_id", how="left")
    if out["stress_score"].isna().any():
        raise RuntimeError("Failed to merge daily stress scores into row-level dataframe.")
    return out


def prepare_dataset(
    name: str,
    csv_path: str | Path,
    scorer: StressScorer,
    steps_per_day: int,
    metadata_path: str | Path | None = None,
) -> PreparedDataset:
    df = prepare_dataframe(csv_path, steps_per_day=steps_per_day, strict_complete_days=True)
    daily_scores = scorer.score_dataframe(df)
    df_with_stress = merge_daily_stress_to_rows(df, daily_scores)
    windows, day_meta = windows_from_prepared_df(df_with_stress, steps_per_day=steps_per_day)
    metadata = None
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path)
    return PreparedDataset(
        name=name,
        csv_path=Path(csv_path),
        df=df_with_stress,
        windows=windows,
        day_meta=day_meta,
        daily_scores=daily_scores,
        metadata=metadata,
    )


def numeric_summary(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    q = np.quantile(arr, [0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "p05": float(q[0]),
        "p25": float(q[1]),
        "p50": float(q[2]),
        "p75": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "max": float(arr.max()),
    }


def approx_wasserstein_1d(x: np.ndarray, y: np.ndarray, n_quantiles: int = 1001) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    qs = np.linspace(0.0, 1.0, n_quantiles)
    xq = np.quantile(x, qs)
    yq = np.quantile(y, qs)
    return float(np.mean(np.abs(xq - yq)))


def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    support = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, support, side="right") / x.size
    cdf_y = np.searchsorted(y, support, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def quantile_rmse(x: np.ndarray, y: np.ndarray, qs: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    if qs is None:
        qs = np.linspace(0.01, 0.99, 99)
    xq = np.quantile(x, qs)
    yq = np.quantile(y, qs)
    return float(np.sqrt(np.mean((xq - yq) ** 2)))


def summarize_dataset_values(dataset: PreparedDataset) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for var in TARGET_COLS:
        summary = numeric_summary(dataset.df[var].to_numpy())
        rows.append({"dataset": dataset.name, "variable": var, **summary})
    thresholds = {
        var: float(np.quantile(dataset.df[var].to_numpy(dtype=np.float64), 0.90)) for var in TARGET_COLS
    }
    exceed_mat = np.column_stack([dataset.df[var].to_numpy(dtype=np.float64) > thresholds[var] for var in TARGET_COLS])
    num_active = exceed_mat.sum(axis=1)
    rows.append({"dataset": dataset.name, "variable": "joint_exceedance>=2", **numeric_summary(num_active >= 2)})
    rows.append({"dataset": dataset.name, "variable": "joint_exceedance==3", **numeric_summary(num_active == 3)})
    return pd.DataFrame(rows)


def compare_value_distributions(real: PreparedDataset, synthetic: PreparedDataset) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for var in TARGET_COLS:
        x = real.df[var].to_numpy(dtype=np.float64)
        y = synthetic.df[var].to_numpy(dtype=np.float64)
        real_stats = numeric_summary(x)
        syn_stats = numeric_summary(y)
        rows.append(
            {
                "synthetic": synthetic.name,
                "variable": var,
                "mean_diff": syn_stats["mean"] - real_stats["mean"],
                "std_diff": syn_stats["std"] - real_stats["std"],
                "p95_diff": syn_stats["p95"] - real_stats["p95"],
                "p99_diff": syn_stats["p99"] - real_stats["p99"],
                "quantile_rmse": quantile_rmse(x, y),
                "wasserstein_approx": approx_wasserstein_1d(x, y),
                "ks_stat": ks_statistic(x, y),
            }
        )
    return pd.DataFrame(rows)


def step_profile(df: pd.DataFrame, value_col: str, subset: str) -> np.ndarray:
    work = df
    if subset == "weekday":
        work = df[df["is_weekend"] == 0]
    elif subset == "weekend":
        work = df[df["is_weekend"] == 1]
    profile = work.groupby("step_in_day")[value_col].mean()
    return profile.sort_index().to_numpy(dtype=np.float64)


def profile_distance(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    if a.size == 0 or b.size == 0:
        return {"mae": np.nan, "rmse": np.nan, "corr": np.nan}
    n = min(a.size, b.size)
    aa = a[:n]
    bb = b[:n]
    if n < 2:
        corr = np.nan
    else:
        aa_std = float(np.std(aa))
        bb_std = float(np.std(bb))
        corr = float(np.corrcoef(aa, bb)[0, 1]) if aa_std > 1e-12 and bb_std > 1e-12 else np.nan
    return {
        "mae": float(np.mean(np.abs(aa - bb))),
        "rmse": float(np.sqrt(np.mean((aa - bb) ** 2))),
        "corr": corr,
    }


def compare_conditional_profiles(real: PreparedDataset, synthetic: PreparedDataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for subset in ["all", "weekday", "weekend"]:
        for var in TARGET_COLS:
            rp = step_profile(real.df, var, subset=subset)
            sp = step_profile(synthetic.df, var, subset=subset)
            dist = profile_distance(rp, sp)
            rows.append({"synthetic": synthetic.name, "subset": subset, "variable": var, **dist})
            n = min(rp.size, sp.size)
            for step in range(n):
                curves.append(
                    {
                        "synthetic": synthetic.name,
                        "subset": subset,
                        "variable": var,
                        "step_in_day": step,
                        "real_mean": float(rp[step]),
                        "synthetic_mean": float(sp[step]),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(curves)


def daily_acf_curve(windows: np.ndarray, channel_idx: int, max_lag: int) -> np.ndarray:
    max_lag = min(int(max_lag), windows.shape[1] - 1)
    curves: List[np.ndarray] = []
    for day in windows:
        x = day[:, channel_idx].astype(np.float64)
        x = x - x.mean()
        denom = float(np.dot(x, x))
        if denom <= 1e-12:
            curves.append(np.concatenate([[1.0], np.zeros(max_lag, dtype=np.float64)]))
            continue
        vals = [1.0]
        for lag in range(1, max_lag + 1):
            num = float(np.dot(x[:-lag], x[lag:]))
            vals.append(num / denom)
        curves.append(np.asarray(vals, dtype=np.float64))
    return np.mean(np.stack(curves, axis=0), axis=0)


def compare_acf(real: PreparedDataset, synthetic: PreparedDataset, max_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for idx, var in enumerate(TARGET_COLS):
        r = daily_acf_curve(real.windows, idx, max_lag=max_lag)
        s = daily_acf_curve(synthetic.windows, idx, max_lag=max_lag)
        rows.append(
            {
                "synthetic": synthetic.name,
                "variable": var,
                "max_lag": int(len(r) - 1),
                "l1": float(np.mean(np.abs(r - s))),
                "rmse": float(np.sqrt(np.mean((r - s) ** 2))),
                "lag1_real": float(r[1]) if len(r) > 1 else np.nan,
                "lag1_synthetic": float(s[1]) if len(s) > 1 else np.nan,
                "lag_last_real": float(r[-1]),
                "lag_last_synthetic": float(s[-1]),
            }
        )
        for lag in range(len(r)):
            curves.append(
                {
                    "synthetic": synthetic.name,
                    "variable": var,
                    "lag": lag,
                    "real_acf": float(r[lag]),
                    "synthetic_acf": float(s[lag]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(curves)


def abs_diff_by_day(df: pd.DataFrame, var: str) -> np.ndarray:
    values: List[np.ndarray] = []
    for _, g in df.groupby("day_id", sort=True):
        x = g.sort_values("step_in_day")[var].to_numpy(dtype=np.float64)
        if x.size >= 2:
            values.append(np.abs(np.diff(x)))
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(values)


def compare_ramps(real: PreparedDataset, synthetic: PreparedDataset) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for var in TARGET_COLS:
        rx = abs_diff_by_day(real.df, var)
        sx = abs_diff_by_day(synthetic.df, var)
        rows.append(
            {
                "synthetic": synthetic.name,
                "variable": var,
                "real_absdiff_p95": float(np.quantile(rx, 0.95)) if rx.size else np.nan,
                "synthetic_absdiff_p95": float(np.quantile(sx, 0.95)) if sx.size else np.nan,
                "quantile_rmse": quantile_rmse(rx, sx),
                "wasserstein_approx": approx_wasserstein_1d(rx, sx),
                "ks_stat": ks_statistic(rx, sx),
            }
        )
    return pd.DataFrame(rows)


def corr_matrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df[TARGET_COLS].corr(method=method)


def compare_corr_matrices(real: PreparedDataset, synthetic: PreparedDataset) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: List[Dict[str, object]] = []
    matrices: Dict[str, pd.DataFrame] = {}
    for method in ["pearson", "spearman"]:
        real_m = corr_matrix(real.df, method)
        syn_m = corr_matrix(synthetic.df, method)
        diff = syn_m - real_m
        rows.append(
            {
                "synthetic": synthetic.name,
                "method": method,
                "fro_norm": float(np.sqrt(np.sum(diff.to_numpy(dtype=np.float64) ** 2))),
                "max_abs_diff": float(np.max(np.abs(diff.to_numpy(dtype=np.float64)))),
            }
        )
        matrices[f"real_{method}"] = real_m
        matrices[f"{synthetic.name}_{method}"] = syn_m
        matrices[f"{synthetic.name}_{method}_diff"] = diff
    return pd.DataFrame(rows), matrices


def daily_cross_corr_curve(
    windows: np.ndarray,
    x_idx: int,
    y_idx: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    curves: List[np.ndarray] = []
    for day in windows:
        x = day[:, x_idx].astype(np.float64)
        y = day[:, y_idx].astype(np.float64)
        vals = []
        for lag in lags:
            if lag < 0:
                xx = x[-lag:]
                yy = y[:lag]
            elif lag > 0:
                xx = x[:-lag]
                yy = y[lag:]
            else:
                xx = x
                yy = y
            if xx.size < 2 or yy.size < 2:
                vals.append(np.nan)
                continue
            xx_std = float(np.std(xx))
            yy_std = float(np.std(yy))
            if xx_std <= 1e-12 or yy_std <= 1e-12:
                vals.append(np.nan)
                continue
            vals.append(float(np.corrcoef(xx, yy)[0, 1]))
        curves.append(np.asarray(vals, dtype=np.float64))
    curve = np.nanmean(np.stack(curves, axis=0), axis=0)
    return lags, curve


def compare_cross_correlation(real: PreparedDataset, synthetic: PreparedDataset, max_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []
    for x_name, y_name in TARGET_PAIRS:
        x_idx = TARGET_COLS.index(x_name)
        y_idx = TARGET_COLS.index(y_name)
        lags, r = daily_cross_corr_curve(real.windows, x_idx, y_idx, max_lag=max_lag)
        _, s = daily_cross_corr_curve(synthetic.windows, x_idx, y_idx, max_lag=max_lag)
        valid = np.isfinite(r) & np.isfinite(s)
        if not np.any(valid):
            rmse = np.nan
            l1 = np.nan
            real_peak_lag = np.nan
            syn_peak_lag = np.nan
            real_peak = np.nan
            syn_peak = np.nan
        else:
            rr = r[valid]
            ss = s[valid]
            vlags = lags[valid]
            rmse = float(np.sqrt(np.mean((rr - ss) ** 2)))
            l1 = float(np.mean(np.abs(rr - ss)))
            real_peak_idx = int(np.nanargmax(np.abs(rr)))
            syn_peak_idx = int(np.nanargmax(np.abs(ss)))
            real_peak_lag = int(vlags[real_peak_idx])
            syn_peak_lag = int(vlags[syn_peak_idx])
            real_peak = float(rr[real_peak_idx])
            syn_peak = float(ss[syn_peak_idx])
        rows.append(
            {
                "synthetic": synthetic.name,
                "pair": f"{x_name}|{y_name}",
                "max_lag": int(max_lag),
                "l1": l1,
                "rmse": rmse,
                "real_peak_lag": real_peak_lag,
                "synthetic_peak_lag": syn_peak_lag,
                "real_peak_corr": real_peak,
                "synthetic_peak_corr": syn_peak,
            }
        )
        for lag, rv, sv in zip(lags, r, s):
            curves.append(
                {
                    "synthetic": synthetic.name,
                    "pair": f"{x_name}|{y_name}",
                    "lag": int(lag),
                    "real_crosscorr": float(rv) if np.isfinite(rv) else np.nan,
                    "synthetic_crosscorr": float(sv) if np.isfinite(sv) else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(curves)


def stress_summary(dataset: PreparedDataset) -> Dict[str, object]:
    summary = numeric_summary(dataset.daily_scores["stress_score"].to_numpy(dtype=np.float64))
    return {
        "dataset": dataset.name,
        "n_days": int(len(dataset.daily_scores)),
        **summary,
    }


def metadata_control_metrics(dataset: PreparedDataset) -> Dict[str, object] | None:
    if dataset.metadata is None or dataset.metadata.empty:
        return None
    meta = dataset.metadata.copy()
    if "day_id" in meta.columns:
        merged = meta.merge(dataset.daily_scores[["day_id", "stress_score"]], on="day_id", how="left", suffixes=("_meta", "_eval"))
    else:
        merged = meta.copy()
        merged["stress_score_eval"] = dataset.daily_scores["stress_score"].to_numpy()[: len(merged)]
    target_col = "target_stress" if "target_stress" in merged.columns else None
    if target_col is None:
        return None
    realized_col = None
    for candidate in ["stress_score_eval", "stress_score", "stress_score_meta"]:
        if candidate in merged.columns:
            realized_col = candidate
            break
    if realized_col is None:
        return None
    err = merged[realized_col].to_numpy(dtype=np.float64) - merged[target_col].to_numpy(dtype=np.float64)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return None
    return {
        "dataset": dataset.name,
        "n_days": int(err.size),
        "mean_abs_target_error": float(np.mean(np.abs(err))),
        "rmse_target_error": float(np.sqrt(np.mean(err ** 2))),
        "within_0.05": float(np.mean(np.abs(err) <= 0.05)),
        "within_0.10": float(np.mean(np.abs(err) <= 0.10)),
    }


def daily_feature_frame(dataset: PreparedDataset) -> pd.DataFrame:
    exclude = {"day_id", "day_of_week", "is_weekend"}
    feature_cols = [
        c for c in dataset.daily_scores.columns if c not in exclude and pd.api.types.is_numeric_dtype(dataset.daily_scores[c])
    ]
    return dataset.daily_scores[feature_cols].copy()


def _sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy().reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def discriminator_metric(real: PreparedDataset, synthetic: PreparedDataset, seed: int) -> Dict[str, object]:
    real_feat = daily_feature_frame(real)
    syn_feat = daily_feature_frame(synthetic)
    n = min(len(real_feat), len(syn_feat))
    if n < 10:
        return {"synthetic": synthetic.name, "n_days_used": int(n), "auc": np.nan, "balanced_accuracy": np.nan}
    real_sub = _sample_rows(real_feat, n, seed)
    syn_sub = _sample_rows(syn_feat, n, seed + 1)
    X = pd.concat([real_sub, syn_sub], axis=0, ignore_index=True)
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "synthetic": synthetic.name,
        "n_days_used": int(n),
        "auc": float(roc_auc_score(y_test, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }


def _random_pairwise_distances(features: np.ndarray, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    n = len(features)
    if n < 2:
        return np.asarray([], dtype=np.float64)
    idx1 = rng.integers(0, n, size=n_pairs)
    idx2 = rng.integers(0, n, size=n_pairs)
    mask = idx1 != idx2
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    if idx1.size == 0:
        return np.asarray([], dtype=np.float64)
    diff = features[idx1] - features[idx2]
    return np.sqrt(np.sum(diff * diff, axis=1))


def memorization_and_diversity_metric(real: PreparedDataset, synthetic: PreparedDataset, seed: int) -> Dict[str, object]:
    real_feat = daily_feature_frame(real)
    syn_feat = daily_feature_frame(synthetic)
    if len(real_feat) == 0 or len(syn_feat) == 0:
        return {
            "synthetic": synthetic.name,
            "nearest_neighbor_mean": np.nan,
            "nearest_neighbor_p10": np.nan,
            "diversity_mean": np.nan,
            "diversity_ratio_to_real": np.nan,
        }
    scaler = StandardScaler().fit(real_feat)
    real_z = scaler.transform(real_feat)
    syn_z = scaler.transform(syn_feat)
    nn = []
    for row in syn_z:
        d = np.sqrt(np.sum((real_z - row) ** 2, axis=1))
        nn.append(float(np.min(d)))
    nn_arr = np.asarray(nn, dtype=np.float64)
    rng = np.random.default_rng(seed)
    syn_div = _random_pairwise_distances(syn_z, n_pairs=min(2000, max(100, len(syn_z) * 20)), rng=rng)
    real_div = _random_pairwise_distances(real_z, n_pairs=min(2000, max(100, len(real_z) * 20)), rng=rng)
    syn_mean = float(np.mean(syn_div)) if syn_div.size else np.nan
    real_mean = float(np.mean(real_div)) if real_div.size else np.nan
    return {
        "synthetic": synthetic.name,
        "nearest_neighbor_mean": float(np.mean(nn_arr)),
        "nearest_neighbor_p10": float(np.quantile(nn_arr, 0.10)),
        "diversity_mean": syn_mean,
        "diversity_ratio_to_real": float(syn_mean / real_mean) if np.isfinite(syn_mean) and np.isfinite(real_mean) and real_mean > 1e-12 else np.nan,
    }


def _split_real_days(df: pd.DataFrame, test_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    day_ids = sorted(df["day_id"].unique())
    if len(day_ids) < 2:
        return df.copy(), df.copy()
    n_test = max(1, int(round(len(day_ids) * test_fraction)))
    train_days = set(day_ids[:-n_test]) if len(day_ids) > n_test else set(day_ids[:1])
    test_days = set(day_ids[-n_test:])
    return df[df["day_id"].isin(train_days)].copy(), df[df["day_id"].isin(test_days)].copy()


def build_next_step_pairs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_rows: List[List[float]] = []
    targets: List[List[float]] = []
    for _, g in df.groupby("day_id", sort=True):
        g = g.sort_values("step_in_day")
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            feature_rows.append(
                [
                    float(cur["price"]),
                    float(cur["load"]),
                    float(cur["lambda"]),
                    float(cur["sin_hour"]),
                    float(cur["cos_hour"]),
                    float(nxt["sin_hour"]),
                    float(nxt["cos_hour"]),
                    float(cur["dow_sin"]),
                    float(cur["dow_cos"]),
                    float(cur["is_weekend"]),
                ]
            )
            targets.append([float(nxt["price"]), float(nxt["load"]), float(nxt["lambda"])])
    if not feature_rows:
        return np.empty((0, 10), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    return np.asarray(feature_rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _fit_predictive_model(X: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(Ridge(alpha=1.0))),
        ]
    )
    model.fit(X, y)
    return model


def predictive_utility_metrics(real: PreparedDataset, synthetic: PreparedDataset) -> tuple[Dict[str, object], pd.DataFrame]:
    real_train_df, real_test_df = _split_real_days(real.df, test_fraction=0.20)
    X_real_train, y_real_train = build_next_step_pairs(real_train_df)
    X_real_test, y_real_test = build_next_step_pairs(real_test_df)
    X_syn, y_syn = build_next_step_pairs(synthetic.df)
    if X_real_train.size == 0 or X_real_test.size == 0 or X_syn.size == 0:
        overall = {
            "synthetic": synthetic.name,
            "baseline_real_overall_mae": np.nan,
            "baseline_real_overall_rmse": np.nan,
            "tstr_overall_mae": np.nan,
            "tstr_overall_rmse": np.nan,
            "mae_ratio_to_real_baseline": np.nan,
            "rmse_ratio_to_real_baseline": np.nan,
        }
        return overall, pd.DataFrame()

    baseline_model = _fit_predictive_model(X_real_train, y_real_train)
    baseline_pred = baseline_model.predict(X_real_test)
    tstr_model = _fit_predictive_model(X_syn, y_syn)
    tstr_pred = tstr_model.predict(X_real_test)

    baseline_mae_overall = float(mean_absolute_error(y_real_test, baseline_pred))
    baseline_rmse_overall = float(np.sqrt(mean_squared_error(y_real_test, baseline_pred)))
    tstr_mae_overall = float(mean_absolute_error(y_real_test, tstr_pred))
    tstr_rmse_overall = float(np.sqrt(mean_squared_error(y_real_test, tstr_pred)))

    per_target_rows = []
    for idx, var in enumerate(TARGET_COLS):
        base_mae = float(mean_absolute_error(y_real_test[:, idx], baseline_pred[:, idx]))
        base_rmse = float(np.sqrt(mean_squared_error(y_real_test[:, idx], baseline_pred[:, idx])))
        syn_mae = float(mean_absolute_error(y_real_test[:, idx], tstr_pred[:, idx]))
        syn_rmse = float(np.sqrt(mean_squared_error(y_real_test[:, idx], tstr_pred[:, idx])))
        per_target_rows.append(
            {
                "synthetic": synthetic.name,
                "target": var,
                "baseline_real_mae": base_mae,
                "baseline_real_rmse": base_rmse,
                "tstr_mae": syn_mae,
                "tstr_rmse": syn_rmse,
                "mae_ratio_to_real_baseline": float(syn_mae / base_mae) if base_mae > 1e-12 else np.nan,
                "rmse_ratio_to_real_baseline": float(syn_rmse / base_rmse) if base_rmse > 1e-12 else np.nan,
            }
        )

    overall = {
        "synthetic": synthetic.name,
        "baseline_real_overall_mae": baseline_mae_overall,
        "baseline_real_overall_rmse": baseline_rmse_overall,
        "tstr_overall_mae": tstr_mae_overall,
        "tstr_overall_rmse": tstr_rmse_overall,
        "mae_ratio_to_real_baseline": float(tstr_mae_overall / baseline_mae_overall) if baseline_mae_overall > 1e-12 else np.nan,
        "rmse_ratio_to_real_baseline": float(tstr_rmse_overall / baseline_rmse_overall) if baseline_rmse_overall > 1e-12 else np.nan,
    }
    return overall, pd.DataFrame(per_target_rows)


def scenario_comparison_metrics(synthetics: Mapping[str, PreparedDataset]) -> pd.DataFrame:
    items = list(synthetics.items())
    rows: List[Dict[str, object]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            name_a, ds_a = items[i]
            name_b, ds_b = items[j]
            a = ds_a.daily_scores["stress_score"].to_numpy(dtype=np.float64)
            b = ds_b.daily_scores["stress_score"].to_numpy(dtype=np.float64)
            pooled = np.concatenate([a, b])
            pooled_std = float(np.std(pooled))
            d = float((np.mean(b) - np.mean(a)) / pooled_std) if pooled_std > 1e-12 else np.nan
            rows.append(
                {
                    "dataset_a": name_a,
                    "dataset_b": name_b,
                    "stress_mean_diff_b_minus_a": float(np.mean(b) - np.mean(a)),
                    "stress_p95_diff_b_minus_a": float(np.quantile(b, 0.95) - np.quantile(a, 0.95)),
                    "stress_cohens_d_b_minus_a": d,
                }
            )
    return pd.DataFrame(rows)


def build_evaluation_context(
    real_csv: str | Path,
    synthetic_paths: Mapping[str, Path],
    metadata_paths: Mapping[str, Path] | None,
    bundle_path: str | Path | None,
    steps_per_day: int,
) -> EvaluationContext:
    real_df = prepare_dataframe(real_csv, steps_per_day=steps_per_day, strict_complete_days=True)
    scorer, source_bundle = build_or_load_scorer(real_df, steps_per_day=steps_per_day, bundle_path=bundle_path)
    real_dataset = prepare_dataset("real", real_csv, scorer=scorer, steps_per_day=steps_per_day, metadata_path=None)
    synthetics: Dict[str, PreparedDataset] = {}
    for name, csv_path in synthetic_paths.items():
        meta_path = metadata_paths.get(name) if metadata_paths else None
        synthetics[name] = prepare_dataset(name, csv_path, scorer=scorer, steps_per_day=steps_per_day, metadata_path=meta_path)
    return EvaluationContext(
        steps_per_day=steps_per_day,
        scorer=scorer,
        real=real_dataset,
        synthetics=synthetics,
        source_bundle=source_bundle,
    )


def run_full_evaluation(
    context: EvaluationContext,
    output_dir: str | Path,
    acf_max_lag: int,
    cross_max_lag: int,
    seed: int,
) -> EvaluationOutputs:
    out_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(out_dir / "tables")
    matrices_dir = ensure_dir(out_dir / "correlation_matrices")
    daily_dir = ensure_dir(out_dir / "daily_scores")

    real_summary = summarize_dataset_values(context.real)
    real_summary.to_csv(tables_dir / "real_value_summary.csv", index=False)
    context.real.daily_scores.to_csv(daily_dir / "real_daily_scores.csv", index=False)

    dataset_value_summaries = [real_summary]
    dist_tables = []
    profile_tables = []
    profile_curve_tables = []
    acf_tables = []
    acf_curve_tables = []
    ramp_tables = []
    corr_diff_tables = []
    cross_tables = []
    cross_curve_tables = []
    stress_summary_rows = [stress_summary(context.real)]
    control_rows: List[Dict[str, object]] = []
    discrim_rows: List[Dict[str, object]] = []
    memo_rows: List[Dict[str, object]] = []
    pred_overall_rows: List[Dict[str, object]] = []
    pred_target_rows: List[pd.DataFrame] = []

    for name, syn in context.synthetics.items():
        dataset_value_summaries.append(summarize_dataset_values(syn))
        syn.daily_scores.to_csv(daily_dir / f"{name}_daily_scores.csv", index=False)

        dist_tables.append(compare_value_distributions(context.real, syn))

        profile_summary, profile_curves = compare_conditional_profiles(context.real, syn)
        profile_tables.append(profile_summary)
        profile_curve_tables.append(profile_curves)

        acf_summary, acf_curves = compare_acf(context.real, syn, max_lag=acf_max_lag)
        acf_tables.append(acf_summary)
        acf_curve_tables.append(acf_curves)

        ramp_tables.append(compare_ramps(context.real, syn))

        corr_summary, matrices = compare_corr_matrices(context.real, syn)
        corr_diff_tables.append(corr_summary)
        for matrix_name, matrix_df in matrices.items():
            matrix_df.to_csv(matrices_dir / f"{matrix_name}.csv")

        cross_summary, cross_curves = compare_cross_correlation(context.real, syn, max_lag=cross_max_lag)
        cross_tables.append(cross_summary)
        cross_curve_tables.append(cross_curves)

        stress_summary_rows.append(stress_summary(syn))
        control = metadata_control_metrics(syn)
        if control is not None:
            control_rows.append(control)

        discrim_rows.append(discriminator_metric(context.real, syn, seed=seed))
        memo_rows.append(memorization_and_diversity_metric(context.real, syn, seed=seed))
        pred_overall, pred_target = predictive_utility_metrics(context.real, syn)
        pred_overall_rows.append(pred_overall)
        if not pred_target.empty:
            pred_target_rows.append(pred_target)

    scenario_df = scenario_comparison_metrics(context.synthetics)

    paths: Dict[str, str] = {}

    def _save(df: pd.DataFrame, filename: str) -> None:
        path = tables_dir / filename
        df.to_csv(path, index=False)
        paths[filename] = str(path)

    _save(pd.concat(dataset_value_summaries, axis=0, ignore_index=True), "dataset_value_summary.csv")
    if dist_tables:
        _save(pd.concat(dist_tables, axis=0, ignore_index=True), "distribution_comparison.csv")
    if profile_tables:
        _save(pd.concat(profile_tables, axis=0, ignore_index=True), "conditional_profile_comparison.csv")
        _save(pd.concat(profile_curve_tables, axis=0, ignore_index=True), "conditional_profile_curves.csv")
    if acf_tables:
        _save(pd.concat(acf_tables, axis=0, ignore_index=True), "acf_comparison.csv")
        _save(pd.concat(acf_curve_tables, axis=0, ignore_index=True), "acf_curves.csv")
    if ramp_tables:
        _save(pd.concat(ramp_tables, axis=0, ignore_index=True), "ramp_comparison.csv")
    if corr_diff_tables:
        _save(pd.concat(corr_diff_tables, axis=0, ignore_index=True), "correlation_matrix_comparison.csv")
    if cross_tables:
        _save(pd.concat(cross_tables, axis=0, ignore_index=True), "cross_correlation_comparison.csv")
        _save(pd.concat(cross_curve_tables, axis=0, ignore_index=True), "cross_correlation_curves.csv")
    _save(pd.DataFrame(stress_summary_rows), "stress_summary.csv")
    if control_rows:
        _save(pd.DataFrame(control_rows), "scenario_target_control.csv")
    if discrim_rows:
        _save(pd.DataFrame(discrim_rows), "discriminator_metrics.csv")
    if memo_rows:
        _save(pd.DataFrame(memo_rows), "memorization_diversity_metrics.csv")
    if pred_overall_rows:
        _save(pd.DataFrame(pred_overall_rows), "predictive_utility_overall.csv")
    if pred_target_rows:
        _save(pd.concat(pred_target_rows, axis=0, ignore_index=True), "predictive_utility_per_target.csv")
    if not scenario_df.empty:
        _save(scenario_df, "scenario_comparison.csv")

    summary = {
        "steps_per_day": int(context.steps_per_day),
        "bundle_used": context.source_bundle,
        "real_csv": str(context.real.csv_path),
        "synthetic_datasets": {name: str(ds.csv_path) for name, ds in context.synthetics.items()},
        "n_real_days": int(len(context.real.daily_scores)),
        "n_synthetic_days": {name: int(len(ds.daily_scores)) for name, ds in context.synthetics.items()},
        "notes": {
            "discriminator_auc": "Closer to 0.5 is better; much larger than 0.5 means synthetic days are easy to distinguish from real days.",
            "predictive_ratio": "Closer to 1.0 is better; values >1 mean training on synthetic hurts downstream next-step prediction relative to training on real.",
            "nearest_neighbor": "Smaller means synthetic days are closer to some real day; extremely small values together with low diversity can indicate memorization risk.",
            "stress_summary": "Compare mainB and stressA stress_score distributions to verify scenario separation.",
        },
        "tables": paths,
    }
    summary_path = out_dir / "evaluation_summary.json"
    save_json(summary, summary_path)
    paths["evaluation_summary.json"] = str(summary_path)
    return EvaluationOutputs(summary=summary, paths=paths)
