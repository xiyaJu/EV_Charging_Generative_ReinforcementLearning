from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .preprocessing import TARGET_COLS
from .utils import ensure_dir, save_json


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc


def _save_plot(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.clf()


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _list_daily_score_files(daily_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not daily_dir.exists():
        return out
    for path in sorted(daily_dir.glob("*_daily_scores.csv")):
        stem = path.stem
        if stem.endswith("_daily_scores"):
            name = stem[: -len("_daily_scores")]
            out[name] = path
    return out


def _plot_stress_distributions(plt, daily_dir: Path, output_dir: Path, image_format: str, dpi: int) -> Dict[str, str]:
    files = _list_daily_score_files(daily_dir)
    if not files:
        return {}

    data = {}
    for name, path in files.items():
        df = pd.read_csv(path)
        if "stress_score" in df.columns and not df.empty:
            data[name] = df["stress_score"].to_numpy(dtype=float)
    if not data:
        return {}

    plot_paths: Dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in data.items():
        bins = np.linspace(0.0, 1.0, 31)
        ax.hist(values, bins=bins, density=True, histtype="step", linewidth=2.0, label=name)
    ax.set_title("Stress score distribution")
    ax.set_xlabel("stress_score")
    ax.set_ylabel("density")
    ax.legend()
    hist_path = output_dir / f"stress_score_distribution.{image_format}"
    _save_plot(fig, hist_path, dpi)
    plt.close(fig)
    plot_paths[hist_path.name] = str(hist_path)

    ordered_names = list(data.keys())
    values_list = [data[name] for name in ordered_names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(values_list, tick_labels=ordered_names, showfliers=True)
    ax.set_title("Stress score boxplot")
    ax.set_ylabel("stress_score")
    box_path = output_dir / f"stress_score_boxplot.{image_format}"
    _save_plot(fig, box_path, dpi)
    plt.close(fig)
    plot_paths[box_path.name] = str(box_path)
    return plot_paths


def _plot_conditional_profiles(plt, tables_dir: Path, output_dir: Path, image_format: str, dpi: int) -> Dict[str, str]:
    curves = _read_csv_if_exists(tables_dir / "conditional_profile_curves.csv")
    if curves is None or curves.empty:
        return {}
    out: Dict[str, str] = {}
    for subset in sorted(curves["subset"].dropna().unique().tolist()):
        for variable in TARGET_COLS:
            sub = curves[(curves["subset"] == subset) & (curves["variable"] == variable)].copy()
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            real_series = sub.sort_values("step_in_day")[["step_in_day", "real_mean"]].drop_duplicates()
            ax.plot(real_series["step_in_day"], real_series["real_mean"], linewidth=2.5, label="real")
            for synthetic in sorted(sub["synthetic"].dropna().unique().tolist()):
                syn = sub[sub["synthetic"] == synthetic].sort_values("step_in_day")
                ax.plot(syn["step_in_day"], syn["synthetic_mean"], linewidth=1.8, label=synthetic)
            ax.set_title(f"Daily profile | subset={subset} | variable={variable}")
            ax.set_xlabel("step_in_day")
            ax.set_ylabel(variable)
            ax.legend()
            path = output_dir / f"profile_{subset}_{variable}.{image_format}"
            _save_plot(fig, path, dpi)
            plt.close(fig)
            out[path.name] = str(path)
    return out


def _plot_acf_curves(plt, tables_dir: Path, output_dir: Path, image_format: str, dpi: int) -> Dict[str, str]:
    curves = _read_csv_if_exists(tables_dir / "acf_curves.csv")
    if curves is None or curves.empty:
        return {}
    out: Dict[str, str] = {}
    for variable in TARGET_COLS:
        sub = curves[curves["variable"] == variable].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        real_series = sub.sort_values("lag")[["lag", "real_acf"]].drop_duplicates()
        ax.plot(real_series["lag"], real_series["real_acf"], marker="o", linewidth=2.2, label="real")
        for synthetic in sorted(sub["synthetic"].dropna().unique().tolist()):
            syn = sub[sub["synthetic"] == synthetic].sort_values("lag")
            ax.plot(syn["lag"], syn["synthetic_acf"], marker="o", linewidth=1.8, label=synthetic)
        ax.set_title(f"Daily mean ACF | variable={variable}")
        ax.set_xlabel("lag")
        ax.set_ylabel("ACF")
        ax.set_ylim(-1.05, 1.05)
        ax.legend()
        path = output_dir / f"acf_{variable}.{image_format}"
        _save_plot(fig, path, dpi)
        plt.close(fig)
        out[path.name] = str(path)
    return out


def _plot_cross_correlation(plt, tables_dir: Path, output_dir: Path, image_format: str, dpi: int) -> Dict[str, str]:
    curves = _read_csv_if_exists(tables_dir / "cross_correlation_curves.csv")
    if curves is None or curves.empty:
        return {}
    out: Dict[str, str] = {}
    for pair in sorted(curves["pair"].dropna().unique().tolist()):
        sub = curves[curves["pair"] == pair].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        real_series = sub.sort_values("lag")[["lag", "real_crosscorr"]].drop_duplicates()
        ax.plot(real_series["lag"], real_series["real_crosscorr"], marker="o", linewidth=2.2, label="real")
        for synthetic in sorted(sub["synthetic"].dropna().unique().tolist()):
            syn = sub[sub["synthetic"] == synthetic].sort_values("lag")
            ax.plot(syn["lag"], syn["synthetic_crosscorr"], marker="o", linewidth=1.8, label=synthetic)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(f"Lead-lag cross correlation | pair={pair}")
        ax.set_xlabel("lag")
        ax.set_ylabel("cross correlation")
        ax.set_ylim(-1.05, 1.05)
        ax.legend()
        safe_name = pair.replace("|", "_")
        path = output_dir / f"crosscorr_{safe_name}.{image_format}"
        _save_plot(fig, path, dpi)
        plt.close(fig)
        out[path.name] = str(path)
    return out


def _plot_metric_bars(
    plt,
    df: pd.DataFrame | None,
    value_cols: Iterable[str],
    group_col: str,
    x_col: str,
    title_prefix: str,
    output_dir: Path,
    image_format: str,
    dpi: int,
) -> Dict[str, str]:
    if df is None or df.empty:
        return {}
    out: Dict[str, str] = {}
    for value_col in value_cols:
        if value_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        groups = sorted(df[group_col].dropna().unique().tolist()) if group_col in df.columns else [None]
        x_labels = sorted(df[x_col].dropna().unique().tolist()) if x_col in df.columns else list(range(len(df)))
        x = np.arange(len(x_labels), dtype=float)
        width = 0.8 / max(1, len(groups))
        for idx, group in enumerate(groups):
            if group is None:
                sub = df.copy()
                label = value_col
            else:
                sub = df[df[group_col] == group].copy()
                label = str(group)
            vals = []
            for item in x_labels:
                row = sub[sub[x_col] == item]
                vals.append(float(row[value_col].iloc[0]) if not row.empty else np.nan)
            offset = (idx - (len(groups) - 1) / 2.0) * width
            ax.bar(x + offset, vals, width=width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x_labels])
        ax.set_title(f"{title_prefix} | {value_col}")
        ax.set_xlabel(x_col)
        ax.set_ylabel(value_col)
        if len(groups) > 1:
            ax.legend()
        safe_name = f"{title_prefix.lower().replace(' ', '_')}_{value_col}.{image_format}"
        path = output_dir / safe_name
        _save_plot(fig, path, dpi)
        plt.close(fig)
        out[path.name] = str(path)
    return out


def _plot_correlation_heatmaps(plt, matrices_dir: Path, output_dir: Path, image_format: str, dpi: int) -> Dict[str, str]:
    if not matrices_dir.exists():
        return {}
    out: Dict[str, str] = {}
    for csv_path in sorted(matrices_dir.glob("*.csv")):
        df = pd.read_csv(csv_path, index_col=0)
        if df.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, 4.5))
        im = ax.imshow(df.to_numpy(dtype=float), aspect="auto")
        ax.set_xticks(np.arange(len(df.columns)))
        ax.set_yticks(np.arange(len(df.index)))
        ax.set_xticklabels(df.columns.tolist())
        ax.set_yticklabels(df.index.tolist())
        ax.set_title(csv_path.stem)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                ax.text(j, i, f"{float(df.iloc[i, j]):.2f}", ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path = output_dir / f"{csv_path.stem}.{image_format}"
        _save_plot(fig, path, dpi)
        plt.close(fig)
        out[path.name] = str(path)
    return out


def create_evaluation_plots(
    eval_dir: str | Path,
    output_dir: str | Path | None = None,
    image_format: str = "png",
    dpi: int = 160,
) -> Dict[str, str]:
    plt = _import_matplotlib()
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_path}")
    out_dir = ensure_dir(output_dir or eval_path / "plots")
    tables_dir = eval_path / "tables"
    matrices_dir = eval_path / "correlation_matrices"
    daily_dir = eval_path / "daily_scores"

    plot_paths: Dict[str, str] = {}
    plot_paths.update(_plot_stress_distributions(plt, daily_dir, out_dir, image_format, dpi))
    plot_paths.update(_plot_conditional_profiles(plt, tables_dir, out_dir, image_format, dpi))
    plot_paths.update(_plot_acf_curves(plt, tables_dir, out_dir, image_format, dpi))
    plot_paths.update(_plot_cross_correlation(plt, tables_dir, out_dir, image_format, dpi))
    plot_paths.update(_plot_correlation_heatmaps(plt, matrices_dir, out_dir, image_format, dpi))

    distribution = _read_csv_if_exists(tables_dir / "distribution_comparison.csv")
    plot_paths.update(
        _plot_metric_bars(
            plt,
            distribution,
            value_cols=["quantile_rmse", "wasserstein_approx", "ks_stat"],
            group_col="synthetic",
            x_col="variable",
            title_prefix="distribution",
            output_dir=out_dir,
            image_format=image_format,
            dpi=dpi,
        )
    )

    ramp = _read_csv_if_exists(tables_dir / "ramp_comparison.csv")
    plot_paths.update(
        _plot_metric_bars(
            plt,
            ramp,
            value_cols=["quantile_rmse", "wasserstein_approx", "ks_stat"],
            group_col="synthetic",
            x_col="variable",
            title_prefix="ramp",
            output_dir=out_dir,
            image_format=image_format,
            dpi=dpi,
        )
    )

    target_control = _read_csv_if_exists(tables_dir / "scenario_target_control.csv")
    plot_paths.update(
        _plot_metric_bars(
            plt,
            target_control,
            value_cols=["mean_abs_target_error", "within_0.05", "within_0.10"],
            group_col="dataset",
            x_col="dataset",
            title_prefix="scenario_control",
            output_dir=out_dir,
            image_format=image_format,
            dpi=dpi,
        )
    )

    discrim = _read_csv_if_exists(tables_dir / "discriminator_metrics.csv")
    plot_paths.update(
        _plot_metric_bars(
            plt,
            discrim,
            value_cols=["auc", "balanced_accuracy"],
            group_col="synthetic",
            x_col="synthetic",
            title_prefix="discriminator",
            output_dir=out_dir,
            image_format=image_format,
            dpi=dpi,
        )
    )

    pred = _read_csv_if_exists(tables_dir / "predictive_utility_overall.csv")
    plot_paths.update(
        _plot_metric_bars(
            plt,
            pred,
            value_cols=["mae_ratio_to_real_baseline", "rmse_ratio_to_real_baseline"],
            group_col="synthetic",
            x_col="synthetic",
            title_prefix="predictive_utility",
            output_dir=out_dir,
            image_format=image_format,
            dpi=dpi,
        )
    )

    memo = _read_csv_if_exists(tables_dir / "memorization_diversity_metrics.csv")
    plot_paths.update(
        _plot_metric_bars(
            plt,
            memo,
            value_cols=["nearest_neighbor_mean", "diversity_ratio_to_real"],
            group_col="synthetic",
            x_col="synthetic",
            title_prefix="memorization_diversity",
            output_dir=out_dir,
            image_format=image_format,
            dpi=dpi,
        )
    )

    save_json({"plot_files": plot_paths}, out_dir / "plot_manifest.json")
    return plot_paths
