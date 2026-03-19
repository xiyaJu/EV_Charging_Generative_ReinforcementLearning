#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

def resolve_path(path: str) -> str:
    p = Path(path)
    if p.exists(): return str(p)
    alt = Path.cwd() / p.name
    if alt.exists(): return str(alt)
    alt2 = Path(__file__).resolve().parent / p.name
    if alt2.exists(): return str(alt2)
    raise FileNotFoundError(f"File not found: {path}")

def standardize(X: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, eps).astype(np.float32)
    Xn = ((X - mean) / std).astype(np.float32)
    return Xn, mean, std

def build_tensor_by_day(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    df = df.sort_values(["day_id", "t"]).reset_index(drop=True)
    counts = df.groupby("day_id")["t"].count()
    bad = counts[counts != 96]
    if len(bad) > 0:
        raise ValueError(f"Found incomplete days (not 96 rows): {bad.head(20).to_dict()}")
    day_ids = df["day_id"].unique()
    N, T, D = len(day_ids), 96, len(cols)
    out = np.zeros((N, T, D), dtype=np.float32)
    for i, did in enumerate(day_ids):
        block = df[df["day_id"] == did].sort_values("t")
        tvals = block["t"].to_numpy()
        if not np.array_equal(tvals, np.arange(96)):
            raise ValueError(f"day_id={did} has non 0..95 t.")
        out[i] = block[cols].to_numpy(dtype=np.float32)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_csv", required=True)
    ap.add_argument("--out_npz", required=True)

    ap.add_argument("--x_cols", default="price,load,lambda",
                    help="Targets to generate, comma-separated.")
    ap.add_argument("--log1p_x_cols", default="lambda",
                    help="Apply log1p to these X cols (recommend: lambda or lambda,load).")

    args = ap.parse_args()
    df = pd.read_csv(resolve_path(args.scene_csv))

    for c in ["day_id", "t"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    # ---------- deterministic/time/calendar in C ----------
    # sin/cos from t if missing
    if "sin_hour" not in df.columns or "cos_hour" not in df.columns:
        t = pd.to_numeric(df["t"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        angle = 2.0 * np.pi * (t / 96.0)
        df["sin_hour"] = np.sin(angle).astype(np.float32)
        df["cos_hour"] = np.cos(angle).astype(np.float32)

    if "day_of_week" not in df.columns:
        raise ValueError("Need 'day_of_week' in scene CSV (0..6).")
    dow = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0).round().clip(0, 6).astype(int)

    # one-hot dow_0..dow_6 + is_weekend
    for k in range(7):
        df[f"dow_{k}"] = (dow == k).astype(np.float32)
    df["is_weekend"] = (dow >= 5).astype(np.float32)

    c_cols = ["sin_hour", "cos_hour"] + [f"dow_{k}" for k in range(7)] + ["is_weekend"]

    # ---------- X (targets to generate) ----------
    x_cols = [c.strip() for c in args.x_cols.split(",") if c.strip()]
    for c in x_cols:
        if c not in df.columns:
            raise ValueError(f"Missing target col '{c}' in CSV")

    # numeric coercion
    for c in x_cols + c_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # keep a copy (raw scale) for day-metrics (extreme sampling)
    df_raw = df.copy()

    log1p_x_cols = [c.strip() for c in args.log1p_x_cols.split(",") if c.strip()] if args.log1p_x_cols.strip() else []
    for c in log1p_x_cols:
        if c not in x_cols:
            raise ValueError(f"log1p_x_cols '{c}' must be in x_cols={x_cols}")
        x = np.clip(df[c].to_numpy(dtype=np.float32), 0.0, None)
        df[c] = np.log1p(x)

    # tensors
    X_raw = build_tensor_by_day(df, x_cols)   # after log1p if applied
    C = build_tensor_by_day(df, c_cols)

    X, x_mean, x_std = standardize(X_raw)

    # day-level metrics on ORIGINAL scale (before log1p)
    # load_sum, lambda_max for extreme-focused sampling
    day_load_sum = None
    day_lambda_max = None
    if "load" in x_cols:
        tmp = build_tensor_by_day(df_raw, ["load"])  # (N,96,1)
        day_load_sum = tmp.sum(axis=1).squeeze(-1).astype(np.float32)
    if "lambda" in x_cols:
        tmp = build_tensor_by_day(df_raw, ["lambda"])
        day_lambda_max = tmp.max(axis=1).squeeze(-1).astype(np.float32)

    meta: Dict[str, Any] = {"scene_csv": str(args.scene_csv), "N_days": int(X.shape[0]), "T": 96,
                            "Dx": int(X.shape[2]), "Dc": int(C.shape[2])}

    np.savez_compressed(
        args.out_npz,
        X=X, x_mean=x_mean, x_std=x_std,
        x_cols=np.array(x_cols, dtype=object),
        log1p_x_cols=np.array(log1p_x_cols, dtype=object),
        C=C, c_cols=np.array(c_cols, dtype=object),
        day_load_sum=day_load_sum if day_load_sum is not None else np.array([], dtype=np.float32),
        day_lambda_max=day_lambda_max if day_lambda_max is not None else np.array([], dtype=np.float32),
        meta=np.array(meta, dtype=object),
    )
    print(f"[OK] Wrote NPZ: {args.out_npz}")
    print(f"     X targets: {x_cols}, log1p: {log1p_x_cols}")
    print(f"     C conds: {c_cols}")

if __name__ == "__main__":
    main()
