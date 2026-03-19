#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_generate_gan_scenarios_v4.py

Fix for PyTorch 2.6 torch.load default weights_only=True.
Loads checkpoint with weights_only=False (fallback for older torch).

Supports both NPZ formats (VAE-style / CVAE-style).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
        )
    def forward(self, x):
        return F.leaky_relu(x + self.net(x), 0.2, inplace=True)

class Generator(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, out_dim: int, base_ch: int = 256, T: int = 96):
        super().__init__()
        self.T = T
        self.fc = nn.Sequential(
            nn.Linear(z_dim + cond_dim, base_ch * T),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = nn.Sequential(
            ResBlock1D(base_ch),
            ResBlock1D(base_ch),
            nn.Conv1d(base_ch, base_ch // 2, 3, padding=1),
            nn.GroupNorm(8, base_ch // 2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(base_ch // 2),
            nn.Conv1d(base_ch // 2, out_dim, 1),
        )
    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        h = self.fc(x).view(x.size(0), -1, self.T)
        return self.conv(h)


def torch_load_any(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sanitize(s: str) -> str:
    s = s.strip().replace(" ", "")
    s = s.replace(",", "-")
    s = re.sub(r"[^a-zA-Z0-9._\-+=]", "_", s)
    return s


def _get_first_existing(data: np.lib.npyio.NpzFile, keys: List[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in data.files:
            return data[k]
    return None


def load_npz_any(npz_path: str) -> Dict[str, object]:
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)
    if "X" not in data.files:
        raise KeyError(f"NPZ must contain 'X'. Found keys={keys}")

    is_cvae = ("x_cols" in data.files) and ("x_mean" in data.files) and ("x_std" in data.files)
    if is_cvae:
        return {
            "format": "cvae",
            "keys": keys,
            "X": data["X"].astype(np.float32),
            "cols": data["x_cols"].tolist(),
            "mean": data["x_mean"].astype(np.float32),
            "std": data["x_std"].astype(np.float32),
            "log1p_cols": data["log1p_x_cols"].tolist() if "log1p_x_cols" in data.files else [],
            "C": data["C"].astype(np.float32) if "C" in data.files else None,
            "c_cols": data["c_cols"].tolist() if "c_cols" in data.files else [],
        }

    if "cols" in data.files:
        cols = data["cols"].tolist()
    elif "columns" in data.files:
        cols = data["columns"].tolist()
    else:
        raise KeyError(f"NPZ must contain 'cols' (or 'columns') for VAE-style. Found keys={keys}")

    mean = _get_first_existing(data, ["mean", "mu", "scaler_mean"])
    std  = _get_first_existing(data, ["std", "sigma", "scaler_std"])

    return {
        "format": "vae",
        "keys": keys,
        "X": data["X"].astype(np.float32),
        "cols": cols,
        "mean": None if mean is None else mean.astype(np.float32),
        "std": None if std is None else std.astype(np.float32),
        "log1p_cols": [],
        "C": None,
        "c_cols": [],
    }


def positional_sincos(T: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(T, dtype=np.float32)
    ang = 2.0 * np.pi * (t / float(T))
    return np.sin(ang).astype(np.float32), np.cos(ang).astype(np.float32)


def build_dow_onehot_from_c(C: np.ndarray, c_cols: List[str]) -> np.ndarray:
    need = [f"dow_{k}" for k in range(7)]
    if not all(k in c_cols for k in need):
        raise ValueError(f"Need dow_0..dow_6 in c_cols. Got c_cols={c_cols}")
    idx = [c_cols.index(k) for k in need]
    dow = C[:, 0, idx]
    hard = np.zeros_like(dow)
    hard[np.arange(dow.shape[0]), np.argmax(dow, axis=1)] = 1.0
    return hard.astype(np.float32)


def build_dow_onehot_from_x(X: np.ndarray, cols: List[str]) -> np.ndarray:
    need = [f"dow_{k}" for k in range(7)]
    if not all(k in cols for k in need):
        raise ValueError("Need dow_0..dow_6 in NPZ cols (VAE-style) for conditioning.")
    idx = [cols.index(k) for k in need]
    dow = X[:, 0, idx]
    hard = np.zeros_like(dow)
    hard[np.arange(dow.shape[0]), np.argmax(dow, axis=1)] = 1.0
    return hard.astype(np.float32)


def inverse_transform(y_std: np.ndarray, cols: List[str], mean: Optional[np.ndarray], std: Optional[np.ndarray], log1p_cols: List[str]) -> np.ndarray:
    if mean is None or std is None:
        return y_std
    y = y_std.copy().astype(np.float32)
    logset = set(log1p_cols or [])
    for j, col in enumerate(cols):
        y[:, :, j] = y[:, :, j] * std[j] + mean[j]
        if col in logset:
            y[:, :, j] = np.expm1(np.clip(y[:, :, j], -20.0, 20.0))
    return y


def extract_price_levels_and_template(npz: Dict[str, object], price_col: str = "price", decimals: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    cols = npz["cols"]
    if price_col not in cols:
        return np.array([], dtype=np.float32), np.zeros((96,), dtype=np.float32)
    pi = cols.index(price_col)
    Xstd = npz["X"][:, :, [pi]]

    mean = npz["mean"]; std = npz["std"]
    if mean is not None and std is not None:
        Xp = Xstd[:, :, 0] * std[pi] + mean[pi]
    else:
        Xp = Xstd[:, :, 0]

    Xp = np.round(Xp.astype(np.float32), decimals=decimals)
    flat = Xp.reshape(-1)
    levels = np.unique(flat[np.isfinite(flat)])
    levels.sort()

    template = np.zeros((96,), dtype=np.float32)
    for t in range(96):
        v = Xp[:, t]
        v = v[np.isfinite(v)]
        if v.size == 0:
            template[t] = 0.0
        else:
            vals, counts = np.unique(v, return_counts=True)
            template[t] = vals[np.argmax(counts)]
    return levels.astype(np.float32), template.astype(np.float32)


def project_to_levels(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    if levels.size == 0:
        return x
    xflat = x.reshape(-1)
    out = np.empty_like(xflat, dtype=np.float32)
    chunk = 200000
    for i in range(0, xflat.size, chunk):
        a = xflat[i:i+chunk][:, None]
        idx = np.argmin(np.abs(a - levels[None, :]), axis=1)
        out[i:i+chunk] = levels[idx]
    return out.reshape(x.shape)


def perturb_price_template(template: np.ndarray, levels: np.ndarray,
                           delta_low: float, delta_mid: float, delta_high: float,
                           global_scale_min: float, global_scale_max: float,
                           rng: np.random.Generator) -> np.ndarray:
    p = template.copy().astype(np.float32)
    p *= rng.uniform(global_scale_min, global_scale_max)
    if levels.size >= 3:
        q1 = np.quantile(levels, 0.33)
        q2 = np.quantile(levels, 0.66)
        for i in range(p.size):
            if p[i] <= q1:
                p[i] *= rng.uniform(1 - delta_low, 1 + delta_low)
            elif p[i] <= q2:
                p[i] *= rng.uniform(1 - delta_mid, 1 + delta_mid)
            else:
                p[i] *= rng.uniform(1 - delta_high, 1 + delta_high)
    return p.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--generate_days", type=int, default=300)
    ap.add_argument("--gen_name", default="mainB")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z_scale", type=float, default=1.0)

    ap.add_argument("--price_mode", default="project_levels", choices=["raw", "project_levels", "template_perturb"])
    ap.add_argument("--price_project_after_perturb", action="store_true")
    ap.add_argument("--price_delta_low", type=float, default=0.08)
    ap.add_argument("--price_delta_mid", type=float, default=0.15)
    ap.add_argument("--price_delta_high", type=float, default=0.40)
    ap.add_argument("--price_global_scale_min", type=float, default=0.75)
    ap.add_argument("--price_global_scale_max", type=float, default=1.35)

    ap.add_argument("--poisson_lambda", action="store_true")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--start_day_id", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    ckpt_path = Path(args.ckpt)
    ckpt = torch_load_any(ckpt_path)

    target_cols = ckpt["target_cols"]
    z_dim = ckpt["latent_dim"]
    g_ch = ckpt["g_channels"]

    npz = load_npz_any(args.npz)
    print("[NPZ format]", npz["format"])
    print("[NPZ keys]", npz["keys"])

    dow_real = build_dow_onehot_from_c(npz["C"], npz["c_cols"]) if npz["format"] == "cvae" else build_dow_onehot_from_x(npz["X"], npz["cols"])

    mean = ckpt.get("mean", None)
    std  = ckpt.get("std", None)
    if isinstance(mean, torch.Tensor): mean = mean.cpu().numpy().astype(np.float32)
    if isinstance(std, torch.Tensor):  std = std.cpu().numpy().astype(np.float32)
    if mean is None: mean = npz["mean"]
    if std is None:  std = npz["std"]
    log1p_cols = ckpt.get("log1p_cols", npz.get("log1p_cols", []))

    price_levels, price_template = extract_price_levels_and_template(npz, "price", decimals=6)

    run_dir = Path(args.out_dir).expanduser() if args.out_dir.strip() else ckpt_path.parent
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"gen_{ts}_{_sanitize(args.gen_name)}"
    gen_dir = run_dir / base
    if gen_dir.exists():
        gen_dir = run_dir / f"{base}_{uuid.uuid4().hex[:6]}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(z_dim, 7, len(target_cols), base_ch=g_ch).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    sin_t, cos_t = positional_sincos(96)

    rows = []
    B = 256
    produced = 0
    while produced < args.generate_days:
        b = min(B, args.generate_days - produced)
        c = dow_real[rng.integers(0, dow_real.shape[0], size=b)]
        z = rng.standard_normal(size=(b, z_dim)).astype(np.float32) * float(args.z_scale)

        ct = torch.from_numpy(c).to(device)
        zt = torch.from_numpy(z).to(device)
        with torch.no_grad():
            y = G(zt, ct).cpu().numpy()
        y = y.transpose(0, 2, 1).astype(np.float32)

        y = inverse_transform(y, target_cols, mean, std, log1p_cols)

        for j, col in enumerate(target_cols):
            if col in ["load", "load_sum", "lambda", "lambda_public"]:
                y[:, :, j] = np.clip(y[:, :, j], 0.0, None)

        if args.poisson_lambda:
            for j, col in enumerate(target_cols):
                if col in ["lambda", "lambda_public"]:
                    y[:, :, j] = rng.poisson(np.clip(y[:, :, j], 0.0, 1e6)).astype(np.float32)

        if "price" in target_cols:
            pj = target_cols.index("price")
            if args.price_mode == "project_levels":
                y[:, :, pj] = project_to_levels(y[:, :, pj], price_levels)
                if price_levels.size > 0:
                    y[:, :, pj] = np.clip(y[:, :, pj], float(price_levels.min()), float(price_levels.max()))
            elif args.price_mode == "template_perturb":
                for bi in range(b):
                    p = perturb_price_template(
                        price_template, price_levels,
                        args.price_delta_low, args.price_delta_mid, args.price_delta_high,
                        args.price_global_scale_min, args.price_global_scale_max,
                        rng
                    )
                    if args.price_project_after_perturb:
                        p = project_to_levels(p, price_levels)
                    y[bi, :, pj] = p

        for bi in range(b):
            dow_k = int(np.argmax(c[bi]))
            is_we = 1 if dow_k >= 5 else 0
            day_id = args.start_day_id + produced + bi
            for t in range(96):
                rec = {
                    "day_id": day_id,
                    "t": t,
                    "day_of_week": dow_k,
                    "is_weekend": is_we,
                    "sin_hour": float(sin_t[t]),
                    "cos_hour": float(cos_t[t]),
                }
                for j, col in enumerate(target_cols):
                    rec[col] = float(y[bi, t, j])
                rows.append(rec)

        produced += b

    df = pd.DataFrame(rows)
    out_csv = gen_dir / "generated_days.csv"
    df.to_csv(out_csv, index=False)
    (gen_dir / "gen_config.json").write_text(json.dumps({
        "ckpt": str(ckpt_path),
        "npz": str(args.npz),
        "target_cols": target_cols,
        "price_mode": args.price_mode,
        "poisson_lambda": bool(args.poisson_lambda),
        "z_scale": float(args.z_scale),
        "generate_days": int(args.generate_days),
        "gen_name": args.gen_name,
    }, indent=2), encoding="utf-8")

    print("[Saved]", out_csv)
    print("[Gen dir]", gen_dir)


if __name__ == "__main__":
    main()
