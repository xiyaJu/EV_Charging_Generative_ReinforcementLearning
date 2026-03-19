#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP-CVAE training + generation for day-level EV scenarios (T=96).
This version adds:
  - price legality / stress-test modes (Scheme B/A)
  - optional --model_dir to skip training and only generate using an existing run
  - generation outputs stored under run_dir/gen_<timestamp>_<tag>/ to avoid overwriting
  - auto run_name built from key hyper-params when --run_name is not provided
"""
from __future__ import annotations
import argparse, json, datetime, uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# ----------------- utils -----------------
def resolve_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    alt = Path.cwd() / p.name
    if alt.exists():
        return str(alt)
    alt2 = Path(__file__).resolve().parent / p.name
    if alt2.exists():
        return str(alt2)
    raise FileNotFoundError(f"File not found: {path}")


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(npz_path: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "meta" in out and isinstance(out["meta"], np.ndarray) and out["meta"].dtype == object:
        out["meta"] = out["meta"].item()
    return out


def split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(n * val_ratio))
    return idx[n_val:], idx[:n_val]


def make_run_dir(base: Path, run_name: str, no_ts: bool) -> Path:
    if no_ts:
        return base
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = f"{ts}_{run_name.strip()}" if run_name.strip() else ts
    rd = base / sub
    if rd.exists():
        rd = base / f"{sub}_{uuid.uuid4().hex[:6]}"
    return rd


def make_gen_dir(run_dir: Path, gen_tag: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = gen_tag.strip().replace(" ", "_")
    sub = f"gen_{ts}_{tag}" if tag else f"gen_{ts}"
    gd = run_dir / sub
    if gd.exists():
        gd = run_dir / f"{sub}_{uuid.uuid4().hex[:6]}"
    gd.mkdir(parents=True, exist_ok=True)
    return gd


def auto_run_name(args: argparse.Namespace) -> str:
    # compact but informative
    hd = "-".join([x.strip() for x in args.hidden_dims.split(",") if x.strip()]) or "na"
    parts = [
        f"cvae",
        f"beta{args.beta:g}",
        f"z{args.latent_dim}",
        f"hd{hd}",
        f"pm{args.price_mode}",
        f"zs{args.z_source}",
        f"zsc{args.z_scale:g}",
    ]
    if args.extreme_metric != "none":
        parts += [f"ext{args.extreme_metric}", f"top{args.extreme_top_pct:g}", f"w{args.extreme_weight:g}"]
    return "_".join(parts)


# ----------------- dataset/model -----------------
class NpzCvaeDataset(Dataset):
    def __init__(self, X: np.ndarray, C: np.ndarray):
        self.X = X.astype(np.float32)
        self.C = C.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.C[i])


class MLP_CVAE(nn.Module):
    def __init__(self, x_dim: int, c_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        in_dim = x_dim + c_dim

        enc = []
        d = in_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)

        dec = []
        d2 = latent_dim + c_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(d2, h), nn.ReLU()]
            d2 = h
        dec += [nn.Linear(d2, x_dim)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor, c: torch.Tensor):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        mu, logvar = self.encode(x, c)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        x_hat = self.decoder(torch.cat([z, c], dim=1))
        return x_hat, mu, logvar

    def decode(self, z: torch.Tensor, c: torch.Tensor):
        return self.decoder(torch.cat([z, c], dim=1))


def loss_fn(x_hat, x, mu, logvar, beta: float):
    recon = torch.mean((x_hat - x) ** 2)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.item(), kl.item()


# ----------------- postprocess helpers -----------------
def inverse_x(X_std: np.ndarray, mean: np.ndarray, std: np.ndarray, cols: List[str], log1p_cols: List[str]) -> np.ndarray:
    X = X_std * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    col_to_i = {c: i for i, c in enumerate(cols)}
    for c in log1p_cols:
        if c in col_to_i:
            i = col_to_i[c]
            X[..., i] = np.expm1(X[..., i])
    return X


def clip_nonneg(X: np.ndarray, cols: List[str]) -> np.ndarray:
    col_to_i = {c: i for i, c in enumerate(cols)}
    for c in cols:
        if c.lower() in ["price", "load", "lambda", "lambda_public", "load_sum", "precip", "rh", "temp"]:
            i = col_to_i[c]
            X[..., i] = np.clip(X[..., i], 0.0, None)
    return X


def poissonize_lambda(X: np.ndarray, cols: List[str], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    col_to_i = {c: i for i, c in enumerate(cols)}
    for cname in ["lambda", "lambda_public"]:
        if cname in col_to_i:
            i = col_to_i[cname]
            lam = np.clip(X[..., i], 0.0, None)
            X[..., i] = rng.poisson(lam).astype(np.float32)
    return X


def get_price_levels_from_training(
    X_std: np.ndarray,
    x_cols: List[str],
    x_mean: np.ndarray,
    x_std_vec: np.ndarray,
    decimals: int = 6,
) -> Optional[np.ndarray]:
    """Extract unique price levels from training data (raw domain), robust to float noise."""
    if "price" not in x_cols:
        return None
    pi = x_cols.index("price")
    prices_raw = X_std[..., pi].astype(np.float64) * float(x_std_vec[pi]) + float(x_mean[pi])
    prices_raw = np.round(prices_raw, decimals=decimals)
    levels = np.unique(prices_raw.reshape(-1))
    levels = np.sort(levels)
    return levels.astype(np.float32)


def project_to_levels(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Project values to nearest element in sorted levels (vectorized)."""
    if levels is None or len(levels) == 0:
        return values
    lv = levels.astype(np.float64)
    v = values.astype(np.float64)
    idx = np.searchsorted(lv, v, side="left")
    idx0 = np.clip(idx - 1, 0, len(lv) - 1)
    idx1 = np.clip(idx, 0, len(lv) - 1)
    v0 = lv[idx0]
    v1 = lv[idx1]
    choose = np.where(np.abs(v - v0) <= np.abs(v - v1), v0, v1)
    return choose.astype(values.dtype)


def template_perturb_prices(
    template_prices: np.ndarray,
    rng: np.random.Generator,
    delta_low: float,
    delta_mid: float,
    delta_high: float,
    global_scale_min: float,
    global_scale_max: float,
    decimals: int = 6,
    add_noise_sigma: float = 0.0,
) -> np.ndarray:
    """
    Scheme A: take a real day step-template price (typ. 3~4 levels),
    then apply group-wise multipliers (low/mid/high) + optional global scaling.
    This preserves the step structure while creating diverse/extreme tariff scenarios.
    """
    p0 = template_prices.astype(np.float64)
    # robust levels
    p0r = np.round(p0, decimals=decimals)
    levels = np.sort(np.unique(p0r))
    if len(levels) == 0:
        return template_prices

    # assign each level to low/mid/high by rank (thirds)
    lev_groups = {"low": [], "mid": [], "high": []}
    if len(levels) == 1:
        lev_groups["mid"] = [levels[0]]
    else:
        for i, lv in enumerate(levels):
            frac = i / max(len(levels) - 1, 1)
            if frac <= 1/3:
                lev_groups["low"].append(lv)
            elif frac <= 2/3:
                lev_groups["mid"].append(lv)
            else:
                lev_groups["high"].append(lv)

    m_low = rng.uniform(1 - delta_low, 1 + delta_low)
    m_mid = rng.uniform(1 - delta_mid, 1 + delta_mid)
    m_high = rng.uniform(1 - delta_high, 1 + delta_high)
    g = rng.uniform(global_scale_min, global_scale_max)

    # map each time point to nearest template level
    p_lvl = project_to_levels(p0r, levels).astype(np.float64)
    p_new = p_lvl.copy()

    if len(lev_groups["low"]) > 0:
        mask = np.isin(p_lvl, np.array(lev_groups["low"]))
        p_new[mask] = p_lvl[mask] * m_low
    if len(lev_groups["mid"]) > 0:
        mask = np.isin(p_lvl, np.array(lev_groups["mid"]))
        p_new[mask] = p_lvl[mask] * m_mid
    if len(lev_groups["high"]) > 0:
        mask = np.isin(p_lvl, np.array(lev_groups["high"]))
        p_new[mask] = p_lvl[mask] * m_high

    p_new = p_new * g
    if add_noise_sigma > 0:
        p_new = p_new + rng.normal(0.0, add_noise_sigma, size=p_new.shape)

    return np.clip(p_new, 0.0, None).astype(np.float32)


def apply_price_mode(
    X_gen_raw: np.ndarray,
    x_cols: List[str],
    mode: str,
    price_levels: Optional[np.ndarray],
    X_train_std: np.ndarray,
    idx_template: np.ndarray,
    x_mean: np.ndarray,
    x_std_vec: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    Modify price column in X_gen_raw in-place according to the selected mode.
      - none: keep decoder output
      - project_levels: Scheme B, snap to observed tariff levels
      - template_perturb: Scheme A, ignore decoder price and use template+perturbation
    """
    if "price" not in x_cols:
        return X_gen_raw
    pi = x_cols.index("price")

    if mode == "none":
        return X_gen_raw

    if mode == "project_levels":
        if price_levels is None or len(price_levels) == 0:
            print("[WARN] price_mode=project_levels but no levels found; keep raw price.")
            return X_gen_raw
        X_gen_raw[:, :, pi] = project_to_levels(X_gen_raw[:, :, pi], price_levels)
        return X_gen_raw

    if mode == "template_perturb":
        # build template from matched real day (same idx as calendar C by default)
        for k, i0 in enumerate(idx_template.tolist()):
            p_tpl = (X_train_std[i0, :, pi].astype(np.float64) * float(x_std_vec[pi]) + float(x_mean[pi])).astype(np.float32)
            p_new = template_perturb_prices(
                template_prices=p_tpl,
                rng=rng,
                delta_low=args.price_delta_low,
                delta_mid=args.price_delta_mid,
                delta_high=args.price_delta_high,
                global_scale_min=args.price_global_scale_min,
                global_scale_max=args.price_global_scale_max,
                decimals=args.price_level_decimals,
                add_noise_sigma=args.price_add_noise_sigma,
            )
            X_gen_raw[k, :, pi] = p_new

        # optional: snap back to original discrete levels (for "legal but stressed")
        if args.price_project_after_perturb and price_levels is not None and len(price_levels) > 0:
            X_gen_raw[:, :, pi] = project_to_levels(X_gen_raw[:, :, pi], price_levels)

        return X_gen_raw

    raise ValueError(f"Unknown price_mode: {mode}")


# ----------------- calendar recovery for output -----------------
def recover_calendar_cols(C_day: np.ndarray, c_cols: List[str]) -> Dict[str, np.ndarray]:
    # C_day: (N,96,Dc)
    col_to_i = {c: i for i, c in enumerate(c_cols)}
    sin_hour = C_day[:, :, col_to_i["sin_hour"]] if "sin_hour" in col_to_i else None
    cos_hour = C_day[:, :, col_to_i["cos_hour"]] if "cos_hour" in col_to_i else None

    # reconstruct day_of_week from one-hot dow_0..dow_6
    dow_idx = [col_to_i.get(f"dow_{k}", -1) for k in range(7)]
    if all(i >= 0 for i in dow_idx):
        onehot = C_day[:, :, dow_idx]  # (N,96,7)
        day_of_week = np.argmax(onehot, axis=2).astype(np.float32)
    else:
        day_of_week = np.zeros((C_day.shape[0], C_day.shape[1]), dtype=np.float32)

    is_weekend = (day_of_week >= 5).astype(np.float32)
    return {"sin_hour": sin_hour, "cos_hour": cos_hour, "day_of_week": day_of_week, "is_weekend": is_weekend}


def write_csv(out_csv: str, X_raw: np.ndarray, x_cols: List[str], C: np.ndarray, c_cols: List[str]) -> None:
    import pandas as pd
    N, T, Dx = X_raw.shape
    cal = recover_calendar_cols(C, c_cols)
    rows = []
    for day in range(N):
        for t in range(T):
            row = {"day_id": day, "t": t}
            # deterministic/calendar cols (guaranteed valid)
            if cal["sin_hour"] is not None:
                row["sin_hour"] = float(cal["sin_hour"][day, t])
            if cal["cos_hour"] is not None:
                row["cos_hour"] = float(cal["cos_hour"][day, t])
            row["day_of_week"] = float(cal["day_of_week"][day, t])
            row["is_weekend"] = float(cal["is_weekend"][day, t])
            # generated targets
            for j, c in enumerate(x_cols):
                row[c] = float(X_raw[day, t, j])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] wrote {out_csv} (days={N}, rows={len(df)})")


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--npz", required=True, help="NPZ produced by prepare script (contains X,C,x_mean,x_std,cols...)")
    ap.add_argument("--out_dir", required=True, help="Base directory to store runs/")
    ap.add_argument("--run_name", default="", help="Optional run name. If empty, auto-generated from args.")
    ap.add_argument("--no_timestamp", action="store_true", help="Do not create timestamp subdir under out_dir.")

    # Resume / generate-only mode
    ap.add_argument("--model_dir", default="", help="If set, skip training and load model.pt from this directory.")

    # training params
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.003)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dims", default="512,512")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    # generation params
    ap.add_argument("--generate_days", type=int, default=0)
    ap.add_argument("--gen_name", default="", help="Tag appended to gen_*/ folder name.")
    ap.add_argument("--out_gen_csv", default="", help="Optional explicit CSV path; default: gen_dir/generated_days.csv")
    ap.add_argument("--poisson_lambda", action="store_true")
    ap.add_argument("--z_scale", type=float, default=1.0, help="Scale z sampling. >1 tends to increase extremes.")
    ap.add_argument(
        "--z_source",
        default="prior",
        choices=["prior", "posterior", "extreme_posterior"],
        help="Where to sample z from. posterior/extreme_posterior uses training posteriors.",
    )

    # Scheme B/A for price
    ap.add_argument(
        "--price_mode",
        default="project_levels",
        choices=["none", "project_levels", "template_perturb"],
        help="Price handling: none (raw decoder), project_levels (Scheme B), template_perturb (Scheme A).",
    )
    ap.add_argument("--price_level_decimals", type=int, default=6, help="Rounding decimals when extracting levels.")
    # template_perturb knobs (Scheme A)
    ap.add_argument("--price_delta_low", type=float, default=0.05)
    ap.add_argument("--price_delta_mid", type=float, default=0.08)
    ap.add_argument("--price_delta_high", type=float, default=0.20)
    ap.add_argument("--price_global_scale_min", type=float, default=1.00)
    ap.add_argument("--price_global_scale_max", type=float, default=1.00)
    ap.add_argument("--price_add_noise_sigma", type=float, default=0.0)
    ap.add_argument("--price_project_after_perturb", action="store_true",
                    help="After template_perturb, snap back to original discrete levels (keeps legality).")

    # extreme-focused training sampler
    ap.add_argument("--extreme_metric", default="none", choices=["none", "load_sum", "lambda_max"])
    ap.add_argument("--extreme_top_pct", type=float, default=0.10)
    ap.add_argument("--extreme_weight", type=float, default=5.0)

    args = ap.parse_args()

    # ---- load data ----
    data = load_npz(resolve_path(args.npz))
    X = data["X"].astype(np.float32)  # standardized targets
    C = data["C"].astype(np.float32)  # conditioning (calendar one-hot + sin/cos + etc)
    x_mean = data["x_mean"].astype(np.float32)
    x_std_vec = data["x_std"].astype(np.float32)
    x_cols = [str(x) for x in data["x_cols"].tolist()]
    c_cols = [str(x) for x in data["c_cols"].tolist()]
    log1p_x_cols = [str(x) for x in data["log1p_x_cols"].tolist()] if "log1p_x_cols" in data else []

    day_load_sum = data.get("day_load_sum", np.array([], dtype=np.float32)).astype(np.float32)
    day_lambda_max = data.get("day_lambda_max", np.array([], dtype=np.float32)).astype(np.float32)

    N, T, Dx = X.shape
    Dc = C.shape[-1]
    x_dim = T * Dx
    c_dim = T * Dc

    # extract price levels once (for Scheme B & optional snap-back)
    price_levels = get_price_levels_from_training(
        X_std=X, x_cols=x_cols, x_mean=x_mean, x_std_vec=x_std_vec, decimals=args.price_level_decimals
    )
    if price_levels is not None:
        print(f"[Price levels] n={len(price_levels)} min={float(price_levels.min()):.6g} max={float(price_levels.max()):.6g}")

    # ---- decide run_dir ----
    base = Path(args.out_dir)
    base.mkdir(parents=True, exist_ok=True)

    # If resume: use existing model_dir as run_dir
    if args.model_dir.strip():
        run_dir = Path(resolve_path(args.model_dir.strip()))
        if not (run_dir / "model.pt").exists():
            raise FileNotFoundError(f"--model_dir provided but model.pt not found under: {run_dir}")
        cfg = load_json(str(run_dir / "config.json")) if (run_dir / "config.json").exists() else None
        if cfg is None:
            raise FileNotFoundError(f"--model_dir provided but config.json not found under: {run_dir}")
        # override architecture from saved config
        hidden_dims = [int(x) for x in cfg.get("hidden_dims", [])]
        latent_dim = int(cfg.get("latent_dim", args.latent_dim))
        beta = float(cfg.get("beta", args.beta))
        print(f"[Resume] run_dir={run_dir} latent_dim={latent_dim} hidden_dims={hidden_dims} beta={beta}")
    else:
        if not args.run_name.strip():
            args.run_name = auto_run_name(args)
        run_dir = make_run_dir(base, args.run_name, args.no_timestamp)
        run_dir.mkdir(parents=True, exist_ok=True)
        hidden_dims = [int(x) for x in args.hidden_dims.split(",") if x.strip()]
        latent_dim = int(args.latent_dim)
        beta = float(args.beta)
        print(f"[Run dir] {run_dir}")

    # ---- build model ----
    model = MLP_CVAE(x_dim=x_dim, c_dim=c_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- training (skip if resume) ----
    if not args.model_dir.strip():
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        train_idx, val_idx = split_indices(N, args.val_ratio, args.seed)

        sampler = None
        if args.extreme_metric != "none":
            if args.extreme_metric == "load_sum" and len(day_load_sum) == N:
                score = day_load_sum
            elif args.extreme_metric == "lambda_max" and len(day_lambda_max) == N:
                score = day_lambda_max
            else:
                raise ValueError("Extreme metric requested but day metrics not available in NPZ. Rebuild NPZ with prepare script.")

            thr = np.quantile(score[train_idx], 1.0 - args.extreme_top_pct)
            w = np.ones(len(train_idx), dtype=np.float32)
            w[score[train_idx] >= thr] = args.extreme_weight
            sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(train_idx), replacement=True)
            print(f"[Extreme sampler] metric={args.extreme_metric} top_pct={args.extreme_top_pct} thr={thr:.4f} weight={args.extreme_weight}")

        ds_train = NpzCvaeDataset(X[train_idx], C[train_idx])
        ds_val = NpzCvaeDataset(X[val_idx], C[val_idx])
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

        best_val = float("inf")
        history = {"train": [], "val": []}

        for epoch in range(1, args.epochs + 1):
            model.train()
            tr_loss = tr_rec = tr_kl = 0.0
            nb = 0
            for xb, cb in dl_train:
                xb = xb.to(device)
                cb = cb.to(device)
                xflat = xb.view(xb.size(0), -1)
                cflat = cb.view(cb.size(0), -1)
                x_hat, mu, logvar = model(xflat, cflat)
                loss, rec, kl = loss_fn(x_hat, xflat, mu, logvar, beta)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tr_loss += loss.item()
                tr_rec += rec
                tr_kl += kl
                nb += 1
            tr_loss /= max(nb, 1)
            tr_rec /= max(nb, 1)
            tr_kl /= max(nb, 1)

            model.eval()
            va_loss = va_rec = va_kl = 0.0
            nb = 0
            with torch.no_grad():
                for xb, cb in dl_val:
                    xb = xb.to(device)
                    cb = cb.to(device)
                    xflat = xb.view(xb.size(0), -1)
                    cflat = cb.view(cb.size(0), -1)
                    x_hat, mu, logvar = model(xflat, cflat)
                    loss, rec, kl = loss_fn(x_hat, xflat, mu, logvar, beta)
                    va_loss += loss.item()
                    va_rec += rec
                    va_kl += kl
                    nb += 1
            va_loss /= max(nb, 1)
            va_rec /= max(nb, 1)
            va_kl /= max(nb, 1)

            history["train"].append({"loss": tr_loss, "recon": tr_rec, "kl": tr_kl})
            history["val"].append({"loss": va_loss, "recon": va_rec, "kl": va_kl})

            if va_loss < best_val:
                best_val = va_loss
                torch.save(model.state_dict(), str(run_dir / "model.pt"))

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:4d}/{args.epochs} | train {tr_loss:.6f} (r={tr_rec:.6f},kl={tr_kl:.6f}) | "
                    f"val {va_loss:.6f} (r={va_rec:.6f},kl={va_kl:.6f})"
                )

        save_json(
            {
                "best_val_loss": float(best_val),
                "x_cols": x_cols,
                "c_cols": c_cols,
                "log1p_x_cols": log1p_x_cols,
                "beta": beta,
                "latent_dim": latent_dim,
                "hidden_dims": hidden_dims,
            },
            str(run_dir / "config.json"),
        )
        save_json(history, str(run_dir / "history.json"))
        print(f"[OK] saved in {run_dir}")

    # ---- generation (can run in both train+gen and resume-only) ----
    if args.generate_days > 0:
        # load best model
        model.load_state_dict(torch.load(str(run_dir / "model.pt"), map_location=device))
        model.eval()

        rng = np.random.default_rng(args.seed)

        # sample real C days with replacement -> guarantees realistic weekday/time distribution
        idx_C = rng.choice(np.arange(N), size=args.generate_days, replace=True)
        C_gen = C[idx_C].copy()
        cflat = torch.from_numpy(C_gen.reshape(args.generate_days, -1)).to(device)

        # z sampling
        if args.z_source == "prior":
            z = torch.randn(args.generate_days, latent_dim, device=device) * float(args.z_scale)
        else:
            # posterior-based: pick (possibly extreme) real days and sample from their posteriors
            if args.z_source == "extreme_posterior":
                if len(day_load_sum) == N:
                    score = day_load_sum
                elif len(day_lambda_max) == N:
                    score = day_lambda_max
                else:
                    score = np.zeros(N, dtype=np.float32)
                thr = np.quantile(score, 0.90)
                pool = np.where(score >= thr)[0]
                if len(pool) == 0:
                    pool = np.arange(N)
                idx_z = rng.choice(pool, size=args.generate_days, replace=True)
            else:
                idx_z = rng.choice(np.arange(N), size=args.generate_days, replace=True)

            X_z = torch.from_numpy(X[idx_z].reshape(args.generate_days, -1)).to(device)
            C_z = torch.from_numpy(C[idx_z].reshape(args.generate_days, -1)).to(device)
            with torch.no_grad():
                mu, logvar = model.encode(X_z, C_z)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std) * float(args.z_scale)
                z = mu + eps * std

        with torch.no_grad():
            x_hat = model.decode(z, cflat).cpu().numpy().astype(np.float32)

        X_gen_std = x_hat.reshape(args.generate_days, T, Dx)
        X_gen_raw = inverse_x(X_gen_std, x_mean, x_std_vec, x_cols, log1p_x_cols)
        X_gen_raw = clip_nonneg(X_gen_raw, x_cols)

        # price handling (Scheme B/A)
        X_gen_raw = apply_price_mode(
            X_gen_raw=X_gen_raw,
            x_cols=x_cols,
            mode=args.price_mode,
            price_levels=price_levels,
            X_train_std=X,
            idx_template=idx_C,
            x_mean=x_mean,
            x_std_vec=x_std_vec,
            rng=rng,
            args=args,
        )

        if args.poisson_lambda:
            X_gen_raw = poissonize_lambda(X_gen_raw, x_cols, seed=args.seed)

        gen_tag = args.gen_name.strip() or f"{args.price_mode}_zs{args.z_source}_zsc{args.z_scale:g}"
        gen_dir = make_gen_dir(run_dir, gen_tag)
        out_csv = args.out_gen_csv.strip() or str(gen_dir / "generated_days.csv")
        write_csv(out_csv, X_gen_raw, x_cols, C_gen, c_cols)

        np.savez_compressed(
            str(gen_dir / "generated_days.npz"),
            X_raw=X_gen_raw,
            x_cols=np.array(x_cols, dtype=object),
            C=C_gen,
            c_cols=np.array(c_cols, dtype=object),
            meta=np.array(
                {
                    "price_mode": args.price_mode,
                    "z_source": args.z_source,
                    "z_scale": float(args.z_scale),
                    "poisson_lambda": bool(args.poisson_lambda),
                    "price_params": {
                        "delta_low": float(args.price_delta_low),
                        "delta_mid": float(args.price_delta_mid),
                        "delta_high": float(args.price_delta_high),
                        "global_scale_min": float(args.price_global_scale_min),
                        "global_scale_max": float(args.price_global_scale_max),
                        "add_noise_sigma": float(args.price_add_noise_sigma),
                        "project_after_perturb": bool(args.price_project_after_perturb),
                    },
                },
                dtype=object,
            ),
        )
        print(f"[OK] also wrote {gen_dir/'generated_days.npz'}")


if __name__ == "__main__":
    main()
