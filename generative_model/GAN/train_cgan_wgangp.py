#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
5_train_cgan_wgangp_v3.py

Conditional WGAN-GP for day-level EV scenario generation.
This version supports TWO NPZ formats:

(A) "VAE-style" NPZ:
    - X:   (N,96,D_all) standardized
    - cols (or columns): list[str]
    - mean/std optional (for real-scale inverse later)

(B) "CVAE-style" NPZ produced by 1_prepare_npz_cvae_from_scene.py:
    - X:   (N,96,Dx) standardized for X targets
    - x_cols: list[str] (targets)
    - x_mean/x_std: (Dx,) for inverse (in log-space if log1p applied)
    - log1p_x_cols: list[str]
    - C:   (N,96,Dc) conditions (sin/cos, dow onehot, is_weekend)
    - c_cols: list[str]

Training only needs standardized X + target column names.
Conditioning uses day-of-week one-hot (dow_0..dow_6) from:
  - CVAE-style: from C/c_cols
  - VAE-style: from X/cols if present

Outputs:
  <out_dir>/<timestamp>__cganWGP__.../
    gan.pt
    config.json
    losses.csv

Example:
  python 5_train_cgan_wgangp_v3.py \
    --npz vae_china_cvae.npz \
    --out_dir runs/china_gan \
    --epochs 300 --batch_size 64 \
    --target_cols price,load,lambda \
    --latent_dim 128 --g_channels 256 --d_channels 256 \
    --n_critic 5 --lambda_gp 10 \
    --run_name china_gan_baseline
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


# ----------------- utils -----------------
def _sanitize(s: str) -> str:
    s = s.strip().replace(" ", "")
    s = s.replace(",", "-")
    s = re.sub(r"[^a-zA-Z0-9._\-+=]", "_", s)
    return s

def make_run_dir(base: Path, args) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    if args.no_timestamp:
        rd = base
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = (
            f"cganWGP__z={args.latent_dim}__g={args.g_channels}__d={args.d_channels}"
            f"__bs={args.batch_size}__ep={args.epochs}__nc={args.n_critic}__gp={args.lambda_gp}"
        )
        if args.run_name.strip():
            tag += f"__note={_sanitize(args.run_name)}"
        rd = base / f"{ts}__{tag}"
        if rd.exists():
            rd = base / f"{ts}__{tag}_{uuid.uuid4().hex[:6]}"
    rd.mkdir(parents=True, exist_ok=True)
    return rd

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

    # detect CVAE-style
    is_cvae = ("x_cols" in data.files) and ("x_mean" in data.files) and ("x_std" in data.files)
    if is_cvae:
        X = data["X"].astype(np.float32)
        x_cols = data["x_cols"].tolist()
        x_mean = data["x_mean"].astype(np.float32)
        x_std  = data["x_std"].astype(np.float32)
        log1p_x_cols = data["log1p_x_cols"].tolist() if "log1p_x_cols" in data.files else []
        C = data["C"].astype(np.float32) if "C" in data.files else None
        c_cols = data["c_cols"].tolist() if "c_cols" in data.files else []
        return {
            "format": "cvae",
            "keys": keys,
            "X": X,
            "cols": x_cols,
            "mean": x_mean,
            "std": x_std,
            "log1p_cols": log1p_x_cols,
            "C": C,
            "c_cols": c_cols,
        }

    # VAE-style
    cols = None
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
        "log1p_cols": [],   # unknown in VAE-style unless you encode it somewhere else
        "C": None,
        "c_cols": [],
    }

def get_indices(cols: List[str], names: List[str]) -> List[int]:
    idx = []
    for n in names:
        if n not in cols:
            raise ValueError(f"Column '{n}' not found. Available={cols}")
        idx.append(cols.index(n))
    return idx

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

def positional_sincos(T: int = 96) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    ang = 2.0 * np.pi * (t / float(T))
    return np.stack([np.sin(ang), np.cos(ang)], axis=0).astype(np.float32)  # (2,T)


# ----------------- models -----------------
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

class Critic(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, base_ch: int = 256):
        super().__init__()
        self.in_ch = in_dim + cond_dim + 2
        self.net = nn.Sequential(
            nn.Conv1d(self.in_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(base_ch),
            nn.Conv1d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(base_ch),
        )
        self.head = nn.Sequential(
            nn.Linear(base_ch, base_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(base_ch, 1),
        )
    def forward(self, x, c, pos):
        cexp = c.unsqueeze(-1).repeat(1, 1, x.size(-1))
        h = torch.cat([x, cexp, pos], dim=1)
        f = self.net(h).mean(dim=2)
        return self.head(f).squeeze(1)

def gradient_penalty(D: Critic, real: torch.Tensor, fake: torch.Tensor, c: torch.Tensor, pos: torch.Tensor, device) -> torch.Tensor:
    B = real.size(0)
    eps = torch.rand(B, 1, 1, device=device)
    inter = eps * real + (1 - eps) * fake
    inter.requires_grad_(True)
    out = D(inter, c, pos)
    grads = torch.autograd.grad(
        outputs=out,
        inputs=inter,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grads = grads.view(B, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--no_timestamp", action="store_true")

    ap.add_argument("--target_cols", default="price,load,lambda")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=1e-4)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--g_channels", type=int, default=256)
    ap.add_argument("--d_channels", type=int, default=256)
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--lambda_gp", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npz = load_npz_any(args.npz)
    X = npz["X"]
    cols = npz["cols"]

    print("[NPZ format]", npz["format"])
    print("[NPZ keys]", npz["keys"])
    if npz["mean"] is None or npz["std"] is None:
        print("[Warn] NPZ has no mean/std. Training OK. Generation to real scale may be limited.")

    # target selection
    target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    tidx = get_indices(cols, target_cols)
    Xt = X[:, :, tidx].transpose(0, 2, 1).astype(np.float32)  # (N,Dt,96)

    # conditioning dow
    if npz["format"] == "cvae":
        dow = build_dow_onehot_from_c(npz["C"], npz["c_cols"])
    else:
        dow = build_dow_onehot_from_x(X, cols)

    N = Xt.shape[0]
    perm = np.random.permutation(N)
    n_train = int(N * 0.9)
    tr, va = perm[:n_train], perm[n_train:]

    Xt_tr = torch.from_numpy(Xt[tr])
    dow_tr = torch.from_numpy(dow[tr])
    Xt_va = torch.from_numpy(Xt[va])
    dow_va = torch.from_numpy(dow[va])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    G = Generator(args.latent_dim, 7, len(target_cols), base_ch=args.g_channels).to(device)
    D = Critic(len(target_cols), 7, base_ch=args.d_channels).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9))

    pos = torch.from_numpy(positional_sincos(96)).unsqueeze(0).to(device)  # (1,2,96)

    run_dir = make_run_dir(Path(args.out_dir), args)
    print("[Run dir]", run_dir)

    # save config
    cfg = {**vars(args), "target_cols": target_cols, "npz": str(args.npz), "npz_format": npz["format"]}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    losses = []
    steps_per_epoch = int(np.ceil(n_train / args.batch_size))

    for ep in range(1, args.epochs + 1):
        G.train(); D.train()
        ep_perm = torch.randperm(n_train)

        for si in range(steps_per_epoch):
            bidx = ep_perm[si * args.batch_size : (si + 1) * args.batch_size]
            real = Xt_tr[bidx].to(device)
            c = dow_tr[bidx].to(device)
            B = real.size(0)
            pos_b = pos.repeat(B, 1, 1)

            for _ in range(args.n_critic):
                z = torch.randn(B, args.latent_dim, device=device)
                fake = G(z, c).detach()

                d_real = D(real, c, pos_b).mean()
                d_fake = D(fake, c, pos_b).mean()
                gp = gradient_penalty(D, real, fake, c, pos_b, device)
                loss_d = (d_fake - d_real) + args.lambda_gp * gp

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            z = torch.randn(B, args.latent_dim, device=device)
            fake = G(z, c)
            loss_g = -D(fake, c, pos_b).mean()

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

        # quick val gap
        G.eval(); D.eval()
        with torch.no_grad():
            vb = min(512, Xt_va.size(0))
            ridx = torch.randperm(Xt_va.size(0))[:vb]
            realv = Xt_va[ridx].to(device)
            cv = dow_va[ridx].to(device)
            pos_v = pos.repeat(vb, 1, 1)
            zv = torch.randn(vb, args.latent_dim, device=device)
            fakev = G(zv, cv)
            gap = (D(realv, cv, pos_v).mean() - D(fakev, cv, pos_v).mean()).item()

        row = {"epoch": ep, "loss_g": float(loss_g.item()), "loss_d": float(loss_d.item()), "val_gap": float(gap)}
        losses.append(row)
        if ep == 1 or ep % 10 == 0:
            print(f"[ep {ep:04d}] loss_g={row['loss_g']:.4f} loss_d={row['loss_d']:.4f} val_gap={row['val_gap']:.4f}")

        if ep % 25 == 0 or ep == args.epochs:
            torch.save({
                "G": G.state_dict(),
                "D": D.state_dict(),
                "target_cols": target_cols,
                "cond_dim": 7,
                "latent_dim": args.latent_dim,
                "g_channels": args.g_channels,
                "d_channels": args.d_channels,
                "npz_path": str(args.npz),
                "npz_format": npz["format"],
                # store scaling/meta if present (important for real-scale generation)
                "cols": cols,
                "mean": npz["mean"],      # can be None
                "std": npz["std"],        # can be None
                "log1p_cols": npz["log1p_cols"],
            }, run_dir / "gan.pt")

    pd.DataFrame(losses).to_csv(run_dir / "losses.csv", index=False)
    print("[Saved]", run_dir / "gan.pt")
    print("[Saved]", run_dir / "losses.csv")


if __name__ == "__main__":
    main()
