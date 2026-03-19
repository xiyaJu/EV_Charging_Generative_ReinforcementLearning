from __future__ import annotations

from typing import Dict

from .diffusion import GaussianDiffusion
from .model import TimeDiTGenerator



def build_model(model_cfg: Dict[str, object], data_cfg: Dict[str, object]) -> TimeDiTGenerator:
    return TimeDiTGenerator(
        seq_len=int(data_cfg["steps_per_day"]),
        target_dim=int(data_cfg.get("target_dim", 3)),
        token_cond_dim=int(data_cfg.get("token_cond_dim", 5)),
        global_cond_dim=int(data_cfg.get("global_cond_dim", 4)),
        hidden_size=int(model_cfg["hidden_size"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
    )



def build_diffusion(diffusion_cfg: Dict[str, object]) -> GaussianDiffusion:
    return GaussianDiffusion(
        timesteps=int(diffusion_cfg["timesteps"]),
        beta_schedule=str(diffusion_cfg["beta_schedule"]),
        observed_loss_weight=float(diffusion_cfg["observed_loss_weight"]),
        clip_denoised=bool(diffusion_cfg.get("clip_denoised", True)),
    )
