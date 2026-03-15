from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn



def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        scale = 1000 / timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas.float(), 1e-5, 0.999)
    raise ValueError(f"Unknown beta schedule: {schedule}")



def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 100,
        beta_schedule: str = "cosine",
        observed_loss_weight: float = 0.1,
        clip_denoised: bool = True,
    ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.observed_loss_weight = observed_loss_weight
        self.clip_denoised = clip_denoised

        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(torch.clamp(1.0 / alphas_cumprod - 1.0, min=0.0)),
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * eps

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        token_cond: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        eps_pred = model(x_t, obs_values, obs_mask, token_cond, global_cond, t)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
        if self.clip_denoised:
            x0_pred = x0_pred.clamp(-1.0, 1.0)
        model_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return {
            "model_mean": model_mean,
            "posterior_variance": posterior_variance,
            "posterior_log_variance": posterior_log_variance,
            "x0_pred": x0_pred,
            "eps_pred": eps_pred,
        }

    def p_losses(self, model: nn.Module, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        x0 = batch["x0"]
        bsz = x0.shape[0]
        t = torch.randint(0, self.timesteps, (bsz,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        eps_pred = model(x_t, batch["obs_values"], batch["obs_mask"], batch["token_cond"], batch["global_cond"], t)
        loss_weight = (1.0 - batch["obs_mask"]) + self.observed_loss_weight * batch["obs_mask"]
        mse = (eps_pred - noise).pow(2)
        loss = (mse * loss_weight).mean()
        return loss, {"loss": float(loss.detach().cpu().item())}

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int],
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        token_cond: torch.Tensor,
        global_cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            out = self.p_mean_variance(model, x_t, t, obs_values, obs_mask, token_cond, global_cond)
            if step > 0:
                noise = torch.randn_like(x_t)
                x_t = out["model_mean"] + torch.exp(0.5 * out["posterior_log_variance"]) * noise
            else:
                x_t = out["model_mean"]
        return x_t
