from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn



def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_size: int = 256) -> None:
        super().__init__()
        self.freq_size = freq_size
        self.proj = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.timestep_embedding(timesteps, self.freq_size)
        return self.proj(emb)


class MLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x


class TimeDiTGenerator(nn.Module):
    """A compact TimeDiT-inspired denoiser for 96-step daily windows.

    Token inputs:
      - noisy target channels (price, load, lambda)
      - observed self-conditioning values
      - observed self-conditioning mask
      - deterministic calendar token conditions

    Global AdaLN conditioning:
      - diffusion timestep embedding
      - daily stress score
      - daily weekday cyclic encoding / weekend indicator
    """

    def __init__(
        self,
        seq_len: int,
        target_dim: int,
        token_cond_dim: int,
        global_cond_dim: int,
        hidden_size: int = 128,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.target_dim = target_dim
        self.token_cond_dim = token_cond_dim
        self.global_cond_dim = global_cond_dim
        input_dim = target_dim + target_dim + target_dim + token_cond_dim
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        self.time_embed = TimestepEmbedder(hidden_size)
        self.global_cond_embed = nn.Sequential(
            nn.Linear(global_cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                TimeDiTBlock(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output_proj = nn.Linear(hidden_size, target_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x_noisy: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        token_cond: torch.Tensor,
        global_cond: torch.Tensor,
        diffusion_step: torch.Tensor,
    ) -> torch.Tensor:
        if x_noisy.ndim != 3:
            raise ValueError("x_noisy must have shape [B, L, C].")
        token_in = torch.cat([x_noisy, obs_values, obs_mask, token_cond], dim=-1)
        h = self.input_proj(token_in) + self.pos_embed
        h = self.dropout(h)
        cond = self.time_embed(diffusion_step) + self.global_cond_embed(global_cond)
        for block in self.blocks:
            h = block(h, cond)
        shift, scale = self.final_mod(cond).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        return self.output_proj(h)
