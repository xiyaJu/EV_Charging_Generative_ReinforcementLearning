from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from .masks import MaskSampler


class DailyWindowDataset(Dataset):
    def __init__(
        self,
        target_windows: np.ndarray,
        token_cond_windows: np.ndarray,
        global_cond: np.ndarray,
        mask_sampler: MaskSampler,
        mode: str = "train",
    ) -> None:
        if target_windows.shape[0] != token_cond_windows.shape[0] or target_windows.shape[0] != global_cond.shape[0]:
            raise ValueError("target_windows, token_cond_windows, and global_cond must have the same first dimension.")
        self.target_windows = target_windows.astype(np.float32)
        self.token_cond_windows = token_cond_windows.astype(np.float32)
        self.global_cond = global_cond.astype(np.float32)
        self.mask_sampler = mask_sampler
        self.mode = mode

    def __len__(self) -> int:
        return int(self.target_windows.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x0 = self.target_windows[idx]
        token_cond = self.token_cond_windows[idx]
        global_cond = self.global_cond[idx]
        if self.mode == "train":
            obs_mask, mask_type = self.mask_sampler.sample()
        else:
            obs_mask, mask_type = self.mask_sampler.sample(force_type="reconstruction")
        obs_values = x0 * obs_mask
        return {
            "x0": torch.from_numpy(x0),
            "token_cond": torch.from_numpy(token_cond),
            "global_cond": torch.from_numpy(global_cond),
            "obs_mask": torch.from_numpy(obs_mask),
            "obs_values": torch.from_numpy(obs_values),
            "mask_type_id": torch.tensor(self._mask_type_id(mask_type), dtype=torch.long),
        }

    @staticmethod
    def _mask_type_id(mask_type: str) -> int:
        mapping = {
            "reconstruction": 0,
            "random": 1,
            "block": 2,
            "stride": 3,
        }
        return mapping[mask_type]
