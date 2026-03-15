from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


DEFAULT_MASK_PROBS = {
    "reconstruction": 0.40,
    "random": 0.25,
    "block": 0.20,
    "stride": 0.15,
}


class MaskSampler:
    def __init__(self, seq_len: int, n_channels: int, probs: Dict[str, float] | None = None, seed: int = 42) -> None:
        self.seq_len = int(seq_len)
        self.n_channels = int(n_channels)
        self.probs = probs or DEFAULT_MASK_PROBS.copy()
        self.rng = np.random.default_rng(seed)
        self.mask_types = list(self.probs.keys())
        weights = np.asarray([self.probs[m] for m in self.mask_types], dtype=np.float64)
        self.prob_array = weights / weights.sum()

    def sample(self, force_type: str | None = None) -> Tuple[np.ndarray, str]:
        mask_type = force_type or str(self.rng.choice(self.mask_types, p=self.prob_array))
        if mask_type == "reconstruction":
            step_mask = np.zeros(self.seq_len, dtype=np.float32)
        elif mask_type == "random":
            observe_ratio = float(self.rng.uniform(0.30, 0.70))
            step_mask = (self.rng.random(self.seq_len) < observe_ratio).astype(np.float32)
            if step_mask.sum() == 0:
                step_mask[self.rng.integers(0, self.seq_len)] = 1.0
            if step_mask.sum() == self.seq_len:
                step_mask[self.rng.integers(0, self.seq_len)] = 0.0
        elif mask_type == "block":
            hidden_len = int(self.rng.integers(max(8, self.seq_len // 8), max(16, self.seq_len // 2) + 1))
            step_mask = np.ones(self.seq_len, dtype=np.float32)
            step_mask[self.seq_len - hidden_len :] = 0.0
        elif mask_type == "stride":
            stride = int(self.rng.choice(np.array([2, 3, 4, 6, 8], dtype=np.int64)))
            offset = int(self.rng.integers(0, stride))
            step_mask = np.ones(self.seq_len, dtype=np.float32)
            step_mask[offset::stride] = 0.0
            if step_mask.sum() == 0:
                step_mask[:] = 1.0
                step_mask[offset] = 0.0
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        full_mask = np.repeat(step_mask[:, None], self.n_channels, axis=1)
        return full_mask.astype(np.float32), mask_type
