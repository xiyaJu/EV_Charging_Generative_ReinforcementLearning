from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


DEFAULT_REQUIRED_COLUMNS = [
    "price",
    "load",
    "lambda",
    "sin_hour",
    "cos_hour",
    "day_of_week",
    "is_weekend",
    "t",
    "day_id",
]


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def pick_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def save_json(data: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def merge_nested_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_nested_dict(out[key], value)
        else:
            out[key] = value
    return out



def cycle_encode(values: np.ndarray | list[float], period: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float32)
    angle = 2.0 * math.pi * arr / float(period)
    return np.sin(angle), np.cos(angle)



def percentile_from_sorted(sorted_values: np.ndarray, value: float) -> float:
    if sorted_values.size == 0:
        return 0.5
    idx = np.searchsorted(sorted_values, value, side="right")
    return float(idx) / float(sorted_values.size)



def topk_mean(values: np.ndarray, frac: float = 0.1) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(np.ceil(values.size * frac)))
    return float(np.mean(np.sort(values)[-k:]))



def tensor_to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
