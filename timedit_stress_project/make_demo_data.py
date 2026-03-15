from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic demo CSV with the required schema.")
    parser.add_argument("--output", type=str, default="demo_data.csv")
    parser.add_argument("--n-days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    steps_per_day = 96
    rows = []
    for day_id in range(args.n_days):
        dow = day_id % 7
        is_weekend = int(dow in (5, 6))
        latent_stress = np.clip(rng.beta(2.5, 3.5) + 0.25 * is_weekend + 0.10 * rng.normal(), 0.02, 0.98)
        for step in range(steps_per_day):
            angle = 2.0 * np.pi * (step / steps_per_day)
            sin_hour = float(np.sin(angle))
            cos_hour = float(np.cos(angle))
            peak = np.exp(-0.5 * ((step - 72) / 10.0) ** 2) + 0.55 * np.exp(-0.5 * ((step - 32) / 14.0) ** 2)
            load = 80 + 16 * peak + 10 * latent_stress * peak + 2.8 * rng.normal()
            lambda_val = 22 + 4.5 * peak + 8.0 * latent_stress * peak + 1.2 * rng.normal()
            price = 35 + 0.45 * load + 1.8 * lambda_val + 22.0 * latent_stress * peak + 4.0 * rng.normal()
            if latent_stress > 0.8 and rng.random() < 0.05:
                price += rng.uniform(8.0, 20.0)
                load += rng.uniform(1.0, 4.0)
                lambda_val += rng.uniform(2.0, 6.0)
            rows.append(
                {
                    "price": float(price),
                    "load": float(load),
                    "lambda": float(lambda_val),
                    "sin_hour": sin_hour,
                    "cos_hour": cos_hour,
                    "day_of_week": int(dow),
                    "is_weekend": int(is_weekend),
                    "t": int(day_id * steps_per_day + step),
                    "day_id": int(day_id),
                }
            )
    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[DONE] Wrote demo data to {args.output}")


if __name__ == "__main__":
    main()
