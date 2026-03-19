from __future__ import annotations

import argparse
from pathlib import Path

from timedit_stress.generation import generate_scenario, load_bundle



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mainB or stressA scenarios from a trained bundle.")
    parser.add_argument("--bundle", type=str, required=True, help="Path to model_bundle.pt")
    parser.add_argument("--scenario", type=str, choices=["mainB", "stressA"], required=True)
    parser.add_argument("--n-days", type=int, required=True, help="How many new days to generate.")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the generated CSV.")
    parser.add_argument("--metadata-path", type=str, default=None, help="Where to save per-day generation metadata.")
    parser.add_argument("--start-day-of-week", type=int, default=None, help="0=Mon style integer. Defaults to training first day.")
    parser.add_argument("--num-candidates", type=int, default=None, help="Candidates per day; best stress match is kept.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-stress", type=float, default=None, help="Optional explicit stress_score in [0,1].")
    parser.add_argument("--start-t", type=int, default=0)
    parser.add_argument("--start-day-id", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    loaded = load_bundle(args.bundle, device=args.device)
    bundle = loaded.bundle
    day_meta = bundle.get("day_meta", {})
    generation_cfg = dict(bundle.get("generation_config", {}))
    default_candidates = int(generation_cfg.get("num_candidates", 4))
    start_day_of_week = args.start_day_of_week
    if start_day_of_week is None:
        if day_meta and "day_of_week" in day_meta and len(day_meta["day_of_week"]) > 0:
            start_day_of_week = int(day_meta["day_of_week"][0])
        else:
            start_day_of_week = 0

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_path.with_name(output_path.stem + "_meta.csv")

    out_df, meta_df = generate_scenario(
        loaded=loaded,
        scenario=args.scenario,
        n_days=args.n_days,
        start_day_of_week=start_day_of_week,
        num_candidates=args.num_candidates or default_candidates,
        seed=args.seed,
        fixed_stress=args.fixed_stress,
        start_t=args.start_t,
        start_day_id=args.start_day_id,
    )
    out_df.to_csv(output_path, index=False)
    meta_df.to_csv(metadata_path, index=False)

    print(f"[DONE] Generated {len(meta_df)} days for scenario={args.scenario}")
    print(f"[DONE] Data CSV: {output_path}")
    print(f"[DONE] Metadata CSV: {metadata_path}")


if __name__ == "__main__":
    main()
