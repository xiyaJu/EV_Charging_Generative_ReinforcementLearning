import argparse
import csv
import gzip
import pickle
import random
from pathlib import Path


def load_trajectories(path):
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_trajectories(path, trajectories):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(trajectories, f)


def parse_component(spec):
    parts = spec.split("::")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid component specification: {spec}. "
            "Use PATH::COUNT or PATH::COUNT::LABEL."
        )
    path = parts[0]
    count_spec = parts[1]
    label = parts[2] if len(parts) >= 3 else Path(path).stem
    return path, count_spec, label


def resolve_sample_count(count_spec, available_count):
    if count_spec.lower() == "all":
        return available_count
    if "." in count_spec:
        ratio = float(count_spec)
        if not 0 < ratio <= 1:
            raise ValueError(f"Ratio count must be in (0, 1], got {count_spec}")
        return max(1, int(round(available_count * ratio)))
    return int(count_spec)


def sample_component(data, sample_count, rng, with_replacement):
    if with_replacement:
        indices = [rng.randrange(len(data)) for _ in range(sample_count)]
        return [data[idx] for idx in indices]

    if sample_count > len(data):
        raise ValueError(
            f"Requested {sample_count} trajectories without replacement, "
            f"but only {len(data)} are available."
        )
    indices = rng.sample(range(len(data)), sample_count)
    return [data[idx] for idx in indices]


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple trajectory banks into one dataset with a reproducible manifest."
    )
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        help="Component spec: PATH::COUNT or PATH::COUNT::LABEL. COUNT can be int, ratio (<=1), or 'all'.",
    )
    parser.add_argument("--output_path", required=True, help="Output .pkl.gz path.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument(
        "--with_replacement",
        action="store_true",
        help="Sample trajectories from each component with replacement.",
    )
    parser.add_argument(
        "--manifest_csv",
        default=None,
        help="Optional manifest CSV path. Defaults to <output_path>.manifest.csv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    merged = []
    manifest_rows = []

    for spec in args.component:
        path, count_spec, label = parse_component(spec)
        data = load_trajectories(path)
        available_count = len(data)
        sample_count = resolve_sample_count(count_spec, available_count)
        sampled = sample_component(
            data=data,
            sample_count=sample_count,
            rng=rng,
            with_replacement=args.with_replacement,
        )
        merged.extend(sampled)
        manifest_rows.append(
            {
                "label": label,
                "source_path": str(Path(path).resolve()),
                "available_count": available_count,
                "requested_count_spec": count_spec,
                "sampled_count": sample_count,
                "sample_mode": "with_replacement" if args.with_replacement else "without_replacement",
                "seed": args.seed,
            }
        )

    rng.shuffle(merged)
    dump_trajectories(args.output_path, merged)

    manifest_path = Path(args.manifest_csv) if args.manifest_csv else Path(f"{args.output_path}.manifest.csv")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Merged {len(merged)} trajectories -> {Path(args.output_path).resolve()}")
    print(f"Manifest saved to {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
