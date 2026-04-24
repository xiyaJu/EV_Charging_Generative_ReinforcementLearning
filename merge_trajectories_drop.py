import argparse
import gzip
import json
import os
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REQUIRED_KEYS = ("observations", "actions", "rewards", "action_mask")


def load_trajectories(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_trajectories(path, trajectories):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if str(path).endswith(".gz"):
        with gzip.open(path, "wb") as f:
            pickle.dump(trajectories, f)
    else:
        with open(path, "wb") as f:
            pickle.dump(trajectories, f)


def summarize_dataset(trajectories):
    if not trajectories:
        return {
            "num_trajectories": 0,
            "num_timesteps": 0,
            "return_mean": None,
            "return_std": None,
            "return_min": None,
            "return_max": None,
            "obs_dims": [],
            "act_dims": [],
            "episode_lens": [],
        }

    returns = np.asarray([float(np.sum(traj["rewards"])) for traj in trajectories], dtype=np.float64)
    obs_dims = sorted({int(traj["observations"].shape[1]) for traj in trajectories})
    act_dims = sorted({int(traj["actions"].shape[1]) for traj in trajectories})
    episode_lens = [int(traj["observations"].shape[0]) for traj in trajectories]

    return {
        "num_trajectories": len(trajectories),
        "num_timesteps": int(sum(episode_lens)),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "obs_dims": obs_dims,
        "act_dims": act_dims,
        "episode_lens": {
            "min": int(min(episode_lens)),
            "max": int(max(episode_lens)),
            "unique_count": len(set(episode_lens)),
        },
    }


def infer_n_cs(traj):
    return int(traj["actions"].shape[1])


def validate_trajectory(traj, source_label):
    missing = [key for key in REQUIRED_KEYS if key not in traj]
    if missing:
        raise ValueError(f"{source_label} 缺少必要字段: {missing}")

    obs = traj["observations"]
    acts = traj["actions"]
    rews = traj["rewards"]
    action_mask = traj["action_mask"]

    t = obs.shape[0]
    if acts.shape[0] != t or rews.shape[0] != t or action_mask.shape[0] != t:
        raise ValueError(
            f"{source_label} 的时序长度不一致: "
            f"obs={obs.shape[0]}, acts={acts.shape[0]}, rews={rews.shape[0]}, mask={action_mask.shape[0]}"
        )


def normalize_output_path(path):
    path = Path(path)
    if path.suffix not in (".pkl", ".gz"):
        path = path.with_suffix(".pkl.gz")
    return str(path)


def build_argparser():
    parser = argparse.ArgumentParser(description="Merge multiple trajectory pickle files into a single dataset.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input trajectory files (.pkl or .pkl.gz).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output trajectory file path. If no suffix is given, .pkl.gz will be used.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle merged trajectories before saving.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling/shuffling.",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Keep at most N trajectories from each input file.",
    )
    parser.add_argument(
        "--total-limit",
        type=int,
        default=None,
        help="Keep at most N trajectories after merge.",
    )
    parser.add_argument(
        "--balance-by-file",
        action="store_true",
        help="When used with --total-limit, sample evenly from each source file first.",
    )
    parser.add_argument(
        "--no-source-metadata",
        action="store_true",
        help="Do not add source metadata fields to each trajectory.",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    rng = random.Random(args.seed)

    merged = []
    per_file_counts = {}
    per_file_stats = {}
    source_groups = defaultdict(list)

    for input_path in args.inputs:
        input_path = str(Path(input_path))
        trajectories = load_trajectories(input_path)
        if not isinstance(trajectories, list):
            raise TypeError(f"{input_path} 不是 trajectory list，而是 {type(trajectories).__name__}")

        for traj_idx, traj in enumerate(trajectories):
            validate_trajectory(traj, f"{input_path}[{traj_idx}]")

        if args.limit_per_file is not None and len(trajectories) > args.limit_per_file:
            chosen_indices = list(range(len(trajectories)))
            rng.shuffle(chosen_indices)
            chosen_indices = chosen_indices[:args.limit_per_file]
            chosen_indices.sort()
            trajectories = [trajectories[i] for i in chosen_indices]

        tagged_trajectories = []
        for traj_idx, traj in enumerate(trajectories):
            traj_copy = dict(traj)
            if not args.no_source_metadata:
                traj_copy["source_file"] = os.path.basename(input_path)
                traj_copy["source_path"] = input_path
                traj_copy["source_index"] = traj_idx
                traj_copy["source_n_cs"] = infer_n_cs(traj)
            tagged_trajectories.append(traj_copy)

        per_file_counts[input_path] = len(tagged_trajectories)
        per_file_stats[input_path] = summarize_dataset(tagged_trajectories)
        source_groups[input_path].extend(tagged_trajectories)
        merged.extend(tagged_trajectories)

    if args.total_limit is not None and len(merged) > args.total_limit:
        if args.balance_by_file:
            selected = []
            file_paths = list(source_groups.keys())
            base_take = args.total_limit // len(file_paths)
            remainder = args.total_limit % len(file_paths)

            for i, file_path in enumerate(file_paths):
                take_n = base_take + (1 if i < remainder else 0)
                candidates = list(source_groups[file_path])
                rng.shuffle(candidates)
                selected.extend(candidates[:take_n])

            merged = selected[:args.total_limit]
        else:
            rng.shuffle(merged)
            merged = merged[:args.total_limit]

    if args.shuffle:
        rng.shuffle(merged)

    output_path = normalize_output_path(args.output)
    dump_trajectories(output_path, merged)

    summary = {
        "output_path": output_path,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "limit_per_file": args.limit_per_file,
        "total_limit": args.total_limit,
        "balance_by_file": args.balance_by_file,
        "include_source_metadata": not args.no_source_metadata,
        "num_input_files": len(args.inputs),
        "input_files": [str(Path(p)) for p in args.inputs],
        "per_file_counts": per_file_counts,
        "per_file_stats": per_file_stats,
        "merged_stats": summarize_dataset(merged),
        "merged_source_n_cs_counts": dict(
            Counter(int(traj["actions"].shape[1]) for traj in merged)
        ),
    }

    summary_path = output_path + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Merged trajectories saved to:", output_path)
    print("Summary saved to:", summary_path)
    print("Merged stats:")
    print(json.dumps(summary["merged_stats"], indent=2, ensure_ascii=False))
    print("Merged source n_cs counts:", summary["merged_source_n_cs_counts"])


if __name__ == "__main__":
    main()
