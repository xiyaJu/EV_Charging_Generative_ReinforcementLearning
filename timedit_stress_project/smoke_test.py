from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent



def run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)



def main() -> None:
    run(["python", "make_demo_data.py", "--output", "demo_data.csv", "--n-days", "50", "--seed", "11"])
    run(["python", "train.py", "--config", "configs/smoke.yaml", "--csv-path", "demo_data.csv", "--output-dir", "runs/smoke"])
    run(
        [
            "python",
            "generate.py",
            "--bundle",
            "runs/smoke/model_bundle.pt",
            "--scenario",
            "mainB",
            "--n-days",
            "3",
            "--output-path",
            "runs/smoke/generated_mainB.csv",
            "--metadata-path",
            "runs/smoke/generated_mainB_meta.csv",
            "--seed",
            "11",
        ]
    )
    run(
        [
            "python",
            "generate.py",
            "--bundle",
            "runs/smoke/model_bundle.pt",
            "--scenario",
            "stressA",
            "--n-days",
            "3",
            "--output-path",
            "runs/smoke/generated_stressA.csv",
            "--metadata-path",
            "runs/smoke/generated_stressA_meta.csv",
            "--seed",
            "12",
        ]
    )
    print("[DONE] Smoke test completed.")


if __name__ == "__main__":
    main()
