# TimeDiT-inspired stress-controllable generator

This project trains **one** conditional diffusion transformer and then generates two selectable scenarios from the same checkpoint:

- `mainB`: closer to the historical mainstream distribution
- `stressA`: more extreme but still history-grounded

It is designed for CSV files with the schema:

```text
price,load,lambda,sin_hour,cos_hour,day_of_week,is_weekend,t,day_id
```

## What the project does

1. Reads your `data.csv`.
2. Treats `price/load/lambda` as the generated targets.
3. Treats `sin_hour/cos_hour/day_of_week/is_weekend` as deterministic calendar conditions.
4. Computes an automatic **daily `stress_score`** from historical data.
5. Trains a compact **TimeDiT-inspired** diffusion transformer with:
   - diffusion denoising on the target channels,
   - token-level calendar conditions,
   - AdaLN-style global conditioning with diffusion time + stress score + day type,
   - mixed mask training (`reconstruction`, `random`, `block`, `stride`).
6. Uses the trained model to generate:
   - `mainB` days by sampling target stress from the central historical band,
   - `stressA` days by sampling target stress from a high historical band.

## Stress score definition

The stress score is computed **per day**.

### 1) Hierarchical robust baseline
For each target variable (`price`, `load`, `lambda`), the code first constructs a robust baseline using this fallback order:

1. `(day_of_week, step_in_day)`
2. `(is_weekend, step_in_day)`
3. `(step_in_day)`

At each step it estimates a median and a robust scale from the IQR.

### 2) Per-day raw metrics
For each day and each target variable:

- `level_raw`: high positive deviation intensity
- `ramp_raw`: strong 15-minute change intensity
- `duration_raw`: fraction of the day spent at elevated positive deviation

A separate `joint_raw` captures how often at least two of the three variables are elevated together.

### 3) Percentile calibration
Every raw metric is converted to its historical percentile rank.

Per-variable stress is then:

```text
variable_stress = 0.50 * level_pct + 0.25 * ramp_pct + 0.25 * duration_pct
```

Final daily stress is:

```text
stress_unscaled =
    0.35 * price_stress +
    0.35 * load_stress +
    0.15 * lambda_stress +
    0.15 * joint_pct
```

Finally, `stress_unscaled` is mapped to its historical percentile rank to obtain:

```text
stress_score in [0, 1]
```

## Project structure

```text
configs/
  default.yaml
  smoke.yaml
make_demo_data.py
smoke_test.py
train.py
generate.py
timedit_stress/
  __init__.py
  dataset.py
  diffusion.py
  factory.py
  generation.py
  masks.py
  model.py
  preprocessing.py
  stress.py
  utils.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Put your CSV in the project directory or pass its path explicitly.

```bash
python train.py --config configs/default.yaml --csv-path /path/to/data.csv --output-dir runs/exp1
```

Main outputs:

- `runs/exp1/model_bundle.pt`
- `runs/exp1/daily_stress_scores.csv`
- `runs/exp1/training_history.csv`
- `runs/exp1/training_data_with_stress.csv`

## Generate `mainB`

```bash
python generate.py \
  --bundle runs/exp1/model_bundle.pt \
  --scenario mainB \
  --n-days 30 \
  --output-path runs/exp1/generated_mainB.csv
```

## Generate `stressA`

```bash
python generate.py \
  --bundle runs/exp1/model_bundle.pt \
  --scenario stressA \
  --n-days 30 \
  --output-path runs/exp1/generated_stressA.csv
```

## Notes on generation

- The generator produces one day at a time (`96` steps per day).
- For each generated day it can sample multiple candidates and keep the one whose realized stress score is closest to the requested target stress.
- Generated CSVs preserve the original column order:

```text
price,load,lambda,sin_hour,cos_hour,day_of_week,is_weekend,t,day_id
```

A second metadata CSV is also written with the requested and realized stress scores.

## Quick smoke test

This creates a synthetic demo dataset, trains for 2 epochs on CPU, and generates both scenarios:

```bash
python smoke_test.py
```

Outputs will be written under `runs/smoke/`.

## Practical tuning advice

If your real dataset is larger or you use a GPU, the first knobs to increase are:

- `training.epochs`
- `model.hidden_size`
- `model.depth`
- `diffusion.timesteps`

If `stressA` is not extreme enough, tighten the `stressA_quantile_range` upward in `configs/default.yaml`.
If `mainB` becomes too calm, widen `mainB_quantile_range` toward the center of the empirical distribution.
