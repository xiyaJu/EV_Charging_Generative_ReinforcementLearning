# Cross-validation and evaluation plots

This add-on keeps the original `train.py`, `generate.py`, and `evaluate.py` unchanged.
Everything is done through new files only.

## New files

- `cross_validate.py`
- `evaluate_with_plots.py`
- `plot_evaluation.py`
- `timedit_stress/cross_validation.py`
- `timedit_stress/evaluation_plots.py`

## 1) Run k-fold cross-validation

Example with 5 blocked folds:

```bash
python cross_validate.py \
  --config configs/default.yaml \
  --csv-path /path/to/data.csv \
  --output-dir runs/cv_exp \
  --n-folds 5
```

Recommended for time series:

- use `--split-mode blocked`
- this keeps each validation fold as a contiguous block of days

Outputs:

- `runs/cv_exp/cv_summary.json`
- `runs/cv_exp/cv_fold_metrics.csv`
- `runs/cv_exp/cv_fold_assignments.csv`
- `runs/cv_exp/fold_01/...`
- `runs/cv_exp/fold_02/...`
- etc.

Inside each fold directory:

- `model_bundle.pt`
- `training_summary.json`
- `training_history.csv`
- `holdout_real.csv`
- `holdout_real_with_stress.csv`

## 2) Final model after cross-validation

Cross-validation is for model selection and reporting.
After you decide the hyperparameters, train one final model on all data with the original script:

```bash
python train.py \
  --config configs/default.yaml \
  --csv-path /path/to/data.csv \
  --output-dir runs/final_exp
```

## 3) Evaluate and create plots in one command

```bash
python evaluate_with_plots.py \
  --real-csv /path/to/data.csv \
  --bundle runs/final_exp/model_bundle.pt \
  --synthetic mainB=runs/final_exp/generated_mainB.csv \
  --synthetic stressA=runs/final_exp/generated_stressA.csv \
  --metadata mainB=runs/final_exp/generated_mainB_meta.csv \
  --metadata stressA=runs/final_exp/generated_stressA_meta.csv \
  --output-dir runs/final_exp/eval
```

Outputs:

- tables in `runs/final_exp/eval/tables/`
- matrices in `runs/final_exp/eval/correlation_matrices/`
- daily scores in `runs/final_exp/eval/daily_scores/`
- plots in `runs/final_exp/eval/plots/`

## 4) If you already ran `evaluate.py`

You can create the plots afterward:

```bash
python plot_evaluation.py \
  --eval-dir runs/final_exp/eval
```

## 5) Evaluate one specific fold

Each fold saves `holdout_real.csv`.
You can compare generated data from that fold against its own holdout split:

```bash
python generate.py \
  --bundle runs/cv_exp/fold_01/model_bundle.pt \
  --scenario mainB \
  --n-days 20 \
  --output-path runs/cv_exp/fold_01/generated_mainB.csv

python evaluate_with_plots.py \
  --real-csv runs/cv_exp/fold_01/holdout_real.csv \
  --bundle runs/cv_exp/fold_01/model_bundle.pt \
  --synthetic mainB=runs/cv_exp/fold_01/generated_mainB.csv \
  --metadata mainB=runs/cv_exp/fold_01/generated_mainB_meta.csv \
  --output-dir runs/cv_exp/fold_01/eval_mainB
```

A practical choice is to generate roughly the same number of days as that fold's holdout size.
You can find the count in `fold_01/training_summary.json` under `n_val_days`.
