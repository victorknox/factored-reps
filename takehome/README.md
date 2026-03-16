# Non-Ergodic Mess3 Mixture: Take-Home Experiment

Belief-state geometry analysis of a transformer trained on a mixture of Mess3 processes.

## Quick Start

```bash
# From the repo root (factored-reps/)
# Ensure dependencies are installed
uv sync  # or pip install -e .

# 1. Generate dataset
uv run python takehome/data/generate_nonergodic_mess3.py

# 2. Train model
uv run python takehome/train/train_transformer.py \
    --config takehome/configs/main.json \
    --data_dir takehome/results \
    --output_dir takehome/results/checkpoints

# 3. Run geometry analysis
uv run python takehome/analysis/analyze_geometry.py \
    --results_dir takehome/results --device cuda

# 4. Run emergence analysis
uv run python takehome/analysis/extra_analysis.py \
    --results_dir takehome/results --device cuda
```

Use `--device cpu` if no GPU is available (slower).

## Smoke Test

```bash
uv run python takehome/data/generate_nonergodic_mess3.py --n_train 1000 --n_val 200
uv run python takehome/train/train_transformer.py \
    --config takehome/configs/smoke.json \
    --data_dir takehome/results \
    --output_dir takehome/results/checkpoints
```

## Outputs

- `results/figures/` — All figures referenced in the report
- `results/checkpoints/` — Model weights and training history
- `results/analysis_results.json` — Probe R² and PCA metrics
- `FINAL_REPORT.md` — Full write-up
- `honor_code_prediction.md` — Pre-registered geometry predictions

## Structure

```
takehome/
├── data/generate_nonergodic_mess3.py   # Data generation
├── train/train_transformer.py          # Training
├── analysis/
│   ├── analyze_geometry.py             # PCA + probes
│   └── extra_analysis.py              # Emergence analysis
├── configs/{main,smoke}.json          # Experiment configs
├── honor_code_prediction.md           # Pre-registration
├── FINAL_REPORT.md                    # Report
└── README.md                          # This file
```
