# Factored World Hypothesis - Reproducibility

This repository contains code to reproduce the experiments from the paper.

## Setup

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## 1. Training

Training configs are in `experiments/training/configs/`. Each `repro_*.yaml` config corresponds to an experiment from the paper.

### Run training

```bash
cd experiments/training

# Independent structure (canonical transformer)
uv run python run.py --config-name=repro_independent

# Conditional structure (chain)
uv run python run.py --config-name=repro_chain

# Noise sweep (vary epsilon)
uv run python run.py --config-name=repro_noise_eps_0
uv run python run.py --config-name=repro_noise_eps_0.1
# ... etc for eps_0.001, 0.01, 0.05, 0.2, 0.3, 0.5

# Architecture variants
uv run python run.py --config-name=repro_rnn_32d
uv run python run.py --config-name=repro_rnn_64d
uv run python run.py --config-name=repro_rnn_256d
uv run python run.py --config-name=repro_lstm_32d
uv run python run.py --config-name=repro_transformer_d480
uv run python run.py --config-name=repro_transformer_nctx33
uv run python run.py --config-name=repro_transformer_nctx101
```

### Local testing (small batch size)

```bash
uv run python run.py --config-name=repro_independent training=smoke training.num_steps=100
```

## 2. Figure Generation

After training, generate figures using the scripts in `experiments/figure_generation/`.

### Setup MLflow tracking

For local MLflow:
```bash
export MLFLOW_TRACKING_URI=./mlruns  # or path to your MLflow tracking directory
```

### 2.1 Belief Grid Figures (Figures 2 & 4)

Edit the config files in `experiments/figure_generation/belief-grid/configs/` to add your experiment and run IDs:

```yaml
# In experiments/figure_generation/belief-grid/configs/fig2.yaml
run_id: "<your-run-id>"        # From repro_independent training
experiment_id: "<your-experiment-id>"
```

Generate figures:

```bash
cd experiments/figure_generation/belief-grid

# Figure 2 (independent structure)
uv run python run_parallel.py --config-name=fig2

# Figure 4 (conditional structure)
uv run python run_parallel.py --config-name=fig4

# Architecture variants (supplemental)
uv run python run_parallel.py --config-name=rnn32
uv run python run_parallel.py --config-name=rnn64
uv run python run_parallel.py --config-name=rnn256d
uv run python run_parallel.py --config-name=lstm32
uv run python run_parallel.py --config-name=transformer_d480
uv run python run_parallel.py --config-name=nctx_33
uv run python run_parallel.py --config-name=nctx_101
```

Output: `experiments/figure_generation/belief-grid/final_figures/`

### 2.2 Orthogonality Figures

Edit the config files in `experiments/figure_generation/orthogonality/configs/` to add your run IDs:

```yaml
# In experiments/figure_generation/orthogonality/configs/default.yaml
run_id: "<your-run-id>"
experiment_id: "<your-experiment-id>"
```

Generate figures:

```bash
cd experiments/figure_generation/orthogonality
uv run python run.py --config-name=default
```

Output: `experiments/figure_generation/orthogonality/`

### 2.3 Noise Sweep Figures

The noise analysis scripts use run names to find experiments. After running all `repro_noise_eps_*` training configs, the run names will be `noise_eps_0`, `noise_eps_0.001`, etc.

Generate figures:

```bash
cd experiments/figure_generation/noise
uv run python analysis_parallel.py
```

Output: `experiments/figure_generation/final_figures/`

## Config Reference

| Config | Description |
|--------|-------------|
| `repro_independent` | Canonical transformer, independent structure |
| `repro_chain` | Canonical transformer, conditional chain |
| `repro_noise_eps_*` | Noise sweep (epsilon 0 to 0.5) |
| `repro_rnn_32d/64d/256d` | RNN variants |
| `repro_lstm_32d` | LSTM baseline |
| `repro_transformer_d480` | Wider transformer |
| `repro_transformer_nctx33` | Longer context (33) |
| `repro_transformer_nctx101` | Longer context (101) |
