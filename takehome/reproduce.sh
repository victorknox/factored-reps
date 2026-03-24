#!/usr/bin/env bash
# reproduce.sh — Full reproduction pipeline for the Mess3 geometry experiment.
#
# Usage:
#   bash takehome/reproduce.sh          # CPU mode
#   bash takehome/reproduce.sh --gpu    # GPU mode (recommended)
#
# Run from the factored-reps repo root.

set -euo pipefail

DEVICE="cpu"
if [[ "${1:-}" == "--gpu" ]]; then
    DEVICE="cuda"
    echo "=== Running with GPU (cuda) ==="
else
    echo "=== Running on CPU (pass --gpu for GPU acceleration) ==="
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo ""
echo "Repo root: $REPO_ROOT"
echo "Device:    $DEVICE"
echo ""

# ─────────────────────────────────────────────
# Step 1: Generate dataset
# ─────────────────────────────────────────────
echo "============================================================"
echo "Step 1/6: Generating dataset (50k train + 5k val sequences)"
echo "============================================================"
python takehome/data/generate_nonergodic_mess3.py \
    --n_train 50000 \
    --n_val 5000 \
    --seq_len 16 \
    --seed 42
echo ""

# ─────────────────────────────────────────────
# Step 2: Train full transformer
# ─────────────────────────────────────────────
echo "============================================================"
echo "Step 2/6: Training full transformer (200 epochs)"
echo "============================================================"
python takehome/train/train_transformer.py \
    --config takehome/configs/main_full.json \
    --data_dir takehome/results \
    --output_dir takehome/results/checkpoints_full
echo ""

# ─────────────────────────────────────────────
# Step 3: Baseline geometry analysis
# ─────────────────────────────────────────────
echo "============================================================"
echo "Step 3/6: Running baseline geometry analysis (PCA + probes)"
echo "============================================================"
python takehome/analysis/analyze_geometry.py \
    --checkpoint_dir takehome/results/checkpoints_full \
    --device "$DEVICE"
echo ""

# ─────────────────────────────────────────────
# Step 4: Emergence analysis
# ─────────────────────────────────────────────
echo "============================================================"
echo "Step 4/6: Running emergence analysis"
echo "============================================================"
python takehome/analysis/extra_analysis.py \
    --checkpoint_dir takehome/results/checkpoints_full \
    --device "$DEVICE"
echo ""

# ─────────────────────────────────────────────
# Step 5: Experiments 1-5, 7
# ─────────────────────────────────────────────
echo "============================================================"
echo "Step 5/6: Running experiments 1-5, 7"
echo "============================================================"
python takehome/analysis/run_all_experiments.py \
    --checkpoint_dir takehome/results/checkpoints_full \
    --device "$DEVICE"
echo ""

# ─────────────────────────────────────────────
# Step 6: Experiment 6 (K-scaling)
# ─────────────────────────────────────────────
echo "============================================================"
echo "Step 6/6: Running experiment 6 (dimensionality scaling with K)"
echo "============================================================"
python takehome/analysis/run_experiment6.py \
    --device "$DEVICE" \
    --d_model 128 \
    --n_layers 4 \
    --n_heads 4 \
    --num_epochs 200 \
    --existing_k3_dir takehome/results
echo ""

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
echo "============================================================"
echo "REPRODUCTION COMPLETE"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  Figures:      takehome/results/figures/"
echo "  Experiments:  takehome/results/experiments/"
echo "  Checkpoints:  takehome/results/checkpoints_full/"
echo "  Report:       takehome/FINAL_REPORT.md"
echo ""
