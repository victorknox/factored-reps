"""Generate combined side-by-side figure for RNN32 and RNN64 models."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np

from plotting import apply_icml_style, _compute_step_ticks


def load_cache(cache_path: str) -> dict:
    """Load cached computation results."""
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def create_combined_figure(
    data_left: dict,
    data_right: dict,
    title_left: str = "RNN 32d",
    title_right: str = "RNN 64d",
    figsize: tuple[float, float] = (6.75, 4.0),
    gt_dims95: float | None = 10,
) -> plt.Figure:
    """Create a side-by-side comparison figure for two models.

    Args:
        data_left: Cached data dict for left model (from run.py cache).
        data_right: Cached data dict for right model.
        title_left: Title for left column.
        title_right: Title for right column.
        figsize: Figure size (width, height).
        gt_dims95: Ground truth dims@95 reference line.

    Returns:
        Matplotlib figure.
    """
    apply_icml_style()

    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Create 2x2 grid: top row = CEV, bottom row = dims@95
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], wspace=0.15, hspace=0.25)

    # Column titles
    fig.text(0.25, 1.02, title_left, fontsize=14, fontweight="bold", ha="center", va="bottom")
    fig.text(0.75, 1.02, title_right, fontsize=14, fontweight="bold", ha="center", va="bottom")

    # Panel labels
    fig.text(0.02, 0.95, "(a)", fontsize=14, fontweight="bold", va="top", ha="left")
    fig.text(0.52, 0.95, "(b)", fontsize=14, fontweight="bold", va="top", ha="left")
    fig.text(0.02, 0.47, "(c)", fontsize=14, fontweight="bold", va="top", ha="left")
    fig.text(0.52, 0.47, "(d)", fontsize=14, fontweight="bold", va="top", ha="left")

    # Plot each model
    for col_idx, (data, title) in enumerate([(data_left, title_left), (data_right, title_right)]):
        # --- Top row: CEV curves ---
        ax_cev = fig.add_subplot(gs[0, col_idx])

        cev_history = data.get("cev_history")
        layer_name = data.get("layer_name")
        belief_baselines = data.get("belief_baselines")

        if cev_history is not None and layer_name is not None:
            # Find the layer in cev_history
            cev_layer = None
            for key in cev_history.keys():
                if layer_name in key or key in layer_name:
                    cev_layer = key
                    break

            if cev_layer is not None:
                layer_data = cev_history[cev_layer]

                # Sort by step
                sorted_data = sorted(layer_data, key=lambda x: x[0])

                # Get colormap
                steps_for_color = [s for s, _ in sorted_data]
                min_step = max(min(steps_for_color), 1)
                max_step = max(steps_for_color)
                norm = mcolors.LogNorm(vmin=min_step, vmax=max_step)
                cmap = plt.cm.viridis_r

                # Plot each CEV curve
                for step, cev_array in sorted_data:
                    components = np.arange(1, len(cev_array) + 1)
                    color = cmap(norm(max(step, 1)))
                    ax_cev.plot(components, cev_array, color=color, linewidth=1.5, alpha=0.8)

                # Plot belief baselines if available
                if belief_baselines is not None:
                    if "factored" in belief_baselines:
                        factored_cev = belief_baselines["factored"]
                        components = np.arange(1, len(factored_cev) + 1)
                        ax_cev.plot(components, factored_cev, "k--", linewidth=1.5,
                                   label="Factored", alpha=0.7)
                    if "product" in belief_baselines:
                        product_cev = belief_baselines["product"]
                        components = np.arange(1, len(product_cev) + 1)
                        ax_cev.plot(components, product_cev, "k:", linewidth=1.5,
                                   label="Product", alpha=0.7)

        ax_cev.set_xlim(1, 64)
        ax_cev.set_ylim(0, 1.05)
        ax_cev.set_xlabel("Dimension", fontsize=12)
        if col_idx == 0:
            ax_cev.set_ylabel("Cumulative Explained Variance", fontsize=12)
        ax_cev.tick_params(axis="both", labelsize=10)
        ax_cev.set_xticks([20, 40, 60])
        ax_cev.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_cev.set_yticklabels(['.0', '.2', '.4', '.6', '.8', '1'])
        ax_cev.spines["top"].set_visible(False)
        ax_cev.spines["right"].set_visible(False)

        # --- Bottom row: dims@95 over training ---
        ax_dims = fig.add_subplot(gs[1, col_idx])

        metrics = data.get("metrics")
        if metrics is not None and "dims95" in metrics:
            dims95_df = metrics["dims95"]
            steps = dims95_df["step"].values
            values = dims95_df["value"].values
            x_max = steps.max()

            # Color by training step
            points = np.array([steps, values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = mcolors.LogNorm(vmin=max(steps.min(), 1), vmax=x_max)
            lc = LineCollection(segments, cmap="viridis_r", norm=norm,
                              capstyle='round', joinstyle='round')
            lc.set_array(steps[:-1])
            lc.set_linewidth(2.5)
            lc.set_zorder(2)
            ax_dims.add_collection(lc)
            ax_dims.autoscale()

            # Reference lines
            if gt_dims95 is not None:
                ax_dims.axhline(y=gt_dims95, color="#c44e52", linewidth=1.5,
                               linestyle="--", zorder=1, label=f"Factored ({gt_dims95:.0f})")

            # Compute joint dims from product baseline
            if belief_baselines is not None and "product" in belief_baselines:
                product_cev = belief_baselines["product"]
                indices_above = np.where(product_cev >= 0.95)[0]
                if len(indices_above) > 0:
                    joint_dims95 = float(indices_above[0] + 1)
                    ax_dims.axhline(y=joint_dims95, color="#c44e52", linewidth=1.5,
                                   linestyle=":", zorder=1, alpha=0.7)
                    ax_dims.text(x_max * 0.95, joint_dims95 + 2, f"Joint ({joint_dims95:.0f})",
                               fontsize=9, color="#c44e52", ha="right", va="bottom", alpha=0.7)

            ax_dims.set_xlabel("Training step", fontsize=12)
            if col_idx == 0:
                ax_dims.set_ylabel("Dimensions for 95%", fontsize=12)
            ax_dims.set_ylim(bottom=0)
            ax_dims.tick_params(axis="both", labelsize=10)
            ax_dims.spines["top"].set_visible(False)
            ax_dims.spines["right"].set_visible(False)
            ax_dims.set_xticks(_compute_step_ticks(x_max))
            ax_dims.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, p: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
            ))
            ax_dims.yaxis.grid(True, linestyle="-", alpha=0.2, linewidth=0.5)

    return fig


def main():
    """Generate combined RNN32 vs RNN64 figure."""
    # Cache paths (generated by run_parallel.py)
    cache_dir = Path(".cache")

    # Find cache files for RNN32 and RNN64
    # RNN32: run_id=b281d45b0a774ef7a8139cd76051e95e
    # RNN64: run_id=88afbc07651947388faab5b4d973c2d2

    rnn32_cache = None
    rnn64_cache = None

    for cache_file in cache_dir.glob("cache_*.pkl"):
        # Load and check which run it belongs to
        data = load_cache(str(cache_file))
        layer_name = data.get("layer_name", "")

        # Check metrics to determine which model
        metrics = data.get("metrics", {})
        if "dims95" in metrics:
            max_step = metrics["dims95"]["step"].max()
            # RNN32 and LSTM32 have 168 checkpoints (max 500k), RNN64 has 609 checkpoints
            # We need another way to distinguish - check layer name pattern or cache key
            print(f"Found cache: {cache_file.name}, layer={layer_name}, max_step={max_step}")

    # For now, use explicit cache keys based on the config hash
    # These are generated based on run_id and other params
    # Let's find them by listing and matching
    print("\nAvailable caches:")
    for cache_file in sorted(cache_dir.glob("cache_*.pkl")):
        print(f"  {cache_file.name}")

    # Use the caches we know about from the recent runs
    # RNN32: b647e5c44842 (from earlier output)
    # RNN64: f06accc7b067 (from earlier output)
    rnn32_path = cache_dir / "cache_b647e5c44842.pkl"
    rnn64_path = cache_dir / "cache_f06accc7b067.pkl"

    if not rnn32_path.exists():
        print(f"RNN32 cache not found at {rnn32_path}")
        print("Please run: uv run python run_parallel.py --config-name=fig2 ...")
        return

    if not rnn64_path.exists():
        print(f"RNN64 cache not found at {rnn64_path}")
        print("Please run: uv run python run_parallel.py --config-name=fig2 ...")
        return

    print(f"\nLoading RNN32 from {rnn32_path}")
    data_rnn32 = load_cache(str(rnn32_path))

    print(f"Loading RNN64 from {rnn64_path}")
    data_rnn64 = load_cache(str(rnn64_path))

    # Create combined figure
    print("\nCreating combined figure...")
    fig = create_combined_figure(
        data_left=data_rnn32,
        data_right=data_rnn64,
        title_left="RNN 32d",
        title_right="RNN 64d",
        figsize=(6.75, 4.0),
        gt_dims95=10,
    )

    output_path = "figure_2_rnn_combined.png"
    fig.savefig(output_path, format="png", dpi=300)
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
