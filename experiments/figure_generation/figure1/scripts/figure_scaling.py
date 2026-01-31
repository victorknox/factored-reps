"""
Standalone scaling diagram: Joint (2^N - 1) vs Factored (N) dimensional scaling.
Generates both log-scale and linear-scale versions.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Style - Publication quality
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'svg.fonttype': 'none',
})

# Colors
COLORS = {
    'good': '#2E7D5A',      # Muted green - factored (efficient)
    'bad': '#9B4D4D',       # Muted red - joint (exponential)
    'text': '#2C3E50',      # Dark blue-gray
    'fill': '#9B4D4D',      # Fill color for savings region
}


def create_scaling_plot(ax, use_log=True, show_title=True):
    """Create the scaling comparison plot on the given axes."""
    N = np.arange(1, 13)
    dim_joint = 2**N - 1
    dim_factored = N

    if use_log:
        ax.semilogy(N, dim_joint, 'o-', color=COLORS['bad'], lw=3, ms=8,
                   label=r'Joint: $2^N - 1$', markerfacecolor='white', markeredgewidth=2)
        ax.semilogy(N, dim_factored, 's-', color=COLORS['good'], lw=3, ms=8,
                   label=r'Factored: $N$', markerfacecolor='white', markeredgewidth=2)
        ax.set_ylim(0.8, 6000)
        # Position "exponential savings" text
        ax.text(8.5, 40, 'exponential\nsavings', ha='center', fontsize=12,
               color=COLORS['text'], style='italic', fontweight='medium')
    else:
        ax.plot(N, dim_joint, 'o-', color=COLORS['bad'], lw=3, ms=8,
               label=r'Joint: $2^N - 1$', markerfacecolor='white', markeredgewidth=2)
        ax.plot(N, dim_factored, 's-', color=COLORS['good'], lw=3, ms=8,
               label=r'Factored: $N$', markerfacecolor='white', markeredgewidth=2)
        ax.set_ylim(-50, 4200)
        # Position text differently for linear scale
        ax.text(9, 1500, 'exponential\nsavings', ha='center', fontsize=12,
               color=COLORS['text'], style='italic', fontweight='medium')

    # Fill between the curves
    ax.fill_between(N, dim_factored, dim_joint, alpha=0.15, color=COLORS['fill'])

    # Labels and styling
    if show_title:
        ax.set_title('Dimensional Scaling', fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('N (number of binary factors)', fontsize=14)
    ax.set_ylabel('Belief dimensions', fontsize=14)

    # Legend
    ax.legend(loc='upper left', fontsize=12, frameon=True, framealpha=0.95,
             edgecolor='#CCC', fancybox=True)

    # Axis limits and ticks
    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(np.arange(2, 13, 2))

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)

    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)


def create_figure_log():
    """Create the log-scale version."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')

    create_scaling_plot(ax, use_log=True)

    plt.tight_layout()

    # Save
    plt.savefig('experiments/figure_generation/figure1/pdf/scaling_log.pdf', dpi=300, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/png/scaling_log.png', dpi=200, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/svg/scaling_log.svg', facecolor='white',
               bbox_inches='tight', pad_inches=0.1)

    print("Saved: experiments/figure_generation/figure1/{pdf,png,svg}/scaling_log.*")
    return fig


def create_figure_log_fitted(width_px=680, height_px=300, dpi=100):
    """Create log-scale version fitted to exact pixel dimensions."""
    # Convert pixels to inches
    width_in = width_px / dpi
    height_in = height_px / dpi

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor('white')

    # Smaller fonts for compact figure
    N = np.arange(1, 13)
    dim_joint = 2**N - 1
    dim_factored = N

    ax.semilogy(N, dim_joint, 'o-', color=COLORS['bad'], lw=2.5, ms=7,
               label=r'Joint: $2^N - 1$', markerfacecolor='white', markeredgewidth=2)
    ax.semilogy(N, dim_factored, 's-', color=COLORS['good'], lw=2.5, ms=7,
               label=r'Factored: $N$', markerfacecolor='white', markeredgewidth=2)

    # Fill between
    ax.fill_between(N, dim_factored, dim_joint, alpha=0.15, color=COLORS['fill'])

    # "exponential savings" text
    ax.text(8.5, 35, 'exponential\nsavings', ha='center', fontsize=14,
           color=COLORS['text'], style='italic', fontweight='medium')

    # Title
    ax.set_title('Factoring Saves Dims', fontsize=18, fontweight='bold', pad=10)

    # Labels
    ax.set_xlabel('N (Number of 2-state Factors)', fontsize=16)
    ax.set_ylabel('Dimensions', fontsize=16)

    # Legend
    ax.legend(loc='upper left', fontsize=14, frameon=True, framealpha=0.95,
             edgecolor='#CCC', fancybox=True, handlelength=1.5)

    # Axis limits and ticks
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0.8, 6000)
    ax.set_xticks(np.arange(2, 13, 2))

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=14)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)

    # Save at exact pixel dimensions
    plt.savefig('experiments/figure_generation/figure1/pdf/scaling_log_fitted.pdf', dpi=dpi, facecolor='white')
    plt.savefig('experiments/figure_generation/figure1/png/scaling_log_fitted.png', dpi=dpi, facecolor='white')
    plt.savefig('experiments/figure_generation/figure1/svg/scaling_log_fitted.svg', facecolor='white')

    print(f"Saved: experiments/figure_generation/figure1/{{pdf,png,svg}}/scaling_log_fitted.* ({width_px}x{height_px}px at {dpi}dpi)")
    return fig


def create_figure_linear():
    """Create the linear-scale version."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')

    create_scaling_plot(ax, use_log=False)

    plt.tight_layout()

    # Save
    plt.savefig('experiments/figure_generation/figure1/pdf/scaling_linear.pdf', dpi=300, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/png/scaling_linear.png', dpi=200, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/svg/scaling_linear.svg', facecolor='white',
               bbox_inches='tight', pad_inches=0.1)

    print("Saved: experiments/figure_generation/figure1/{pdf,png,svg}/scaling_linear.*")
    return fig


def create_figure_both():
    """Create a side-by-side comparison with both scales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    create_scaling_plot(ax1, use_log=False, show_title=True)
    ax1.set_title('Linear Scale', fontsize=16, fontweight='bold', pad=12)

    create_scaling_plot(ax2, use_log=True, show_title=True)
    ax2.set_title('Log Scale', fontsize=16, fontweight='bold', pad=12)

    plt.tight_layout()

    # Save
    plt.savefig('experiments/figure_generation/figure1/pdf/scaling_both.pdf', dpi=300, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/png/scaling_both.png', dpi=200, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/svg/scaling_both.svg', facecolor='white',
               bbox_inches='tight', pad_inches=0.1)

    print("Saved: experiments/figure_generation/figure1/{pdf,png,svg}/scaling_both.*")
    return fig


if __name__ == '__main__':
    create_figure_log()
    create_figure_linear()
    create_figure_both()
    create_figure_log_fitted(width_px=723, height_px=328, dpi=100)
