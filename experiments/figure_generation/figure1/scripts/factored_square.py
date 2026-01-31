"""
Figure 1: 2D Factored Square view for indecomposable beliefs.
Matches the second column of figure1_v3.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Geometry (same as Blender script)
# =============================================================================

TETRA_VERTICES = np.array([
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
], dtype=float) * 0.65


def belief_to_simplex(a1, a2):
    return np.array([a1*a2, a1*(1-a2), (1-a1)*a2, (1-a1)*(1-a2)])


def simplex_to_3d(probs):
    return probs @ TETRA_VERTICES


def compute_segre_normal(a1, a2, eps=1e-5):
    """Compute the unit normal to the Segre surface at (a1, a2)."""
    p_a1_plus = simplex_to_3d(belief_to_simplex(min(a1 + eps, 1), a2))
    p_a1_minus = simplex_to_3d(belief_to_simplex(max(a1 - eps, 0), a2))
    dp_da1 = (p_a1_plus - p_a1_minus) / (2 * eps)

    p_a2_plus = simplex_to_3d(belief_to_simplex(a1, min(a2 + eps, 1)))
    p_a2_minus = simplex_to_3d(belief_to_simplex(a1, max(a2 - eps, 0)))
    dp_da2 = (p_a2_plus - p_a2_minus) / (2 * eps)

    normal = np.cross(dp_da1, dp_da2)
    normal = normal / np.linalg.norm(normal)
    return normal


# =============================================================================
# Color mapping
# =============================================================================

def pos_to_color(a1, a2):
    """Map (a1, a2) to RGB using bilinear interpolation."""
    c00 = np.array([0.40, 0.65, 0.70])  # teal
    c10 = np.array([0.85, 0.55, 0.50])  # coral
    c01 = np.array([0.55, 0.70, 0.85])  # blue
    c11 = np.array([0.95, 0.85, 0.60])  # gold

    c_bottom = (1 - a1) * c00 + a1 * c10
    c_top = (1 - a1) * c01 + a1 * c11
    return tuple((1 - a2) * c_bottom + a2 * c_top)


def make_colored_square_image(res=100):
    """Create image array for colored square background."""
    img = np.zeros((res, res, 3))
    for i in range(res):
        for j in range(res):
            a1 = j / (res - 1)
            a2 = 1 - i / (res - 1)
            img[i, j] = pos_to_color(a1, a2)
    return img


# =============================================================================
# Indecomposable beliefs (same as Blender script)
# =============================================================================

def compute_beliefs_indecomposable_random(n_points=25, offset_strength=0.25, seed=303):
    """Create indecomposable beliefs from random positions on the Segre surface."""
    rng = np.random.default_rng(seed)

    z_min = TETRA_VERTICES[:, 2].min()
    z_max = TETRA_VERTICES[:, 2].max()

    beliefs = []
    attempts = 0
    while len(beliefs) < n_points and attempts < n_points * 10:
        attempts += 1
        a1 = rng.uniform(0.05, 0.95)
        a2 = rng.uniform(0.05, 0.95)

        factored_simplex = belief_to_simplex(a1, a2)
        factored_pos = simplex_to_3d(factored_simplex)

        normal = compute_segre_normal(a1, a2)
        magnitude = rng.standard_normal() * offset_strength
        true_pos = factored_pos + magnitude * normal

        if true_pos[2] < z_min + 0.02 or true_pos[2] > z_max - 0.02:
            continue

        beliefs.append({
            'alpha1': a1,
            'alpha2': a2,
            'pos3d': true_pos,
            'surface_pos': factored_pos,
        })

    return beliefs


# =============================================================================
# Drawing
# =============================================================================

def draw_factored_square(beliefs, output_path, faded=True):
    """Draw factored view as colored 2D square with corner labels."""
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect('equal', adjustable='box')

    # Colored background
    img = make_colored_square_image(100)
    img_alpha = 0.4 if faded else 1.0
    ax.imshow(img, extent=[0, 1, 0, 1], origin='lower', aspect='auto', alpha=img_alpha, zorder=0)

    # Grid lines
    grid_alpha = 0.2 if faded else 0.3
    for t in [0.25, 0.5, 0.75]:
        ax.axhline(t, color='white', lw=0.8, alpha=grid_alpha)
        ax.axvline(t, color='white', lw=0.8, alpha=grid_alpha)

    # Border
    border_alpha = 0.5 if faded else 0.8
    ax.plot([0,1,1,0,0], [0,0,1,1,0], color='#1E293B', lw=2, alpha=border_alpha)

    # Corner labels (matching figure1_v3 style)
    label_alpha = 0.6 if faded else 0.9
    fs = 14
    ax.text(1.04, 0, r'$S_\alpha$', fontsize=fs, ha='left', va='center', alpha=label_alpha)
    ax.text(1.04, 1, r'$S_\gamma$', fontsize=fs, ha='left', va='center', alpha=label_alpha)
    ax.text(-0.04, 0, r'$S_\beta$', fontsize=fs, ha='right', va='center', alpha=label_alpha)
    ax.text(-0.04, 1, r'$S_\delta$', fontsize=fs, ha='right', va='center', alpha=label_alpha)

    ax.axis('off')

    # Draw belief points
    for b in beliefs:
        # Transform coordinates to match the square layout
        # In figure1_v3, the mapping seems to be: x = alpha1, y = alpha2
        x = b['alpha1']
        y = b['alpha2']

        # Color based on position
        c = pos_to_color(b['alpha1'], b['alpha2'])

        ax.scatter([x], [y], c=[c], s=80,
                  edgecolors='#333333', linewidths=1.5, zorder=10, alpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(output_path.replace('/png/', '/pdf/').replace('.png', '.pdf'), facecolor='white', bbox_inches='tight', pad_inches=0.05)
    print(f"Saved: {output_path} and .pdf")
    plt.close()


if __name__ == '__main__':
    beliefs = compute_beliefs_indecomposable_random(n_points=25, offset_strength=0.25, seed=303)
    draw_factored_square(beliefs, 'experiments/figure_generation/figure1/png/factored_square.png', faded=True)
