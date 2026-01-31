"""
Figure 1: Factored Belief Geometry (v3)

- Factored view: 2D colored square
- Joint view: 3D Segre surface with matching colors
- Clear off-surface for indecomposable
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Arc
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
import colorsys
from matplotlib.colors import LightSource
from scipy.optimize import minimize

# Font for Unicode arrows (DejaVu Sans has full arrow support)
ARROW_FONT = FontProperties(family='DejaVu Sans', weight='bold', size=10)

# =============================================================================
# Style
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',  # Better arrow symbol support
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
})

COLORS = {
    'hmm1': '#C9A227',      # Gold/amber for HMM₁ (factor 1)
    'hmm2': '#8E4585',      # Magenta/plum for HMM₂ (factor 2)
    'joint': '#D4654A',     # Coral/terracotta for joint states
    'edge': '#4A5568',
    'text': '#2D3748',
    'good': '#2E7D5A',      # Muted green
    'bad': '#9B4D4D',       # Muted red
    'light_bg': '#F7FAFC',
    'arrow': '#2D3748',     # Dark gray for arrows
}

# =============================================================================
# HMM Diagram Drawing
# =============================================================================

def draw_curved_arrow(ax, start, end, color='#2D3748', lw=1.5, connectionstyle='arc3,rad=0.2'):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(start, end,
                            connectionstyle=connectionstyle,
                            arrowstyle='->,head_length=8,head_width=5',
                            color=color, lw=lw, zorder=5)
    ax.add_patch(arrow)
    return arrow


def draw_self_loop(ax, center, radius, angle=90, color='#2D3748', lw=1.5):
    """Draw a self-loop arrow at a state using a simple curved path."""
    angle_rad = np.radians(angle)

    # Start and end points on the circle edge
    offset = 25  # degrees offset from the main angle
    start_angle = angle + offset
    end_angle = angle - offset

    start_rad = np.radians(start_angle)
    end_rad = np.radians(end_angle)

    start_x = center[0] + radius * np.cos(start_rad)
    start_y = center[1] + radius * np.sin(start_rad)
    end_x = center[0] + radius * np.cos(end_rad)
    end_y = center[1] + radius * np.sin(end_rad)

    # Draw curved arrow
    arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                            connectionstyle=f'arc3,rad=-0.8',
                            arrowstyle='->,head_length=6,head_width=4',
                            color=color, lw=lw, zorder=5)
    ax.add_patch(arrow)


def draw_joint_hmm(ax):
    """Draw the joint 4-state HMM (2x2 arrangement) with emissions on arrows."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    # State positions (2x2 grid) - centered
    positions = {
        '00': (0.28, 0.28),
        '10': (0.72, 0.28),
        '01': (0.28, 0.72),
        '11': (0.72, 0.72),
    }

    # State names for display
    state_names = {'00': 'A', '10': 'B', '01': 'C', '11': 'D'}

    radius = 0.10
    state_color = COLORS['joint']

    # Draw states with letter labels
    for label, pos in positions.items():
        circle = Circle(pos, radius, facecolor=state_color, edgecolor='white',
                       linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state_names[label], ha='center', va='center',
               fontsize=14, fontweight='bold', color='white', zorder=11)

    # Draw self-loops with emission labels
    loop_angles = {'00': 225, '10': 315, '01': 135, '11': 45}
    emissions = {'00': 'a', '10': 'b', '01': 'c', '11': 'd'}
    for label, pos in positions.items():
        draw_self_loop(ax, pos, radius, angle=loop_angles[label],
                      color=COLORS['arrow'], lw=1.3)
        # Emission label near self-loop
        angle_rad = np.radians(loop_angles[label])
        label_dist = radius + 0.12
        lx = pos[0] + label_dist * np.cos(angle_rad)
        ly = pos[1] + label_dist * np.sin(angle_rad)
        ax.text(lx, ly, f':{emissions[label]}', fontsize=9, ha='center', va='center',
               color=COLORS['text'], style='italic')

    # Draw transition arrows between states
    arrow_color = COLORS['arrow']
    r = radius

    # Horizontal transitions (top row: C↔D)
    draw_curved_arrow(ax, (0.28 + r, 0.74), (0.72 - r, 0.74),
                     color=arrow_color, connectionstyle='arc3,rad=0.2')
    draw_curved_arrow(ax, (0.72 - r, 0.70), (0.28 + r, 0.70),
                     color=arrow_color, connectionstyle='arc3,rad=0.2')

    # Horizontal transitions (bottom row: A↔B)
    draw_curved_arrow(ax, (0.28 + r, 0.30), (0.72 - r, 0.30),
                     color=arrow_color, connectionstyle='arc3,rad=0.2')
    draw_curved_arrow(ax, (0.72 - r, 0.26), (0.28 + r, 0.26),
                     color=arrow_color, connectionstyle='arc3,rad=0.2')

    # Vertical transitions (left column: A↔C)
    draw_curved_arrow(ax, (0.26, 0.28 + r), (0.26, 0.72 - r),
                     color=arrow_color, connectionstyle='arc3,rad=-0.2')
    draw_curved_arrow(ax, (0.30, 0.72 - r), (0.30, 0.28 + r),
                     color=arrow_color, connectionstyle='arc3,rad=-0.2')

    # Vertical transitions (right column: B↔D)
    draw_curved_arrow(ax, (0.70, 0.28 + r), (0.70, 0.72 - r),
                     color=arrow_color, connectionstyle='arc3,rad=-0.2')
    draw_curved_arrow(ax, (0.74, 0.72 - r), (0.74, 0.28 + r),
                     color=arrow_color, connectionstyle='arc3,rad=-0.2')

    # Emission sequence example at bottom
    ax.text(0.50, -0.02, '...abcdabcc...', fontsize=10, ha='center', va='top',
           color=COLORS['text'], family='monospace')

    # Title
    ax.set_title('Joint HMM (4 states)', fontsize=14, fontweight='bold', pad=8)


def draw_factored_hmm(ax):
    """Draw the factored representation: two independent binary HMMs with emissions."""
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.1, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    radius = 0.085

    # HMM 1 (gold/amber) - top row
    hmm1_color = COLORS['hmm1']
    hmm1_y = 0.72
    hmm1_positions = [(0.20, hmm1_y), (0.50, hmm1_y)]

    for i, pos in enumerate(hmm1_positions):
        circle = Circle(pos, radius, facecolor=hmm1_color, edgecolor='white',
                       linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], str(i), ha='center', va='center',
               fontsize=12, fontweight='bold', color='white', zorder=11)

    # Self-loops for HMM1 with emission labels
    draw_self_loop(ax, hmm1_positions[0], radius, angle=135, color=COLORS['arrow'], lw=1.3)
    ax.text(0.08, 0.84, ':0', fontsize=9, ha='center', va='center',
           color=hmm1_color, style='italic')
    draw_self_loop(ax, hmm1_positions[1], radius, angle=45, color=COLORS['arrow'], lw=1.3)
    ax.text(0.62, 0.84, ':1', fontsize=9, ha='center', va='center',
           color=hmm1_color, style='italic')

    # Transitions for HMM1
    draw_curved_arrow(ax, (0.20 + radius, hmm1_y + 0.015), (0.50 - radius, hmm1_y + 0.015),
                     color=COLORS['arrow'], connectionstyle='arc3,rad=0.25')
    draw_curved_arrow(ax, (0.50 - radius, hmm1_y - 0.015), (0.20 + radius, hmm1_y - 0.015),
                     color=COLORS['arrow'], connectionstyle='arc3,rad=0.25')

    # HMM 2 (magenta/plum) - bottom row
    hmm2_color = COLORS['hmm2']
    hmm2_y = 0.28
    hmm2_positions = [(0.20, hmm2_y), (0.50, hmm2_y)]

    for i, pos in enumerate(hmm2_positions):
        circle = Circle(pos, radius, facecolor=hmm2_color, edgecolor='white',
                       linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], str(i), ha='center', va='center',
               fontsize=12, fontweight='bold', color='white', zorder=11)

    # Self-loops for HMM2 with emission labels
    draw_self_loop(ax, hmm2_positions[0], radius, angle=225, color=COLORS['arrow'], lw=1.3)
    ax.text(0.08, 0.16, ':0', fontsize=9, ha='center', va='center',
           color=hmm2_color, style='italic')
    draw_self_loop(ax, hmm2_positions[1], radius, angle=315, color=COLORS['arrow'], lw=1.3)
    ax.text(0.62, 0.16, ':1', fontsize=9, ha='center', va='center',
           color=hmm2_color, style='italic')

    # Transitions for HMM2
    draw_curved_arrow(ax, (0.20 + radius, hmm2_y + 0.015), (0.50 - radius, hmm2_y + 0.015),
                     color=COLORS['arrow'], connectionstyle='arc3,rad=0.25')
    draw_curved_arrow(ax, (0.50 - radius, hmm2_y - 0.015), (0.20 + radius, hmm2_y - 0.015),
                     color=COLORS['arrow'], connectionstyle='arc3,rad=0.25')

    # Labels for HMMs
    ax.text(0.03, hmm1_y, r'$x_1$', fontsize=12, fontweight='bold', color=hmm1_color,
           ha='right', va='center')
    ax.text(0.03, hmm2_y, r'$x_2$', fontsize=12, fontweight='bold', color=hmm2_color,
           ha='right', va='center')

    # Tensor product symbol between the two HMMs
    ax.text(0.35, 0.50, r'$\otimes$', fontsize=16, ha='center', va='center',
           color=COLORS['text'])

    # State mapping on the right side
    mapping_x = 0.70
    label_size = 10

    ax.text(mapping_x, 0.82, '00→a', fontsize=label_size, ha='left', va='center',
           color=COLORS['text'], family='monospace')
    ax.text(mapping_x, 0.70, '01→b', fontsize=label_size, ha='left', va='center',
           color=COLORS['text'], family='monospace')
    ax.text(mapping_x, 0.30, '10→c', fontsize=label_size, ha='left', va='center',
           color=COLORS['text'], family='monospace')
    ax.text(mapping_x, 0.18, '11→d', fontsize=label_size, ha='left', va='center',
           color=COLORS['text'], family='monospace')

    # Binary emission sequences at bottom
    ax.text(0.35, -0.02, '...01100...', fontsize=9, ha='center', va='top',
           color=hmm1_color, family='monospace')
    ax.text(0.35, -0.08, '...00110...', fontsize=9, ha='center', va='top',
           color=hmm2_color, family='monospace')

    # Title
    ax.set_title('Factored: 2 binary HMMs', fontsize=14, fontweight='bold', pad=8)

# =============================================================================
# Color Mapping - proper 2D colormap
# =============================================================================

def pos_to_color(a1, a2):
    """Map (α₁, α₂) to RGB using bilinear interpolation of 4 corner colors.

    Corner colors (lighter, professional):
    - (0,0): light teal
    - (1,0): warm coral/salmon
    - (0,1): soft sky blue
    - (1,1): light gold/cream
    """
    # Define corner colors (RGB, 0-1) - lighter palette
    c00 = np.array([0.40, 0.65, 0.70])  # light teal (α₁=0, α₂=0)
    c10 = np.array([0.85, 0.55, 0.50])  # warm coral (α₁=1, α₂=0)
    c01 = np.array([0.55, 0.70, 0.85])  # sky blue (α₁=0, α₂=1)
    c11 = np.array([0.95, 0.85, 0.60])  # light gold (α₁=1, α₂=1)

    # Bilinear interpolation
    c_bottom = (1 - a1) * c00 + a1 * c10
    c_top = (1 - a1) * c01 + a1 * c11
    return tuple((1 - a2) * c_bottom + a2 * c_top)


def make_colored_square_image(res=100, rotate=-90, flip_vertical=True, extra_rotate=180):
    """Create image array for colored square background.

    rotate: degrees to rotate the square (-90 = clockwise 90°)
    flip_vertical: if True, mirror vertically after rotation
    extra_rotate: additional rotation in degrees (0, 90, 180, 270)
    """
    img = np.zeros((res, res, 3))
    for i in range(res):
        for j in range(res):
            a1 = j / (res - 1)
            a2 = 1 - i / (res - 1)  # Flip so α₂ increases upward
            img[i, j] = pos_to_color(a1, a2)

    # Apply rotation (-90 degrees = clockwise = k=-1 in numpy)
    if rotate == -90:
        img = np.rot90(img, k=-1)
    elif rotate == 90:
        img = np.rot90(img, k=1)
    elif rotate == 180:
        img = np.rot90(img, k=2)

    # Apply vertical flip
    if flip_vertical:
        img = np.flipud(img)

    # Apply extra rotation
    if extra_rotate == 90:
        img = np.rot90(img, k=1)
    elif extra_rotate == 180:
        img = np.rot90(img, k=2)
    elif extra_rotate == 270:
        img = np.rot90(img, k=3)

    return img


# =============================================================================
# HMM Math
# =============================================================================

class TwoStateHMM:
    """SNS (Simple Nonunifilar Source) with transition-based emissions.

    Emissions:
    - A→A: emit 1 (self-loop)
    - A→B: emit 1 (transition)
    - B→A: emit 0 (the only way to emit 0)
    - B→B: emit 1 (self-loop)
    """
    def __init__(self, t_AA, t_BA):
        self.t_AA = t_AA
        self.t_BA = t_BA

    def steady_state(self):
        denom = self.t_BA + (1 - self.t_AA)
        if abs(denom) < 1e-10:
            return 0.5
        return self.t_BA / denom

    def update_belief(self, alpha, y):
        """Update belief for SNS with transition-based emissions."""
        if y == 0:
            # Only B→A produces y=0, so we're definitely in A now
            return 1.0
        else:  # y == 1
            # y=1 from A→A, A→B, or B→B
            # P(y=1) = α + (1-α)*(1-t_BA) = 1 - (1-α)*t_BA
            p_y1 = 1 - (1 - alpha) * self.t_BA
            if p_y1 < 1e-10:
                return alpha
            # P(now in A | y=1) = P(A→A) / P(y=1) = α*t_AA / p_y1
            return alpha * self.t_AA / p_y1


def belief_to_simplex(a1, a2):
    return np.array([a1*a2, a1*(1-a2), (1-a1)*a2, (1-a1)*(1-a2)])


# Tetrahedron
TETRA_VERTICES = np.array([
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
], dtype=float) * 0.65

VERTEX_LABELS = [r'\alpha', r'\beta', r'\gamma', r'\delta']  # 00=α, 01=β, 10=γ, 11=δ
TETRA_EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]


def simplex_to_3d(probs):
    return probs @ TETRA_VERTICES


def compute_beliefs_independent(hmm1, hmm2, max_len, init_belief=None):
    """Compute beliefs starting from specified initial belief (or steady state)."""
    beliefs = []

    if init_belief is None:
        a1_init, a2_init = hmm1.steady_state(), hmm2.steady_state()
    else:
        a1_init, a2_init = init_belief

    def recurse(seq, a1, a2):
        beliefs.append({
            'len': len(seq), 'alpha1': a1, 'alpha2': a2,
            'pos3d': simplex_to_3d(belief_to_simplex(a1, a2))
        })
        if len(seq) >= max_len:
            return
        for y1 in [0, 1]:
            for y2 in [0, 1]:
                recurse(seq + [(y1,y2)], hmm1.update_belief(a1, y1), hmm2.update_belief(a2, y2))

    recurse([], a1_init, a2_init)
    return beliefs


def compute_beliefs_coupled(hmm1, hmm2, max_len, coupling, init_belief=None):
    """Coupled but still product-preserving - stays on surface."""
    beliefs = []

    if init_belief is None:
        a1_init, a2_init = hmm1.steady_state(), hmm2.steady_state()
    else:
        a1_init, a2_init = init_belief

    def recurse(seq, a1, a2):
        beliefs.append({
            'len': len(seq), 'alpha1': a1, 'alpha2': a2,
            'pos3d': simplex_to_3d(belief_to_simplex(a1, a2))
        })
        if len(seq) >= max_len:
            return
        for y1 in [0, 1]:
            for y2 in [0, 1]:
                new_a1 = hmm1.update_belief(a1, y1)
                # Coupling: y1 affects HMM2's update
                mod_t_AA = hmm2.t_AA + coupling * (1 if y1 == 0 else -1) * 0.2
                mod_t_BA = hmm2.t_BA + coupling * (1 if y1 == 0 else -1) * 0.2
                mod_t_AA = np.clip(mod_t_AA, 0.05, 0.95)
                mod_t_BA = np.clip(mod_t_BA, 0.05, 0.95)
                mod_hmm2 = TwoStateHMM(mod_t_AA, mod_t_BA)
                new_a2 = mod_hmm2.update_belief(a2, y2)
                recurse(seq + [(y1,y2)], new_a1, new_a2)

    recurse([], a1_init, a2_init)
    return beliefs


def project_to_segre(simplex_point):
    """Find the closest point on the Segre surface to a given simplex point.

    Segre surface: p(α₁,α₂) = (α₁α₂, α₁(1-α₂), (1-α₁)α₂, (1-α₁)(1-α₂))
    Minimize ||q - p(α₁,α₂)||² over (α₁,α₂) ∈ [0,1]²
    """
    def segre_point(alphas):
        a1, a2 = alphas
        return np.array([a1*a2, a1*(1-a2), (1-a1)*a2, (1-a1)*(1-a2)])

    def objective(alphas):
        return np.sum((simplex_point - segre_point(alphas))**2)

    # Try multiple starting points to avoid local minima
    best_result = None
    best_val = float('inf')
    for a1_init in [0.25, 0.5, 0.75]:
        for a2_init in [0.25, 0.5, 0.75]:
            result = minimize(objective, [a1_init, a2_init],
                            bounds=[(0.001, 0.999), (0.001, 0.999)],
                            method='L-BFGS-B')
            if result.fun < best_val:
                best_val = result.fun
                best_result = result

    closest_alphas = best_result.x
    closest_simplex = segre_point(closest_alphas)
    return closest_simplex, closest_alphas


def project_to_segre_3d(point_3d):
    """Find the closest point on the Segre surface to a given 3D point.

    Searches over (α₁, α₂) to minimize 3D Euclidean distance.
    """
    def objective(alphas):
        a1, a2 = alphas
        surface_simplex = belief_to_simplex(a1, a2)
        surface_3d = simplex_to_3d(surface_simplex)
        return np.sum((point_3d - surface_3d)**2)

    # Try multiple starting points
    best_result = None
    best_val = float('inf')
    for a1_init in [0.25, 0.5, 0.75]:
        for a2_init in [0.25, 0.5, 0.75]:
            result = minimize(objective, [a1_init, a2_init],
                            bounds=[(0.001, 0.999), (0.001, 0.999)],
                            method='L-BFGS-B')
            if result.fun < best_val:
                best_val = result.fun
                best_result = result

    closest_alphas = best_result.x
    closest_3d = simplex_to_3d(belief_to_simplex(*closest_alphas))
    return closest_3d, closest_alphas


def compute_beliefs_indecomposable_random(n_points=30, offset_strength=0.25, seed=42):
    """Create indecomposable beliefs from random positions on the Segre surface.

    Generates random (α₁, α₂) positions on the surface, then pushes them in z.
    """
    rng = np.random.default_rng(seed)

    # Get simplex bounds (tetrahedron vertices define the valid region)
    z_min = TETRA_VERTICES[:, 2].min()
    z_max = TETRA_VERTICES[:, 2].max()

    beliefs = []
    for i in range(n_points):
        # Random position on the Segre surface
        a1 = rng.uniform(0.05, 0.95)
        a2 = rng.uniform(0.05, 0.95)

        factored_simplex = belief_to_simplex(a1, a2)
        factored_pos = simplex_to_3d(factored_simplex)

        # Push OFF surface in z with varying magnitude (positive and negative)
        magnitude = rng.standard_normal() * offset_strength
        true_pos = factored_pos + np.array([0, 0, magnitude])

        # Clamp z to stay within simplex bounds (with small margin)
        true_pos[2] = np.clip(true_pos[2], z_min + 0.05, z_max - 0.05)

        # Recompute actual magnitude after clamping
        actual_magnitude = true_pos[2] - factored_pos[2]

        # Find orthogonal projection back onto Segre surface (perpendicular to surface)
        closest_pos, proj_alphas = project_to_segre_3d(true_pos)

        beliefs.append({
            'len': i,  # Use index as "length" for sizing
            'alpha1': a1,
            'alpha2': a2,
            'pos3d': true_pos,
            'factored_pos3d': closest_pos,
            'proj_alpha1': proj_alphas[0],
            'proj_alpha2': proj_alphas[1],
            'offset_magnitude': actual_magnitude,  # Store for occlusion check
        })

    return beliefs


# =============================================================================
# Depth cueing for 3D visualization
# =============================================================================

def compute_camera_depth(points, elev, azim):
    """Compute depth of 3D points relative to camera position.

    Uses the matplotlib 3D camera model: the camera looks from a direction
    determined by elevation and azimuth angles towards the origin.
    """
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)

    cam_dir = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])

    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    return points @ cam_dir


def depth_to_alpha(depths, min_alpha=0.25, max_alpha=1.0):
    """Map depths to alpha values (further from camera = more transparent)."""
    if len(depths) <= 1:
        return np.array([max_alpha])

    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min < 1e-10:
        return np.ones(len(depths)) * max_alpha

    # Larger depth = closer to camera (higher projection onto cam direction)
    # So larger depth should get higher alpha
    normalized = (depths - d_min) / (d_max - d_min)
    return min_alpha + normalized * (max_alpha - min_alpha)


def depth_to_size(depths, min_size=35, max_size=100):
    """Map depths to point sizes (further from camera = smaller)."""
    if len(depths) <= 1:
        return np.array([max_size])

    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min < 1e-10:
        return np.ones(len(depths)) * max_size

    # Larger depth = closer to camera, should be larger size
    normalized = (depths - d_min) / (d_max - d_min)
    return min_size + normalized * (max_size - min_size)


# =============================================================================
# Segre Surface
# =============================================================================

def segre_mesh(res=35):
    a1 = np.linspace(0, 1, res)
    a2 = np.linspace(0, 1, res)
    A1, A2 = np.meshgrid(a1, a2)
    X, Y, Z = np.zeros_like(A1), np.zeros_like(A1), np.zeros_like(A1)
    C = np.zeros((res, res, 4))  # RGBA colors
    
    for i in range(res):
        for j in range(res):
            pt = simplex_to_3d(belief_to_simplex(A1[i,j], A2[i,j]))
            X[i,j], Y[i,j], Z[i,j] = pt
            rgb = pos_to_color(A1[i,j], A2[i,j])
            C[i, j] = (*rgb, 0.7)  # Add alpha
    
    return X, Y, Z, C


def segre_grid(res=6):
    lines = []
    for i in range(res + 1):
        a = i / res
        lines.append(np.array([simplex_to_3d(belief_to_simplex(a, j/30)) for j in range(31)]))
        lines.append(np.array([simplex_to_3d(belief_to_simplex(j/30, a)) for j in range(31)]))
    return lines


def segre_boundary():
    r = 50
    return [
        np.array([simplex_to_3d(belief_to_simplex(0, i/r)) for i in range(r+1)]),
        np.array([simplex_to_3d(belief_to_simplex(1, i/r)) for i in range(r+1)]),
        np.array([simplex_to_3d(belief_to_simplex(i/r, 0)) for i in range(r+1)]),
        np.array([simplex_to_3d(belief_to_simplex(i/r, 1)) for i in range(r+1)]),
    ]


# =============================================================================
# Drawing
# =============================================================================

def setup_3d_ax(ax):
    ax.set_box_aspect([1, 1, 1])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.line.set_color('none')
    ax.yaxis.line.set_color('none')
    ax.zaxis.line.set_color('none')
    ax.computed_zorder = False


def draw_tetra(ax, alpha=0.35):
    for i, j in TETRA_EDGES:
        ax.plot3D(*zip(TETRA_VERTICES[i], TETRA_VERTICES[j]),
                  color=COLORS['edge'], alpha=alpha, lw=1.0, zorder=0)

    # Vertex labels - bold S with subscript Greek letters
    for v, lbl in zip(TETRA_VERTICES, VERTEX_LABELS):
        vx, vy, vz = v[0]*1.15, v[1]*1.15, v[2]*1.15
        ax.text(vx, vy, vz, r'$\mathbf{S}_{' + lbl + r'}$', ha='center', va='center',
                color=COLORS['edge'], fontsize=14, zorder=200)


def draw_segre_colored(ax, show_grid=True, show_shadow=True, surface_alpha=0.7, faded=False):
    """Draw Segre surface with position-matched colors."""
    X, Y, Z, C = segre_mesh(res=40)
    
    if show_shadow:
        z_floor = TETRA_VERTICES[:, 2].min() * 0.98
        shadow_alpha = 0.08 if faded else 0.15
        ax.plot_surface(X, Y, np.full_like(Z, z_floor), 
                       alpha=shadow_alpha, color='#334155',
                       edgecolor='none', shade=False, zorder=1)
    
    # Apply fading
    if faded:
        C[:, :, 3] = 0.25  # Reduce alpha
    else:
        C[:, :, 3] = surface_alpha
    
    # Draw colored surface
    ax.plot_surface(X, Y, Z, facecolors=C, edgecolor='none',
                   shade=True, zorder=2, rcount=40, ccount=40)
    
    if show_grid:
        grid_alpha = 0.15 if faded else 0.4
        for line in segre_grid(5):
            ax.plot3D(line[:,0], line[:,1], line[:,2],
                     color='#1E293B', alpha=grid_alpha, lw=0.5, zorder=3)
    
    # Boundary
    boundary_alpha = 0.3 if faded else 0.7
    for b in segre_boundary():
        ax.plot3D(b[:,0], b[:,1], b[:,2],
                 color='#1E293B', alpha=boundary_alpha, lw=1.5, zorder=4)


def draw_factored_square(ax, beliefs, title="", faded=False):
    """Draw factored view as colored 2D square with corner labels only.

    The square is rotated -90 degrees (clockwise), flipped vertically, then
    rotated 180° more to align with the Segre variety orientation.

    Combined transformation: (α₁, α₂) → (1-α₂, α₁)
    """
    # Smaller square with more padding
    sq_min, sq_max = 0.2, 0.8  # Square from 0.2 to 0.8 (smaller)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal', adjustable='box')

    # Colored background (rotated -90 degrees + vertical flip + 180° extra)
    img = make_colored_square_image(100, rotate=-90, flip_vertical=True, extra_rotate=180)
    img_alpha = 0.3 if faded else 1.0
    ax.imshow(img, extent=[sq_min, sq_max, sq_min, sq_max], origin='lower', aspect='auto', alpha=img_alpha, zorder=0)

    # Grid
    grid_alpha = 0.15 if faded else 0.3
    sq_range = sq_max - sq_min
    for t in [0.25, 0.5, 0.75]:
        ax.axhline(sq_min + t * sq_range, xmin=sq_min, xmax=sq_max, color='white', lw=0.5, alpha=grid_alpha)
        ax.axvline(sq_min + t * sq_range, ymin=sq_min, ymax=sq_max, color='white', lw=0.5, alpha=grid_alpha)

    # Border
    border_alpha = 0.3 if faded else 0.8
    ax.plot([sq_min, sq_max, sq_max, sq_min, sq_min],
            [sq_min, sq_min, sq_max, sq_max, sq_min], color='#1E293B', lw=1.5, alpha=border_alpha)

    # Corner labels - bold S with Greek letter subscripts
    label_alpha = 0.4 if faded else 0.9

    # Corner positions: (x, y, greek, ha)
    # Mapping: 00=α (bottom-left origin), 01=β, 10=γ, 11=δ
    # But square is transformed, so positions are different
    corners = [
        (sq_max + 0.02, sq_min, r'\gamma', 'left'),   # bottom-right = 10
        (sq_max + 0.02, sq_max, r'\delta', 'left'),   # top-right = 11
        (sq_min - 0.02, sq_min, r'\alpha', 'right'),  # bottom-left = 00
        (sq_min - 0.02, sq_max, r'\beta', 'right'),   # top-left = 01
    ]

    for x, y, greek, ha in corners:
        ax.text(x, y, r'$\mathbf{S}_{' + greek + r'}$', ha=ha, va='center',
               color=COLORS['text'], alpha=label_alpha, fontsize=14)

    # Remove axes completely
    ax.axis('off')

    # Beliefs - larger with dark edges for visibility
    # Transform coordinates: (α₁, α₂) → (1-α₂, α₁) then scale to square bounds
    # Use projected alphas if available (for indecomposable)
    point_alpha = 0.5 if faded else 1.0
    for b in beliefs:
        # Use projected alphas if available, otherwise use original
        a1 = b.get('proj_alpha1', b['alpha1'])
        a2 = b.get('proj_alpha2', b['alpha2'])
        c = pos_to_color(a1, a2)
        s = 50
        ec = '#333333'
        ew = 1.0
        # Transformed coordinates to match background, scaled to smaller square
        x_trans = sq_min + (1 - a2) * sq_range
        y_trans = sq_min + a1 * sq_range
        ax.scatter([x_trans], [y_trans], c=[c], s=s,
                  edgecolors=ec, linewidths=ew, zorder=10, alpha=point_alpha)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)


def draw_beliefs_3d_colored(ax, beliefs, show_projection_lines=False,
                             elev=20, azim=160, depth_cue=True,
                             min_alpha=0.25, max_alpha=1.0,
                             min_size=35, max_size=100):
    """Draw 3D beliefs with position-matched colors and depth cueing.

    Args:
        ax: 3D matplotlib axis
        beliefs: list of belief dicts with 'pos3d', 'alpha1', 'alpha2', 'len'
        show_projection_lines: if True, draw lines from points to surface
        elev: camera elevation angle (for depth cueing)
        azim: camera azimuth angle (for depth cueing)
        depth_cue: if True, fade and shrink points further from camera
        min_alpha: minimum alpha for furthest points
        max_alpha: maximum alpha for closest points
        min_size: minimum size for furthest points
        max_size: maximum size for closest points
    """
    # Projection lines (red lines from off-surface points to surface)
    # Draw with per-segment occlusion-based alpha
    if show_projection_lines:
        n_segments = 10  # Number of segments to divide each line into

        # Camera direction for occlusion
        elev_rad = np.radians(elev)
        cam_dir_z = np.sin(elev_rad)  # positive means camera is above

        for b in beliefs:
            if 'factored_pos3d' in b and 'offset_magnitude' in b:
                p = b['pos3d']
                fp = b['factored_pos3d']
                a1, a2 = b['alpha1'], b['alpha2']

                # Surface z at the original (α₁, α₂)
                surface_pos = simplex_to_3d(belief_to_simplex(a1, a2))
                surface_z = surface_pos[2]

                dist = np.linalg.norm(p - fp)
                if dist > 0.01:
                    # Sample points along the line
                    for i in range(n_segments):
                        t0 = i / n_segments
                        t1 = (i + 1) / n_segments
                        q0 = (1 - t0) * p + t0 * fp
                        q1 = (1 - t1) * p + t1 * fp
                        mid = (q0 + q1) / 2

                        # Check occlusion: is midpoint z above or below surface z?
                        # If camera is above (cam_dir_z > 0):
                        #   - points below surface (mid_z < surface_z) are occluded
                        # If camera is below (cam_dir_z < 0):
                        #   - points above surface (mid_z > surface_z) are occluded
                        z_diff = mid[2] - surface_z
                        is_occluded = (z_diff * cam_dir_z) < 0

                        seg_alpha = 0.15 if is_occluded else 0.85
                        ax.plot3D([q0[0], q1[0]], [q0[1], q1[1]], [q0[2], q1[2]],
                                 color='#C44', alpha=seg_alpha, lw=1.2, ls='-', zorder=50)

    # Collect all 3D positions for depth calculation
    all_points = np.array([b['pos3d'] for b in beliefs])

    # Compute depth-based styling if enabled
    if depth_cue and len(all_points) > 1:
        depths = compute_camera_depth(all_points, elev, azim)
        alphas = depth_to_alpha(depths, min_alpha=min_alpha, max_alpha=max_alpha)
        sizes = depth_to_size(depths, min_size=min_size, max_size=max_size)
    else:
        alphas = np.ones(len(beliefs)) * max_alpha
        sizes = np.ones(len(beliefs)) * max_size

    # Compute zorder based on depth - closer points get higher zorder
    if depth_cue and len(all_points) > 1:
        depths = compute_camera_depth(all_points, elev, azim)
        # Normalize depths to zorder range (100-200)
        d_min, d_max = depths.min(), depths.max()
        if d_max - d_min > 1e-10:
            zorders = 100 + 100 * (depths - d_min) / (d_max - d_min)
        else:
            zorders = np.ones(len(beliefs)) * 150
        sort_order = np.argsort(-depths)  # draw furthest first
    else:
        zorders = np.ones(len(beliefs)) * 150
        sort_order = range(len(beliefs))

    for idx in sort_order:
        b = beliefs[idx]
        c = pos_to_color(b['alpha1'], b['alpha2'])
        p = b['pos3d']

        # Use depth-based size
        s = sizes[idx]
        ec = '#333333'
        ew = 1.2

        # For indecomposable points, only use occlusion-based alpha (not depth cueing)
        if 'factored_pos3d' in b and 'offset_magnitude' in b:
            # Occlusion: is point above or below surface z at its (α₁, α₂)?
            elev_rad = np.radians(elev)
            cam_dir_z = np.sin(elev_rad)
            surface_pos = simplex_to_3d(belief_to_simplex(b['alpha1'], b['alpha2']))
            z_diff = p[2] - surface_pos[2]
            is_occluded = (z_diff * cam_dir_z) < 0
            point_alpha = 0.25 if is_occluded else 1.0
        else:
            point_alpha = alphas[idx]  # Use depth cueing for non-indecomposable

        ax.scatter([p[0]], [p[1]], [p[2]], c=[c], s=s,
                  edgecolors=ec, linewidths=ew, zorder=zorders[idx], depthshade=False, alpha=point_alpha)


# =============================================================================
# Main Figure
# =============================================================================

def create_figure():
    # SNS HMMs - lower t_BA means less P(emit 0), more gradual belief spread
    hmm1 = TwoStateHMM(t_AA=0.7, t_BA=0.3)
    hmm2 = TwoStateHMM(t_AA=0.6, t_BA=0.35)

    max_len = 3
    init_belief = (0.5, 0.5)  # Start from center for good spread
    beliefs_ind = compute_beliefs_independent(hmm1, hmm2, max_len, init_belief=init_belief)
    beliefs_dep = compute_beliefs_coupled(hmm1, hmm2, max_len, coupling=-1.5, init_belief=init_belief)
    beliefs_decomp = compute_beliefs_indecomposable_random(n_points=25, offset_strength=0.25, seed=303)
    
    # Check off-surface deviation
    deviations = [np.linalg.norm(b['pos3d'] - b['factored_pos3d']) for b in beliefs_decomp]
    print(f"Indecomposable: max deviation = {max(deviations):.3f}, mean = {np.mean(deviations):.3f}")
    
    # Figure: 4 rows × 2 columns - each part (a,b,c,d) in its own row
    # Col 0: 3D views
    # Col 1: 2D squares (or scaling plot for d)
    fig = plt.figure(figsize=(8, 14))

    gs = GridSpec(4, 2, figure=fig,
                  height_ratios=[1, 1, 1, 0.8],
                  width_ratios=[1.2, 1],
                  hspace=0.2, wspace=0.1,
                  left=0.02, right=0.98, top=0.97, bottom=0.03)

    view_elev, view_azim = 20, 160

    # =========================================================================
    # (a) Independent - Row 0
    # =========================================================================

    ax_j_a = fig.add_subplot(gs[0, 0], projection='3d')
    setup_3d_ax(ax_j_a)
    ax_j_a.view_init(elev=view_elev, azim=view_azim)
    draw_tetra(ax_j_a)
    draw_segre_colored(ax_j_a, show_shadow=True)
    draw_beliefs_3d_colored(ax_j_a, beliefs_ind, elev=view_elev, azim=view_azim)
    ax_j_a.set_title('(a) Independent\non surface', fontsize=12, fontweight='bold',
                     color=COLORS['good'], pad=10)

    ax_f_a = fig.add_subplot(gs[0, 1])
    draw_factored_square(ax_f_a, beliefs_ind, title='lossless')

    # =========================================================================
    # (b) Dependent - Row 1
    # =========================================================================

    ax_j_b = fig.add_subplot(gs[1, 0], projection='3d')
    setup_3d_ax(ax_j_b)
    ax_j_b.view_init(elev=view_elev, azim=view_azim)
    draw_tetra(ax_j_b)
    draw_segre_colored(ax_j_b, show_shadow=True)
    draw_beliefs_3d_colored(ax_j_b, beliefs_dep, elev=view_elev, azim=view_azim)
    ax_j_b.set_title('(b) Dependent\non surface', fontsize=12, fontweight='bold',
                     color=COLORS['good'], pad=10)

    ax_f_b = fig.add_subplot(gs[1, 1])
    draw_factored_square(ax_f_b, beliefs_dep, title='lossless')

    # =========================================================================
    # (c) Indecomposable - Row 2
    # =========================================================================

    ax_j_c = fig.add_subplot(gs[2, 0], projection='3d')
    setup_3d_ax(ax_j_c)
    ax_j_c.view_init(elev=view_elev, azim=view_azim)
    draw_tetra(ax_j_c)
    draw_segre_colored(ax_j_c, show_shadow=True, faded=False)
    draw_beliefs_3d_colored(ax_j_c, beliefs_decomp, show_projection_lines=True,
                             elev=view_elev, azim=view_azim)
    ax_j_c.set_title('(c) Indecomposable\nOFF surface', fontsize=12, fontweight='bold',
                     color=COLORS['bad'], pad=10)

    ax_f_c = fig.add_subplot(gs[2, 1])
    draw_factored_square(ax_f_c, beliefs_decomp, title='lossy', faded=False)

    # =========================================================================
    # (d) Scaling - Row 3
    # =========================================================================

    ax_scale = fig.add_subplot(gs[3, 0])  # Single column

    N = np.arange(1, 13)
    dim_joint = 2**N - 1
    dim_factored = N

    ax_scale.semilogy(N, dim_joint, 'o-', color=COLORS['bad'], lw=2.5, ms=7,
                     label=r'Joint: $2^N - 1$')
    ax_scale.semilogy(N, dim_factored, 's-', color=COLORS['good'], lw=2.5, ms=7,
                     label=r'Factored: $N$')

    ax_scale.fill_between(N, dim_factored, dim_joint, alpha=0.15, color=COLORS['bad'])
    ax_scale.text(9, 40, 'exponential\nsavings', ha='center', fontsize=10,
                 color=COLORS['text'], style='italic')

    ax_scale.set_title('(d) Dimensional Scaling', fontsize=12, fontweight='bold', pad=12)
    ax_scale.set_xlabel('N (number of binary factors)', fontsize=11)
    ax_scale.set_ylabel('Dimensions needed', fontsize=11)
    ax_scale.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9,
                    edgecolor='#DDD', fancybox=True)
    ax_scale.set_xlim(0.5, 12.5)
    ax_scale.set_ylim(0.5, 6000)
    ax_scale.spines['top'].set_visible(False)
    ax_scale.spines['right'].set_visible(False)
    ax_scale.tick_params(labelsize=10)
    
    # =========================================================================
    # Save
    # =========================================================================

    plt.savefig('experiments/figure_generation/figure1/pdf/segre_geometry.pdf', dpi=300, facecolor='white')
    plt.savefig('experiments/figure_generation/figure1/png/segre_geometry.png', dpi=200, facecolor='white')
    plt.savefig('experiments/figure_generation/figure1/svg/segre_geometry.svg', facecolor='white')
    print("Saved: experiments/figure_generation/figure1/{pdf,png,svg}/segre_geometry.*")
    
    return fig


if __name__ == '__main__':
    create_figure()