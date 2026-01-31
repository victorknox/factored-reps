"""
Figure 1 HMMs: Joint and Factored HMM diagrams for ICML submission.
Labels show "emission | probability%" format.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle, ConnectionStyle, PathPatch, Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe

# Font for Unicode arrows (DejaVu Sans has full arrow support)
ARROW_FONT = FontProperties(family='DejaVu Sans', weight='bold', size=11)

# =============================================================================
# Style - Publication quality
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',  # Better arrow symbol support
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'svg.fonttype': 'none',  # Keep text editable in Figma (may need to substitute fonts)
})

# Refined color palette
COLORS = {
    'hmm1': '#E2C16B',      # Warm gold (lighter)
    'hmm2': '#A67BB3',      # Rich purple (lighter)
    'joint': '#7A98A3',     # Steel blue (lighter)
    'hmm1_text': '#D4A84B', # Warm gold (darker)
    'hmm2_text': '#8B5A9C', # Rich purple (darker)
    'edge': '#4A5568',      # Consistent edge color
    'text': '#2C3E50',      # Dark blue-gray
}

# HMM parameters
# SNS (Simple Nonunifilar Source) parameters
# Emissions: S₋ self-loop emits 1, S₋→S₊ emits 1, S₊→S₋ emits 0, S₊ self-loop emits 1
HMM1_PARAMS = {'t_AA': 0.7, 't_AB': 0.3, 't_BA': 0.3, 't_BB': 0.7}
HMM2_PARAMS = {'t_AA': 0.6, 't_AB': 0.4, 't_BA': 0.35, 't_BB': 0.65}
JOINT_EMISSIONS = {(0, 0): 'A', (0, 1): 'B', (1, 0): 'C', (1, 1): 'D'}
BOLD_DIGITS = {'0': '𝟎', '1': '𝟏'}

# =============================================================================
# Display options (easy to toggle)
# =============================================================================
SHOW_PROBS = False      # Set to True to show "D:42%", False for just "D"
WEIGHT_ARROWS = True   # Set to True to scale arrow width by probability
SHADE_ARROWS = True    # Set to True to use grayscale shading (darker = higher prob)
MIN_ARROW_WIDTH = 0.8  # Minimum arrow width when WEIGHT_ARROWS is True
MAX_ARROW_WIDTH = 2.5  # Maximum arrow width when WEIGHT_ARROWS is True
MIN_GRAY = 0.75        # Lightest gray (0=black, 1=white) for lowest prob
MAX_GRAY = 0.25        # Darkest gray for highest prob


def get_arrow_width(prob):
    """Calculate arrow width based on probability (linear interpolation)."""
    if not WEIGHT_ARROWS:
        return 1.5  # Default width
    # Linear interpolation between MIN and MAX based on probability
    return MIN_ARROW_WIDTH + (MAX_ARROW_WIDTH - MIN_ARROW_WIDTH) * prob


def get_arrow_color(prob, base_color=None):
    """Calculate arrow color based on probability (grayscale interpolation).

    If SHADE_ARROWS is False, returns the base_color or default edge color.
    If SHADE_ARROWS is True, returns a grayscale color (darker = higher prob).
    """
    if not SHADE_ARROWS:
        return base_color if base_color else COLORS['edge']
    # Linear interpolation: high prob -> dark gray, low prob -> light gray
    gray = MIN_GRAY + (MAX_GRAY - MIN_GRAY) * prob
    return f'#{int(gray*255):02x}{int(gray*255):02x}{int(gray*255):02x}'


def sample_binary_emissions(params, length, rng, start_state=None):
    """Sample a binary emission sequence from a 2-state SNS.

    SNS emissions:
    - Self-loop (stay in same state) → emits 1
    - S_+ → S_- transition → emits 0
    """
    if start_state is None:
        state = int(rng.random() < 0.5)
    else:
        state = int(start_state)

    emissions = []
    for _ in range(length):
        old_state = state
        if state == 0:  # State A (S_-)
            state = 0 if rng.random() < params['t_AA'] else 1
        else:  # State B (S_+)
            state = 0 if rng.random() < params['t_BA'] else 1

        # SNS emission: 0 only when transitioning from S_+ to S_-
        if old_state == 1 and state == 0:
            emissions.append(0)
        else:
            emissions.append(1)
    return emissions


def format_sequence(seq):
    """Format a sequence as space-separated tokens."""
    return ' '.join(str(token) for token in seq)


def bold_digit(token):
    """Return a bold unicode digit for 0/1 tokens."""
    return BOLD_DIGITS.get(str(token), str(token))


def decorate_sequence(seq):
    """Wrap a sequence with leading/trailing dots."""
    return f'... {format_sequence(seq)} ...'


def sequence_tokens(seq):
    """Return dot-prefixed/suffixed tokens for alignment."""
    return ['.', '.', '.'] + [str(token) for token in seq] + ['.', '.', '.']


def draw_sequence(ax, seq, y, color, fontsize, fontfamily, fontweight,
                  x_center=0.5, dx=0.045):
    """Draw an aligned token sequence with consistent spacing."""
    tokens = sequence_tokens(seq)
    x0 = x_center - (len(tokens) - 1) * dx / 2
    for i, token in enumerate(tokens):
        ax.text(x0 + i * dx, y, token, fontsize=fontsize, ha='center', va='center',
               color=color, fontfamily=fontfamily, fontweight=fontweight)


def draw_decode_table(ax, origin=(0.72, 0.40), size=0.22,
                      col_color=None, row_color=None):
    """Draw a small decoding table mapping (x1, x2) -> emission."""
    x0, y0 = origin
    w = h = size
    line_color = COLORS['edge']
    text_color = COLORS['text']
    col_color = col_color or text_color
    row_color = row_color or text_color

    # Outer box and grid lines
    ax.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 + h, y0 + h, y0],
            color=line_color, lw=0.9, zorder=4)
    ax.plot([x0 + w / 2, x0 + w / 2], [y0, y0 + h],
            color=line_color, lw=0.9, zorder=4)
    ax.plot([x0, x0 + w], [y0 + h / 2, y0 + h / 2],
            color=line_color, lw=0.9, zorder=4)

    # Column/row headers
    ax.text(x0 + w * 0.25, y0 + h + 0.03, '0', fontsize=9, ha='center',
           va='bottom', color=col_color, fontweight='bold')
    ax.text(x0 + w * 0.75, y0 + h + 0.03, '1', fontsize=9, ha='center',
           va='bottom', color=col_color, fontweight='bold')
    ax.text(x0 - 0.03, y0 + h * 0.75, '0', fontsize=9, ha='right',
           va='center', color=row_color, fontweight='bold')
    ax.text(x0 - 0.03, y0 + h * 0.25, '1', fontsize=9, ha='right',
           va='center', color=row_color, fontweight='bold')

    # Cell values
    cells = {
        (0, 0): 'A',
        (1, 0): 'C',
        (0, 1): 'B',
        (1, 1): 'D',
    }
    for x1 in (0, 1):
        for x2 in (0, 1):
            cx = x0 + (0.25 if x1 == 0 else 0.75) * w
            cy = y0 + (0.75 if x2 == 0 else 0.25) * h
            ax.text(cx, cy, cells[(x1, x2)], fontsize=10, ha='center',
                   va='center', color=text_color, fontweight='bold')


# =============================================================================
# Drawing utilities
# =============================================================================

def add_triangle_arrow(ax, tip, direction, color, size=0.025, zorder=6):
    """Add a filled triangular arrowhead at the given tip."""
    direction = np.array(direction, dtype=float)
    if np.linalg.norm(direction) == 0:
        return None
    tangent = direction / np.linalg.norm(direction)
    left = np.array([-tangent[1], tangent[0]])

    arrow_tip = np.array(tip, dtype=float)
    arrow_base1 = arrow_tip - tangent * size * 1.5 + left * size
    arrow_base2 = arrow_tip - tangent * size * 1.5 - left * size

    triangle = Polygon([arrow_tip, arrow_base1, arrow_base2],
                       facecolor=color, edgecolor=color, zorder=zorder)
    ax.add_patch(triangle)
    return triangle


def draw_arrow(ax, start, end, color='#4A5568', rad=0.25, lw=1.15, zorder=5, head_size=0.020):
    """Draw a curved arrow with a triangular head."""
    connection = ConnectionStyle.Arc3(rad=rad)
    path = connection(start, end)
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, zorder=zorder)
    ax.add_patch(patch)

    verts = path.vertices
    if len(verts) >= 2:
        tip = verts[-1]
        direction = verts[-1] - verts[-2]
    else:
        tip = np.array(end)
        direction = np.array(end) - np.array(start)

    add_triangle_arrow(ax, tip, direction, color, size=head_size, zorder=zorder + 1)
    return patch


def draw_self_loop_bezier(ax, center, radius, angle=90, color='#2D3748', lw=1.15,
                          loop_size=0.18, attach_gap_mult=0.12, head_size=0.020,
                          label_pad=0.0):
    """Custom bezier for precise loop shape"""
    offset = 35
    start_rad = np.radians(angle + offset)
    end_rad = np.radians(angle - offset)

    # Start and end on circle edge
    attach_gap = radius * attach_gap_mult
    start_r = radius + attach_gap
    end_r = radius + attach_gap
    p0 = np.array([center[0] + start_r * np.cos(start_rad),
                   center[1] + start_r * np.sin(start_rad)])
    p3 = np.array([center[0] + end_r * np.cos(end_rad),
                   center[1] + end_r * np.sin(end_rad)])

    # Control points pushed outward
    angle_rad = np.radians(angle)
    outward = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    # Control points create the loop bulge
    p1 = p0 + outward * loop_size + np.array([-outward[1], outward[0]]) * loop_size * 0.5
    p2 = p3 + outward * loop_size + np.array([outward[1], -outward[0]]) * loop_size * 0.5

    # Draw cubic bezier curve
    path = Path([p0, p1, p2, p3], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, zorder=5)
    ax.add_patch(patch)

    # Add arrowhead at end (tangent of bezier at t=1 is 3*(p3-p2))
    tangent = 3 * (p3 - p2)
    add_triangle_arrow(ax, p3, tangent, color, size=head_size, zorder=6)

    label_dist = radius + attach_gap + loop_size + radius * 0.4 + label_pad
    return (center[0] + label_dist * np.cos(angle_rad),
            center[1] + label_dist * np.sin(angle_rad))


def draw_state(ax, pos, radius, color):
    """Draw a state circle."""
    circle = Circle(pos, radius, facecolor=color, edgecolor='none', zorder=10)
    ax.add_patch(circle)


def get_curve_point(start, end, rad, t=0.5):
    """Get point on a curved arrow path (quadratic bezier).

    For matplotlib's arc3 connection style, the control point is offset
    perpendicular to the chord. When rad > 0, the curve bends to the right
    (looking from start to end), which means the control point is on the left.

    t=0.0 gives start, t=1.0 gives end, t=0.5 gives the midpoint of the curve.
    """
    x1, y1 = start
    x2, y2 = end

    # Chord length and perpendicular direction
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return x1, y1

    # Perpendicular unit vector (matches matplotlib's convention)
    perp_x, perp_y = -dy / length, dx / length

    # Control point is at: midpoint - rad * length * perpendicular
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ctrl_x = mx - perp_x * rad * length
    ctrl_y = my - perp_y * rad * length

    # Quadratic bezier: B(t) = (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
    curve_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
    curve_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2

    return curve_x, curve_y


def get_curve_tangent_angle(start, end, rad, t=0.5):
    """Get the tangent angle (in degrees) at point t on a curved arrow path.

    Returns the angle that a label should be rotated to follow the curve.
    """
    x1, y1 = start
    x2, y2 = end

    # Chord length and perpendicular direction
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return 0

    # Perpendicular unit vector (matches matplotlib's convention)
    perp_x, perp_y = -dy / length, dx / length

    # Control point
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    ctrl_x = mx - perp_x * rad * length
    ctrl_y = my - perp_y * rad * length

    # Tangent of quadratic bezier: B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
    tangent_x = 2*(1-t)*(ctrl_x - x1) + 2*t*(x2 - ctrl_x)
    tangent_y = 2*(1-t)*(ctrl_y - y1) + 2*t*(y2 - ctrl_y)

    # Convert to angle in degrees
    angle = np.degrees(np.arctan2(tangent_y, tangent_x))

    # Keep angle in readable range (-90 to 90) so text isn't upside down
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def add_label(ax, x, y, emission, prob, fontsize=10, color='#2C3E50', rotation=0,
              use_mathtext=True, emission_text=None, fontfamily=None):
    """Add label with bold emission and optionally probability, with white background."""
    emission_str = emission_text if emission_text is not None else str(emission)
    if SHOW_PROBS:
        if use_mathtext:
            label = rf'$\mathbf{{{emission_str}}}:\mathrm{{{int(prob*100)}\%}}$'
        else:
            label = f'{emission_str}:{int(prob*100)}%'
    else:
        # Just show the emission token
        if use_mathtext:
            label = rf'$\mathbf{{{emission_str}}}$'
        else:
            label = f'{emission_str}'
    bbox_props = dict(boxstyle='round,pad=0.15', facecolor='white', alpha=1.0, edgecolor='none')
    txt = ax.text(x, y, label, fontsize=fontsize, ha='center', va='center',
                  color=color, zorder=15, rotation=rotation, fontfamily=fontfamily,
                  bbox=bbox_props)
    return txt


def add_label_on_curve(ax, start, end, rad, emission, prob, fontsize=10, color='#2C3E50', rotation=0,
                       t=0.5, use_mathtext=True, emission_text=None, fontfamily=None):
    """Add label directly on a curved arrow at position t (0=start, 1=end, 0.5=middle)."""
    x, y = get_curve_point(start, end, rad, t)
    return add_label(ax, x, y, emission, prob, fontsize=fontsize, color=color, rotation=rotation,
                     use_mathtext=use_mathtext, emission_text=emission_text, fontfamily=fontfamily)


# =============================================================================
# Joint HMM (4 states)
# =============================================================================

def draw_joint_hmm(ax, joint_seq=None):
    """Draw the joint 4-state HMM with emission|prob labels."""
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    # State positions (2x2 grid)
    spacing = 0.6
    cx, cy = 0.5, 0.5
    positions = {
        '00': (cx - spacing/2, cy - spacing/2),
        '10': (cx + spacing/2, cy - spacing/2),
        '01': (cx - spacing/2, cy + spacing/2),
        '11': (cx + spacing/2, cy + spacing/2),
    }

    radius = 0.085
    edge_color = COLORS['edge']
    # SNS self-loop emissions: all self-loops emit (1,1) → D
    # (Both factors emit 1 on self-loops)
    emissions = {'00': 'D', '01': 'D', '10': 'D', '11': 'D'}

    # Draw states
    for label, pos in positions.items():
        draw_state(ax, pos, radius, COLORS['joint'])

    # State labels inside circles - Greek letters for joint states
    # '00' = α, '01' = β, '10' = γ, '11' = δ
    state_labels = {'00': r'\alpha', '01': r'\beta', '10': r'\gamma', '11': r'\delta'}
    for label, pos in positions.items():
        ax.text(pos[0], pos[1], r'$\mathbf{S}_{' + state_labels[label] + r'}$', ha='center', va='center',
               color='white', fontsize=11, zorder=11)

    # Self-loop probabilities
    self_probs = {
        '00': HMM1_PARAMS['t_AA'] * HMM2_PARAMS['t_AA'],
        '01': HMM1_PARAMS['t_AA'] * HMM2_PARAMS['t_BB'],
        '10': HMM1_PARAMS['t_BB'] * HMM2_PARAMS['t_AA'],
        '11': HMM1_PARAMS['t_BB'] * HMM2_PARAMS['t_BB'],
    }
    loop_angles = {'00': 225, '10': 315, '01': 135, '11': 45}

    # Draw self-loops with labels
    loop_label_pad_top = -radius * 0.12
    for label, pos in positions.items():
        label_pad = loop_label_pad_top if label in ('01', '11') else 0.0
        prob = self_probs[label]
        lx, ly = draw_self_loop_bezier(ax, pos, radius, loop_angles[label],
                                       color=get_arrow_color(prob, edge_color), label_pad=label_pad,
                                       lw=get_arrow_width(prob))
        add_label(ax, lx, ly, emissions[label], prob,
                 fontsize=10, color=COLORS['text'])

    r = radius
    rad_transition = 0.18

    # Horizontal transitions (00 <-> 10, bottom row)
    p_00_10 = HMM1_PARAMS['t_AB'] * HMM2_PARAMS['t_AA']
    p_10_00 = HMM1_PARAMS['t_BA'] * HMM2_PARAMS['t_AA']

    y_bottom = positions['00'][1]
    # 00->10 (outer, below)
    start_00_10 = (positions['00'][0] + r, y_bottom - r*0.4)
    end_00_10 = (positions['10'][0] - r, y_bottom - r*0.4)
    draw_arrow(ax, start_00_10, end_00_10, get_arrow_color(p_00_10, edge_color), rad=rad_transition, lw=get_arrow_width(p_00_10))
    add_label_on_curve(ax, start_00_10, end_00_10, rad_transition, 'D', p_00_10, fontsize=10)

    # 10->00 (inner, above)
    start_10_00 = (positions['10'][0] - r, y_bottom + r*0.4)
    end_10_00 = (positions['00'][0] + r, y_bottom + r*0.4)
    draw_arrow(ax, start_10_00, end_10_00, get_arrow_color(p_10_00, edge_color), rad=-rad_transition, lw=get_arrow_width(p_10_00))
    add_label_on_curve(ax, start_10_00, end_10_00, -rad_transition, 'B', p_10_00, fontsize=10)

    # Horizontal transitions (01 <-> 11, top row)
    p_01_11 = HMM1_PARAMS['t_AB'] * HMM2_PARAMS['t_BB']
    p_11_01 = HMM1_PARAMS['t_BA'] * HMM2_PARAMS['t_BB']

    y_top = positions['01'][1]
    # 01->11 (outer, above)
    start_01_11 = (positions['01'][0] + r, y_top + r*0.4)
    end_01_11 = (positions['11'][0] - r, y_top + r*0.4)
    draw_arrow(ax, start_01_11, end_01_11, get_arrow_color(p_01_11, edge_color), rad=-rad_transition, lw=get_arrow_width(p_01_11))
    add_label_on_curve(ax, start_01_11, end_01_11, -rad_transition, 'D', p_01_11, fontsize=10)

    # 11->01 (inner, below)
    start_11_01 = (positions['11'][0] - r, y_top - r*0.4)
    end_11_01 = (positions['01'][0] + r, y_top - r*0.4)
    draw_arrow(ax, start_11_01, end_11_01, get_arrow_color(p_11_01, edge_color), rad=rad_transition, lw=get_arrow_width(p_11_01))
    add_label_on_curve(ax, start_11_01, end_11_01, rad_transition, 'B', p_11_01, fontsize=10)

    # Vertical transitions (00 <-> 01, left column)
    p_00_01 = HMM1_PARAMS['t_AA'] * HMM2_PARAMS['t_AB']
    p_01_00 = HMM1_PARAMS['t_AA'] * HMM2_PARAMS['t_BA']

    x_left = positions['00'][0]
    # 00->01 (outer, left side)
    start_00_01 = (x_left - r*0.4, positions['00'][1] + r)
    end_00_01 = (x_left - r*0.4, positions['01'][1] - r)
    draw_arrow(ax, start_00_01, end_00_01, get_arrow_color(p_00_01, edge_color), rad=-rad_transition, lw=get_arrow_width(p_00_01))
    add_label_on_curve(ax, start_00_01, end_00_01, -rad_transition, 'D', p_00_01, fontsize=10, rotation=90)

    # 01->00 (inner, right side)
    start_01_00 = (x_left + r*0.4, positions['01'][1] - r)
    end_01_00 = (x_left + r*0.4, positions['00'][1] + r)
    draw_arrow(ax, start_01_00, end_01_00, get_arrow_color(p_01_00, edge_color), rad=rad_transition, lw=get_arrow_width(p_01_00))
    add_label_on_curve(ax, start_01_00, end_01_00, rad_transition, 'C', p_01_00, fontsize=10, rotation=90)

    # Vertical transitions (10 <-> 11, right column)
    p_10_11 = HMM1_PARAMS['t_BB'] * HMM2_PARAMS['t_AB']
    p_11_10 = HMM1_PARAMS['t_BB'] * HMM2_PARAMS['t_BA']

    x_right = positions['10'][0]
    # 10->11 (inner, left side)
    start_10_11 = (x_right + r*0.4, positions['10'][1] + r)
    end_10_11 = (x_right + r*0.4, positions['11'][1] - r)
    draw_arrow(ax, start_10_11, end_10_11, get_arrow_color(p_10_11, edge_color), rad=rad_transition, lw=get_arrow_width(p_10_11))
    add_label_on_curve(ax, start_10_11, end_10_11, rad_transition, 'D', p_10_11, fontsize=10, rotation=90)

    # 11->10 (outer, right side)
    start_11_10 = (x_right - r*0.4, positions['11'][1] - r)
    end_11_10 = (x_right - r*0.4, positions['10'][1] + r)
    draw_arrow(ax, start_11_10, end_11_10, get_arrow_color(p_11_10, edge_color), rad=-rad_transition, lw=get_arrow_width(p_11_10))
    add_label_on_curve(ax, start_11_10, end_11_10, -rad_transition, 'C', p_11_10, fontsize=10, rotation=90)

    # Diagonal transitions (both factors transition simultaneously)
    # 00 <-> 11 diagonal
    p_00_11 = HMM1_PARAMS['t_AB'] * HMM2_PARAMS['t_AB']  # Both 0→1
    p_11_00 = HMM1_PARAMS['t_BA'] * HMM2_PARAMS['t_BA']  # Both 1→0

    # 01 <-> 10 diagonal
    p_01_10 = HMM1_PARAMS['t_AB'] * HMM2_PARAMS['t_BA']  # factor1: 0→1, factor2: 1→0
    p_10_01 = HMM1_PARAMS['t_BA'] * HMM2_PARAMS['t_AB']  # factor1: 1→0, factor2: 0→1

    # Draw diagonal arrows (00 <-> 11) - opposite curvature with perpendicular offset
    diag_offset = r * 1.0  # Offset from circle edge along diagonal
    rad_diag = 0.2
    sep = 0.025  # Perpendicular separation
    # Perpendicular to (1,1) diagonal is (-1,1)/sqrt(2)
    perp_00_11 = (-0.707 * sep, 0.707 * sep)

    # 00 -> 11 (curves one way, offset perpendicular)
    start_00_11 = (positions['00'][0] + diag_offset + perp_00_11[0], positions['00'][1] + diag_offset + perp_00_11[1])
    end_00_11 = (positions['11'][0] - diag_offset + perp_00_11[0], positions['11'][1] - diag_offset + perp_00_11[1])
    draw_arrow(ax, start_00_11, end_00_11, get_arrow_color(p_00_11, edge_color), rad=rad_diag, lw=get_arrow_width(p_00_11))

    # 11 -> 00 (curves the other way, offset opposite)
    start_11_00 = (positions['11'][0] - diag_offset - perp_00_11[0], positions['11'][1] - diag_offset - perp_00_11[1])
    end_11_00 = (positions['00'][0] + diag_offset - perp_00_11[0], positions['00'][1] + diag_offset - perp_00_11[1])
    draw_arrow(ax, start_11_00, end_11_00, get_arrow_color(p_11_00, edge_color), rad=-rad_diag, lw=get_arrow_width(p_11_00))

    # Labels on the diagonal curves - place at t=0.3 (closer to start) to avoid center overlap
    # Labels follow the curve tangent angle
    t_label = 0.3
    # D: on 00->11 curve (both emit 1: D)
    d_x, d_y = get_curve_point(start_00_11, end_00_11, rad_diag, t=t_label)
    d_angle = get_curve_tangent_angle(start_00_11, end_00_11, rad_diag, t=t_label)
    add_label(ax, d_x, d_y, 'D', p_00_11, fontsize=9, color=COLORS['text'], rotation=d_angle)
    # A: on 11->00 curve (both emit 0: A)
    a_x, a_y = get_curve_point(start_11_00, end_11_00, -rad_diag, t=t_label)
    a_angle = get_curve_tangent_angle(start_11_00, end_11_00, -rad_diag, t=t_label)
    add_label(ax, a_x, a_y, 'A', p_11_00, fontsize=9, color=COLORS['text'], rotation=a_angle)

    # Draw diagonal arrows (01 <-> 10) - opposite curvature with perpendicular offset
    # Perpendicular to (1,-1) diagonal is (1,1)/sqrt(2)
    perp_01_10 = (0.707 * sep, 0.707 * sep)

    # 01 -> 10 (curves one way, offset perpendicular)
    start_01_10 = (positions['01'][0] + diag_offset + perp_01_10[0], positions['01'][1] - diag_offset + perp_01_10[1])
    end_01_10 = (positions['10'][0] - diag_offset + perp_01_10[0], positions['10'][1] + diag_offset + perp_01_10[1])
    draw_arrow(ax, start_01_10, end_01_10, get_arrow_color(p_01_10, edge_color), rad=rad_diag, lw=get_arrow_width(p_01_10))

    # 10 -> 01 (curves the other way, offset opposite)
    start_10_01 = (positions['10'][0] - diag_offset - perp_01_10[0], positions['10'][1] + diag_offset - perp_01_10[1])
    end_10_01 = (positions['01'][0] + diag_offset - perp_01_10[0], positions['01'][1] - diag_offset - perp_01_10[1])
    draw_arrow(ax, start_10_01, end_10_01, get_arrow_color(p_10_01, edge_color), rad=-rad_diag, lw=get_arrow_width(p_10_01))

    # C: on 01->10 curve (factor1 emits 1, factor2 emits 0: C)
    c_x, c_y = get_curve_point(start_01_10, end_01_10, rad_diag, t=t_label)
    c_angle = get_curve_tangent_angle(start_01_10, end_01_10, rad_diag, t=t_label)
    add_label(ax, c_x, c_y, 'C', p_01_10, fontsize=9, color=COLORS['text'], rotation=c_angle)
    # B: on 10->01 curve (factor1 emits 0, factor2 emits 1: B)
    b_x, b_y = get_curve_point(start_10_01, end_10_01, -rad_diag, t=t_label)
    b_angle = get_curve_tangent_angle(start_10_01, end_10_01, -rad_diag, t=t_label)
    add_label(ax, b_x, b_y, 'B', p_10_01, fontsize=9, color=COLORS['text'], rotation=b_angle)

    # Title
    ax.text(0.5, 1.15, 'Joint HMM', fontsize=13, fontweight='bold',
           ha='center', va='bottom', color=COLORS['text'])

    if joint_seq:
        draw_sequence(ax, joint_seq, y=-0.22, color=COLORS['text'],
                     fontsize=11, fontfamily='DejaVu Sans Mono', fontweight='bold')


# =============================================================================
# Factored HMMs
# =============================================================================

def draw_factored_hmm(ax, seq1=None, seq2=None, joint_seq=None):
    """Draw the factored representation: two independent binary HMMs."""
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')

    radius = 0.075
    edge_color = COLORS['edge']
    spacing = 0.6
    cx = 0.5
    hmm_x = [cx - spacing / 2, cx + spacing / 2]
    loop_label_pad = -radius * 0.12
    # =========== HMM 1 (gold) - top ===========
    hmm1_color = COLORS['hmm1']
    hmm1_label_color = COLORS['hmm1_text']
    hmm1_y = 0.75
    hmm1_x = hmm_x

    draw_state(ax, (hmm1_x[0], hmm1_y), radius, hmm1_color)
    draw_state(ax, (hmm1_x[1], hmm1_y), radius, hmm1_color)
    # State labels inside circles - +/- for binary states
    ax.text(hmm1_x[0], hmm1_y, r'$\mathbf{S}_{-}$', ha='center', va='center',
           color='white', fontsize=11, zorder=11)
    ax.text(hmm1_x[1], hmm1_y, r'$\mathbf{S}_{+}$', ha='center', va='center',
           color='white', fontsize=11, zorder=11)

    # Self-loops
    lx, ly = draw_self_loop_bezier(ax, (hmm1_x[0], hmm1_y), radius, 135,
                                   color=get_arrow_color(HMM1_PARAMS['t_AA'], edge_color), label_pad=loop_label_pad,
                                   lw=get_arrow_width(HMM1_PARAMS['t_AA']))
    add_label(ax, lx, ly, '1', HMM1_PARAMS['t_AA'], fontsize=11, color=hmm1_label_color,
              use_mathtext=False, emission_text=bold_digit('1'), fontfamily='STIXGeneral')

    lx, ly = draw_self_loop_bezier(ax, (hmm1_x[1], hmm1_y), radius, 45,
                                   color=get_arrow_color(HMM1_PARAMS['t_BB'], edge_color), label_pad=loop_label_pad,
                                   lw=get_arrow_width(HMM1_PARAMS['t_BB']))
    add_label(ax, lx, ly, '1', HMM1_PARAMS['t_BB'], fontsize=11, color=hmm1_label_color,
              use_mathtext=False, emission_text=bold_digit('1'), fontfamily='STIXGeneral')

    # Transitions
    r = radius
    rad_pair = -0.14
    # Top arrow: left to right
    start_hmm1_top = (hmm1_x[0] + r, hmm1_y + r*0.5)
    end_hmm1_top = (hmm1_x[1] - r, hmm1_y + r*0.5)
    draw_arrow(ax, start_hmm1_top, end_hmm1_top, get_arrow_color(HMM1_PARAMS['t_AB'], edge_color), rad=rad_pair, lw=get_arrow_width(HMM1_PARAMS['t_AB']))
    # Bottom arrow: right to left
    start_hmm1_bot = (hmm1_x[1] - r, hmm1_y - r*0.5)
    end_hmm1_bot = (hmm1_x[0] + r, hmm1_y - r*0.5)
    draw_arrow(ax, start_hmm1_bot, end_hmm1_bot, get_arrow_color(HMM1_PARAMS['t_BA'], edge_color), rad=rad_pair, lw=get_arrow_width(HMM1_PARAMS['t_BA']))

    # Labels on the curves
    add_label_on_curve(ax, start_hmm1_top, end_hmm1_top, rad_pair, '1', HMM1_PARAMS['t_AB'],
                       fontsize=11, color=hmm1_label_color, use_mathtext=False,
                       emission_text=bold_digit('1'), fontfamily='STIXGeneral')
    add_label_on_curve(ax, start_hmm1_bot, end_hmm1_bot, rad_pair, '0', HMM1_PARAMS['t_BA'],
                       fontsize=11, color=hmm1_label_color, use_mathtext=False,
                       emission_text=bold_digit('0'), fontfamily='STIXGeneral')

    # =========== HMM 2 (purple) - bottom ===========
    hmm2_color = COLORS['hmm2']
    hmm2_label_color = COLORS['hmm2_text']
    hmm2_y = 0.25
    hmm2_x = hmm_x

    draw_state(ax, (hmm2_x[0], hmm2_y), radius, hmm2_color)
    draw_state(ax, (hmm2_x[1], hmm2_y), radius, hmm2_color)
    # State labels inside circles - +/- for binary states
    ax.text(hmm2_x[0], hmm2_y, r'$\mathbf{S}_{-}$', ha='center', va='center',
           color='white', fontsize=11, zorder=11)
    ax.text(hmm2_x[1], hmm2_y, r'$\mathbf{S}_{+}$', ha='center', va='center',
           color='white', fontsize=11, zorder=11)

    # Self-loops
    lx, ly = draw_self_loop_bezier(ax, (hmm2_x[0], hmm2_y), radius, 135,
                                   color=get_arrow_color(HMM2_PARAMS['t_AA'], edge_color), label_pad=loop_label_pad,
                                   lw=get_arrow_width(HMM2_PARAMS['t_AA']))
    add_label(ax, lx, ly, '1', HMM2_PARAMS['t_AA'], fontsize=11, color=hmm2_label_color,
              use_mathtext=False, emission_text=bold_digit('1'), fontfamily='STIXGeneral')

    lx, ly = draw_self_loop_bezier(ax, (hmm2_x[1], hmm2_y), radius, 45,
                                   color=get_arrow_color(HMM2_PARAMS['t_BB'], edge_color), label_pad=loop_label_pad,
                                   lw=get_arrow_width(HMM2_PARAMS['t_BB']))
    add_label(ax, lx, ly, '1', HMM2_PARAMS['t_BB'], fontsize=11, color=hmm2_label_color,
              use_mathtext=False, emission_text=bold_digit('1'), fontfamily='STIXGeneral')

    # Transitions
    # Top arrow: left to right
    start_hmm2_top = (hmm2_x[0] + r, hmm2_y + r*0.5)
    end_hmm2_top = (hmm2_x[1] - r, hmm2_y + r*0.5)
    draw_arrow(ax, start_hmm2_top, end_hmm2_top, get_arrow_color(HMM2_PARAMS['t_AB'], edge_color), rad=rad_pair, lw=get_arrow_width(HMM2_PARAMS['t_AB']))
    # Bottom arrow: right to left
    start_hmm2_bot = (hmm2_x[1] - r, hmm2_y - r*0.5)
    end_hmm2_bot = (hmm2_x[0] + r, hmm2_y - r*0.5)
    draw_arrow(ax, start_hmm2_bot, end_hmm2_bot, get_arrow_color(HMM2_PARAMS['t_BA'], edge_color), rad=rad_pair, lw=get_arrow_width(HMM2_PARAMS['t_BA']))

    # Labels on the curves
    add_label_on_curve(ax, start_hmm2_top, end_hmm2_top, rad_pair, '1', HMM2_PARAMS['t_AB'],
                       fontsize=11, color=hmm2_label_color, use_mathtext=False,
                       emission_text=bold_digit('1'), fontfamily='STIXGeneral')
    add_label_on_curve(ax, start_hmm2_bot, end_hmm2_bot, rad_pair, '0', HMM2_PARAMS['t_BA'],
                       fontsize=11, color=hmm2_label_color, use_mathtext=False,
                       emission_text=bold_digit('0'), fontfamily='STIXGeneral')

    seq_fontsize = 11
    seq_fontfamily = 'DejaVu Sans Mono'
    seq_fontweight = 'bold'
    seq_dx = 0.045
    if seq1:
        draw_sequence(ax, seq1, y=-0.04, color=hmm1_label_color,
                     fontsize=seq_fontsize, fontfamily=seq_fontfamily,
                     fontweight=seq_fontweight, dx=seq_dx)
    if seq2:
        draw_sequence(ax, seq2, y=-0.14, color=hmm2_label_color,
                     fontsize=seq_fontsize, fontfamily=seq_fontfamily,
                     fontweight=seq_fontweight, dx=seq_dx)
    if joint_seq:
        ax.annotate('', xy=(0.5, -0.32), xytext=(0.5, -0.18),
                    arrowprops=dict(arrowstyle='-|>', lw=1.0, color='#666666'))
        draw_sequence(ax, joint_seq, y=-0.40, color=COLORS['text'],
                     fontsize=seq_fontsize, fontfamily=seq_fontfamily,
                     fontweight=seq_fontweight, dx=seq_dx)

    # =========== Tensor product symbol ===========
    ax.text(0.5, 0.5, r'$\otimes$', fontsize=18, ha='center', va='center',
           color='#888888')

    # =========== Decode table ===========
    draw_decode_table(ax, origin=(-0.14, -0.20), size=0.22,
                      col_color=hmm1_label_color, row_color=hmm2_label_color)

    # Title
    ax.text(0.5, 1.1, 'Factored HMMs', fontsize=13, fontweight='bold',
           ha='center', va='bottom', color=COLORS['text'])


# =============================================================================
# Main Figure
# =============================================================================

def create_figure():
    """Create publication-quality figure."""
    rng = np.random.default_rng(7)
    seq_len = 10
    seq1 = sample_binary_emissions(HMM1_PARAMS, seq_len, rng)
    seq2 = sample_binary_emissions(HMM2_PARAMS, seq_len, rng)
    joint_seq = [JOINT_EMISSIONS[(seq1[i], seq2[i])] for i in range(seq_len)]

    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor('white')

    gs = GridSpec(1, 2, figure=fig, wspace=0.02,
                  left=0.02, right=0.98, top=0.90, bottom=0.05)

    ax_joint = fig.add_subplot(gs[0, 0])
    draw_joint_hmm(ax_joint, joint_seq=joint_seq)

    ax_factored = fig.add_subplot(gs[0, 1])
    draw_factored_hmm(ax_factored, seq1=seq1, seq2=seq2, joint_seq=joint_seq)

    # Save
    plt.savefig('experiments/figure_generation/figure1/pdf/hmm_diagrams.pdf', dpi=300, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/png/hmm_diagrams.png', dpi=200, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/svg/hmm_diagrams.svg', facecolor='white',
               bbox_inches='tight', pad_inches=0.1)

    print("Saved: experiments/figure_generation/figure1/{pdf,png,svg}/hmm_diagrams.*")
    return fig


if __name__ == '__main__':
    create_figure()
