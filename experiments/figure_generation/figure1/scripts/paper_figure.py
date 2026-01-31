"""
Figure 1 Combined: Complete figure for ICML paper.

Layout:
- Row 1: (a) Joint HMM, (b) Factored HMMs
- Row 2: (c) Segre Surface 3D, (d) Factor Square 2D
- Row 3: (e) Dependent, (f) Indecomposable, (g) Scaling
- Row 4: (h) Experimental - Factor vs Joint contrast
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Circle, PathPatch, Polygon
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA

# =============================================================================
# Style - Publication quality
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'svg.fonttype': 'none',
})

# Unified color palette
COLORS = {
    'hmm1': '#E2C16B',      # Warm gold (lighter) - factor 1 states
    'hmm2': '#A67BB3',      # Rich purple (lighter) - factor 2 states
    'hmm1_text': '#D4A84B', # Warm gold (darker) - factor 1 labels
    'hmm2_text': '#8B5A9C', # Rich purple (darker) - factor 2 labels
    'joint': '#7A98A3',     # Steel blue - joint states
    'edge': '#4A5568',      # Consistent edge color
    'text': '#2C3E50',      # Dark blue-gray
    'good': '#2E7D5A',      # Muted green - for "on surface"
    'bad': '#9B4D4D',       # Muted red - for "off surface"
}

# HMM parameters (shared)
HMM1_PARAMS = {'t_AA': 0.75, 't_AB': 0.25, 't_BA': 0.25, 't_BB': 0.75}
HMM2_PARAMS = {'t_AA': 0.35, 't_AB': 0.65, 't_BA': 0.65, 't_BB': 0.35}
JOINT_EMISSIONS = {(0, 0): 'A', (0, 1): 'B', (1, 0): 'C', (1, 1): 'D'}
BOLD_DIGITS = {'0': '𝟎', '1': '𝟏'}

# Tetrahedron vertices for 3D visualization
TETRA_VERTICES = np.array([
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
], dtype=float) * 0.65

VERTEX_LABELS = ['00', '01', '10', '11']
TETRA_EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]


# =============================================================================
# Sequence generation
# =============================================================================

def sample_binary_emissions(params, length, rng, start_state=None):
    """Sample a binary emission sequence from a 2-state HMM."""
    if start_state is None:
        state = int(rng.random() < 0.5)
    else:
        state = int(start_state)

    emissions = []
    for _ in range(length):
        if state == 0:
            state = 0 if rng.random() < params['t_AA'] else 1
        else:
            state = 0 if rng.random() < params['t_BA'] else 1
        emissions.append(state)
    return emissions


def format_sequence(seq):
    """Format a sequence as space-separated tokens."""
    return ' '.join(str(token) for token in seq)


def bold_digit(token):
    """Return a bold unicode digit for 0/1 tokens."""
    return BOLD_DIGITS.get(str(token), str(token))


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


# =============================================================================
# Arrow and shape drawing utilities
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
    from matplotlib.patches import ConnectionStyle
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
    """Custom bezier for precise loop shape."""
    offset = 35
    start_rad = np.radians(angle + offset)
    end_rad = np.radians(angle - offset)

    attach_gap = radius * attach_gap_mult
    start_r = radius + attach_gap
    end_r = radius + attach_gap
    p0 = np.array([center[0] + start_r * np.cos(start_rad),
                   center[1] + start_r * np.sin(start_rad)])
    p3 = np.array([center[0] + end_r * np.cos(end_rad),
                   center[1] + end_r * np.sin(end_rad)])

    angle_rad = np.radians(angle)
    outward = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    p1 = p0 + outward * loop_size + np.array([-outward[1], outward[0]]) * loop_size * 0.5
    p2 = p3 + outward * loop_size + np.array([outward[1], -outward[0]]) * loop_size * 0.5

    path = Path([p0, p1, p2, p3], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    patch = PathPatch(path, facecolor='none', edgecolor=color, lw=lw, zorder=5)
    ax.add_patch(patch)

    tangent = 3 * (p3 - p2)
    add_triangle_arrow(ax, p3, tangent, color, size=head_size, zorder=6)

    label_dist = radius + attach_gap + loop_size + radius * 0.4 + label_pad
    return (center[0] + label_dist * np.cos(angle_rad),
            center[1] + label_dist * np.sin(angle_rad))


def draw_state(ax, pos, radius, color):
    """Draw a state circle."""
    circle = Circle(pos, radius, facecolor=color, edgecolor='none', zorder=10)
    ax.add_patch(circle)


def add_label(ax, x, y, emission, prob, fontsize=10, color='#2C3E50', rotation=0,
              use_mathtext=True, emission_text=None, fontfamily=None):
    """Add label with bold emission and normal probability."""
    emission_str = emission_text if emission_text is not None else str(emission)
    if use_mathtext:
        label = rf'$\mathbf{{{emission_str}}}\,|\,\mathrm{{{int(prob*100)}\%}}$'
    else:
        label = f'{emission_str} | {int(prob*100)}%'
    txt = ax.text(x, y, label, fontsize=fontsize, ha='center', va='center',
                  color=color, zorder=15, rotation=rotation, fontfamily=fontfamily)
    txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white')])
    return txt


def draw_decode_table(ax, origin=(0.72, 0.40), size=0.22,
                      col_color=None, row_color=None):
    """Draw a small decoding table mapping (x1, x2) -> emission."""
    x0, y0 = origin
    w = h = size
    line_color = COLORS['edge']
    text_color = COLORS['text']
    col_color = col_color or text_color
    row_color = row_color or text_color

    ax.plot([x0, x0 + w, x0 + w, x0, x0], [y0, y0, y0 + h, y0 + h, y0],
            color=line_color, lw=0.9, zorder=4)
    ax.plot([x0 + w / 2, x0 + w / 2], [y0, y0 + h],
            color=line_color, lw=0.9, zorder=4)
    ax.plot([x0, x0 + w], [y0 + h / 2, y0 + h / 2],
            color=line_color, lw=0.9, zorder=4)

    ax.text(x0 + w * 0.25, y0 + h + 0.03, '0', fontsize=9, ha='center',
           va='bottom', color=col_color, fontweight='bold')
    ax.text(x0 + w * 0.75, y0 + h + 0.03, '1', fontsize=9, ha='center',
           va='bottom', color=col_color, fontweight='bold')
    ax.text(x0 - 0.03, y0 + h * 0.75, '0', fontsize=9, ha='right',
           va='center', color=row_color, fontweight='bold')
    ax.text(x0 - 0.03, y0 + h * 0.25, '1', fontsize=9, ha='right',
           va='center', color=row_color, fontweight='bold')

    cells = {(0, 0): 'A', (1, 0): 'C', (0, 1): 'B', (1, 1): 'D'}
    for x1 in (0, 1):
        for x2 in (0, 1):
            cx = x0 + (0.25 if x1 == 0 else 0.75) * w
            cy = y0 + (0.75 if x2 == 0 else 0.25) * h
            ax.text(cx, cy, cells[(x1, x2)], fontsize=10, ha='center',
                   va='center', color=text_color, fontweight='bold')


# =============================================================================
# HMM Diagrams
# =============================================================================

def draw_joint_hmm(ax, joint_seq=None):
    """Draw the joint 4-state HMM with emission|prob labels."""
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

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
    emissions = {'00': 'A', '01': 'B', '10': 'C', '11': 'D'}

    for label, pos in positions.items():
        draw_state(ax, pos, radius, COLORS['joint'])

    self_probs = {
        '00': HMM1_PARAMS['t_AA'] * HMM2_PARAMS['t_AA'],
        '01': HMM1_PARAMS['t_AA'] * HMM2_PARAMS['t_BB'],
        '10': HMM1_PARAMS['t_BB'] * HMM2_PARAMS['t_AA'],
        '11': HMM1_PARAMS['t_BB'] * HMM2_PARAMS['t_BB'],
    }
    loop_angles = {'00': 225, '10': 315, '01': 135, '11': 45}

    loop_label_pad_top = -radius * 0.12
    for label, pos in positions.items():
        label_pad = loop_label_pad_top if label in ('01', '11') else 0.0
        lx, ly = draw_self_loop_bezier(ax, pos, radius, loop_angles[label],
                                       color=edge_color, label_pad=label_pad)
        add_label(ax, lx, ly, emissions[label], self_probs[label],
                 fontsize=10, color=COLORS['text'])

    r = radius
    rad_transition = 0.18
    label_offset_h = r * 1.25
    label_offset_v = r * 1.5

    # Horizontal transitions (00 <-> 10, bottom row)
    p_00_10 = HMM1_PARAMS['t_AB'] * HMM2_PARAMS['t_AA']
    p_10_00 = HMM1_PARAMS['t_BA'] * HMM2_PARAMS['t_AA']

    y_bottom = positions['00'][1]
    draw_arrow(ax, (positions['00'][0] + r, y_bottom - r*0.4),
               (positions['10'][0] - r, y_bottom - r*0.4),
               edge_color, rad=rad_transition)
    draw_arrow(ax, (positions['10'][0] - r, y_bottom + r*0.4),
               (positions['00'][0] + r, y_bottom + r*0.4),
               edge_color, rad=rad_transition)

    mid_x = (positions['00'][0] + positions['10'][0]) / 2
    add_label(ax, mid_x, y_bottom - label_offset_h, 'C', p_00_10, fontsize=10)
    add_label(ax, mid_x, y_bottom + label_offset_h, 'A', p_10_00, fontsize=10)

    # Horizontal transitions (01 <-> 11, top row)
    p_01_11 = HMM1_PARAMS['t_AB'] * HMM2_PARAMS['t_BB']
    p_11_01 = HMM1_PARAMS['t_BA'] * HMM2_PARAMS['t_BB']

    y_top = positions['01'][1]
    draw_arrow(ax, (positions['01'][0] + r, y_top + r*0.4),
               (positions['11'][0] - r, y_top + r*0.4),
               edge_color, rad=-rad_transition)
    draw_arrow(ax, (positions['11'][0] - r, y_top - r*0.4),
               (positions['01'][0] + r, y_top - r*0.4),
               edge_color, rad=-rad_transition)

    mid_x = (positions['01'][0] + positions['11'][0]) / 2
    add_label(ax, mid_x, y_top + label_offset_h, 'D', p_01_11, fontsize=10)
    add_label(ax, mid_x, y_top - label_offset_h, 'B', p_11_01, fontsize=10)

    # Vertical transitions (00 <-> 01, left column)
    p_00_01 = HMM1_PARAMS['t_AA'] * HMM2_PARAMS['t_AB']
    p_01_00 = HMM1_PARAMS['t_AA'] * HMM2_PARAMS['t_BA']

    x_left = positions['00'][0]
    draw_arrow(ax, (x_left - r*0.4, positions['00'][1] + r),
               (x_left - r*0.4, positions['01'][1] - r),
               edge_color, rad=-rad_transition)
    draw_arrow(ax, (x_left + r*0.4, positions['01'][1] - r),
               (x_left + r*0.4, positions['00'][1] + r),
               edge_color, rad=-rad_transition)

    mid_y = (positions['00'][1] + positions['01'][1]) / 2
    add_label(ax, x_left - label_offset_v, mid_y, 'B', p_00_01, fontsize=10,
             rotation=90)
    add_label(ax, x_left + label_offset_v, mid_y, 'A', p_01_00, fontsize=10,
             rotation=90)

    # Vertical transitions (10 <-> 11, right column)
    p_10_11 = HMM1_PARAMS['t_BB'] * HMM2_PARAMS['t_AB']
    p_11_10 = HMM1_PARAMS['t_BB'] * HMM2_PARAMS['t_BA']

    x_right = positions['10'][0]
    draw_arrow(ax, (x_right + r*0.4, positions['10'][1] + r),
               (x_right + r*0.4, positions['11'][1] - r),
               edge_color, rad=rad_transition)
    draw_arrow(ax, (x_right - r*0.4, positions['11'][1] - r),
               (x_right - r*0.4, positions['10'][1] + r),
               edge_color, rad=rad_transition)

    add_label(ax, x_right + label_offset_v, mid_y, 'D', p_10_11, fontsize=10,
             rotation=90)
    add_label(ax, x_right - label_offset_v, mid_y, 'C', p_11_10, fontsize=10,
             rotation=90)

    ax.text(0.5, 1.15, 'Joint HMM', fontsize=13, fontweight='bold',
           ha='center', va='bottom', color=COLORS['text'])

    if joint_seq:
        draw_sequence(ax, joint_seq, y=-0.22, color=COLORS['text'],
                     fontsize=11, fontfamily='DejaVu Sans Mono', fontweight='bold')


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

    # HMM 1 (gold) - top
    hmm1_color = COLORS['hmm1']
    hmm1_label_color = COLORS['hmm1_text']
    hmm1_y = 0.75

    draw_state(ax, (hmm_x[0], hmm1_y), radius, hmm1_color)
    draw_state(ax, (hmm_x[1], hmm1_y), radius, hmm1_color)

    lx, ly = draw_self_loop_bezier(ax, (hmm_x[0], hmm1_y), radius, 135,
                                   color=edge_color, label_pad=loop_label_pad)
    add_label(ax, lx, ly, '0', HMM1_PARAMS['t_AA'], fontsize=11, color=hmm1_label_color,
              use_mathtext=False, emission_text=bold_digit('0'), fontfamily='STIXGeneral')

    lx, ly = draw_self_loop_bezier(ax, (hmm_x[1], hmm1_y), radius, 45,
                                   color=edge_color, label_pad=loop_label_pad)
    add_label(ax, lx, ly, '1', HMM1_PARAMS['t_BB'], fontsize=11, color=hmm1_label_color,
              use_mathtext=False, emission_text=bold_digit('1'), fontfamily='STIXGeneral')

    r = radius
    rad_pair = -0.14
    label_offset_factored_top = r * 1.35
    label_offset_factored_bottom = r * 0.4
    draw_arrow(ax, (hmm_x[0] + r, hmm1_y + r*0.5),
               (hmm_x[1] - r, hmm1_y + r*0.5),
               edge_color, rad=rad_pair)
    draw_arrow(ax, (hmm_x[1] - r, hmm1_y - r*0.5),
               (hmm_x[0] + r, hmm1_y - r*0.5),
               edge_color, rad=rad_pair)

    mid_x = (hmm_x[0] + hmm_x[1]) / 2
    add_label(ax, mid_x, hmm1_y + label_offset_factored_top, '1', HMM1_PARAMS['t_AB'],
              fontsize=11, color=hmm1_label_color, use_mathtext=False, emission_text=bold_digit('1'),
              fontfamily='STIXGeneral')
    add_label(ax, mid_x, hmm1_y - label_offset_factored_bottom, '0', HMM1_PARAMS['t_BA'],
              fontsize=11, color=hmm1_label_color, use_mathtext=False, emission_text=bold_digit('0'),
              fontfamily='STIXGeneral')

    # HMM 2 (purple) - bottom
    hmm2_color = COLORS['hmm2']
    hmm2_label_color = COLORS['hmm2_text']
    hmm2_y = 0.25

    draw_state(ax, (hmm_x[0], hmm2_y), radius, hmm2_color)
    draw_state(ax, (hmm_x[1], hmm2_y), radius, hmm2_color)

    lx, ly = draw_self_loop_bezier(ax, (hmm_x[0], hmm2_y), radius, 135,
                                   color=edge_color, label_pad=loop_label_pad)
    add_label(ax, lx, ly, '0', HMM2_PARAMS['t_AA'], fontsize=11, color=hmm2_label_color,
              use_mathtext=False, emission_text=bold_digit('0'), fontfamily='STIXGeneral')

    lx, ly = draw_self_loop_bezier(ax, (hmm_x[1], hmm2_y), radius, 45,
                                   color=edge_color, label_pad=loop_label_pad)
    add_label(ax, lx, ly, '1', HMM2_PARAMS['t_BB'], fontsize=11, color=hmm2_label_color,
              use_mathtext=False, emission_text=bold_digit('1'), fontfamily='STIXGeneral')

    draw_arrow(ax, (hmm_x[0] + r, hmm2_y + r*0.5),
               (hmm_x[1] - r, hmm2_y + r*0.5),
               edge_color, rad=rad_pair)
    draw_arrow(ax, (hmm_x[1] - r, hmm2_y - r*0.5),
               (hmm_x[0] + r, hmm2_y - r*0.5),
               edge_color, rad=rad_pair)

    mid_x = (hmm_x[0] + hmm_x[1]) / 2
    add_label(ax, mid_x, hmm2_y + label_offset_factored_top, '1', HMM2_PARAMS['t_AB'],
              fontsize=11, color=hmm2_label_color, use_mathtext=False, emission_text=bold_digit('1'),
              fontfamily='STIXGeneral')
    add_label(ax, mid_x, hmm2_y - label_offset_factored_bottom, '0', HMM2_PARAMS['t_BA'],
              fontsize=11, color=hmm2_label_color, use_mathtext=False, emission_text=bold_digit('0'),
              fontfamily='STIXGeneral')

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

    ax.text(0.5, 0.5, r'$\otimes$', fontsize=18, ha='center', va='center',
           color='#888888')

    draw_decode_table(ax, origin=(-0.14, -0.20), size=0.22,
                      col_color=hmm1_label_color, row_color=hmm2_label_color)

    ax.text(0.5, 1.1, 'Factored HMMs', fontsize=13, fontweight='bold',
           ha='center', va='bottom', color=COLORS['text'])


# =============================================================================
# Color mapping for geometry
# =============================================================================

def pos_to_color(a1, a2):
    """Map (α₁, α₂) to RGB using bilinear interpolation of 4 corner colors."""
    c00 = np.array([0.40, 0.65, 0.70])  # light teal
    c10 = np.array([0.85, 0.55, 0.50])  # warm coral
    c01 = np.array([0.55, 0.70, 0.85])  # sky blue
    c11 = np.array([0.95, 0.85, 0.60])  # light gold

    c_bottom = (1 - a1) * c00 + a1 * c10
    c_top = (1 - a1) * c01 + a1 * c11
    return tuple((1 - a2) * c_bottom + a2 * c_top)


def make_colored_square_image(res=100, rotate=-90, flip_vertical=True, extra_rotate=180):
    """Create image array for colored square background."""
    img = np.zeros((res, res, 3))
    for i in range(res):
        for j in range(res):
            a1 = j / (res - 1)
            a2 = 1 - i / (res - 1)
            img[i, j] = pos_to_color(a1, a2)

    if rotate == -90:
        img = np.rot90(img, k=-1)
    elif rotate == 90:
        img = np.rot90(img, k=1)
    elif rotate == 180:
        img = np.rot90(img, k=2)

    if flip_vertical:
        img = np.flipud(img)

    if extra_rotate == 90:
        img = np.rot90(img, k=1)
    elif extra_rotate == 180:
        img = np.rot90(img, k=2)
    elif extra_rotate == 270:
        img = np.rot90(img, k=3)

    return img


# =============================================================================
# HMM belief computation
# =============================================================================

class TwoStateHMM:
    def __init__(self, e_A, e_B, t_AA, t_BA):
        self.e_A = e_A
        self.e_B = e_B
        self.t_AA = t_AA
        self.t_BA = t_BA

    def steady_state(self):
        denom = self.t_BA + (1 - self.t_AA)
        if abs(denom) < 1e-10:
            return 0.5
        return self.t_BA / denom

    def update_belief(self, alpha, y):
        p_y_given_A = self.e_A if y == 0 else (1 - self.e_A)
        p_y_given_B = self.e_B if y == 0 else (1 - self.e_B)
        p_y = alpha * p_y_given_A + (1 - alpha) * p_y_given_B
        if p_y < 1e-10:
            return alpha
        p_A_was = (alpha * p_y_given_A) / p_y
        p_B_was = ((1 - alpha) * p_y_given_B) / p_y
        return p_A_was * self.t_AA + p_B_was * self.t_BA


def belief_to_simplex(a1, a2):
    return np.array([a1*a2, a1*(1-a2), (1-a1)*a2, (1-a1)*(1-a2)])


def simplex_to_3d(probs):
    return probs @ TETRA_VERTICES


def compute_beliefs_independent(hmm1, hmm2, max_len):
    beliefs = []
    ss1, ss2 = hmm1.steady_state(), hmm2.steady_state()

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

    recurse([], ss1, ss2)
    return beliefs


def compute_beliefs_coupled(hmm1, hmm2, max_len, coupling):
    """Coupled but still product-preserving - stays on surface."""
    beliefs = []
    ss1, ss2 = hmm1.steady_state(), hmm2.steady_state()

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
                mod_t_AA = hmm2.t_AA + coupling * (1 if y1 == 0 else -1) * 0.2
                mod_t_BA = hmm2.t_BA + coupling * (1 if y1 == 0 else -1) * 0.2
                mod_t_AA = np.clip(mod_t_AA, 0.05, 0.95)
                mod_t_BA = np.clip(mod_t_BA, 0.05, 0.95)
                mod_hmm2 = TwoStateHMM(hmm2.e_A, hmm2.e_B, mod_t_AA, mod_t_BA)
                new_a2 = mod_hmm2.update_belief(a2, y2)
                recurse(seq + [(y1,y2)], new_a1, new_a2)

    recurse([], ss1, ss2)
    return beliefs


def compute_beliefs_indecomposable(hmm1, hmm2, max_len, offset_strength=0.25):
    """Indecomposable - manually offset beliefs off surface."""
    beliefs = []
    ss1, ss2 = hmm1.steady_state(), hmm2.steady_state()
    np.random.seed(42)

    def recurse(seq, a1, a2):
        factored_simplex = belief_to_simplex(a1, a2)
        factored_pos = simplex_to_3d(factored_simplex)

        offset = offset_strength * np.random.uniform(-1, 1)
        true_pos = factored_pos.copy()
        true_pos[2] += offset

        beliefs.append({
            'len': len(seq), 'alpha1': a1, 'alpha2': a2,
            'pos3d': true_pos,
            'factored_pos3d': factored_pos
        })
        if len(seq) >= max_len:
            return
        for y1 in [0, 1]:
            for y2 in [0, 1]:
                recurse(seq + [(y1,y2)], hmm1.update_belief(a1, y1), hmm2.update_belief(a2, y2))

    recurse([], ss1, ss2)
    return beliefs


# =============================================================================
# Depth cueing for 3D visualization
# =============================================================================

def compute_camera_depth(points, elev, azim):
    """Compute depth of 3D points relative to camera position.

    Uses the matplotlib 3D camera model: the camera looks from a direction
    determined by elevation and azimuth angles towards the origin.

    Args:
        points: (N, 3) array of 3D points
        elev: elevation angle in degrees
        azim: azimuth angle in degrees

    Returns:
        depths: (N,) array of depth values (larger = further from camera)
    """
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)

    # Camera direction vector (pointing from camera towards scene)
    cam_dir = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])

    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Depth is projection onto camera direction
    depths = points @ cam_dir
    return depths


def depth_to_alpha(depths, min_alpha=0.25, max_alpha=1.0):
    """Map depths to alpha values (further from camera = more transparent).

    Args:
        depths: array of depth values
        min_alpha: alpha for furthest points
        max_alpha: alpha for closest points

    Returns:
        alphas: array of alpha values
    """
    if len(depths) <= 1:
        return np.array([max_alpha])

    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min < 1e-10:
        return np.ones(len(depths)) * max_alpha

    # Larger depth = closer to camera (higher projection onto cam direction)
    # So larger depth should get higher alpha
    normalized = (depths - d_min) / (d_max - d_min)
    return min_alpha + normalized * (max_alpha - min_alpha)


def depth_to_size(depths, min_size=40, max_size=120):
    """Map depths to point sizes (further from camera = smaller).

    Args:
        depths: array of depth values
        min_size: size for furthest points
        max_size: size for closest points

    Returns:
        sizes: array of point sizes
    """
    if len(depths) <= 1:
        return np.array([max_size])

    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min < 1e-10:
        return np.ones(len(depths)) * max_size

    # Larger depth = closer to camera, should be larger size
    normalized = (depths - d_min) / (d_max - d_min)
    return min_size + normalized * (max_size - min_size)


# =============================================================================
# Segre surface and 3D geometry
# =============================================================================

def segre_mesh(res=35):
    a1 = np.linspace(0, 1, res)
    a2 = np.linspace(0, 1, res)
    A1, A2 = np.meshgrid(a1, a2)
    X, Y, Z = np.zeros_like(A1), np.zeros_like(A1), np.zeros_like(A1)
    C = np.zeros((res, res, 4))

    for i in range(res):
        for j in range(res):
            pt = simplex_to_3d(belief_to_simplex(A1[i,j], A2[i,j]))
            X[i,j], Y[i,j], Z[i,j] = pt
            rgb = pos_to_color(A1[i,j], A2[i,j])
            C[i, j] = (*rgb, 0.7)

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
    for v, lbl in zip(TETRA_VERTICES, VERTEX_LABELS):
        ax.text(v[0]*1.22, v[1]*1.22, v[2]*1.22, lbl,
                fontsize=10, ha='center', va='center', color=COLORS['edge'], zorder=200)


def draw_segre_colored(ax, show_grid=True, show_shadow=True, surface_alpha=0.7, faded=False):
    """Draw Segre surface with position-matched colors."""
    X, Y, Z, C = segre_mesh(res=40)

    if show_shadow:
        z_floor = TETRA_VERTICES[:, 2].min() * 0.98
        shadow_alpha = 0.08 if faded else 0.15
        ax.plot_surface(X, Y, np.full_like(Z, z_floor),
                       alpha=shadow_alpha, color='#334155',
                       edgecolor='none', shade=False, zorder=1)

    if faded:
        C[:, :, 3] = 0.25
    else:
        C[:, :, 3] = surface_alpha

    ax.plot_surface(X, Y, Z, facecolors=C, edgecolor='none',
                   shade=True, zorder=2, rcount=40, ccount=40)

    if show_grid:
        grid_alpha = 0.15 if faded else 0.4
        for line in segre_grid(5):
            ax.plot3D(line[:,0], line[:,1], line[:,2],
                     color='#1E293B', alpha=grid_alpha, lw=0.5, zorder=3)

    boundary_alpha = 0.3 if faded else 0.7
    for b in segre_boundary():
        ax.plot3D(b[:,0], b[:,1], b[:,2],
                 color='#1E293B', alpha=boundary_alpha, lw=1.5, zorder=4)


def draw_factored_square(ax, beliefs, title="", faded=False):
    """Draw factored view as colored 2D square with corner labels."""
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, 1.12)
    ax.set_aspect('equal', adjustable='box')

    img = make_colored_square_image(100, rotate=-90, flip_vertical=True, extra_rotate=180)
    img_alpha = 0.3 if faded else 1.0
    ax.imshow(img, extent=[0, 1, 0, 1], origin='lower', aspect='auto', alpha=img_alpha, zorder=0)

    grid_alpha = 0.15 if faded else 0.3
    for t in [0.25, 0.5, 0.75]:
        ax.axhline(t, color='white', lw=0.5, alpha=grid_alpha)
        ax.axvline(t, color='white', lw=0.5, alpha=grid_alpha)

    border_alpha = 0.3 if faded else 0.8
    ax.plot([0,1,1,0,0], [0,0,1,1,0], color='#1E293B', lw=1.5, alpha=border_alpha)

    label_alpha = 0.4 if faded else 0.9
    ax.text(1.08, 0, '00', fontsize=11, ha='left', va='center', color=COLORS['edge'], alpha=label_alpha)
    ax.text(1.08, 1, '10', fontsize=11, ha='left', va='center', color=COLORS['edge'], alpha=label_alpha)
    ax.text(-0.08, 0, '01', fontsize=11, ha='right', va='center', color=COLORS['edge'], alpha=label_alpha)
    ax.text(-0.08, 1, '11', fontsize=11, ha='right', va='center', color=COLORS['edge'], alpha=label_alpha)

    ax.axis('off')

    point_alpha = 0.5 if faded else 1.0
    for b in beliefs:
        c = pos_to_color(b['alpha1'], b['alpha2'])
        s = 100 if b['len'] == 0 else 50
        ec = '#1a1a1a' if b['len'] == 0 else '#333333'
        ew = 2.5 if b['len'] == 0 else 1.0
        x_trans = 1 - b['alpha2']
        y_trans = b['alpha1']
        ax.scatter([x_trans], [y_trans], c=[c], s=s,
                  edgecolors=ec, linewidths=ew, zorder=10, alpha=point_alpha)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)


def draw_beliefs_3d_colored(ax, beliefs, show_projection_lines=False,
                             elev=20, azim=160, depth_cue=True,
                             min_alpha=0.25, max_alpha=1.0,
                             min_size=35, max_size=120):
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
    if show_projection_lines:
        for b in beliefs:
            if 'factored_pos3d' in b:
                p = b['pos3d']
                fp = b['factored_pos3d']
                dist = np.linalg.norm(p - fp)
                if dist > 0.01:
                    ax.plot3D([p[0], fp[0]], [p[1], fp[1]], [p[2], fp[2]],
                             color='#C44', alpha=0.7, lw=1.0, ls='-', zorder=50)

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

        # Base size depends on whether it's the root belief
        if b['len'] == 0:
            s = sizes[idx] * 1.2  # root point slightly larger
            ec = '#1a1a1a'
            ew = 2.5
        else:
            s = sizes[idx] * 0.6
            ec = '#333333'
            ew = 1.2

        point_alpha = alphas[idx]

        # Additional alpha reduction for indecomposable points below surface
        if 'factored_pos3d' in b:
            fp = b['factored_pos3d']
            if p[2] < fp[2]:
                point_alpha *= 0.5

        ax.scatter([p[0]], [p[1]], [p[2]], c=[c], s=s,
                  edgecolors=ec, linewidths=ew, zorder=zorders[idx], depthshade=False, alpha=point_alpha)


# =============================================================================
# Experimental: mess3 beliefs (simplified for this figure)
# =============================================================================

def simplex_to_triangle(beliefs):
    """Map 3-simplex coordinates to equilateral triangle in 2D."""
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3) / 2.0) * beliefs[:, 2]
    return np.stack([x, y], axis=1)


def simplex_vertices_2d():
    return simplex_to_triangle(np.eye(3))


def embed_factors_orthogonal(b0, b1):
    """Embed two 2-simplexes in orthogonal 3D subspaces."""
    tri0 = simplex_to_triangle(b0)
    tri1 = simplex_to_triangle(b1)
    verts2d = simplex_vertices_2d()

    n0, n1 = tri0.shape[0], tri1.shape[0]
    centroid_2d = verts2d.mean(axis=0)

    tri0_c = tri0 - centroid_2d
    tri1_c = tri1 - centroid_2d
    verts_c = verts2d - centroid_2d

    # Factor 1: XY plane, Factor 2: YZ plane
    coords0 = np.column_stack([tri0_c[:, 0], tri0_c[:, 1], np.zeros(n0)])
    coords1 = np.column_stack([np.zeros(n1), tri1_c[:, 0], tri1_c[:, 1]])
    verts0 = np.column_stack([verts_c[:, 0], verts_c[:, 1], np.zeros(3)])
    verts1 = np.column_stack([np.zeros(3), verts_c[:, 0], verts_c[:, 1]])

    combined_center = np.vstack([coords0, coords1]).mean(axis=0)
    coords0 -= combined_center
    coords1 -= combined_center
    verts0 -= combined_center
    verts1 -= combined_center

    return coords0, coords1, verts0, verts1


def compute_joint_pca(b0, b1, seed=42):
    """Compute PCA of joint belief space."""
    joint = np.einsum("ni,nj->nij", b0, b1).reshape(-1, 9)
    pca = PCA(n_components=3, random_state=seed)
    projected = pca.fit_transform(joint)
    return projected, pca


def generate_synthetic_beliefs(n_points=500, seed=42):
    """Generate synthetic 3-state beliefs for visualization."""
    rng = np.random.default_rng(seed)

    # Generate points that cluster near simplex vertices
    b0 = rng.dirichlet([0.3, 0.3, 0.3], size=n_points)
    b1 = rng.dirichlet([0.3, 0.3, 0.3], size=n_points)

    return b0, b1


def draw_factor_simplexes(ax, coords0, coords1, verts0, verts1, colors0, colors1):
    """Draw two factor simplexes in 3D."""
    ax.scatter(coords0[:, 0], coords0[:, 1], coords0[:, 2],
              c=colors0, s=8, alpha=0.6, linewidths=0)
    ax.scatter(coords1[:, 0], coords1[:, 1], coords1[:, 2],
              c=colors1, s=8, alpha=0.6, linewidths=0)

    # Draw simplex edges
    for i in range(3):
        j = (i + 1) % 3
        ax.plot3D([verts0[i, 0], verts0[j, 0]],
                 [verts0[i, 1], verts0[j, 1]],
                 [verts0[i, 2], verts0[j, 2]],
                 color=COLORS['hmm1_text'], alpha=0.6, lw=1.5)
        ax.plot3D([verts1[i, 0], verts1[j, 0]],
                 [verts1[i, 1], verts1[j, 1]],
                 [verts1[i, 2], verts1[j, 2]],
                 color=COLORS['hmm2_text'], alpha=0.6, lw=1.5)


def draw_joint_simplex(ax, projected, colors):
    """Draw joint 8-simplex (PCA projection)."""
    ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
              c=colors, s=8, alpha=0.6, linewidths=0)


def get_factor_colors(beliefs, anchors):
    """Get colors for factor beliefs based on anchor colors."""
    weighted = beliefs ** 1.5
    normalization = weighted.sum(axis=1, keepdims=True)
    normalization = np.where(normalization > 0, normalization, 1.0)
    weighted = weighted / normalization
    return weighted @ anchors


# Anchor colors for factors
FACTOR1_ANCHORS = np.array([
    [0.89, 0.10, 0.11],  # Red
    [0.99, 0.55, 0.00],  # Orange
    [0.99, 0.91, 0.15],  # Yellow
])

FACTOR2_ANCHORS = np.array([
    [0.12, 0.47, 0.71],  # Blue
    [0.00, 0.75, 0.75],  # Teal
    [0.17, 0.63, 0.17],  # Green
])


# =============================================================================
# Main Figure
# =============================================================================

def create_figure():
    """Create the combined publication-quality figure."""

    # Generate HMM sequences
    rng = np.random.default_rng(7)
    seq_len = 10
    seq1 = sample_binary_emissions(HMM1_PARAMS, seq_len, rng)
    seq2 = sample_binary_emissions(HMM2_PARAMS, seq_len, rng)
    joint_seq = [JOINT_EMISSIONS[(seq1[i], seq2[i])] for i in range(seq_len)]

    # Generate beliefs
    hmm1 = TwoStateHMM(e_A=0.9, e_B=0.1, t_AA=0.75, t_BA=0.25)
    hmm2 = TwoStateHMM(e_A=0.9, e_B=0.1, t_AA=0.35, t_BA=0.65)

    max_len = 2
    beliefs_ind = compute_beliefs_independent(hmm1, hmm2, max_len)
    beliefs_dep = compute_beliefs_coupled(hmm1, hmm2, max_len, coupling=-0.5)
    beliefs_decomp = compute_beliefs_indecomposable(hmm1, hmm2, max_len, offset_strength=0.5)

    # Generate synthetic experimental data
    b0_exp, b1_exp = generate_synthetic_beliefs(n_points=800, seed=42)
    coords0, coords1, verts0, verts1 = embed_factors_orthogonal(b0_exp, b1_exp)
    joint_pca, pca_model = compute_joint_pca(b0_exp, b1_exp, seed=42)

    colors0 = get_factor_colors(b0_exp, FACTOR1_ANCHORS)
    colors1 = get_factor_colors(b1_exp, FACTOR2_ANCHORS)

    # Joint colors from outer product
    joint_probs = np.einsum("ni,nj->nij", b0_exp, b1_exp)
    joint_colors = joint_probs.reshape(-1, 9) @ np.random.default_rng(42).random((9, 3))
    joint_colors = joint_colors / joint_colors.max(axis=0, keepdims=True)

    # Create figure
    fig = plt.figure(figsize=(13.5, 16))

    gs_main = GridSpec(4, 1, figure=fig, height_ratios=[1.2, 1.2, 0.9, 1.3],
                       hspace=0.30, left=0.02, right=0.98, top=0.97, bottom=0.04)

    view_elev, view_azim = 20, 160

    # =========================================================================
    # Row 0: HMM diagrams (a, b)
    # =========================================================================

    gs_hmm = gs_main[0].subgridspec(1, 2, wspace=0.02)
    ax_joint_hmm = fig.add_subplot(gs_hmm[0])
    ax_factored_hmm = fig.add_subplot(gs_hmm[1])

    draw_joint_hmm(ax_joint_hmm, joint_seq=joint_seq)
    ax_joint_hmm.text(-0.15, 1.2, '(a)', fontsize=14, fontweight='bold',
                     transform=ax_joint_hmm.transAxes, va='top')

    draw_factored_hmm(ax_factored_hmm, seq1=seq1, seq2=seq2, joint_seq=joint_seq)
    ax_factored_hmm.text(-0.05, 1.15, '(b)', fontsize=14, fontweight='bold',
                        transform=ax_factored_hmm.transAxes, va='top')

    # =========================================================================
    # Row 1: Geometry - Segre 3D + Square 2D (c, d)
    # =========================================================================

    gs_geom = gs_main[1].subgridspec(1, 2, wspace=0.05)
    ax_segre = fig.add_subplot(gs_geom[0], projection='3d')
    ax_square = fig.add_subplot(gs_geom[1])

    setup_3d_ax(ax_segre)
    ax_segre.view_init(elev=view_elev, azim=view_azim)
    draw_tetra(ax_segre)
    draw_segre_colored(ax_segre, show_shadow=True)
    draw_beliefs_3d_colored(ax_segre, beliefs_ind, elev=view_elev, azim=view_azim)
    ax_segre.set_title('Joint Belief Space\n(Segre surface)', fontsize=12, fontweight='bold', pad=5)
    ax_segre.text2D(-0.05, 0.95, '(c)', fontsize=14, fontweight='bold',
                   transform=ax_segre.transAxes, va='top')

    draw_factored_square(ax_square, beliefs_ind, title='Factored Belief Space\n(lossless)')
    ax_square.text(-0.1, 1.05, '(d)', fontsize=14, fontweight='bold',
                  transform=ax_square.transAxes, va='top')

    # =========================================================================
    # Row 2: Dependent + Indecomposable + Scaling (e, f, g)
    # =========================================================================

    gs_mid = gs_main[2].subgridspec(1, 5, width_ratios=[1, 0.7, 1, 0.7, 1.2], wspace=0.08)

    # Dependent: 3D + 2D
    ax_dep_3d = fig.add_subplot(gs_mid[0], projection='3d')
    ax_dep_2d = fig.add_subplot(gs_mid[1])

    setup_3d_ax(ax_dep_3d)
    ax_dep_3d.view_init(elev=view_elev, azim=view_azim)
    draw_tetra(ax_dep_3d, alpha=0.2)
    draw_segre_colored(ax_dep_3d, show_shadow=False, show_grid=False, surface_alpha=0.4)
    draw_beliefs_3d_colored(ax_dep_3d, beliefs_dep, elev=view_elev, azim=view_azim)
    ax_dep_3d.set_title('Dependent\n(on surface)', fontsize=10, fontweight='bold',
                       color=COLORS['good'], pad=0)
    ax_dep_3d.text2D(-0.1, 1.0, '(e)', fontsize=12, fontweight='bold',
                    transform=ax_dep_3d.transAxes, va='top')

    draw_factored_square(ax_dep_2d, beliefs_dep, title='')

    # Indecomposable: 3D + 2D
    ax_indec_3d = fig.add_subplot(gs_mid[2], projection='3d')
    ax_indec_2d = fig.add_subplot(gs_mid[3])

    setup_3d_ax(ax_indec_3d)
    ax_indec_3d.view_init(elev=view_elev, azim=view_azim)
    draw_tetra(ax_indec_3d, alpha=0.2)
    draw_segre_colored(ax_indec_3d, show_shadow=False, show_grid=False, surface_alpha=0.4, faded=False)
    draw_beliefs_3d_colored(ax_indec_3d, beliefs_decomp, show_projection_lines=True,
                             elev=view_elev, azim=view_azim)
    ax_indec_3d.set_title('Indecomposable\n(OFF surface)', fontsize=10, fontweight='bold',
                         color=COLORS['bad'], pad=0)
    ax_indec_3d.text2D(-0.1, 1.0, '(f)', fontsize=12, fontweight='bold',
                      transform=ax_indec_3d.transAxes, va='top')

    draw_factored_square(ax_indec_2d, beliefs_decomp, title='', faded=True)

    # Scaling plot
    ax_scaling = fig.add_subplot(gs_mid[4])

    N = np.arange(1, 13)
    dim_joint = 2**N - 1
    dim_factored = N

    ax_scaling.semilogy(N, dim_joint, 'o-', color=COLORS['bad'], lw=2.5, ms=6,
                       label=r'Joint: $2^N - 1$')
    ax_scaling.semilogy(N, dim_factored, 's-', color=COLORS['good'], lw=2.5, ms=6,
                       label=r'Factored: $N$')

    ax_scaling.fill_between(N, dim_factored, dim_joint, alpha=0.12, color=COLORS['bad'])
    ax_scaling.text(8.5, 35, 'exponential\nsavings', ha='center', fontsize=9,
                   color=COLORS['text'], style='italic')

    ax_scaling.set_title('Dimensional Scaling', fontsize=10, fontweight='bold', pad=8)
    ax_scaling.set_xlabel('N (binary factors)', fontsize=9)
    ax_scaling.set_ylabel('Dimensions', fontsize=9)
    ax_scaling.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.9,
                     edgecolor='#DDD', fancybox=True)
    ax_scaling.set_xlim(0.5, 12.5)
    ax_scaling.set_ylim(0.5, 6000)
    ax_scaling.spines['top'].set_visible(False)
    ax_scaling.spines['right'].set_visible(False)
    ax_scaling.tick_params(labelsize=8)
    ax_scaling.text(-0.15, 1.05, '(g)', fontsize=12, fontweight='bold',
                   transform=ax_scaling.transAxes, va='top')

    # =========================================================================
    # Row 3: Experimental (h) - Factor vs Joint contrast
    # =========================================================================

    gs_exp = gs_main[3].subgridspec(1, 2, wspace=0.08)
    ax_factor_exp = fig.add_subplot(gs_exp[0], projection='3d')
    ax_joint_exp = fig.add_subplot(gs_exp[1], projection='3d')

    # Factor simplexes (compact, orthogonal)
    setup_3d_ax(ax_factor_exp)
    ax_factor_exp.view_init(elev=25, azim=-50)
    draw_factor_simplexes(ax_factor_exp, coords0, coords1, verts0, verts1, colors0, colors1)
    ax_factor_exp.set_title('Factor Representations\n(compact, orthogonal)', fontsize=12, fontweight='bold',
                           color=COLORS['good'], pad=10)
    ax_factor_exp.text2D(-0.02, 0.95, '(h)', fontsize=14, fontweight='bold',
                        transform=ax_factor_exp.transAxes, va='top')

    # Joint simplex (sprawling)
    setup_3d_ax(ax_joint_exp)
    ax_joint_exp.view_init(elev=25, azim=-50)
    draw_joint_simplex(ax_joint_exp, joint_pca, joint_colors)
    ax_joint_exp.set_title('Joint Representation\n(sprawling, high-dimensional)', fontsize=12, fontweight='bold',
                          color=COLORS['bad'], pad=10)

    # Save
    plt.savefig('experiments/figure_generation/figure1/pdf/paper_figure.pdf', dpi=300, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/png/paper_figure.png', dpi=200, facecolor='white',
               bbox_inches='tight', pad_inches=0.1)
    plt.savefig('experiments/figure_generation/figure1/svg/paper_figure.svg', facecolor='white',
               bbox_inches='tight', pad_inches=0.1)

    print("Saved: experiments/figure_generation/figure1/{pdf,png,svg}/paper_figure.*")
    return fig


if __name__ == '__main__':
    create_figure()
