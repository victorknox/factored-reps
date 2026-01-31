#!/usr/bin/env python3
"""Generate 3D belief geometry plots for a two-factor mess3 process."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA

from fwh_core.generative_processes.builder import build_factored_process_from_spec
from fwh_core.generative_processes.generator import generate_data_batch_with_full_history

# ---------------------------------------------------------------------------
# CIELAB color utilities and 9-vertex interpolation
# ---------------------------------------------------------------------------

# D65 white point reference
_D65_WHITE = np.array([95.047, 100.0, 108.883])

# sRGB to XYZ matrix (D65)
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

# XYZ to sRGB matrix (inverse)
_XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB [0,1] to CIELAB.

    Args:
        rgb: RGB values, shape [..., 3], values in [0, 1]

    Returns:
        LAB values, shape [..., 3] with L in [0, 100], a/b in ~[-128, 128]
    """
    # Linearize sRGB (remove gamma)
    mask = rgb > 0.04045
    rgb_linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # RGB to XYZ (scale to 0-100 range)
    xyz = rgb_linear @ _RGB_TO_XYZ.T * 100.0

    # Normalize by D65 white point
    xyz_norm = xyz / _D65_WHITE

    # Apply f function for LAB conversion
    delta = 6.0 / 29.0
    delta_cubed = delta**3
    f = np.where(xyz_norm > delta_cubed, xyz_norm ** (1.0 / 3.0), xyz_norm / (3.0 * delta**2) + 4.0 / 29.0)

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    Convert CIELAB to RGB [0,1] with clamping.

    Args:
        lab: LAB values, shape [..., 3]

    Returns:
        RGB values, shape [..., 3], clamped to [0, 1]
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # LAB to XYZ
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def f_inv(t):
        return np.where(t > delta, t**3, 3.0 * delta**2 * (t - 4.0 / 29.0))

    X = f_inv(fx) * _D65_WHITE[0]
    Y = f_inv(fy) * _D65_WHITE[1]
    Z = f_inv(fz) * _D65_WHITE[2]

    xyz = np.stack([X, Y, Z], axis=-1) / 100.0

    # XYZ to linear RGB
    rgb_linear = xyz @ _XYZ_TO_RGB.T

    # Clamp negatives before gamma correction
    rgb_linear = np.clip(rgb_linear, 0.0, None)

    # Apply sRGB gamma
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * rgb_linear ** (1.0 / 2.4) - 0.055, 12.92 * rgb_linear)

    return np.clip(rgb, 0.0, 1.0)


# 9 anchor colors for joint states, organized as [factor1_state, factor2_state]
# Chosen to be maximally distinguishable and well-spread in CIELAB space
NINE_VERTEX_ANCHORS_RGB = np.array(
    [
        # Factor 1 = 0
        [
            [0.89, 0.10, 0.11],  # (0,0) Crimson red
            [0.17, 0.63, 0.17],  # (0,1) Forest green
            [0.12, 0.47, 0.71],  # (0,2) Steel blue
        ],
        # Factor 1 = 1
        [
            [0.99, 0.55, 0.00],  # (1,0) Vivid orange
            [0.00, 0.75, 0.75],  # (1,1) Teal/cyan
            [0.58, 0.40, 0.74],  # (1,2) Medium purple
        ],
        # Factor 1 = 2
        [
            [0.99, 0.91, 0.15],  # (2,0) Bright yellow
            [0.89, 0.47, 0.76],  # (2,1) Orchid pink
            [0.55, 0.22, 0.35],  # (2,2) Deep burgundy
        ],
    ]
)

# Pre-compute anchor colors in LAB space
_NINE_VERTEX_ANCHORS_LAB = rgb_to_lab(NINE_VERTEX_ANCHORS_RGB)


def nine_vertex_colors(b0: np.ndarray, b1: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Compute colors for joint belief states using 9-vertex interpolation.

    Blends 9 anchor colors (one per joint state) weighted by the joint
    probability distribution, with power weighting for emphasis.
    Blending is done in CIELAB space for perceptual uniformity.

    Args:
        b0: Belief states for factor 1, shape [N, 3]
        b1: Belief states for factor 2, shape [N, 3]
        alpha: Power for probability weighting (higher = more emphasis on dominant states)

    Returns:
        RGB colors, shape [N, 3], values in [0, 1]
    """
    # Compute joint probabilities: p[n, i, j] = b0[n, i] * b1[n, j]
    joint_probs = np.einsum("ni,nj->nij", b0, b1)  # Shape [N, 3, 3]

    # Apply power weighting to emphasize dominant states
    weighted_probs = joint_probs**alpha

    # Renormalize after power transform
    normalization = weighted_probs.sum(axis=(1, 2), keepdims=True)
    normalization = np.where(normalization > 0, normalization, 1.0)  # Avoid division by zero
    weighted_probs = weighted_probs / normalization

    # Blend anchor colors in LAB space
    # weighted_probs: [N, 3, 3], _NINE_VERTEX_ANCHORS_LAB: [3, 3, 3]
    blended_lab = np.einsum("nij,ijk->nk", weighted_probs, _NINE_VERTEX_ANCHORS_LAB)

    # Convert back to RGB
    return lab_to_rgb(blended_lab)


# Anchor colors for individual factors (3 vertices each)
# Factor 1 uses warm colors, Factor 2 uses cool colors
FACTOR1_ANCHORS_RGB = np.array([
    [0.89, 0.10, 0.11],  # State 0: Crimson red
    [0.99, 0.55, 0.00],  # State 1: Vivid orange
    [0.99, 0.91, 0.15],  # State 2: Bright yellow
])

FACTOR2_ANCHORS_RGB = np.array([
    [0.12, 0.47, 0.71],  # State 0: Steel blue
    [0.00, 0.75, 0.75],  # State 1: Teal/cyan
    [0.17, 0.63, 0.17],  # State 2: Forest green
])

_FACTOR1_ANCHORS_LAB = rgb_to_lab(FACTOR1_ANCHORS_RGB)
_FACTOR2_ANCHORS_LAB = rgb_to_lab(FACTOR2_ANCHORS_RGB)


def three_vertex_colors(beliefs: np.ndarray, anchors_lab: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Compute colors for belief states using 3-vertex interpolation.

    Args:
        beliefs: Belief states, shape [N, 3]
        anchors_lab: Anchor colors in LAB space, shape [3, 3]
        alpha: Power for probability weighting

    Returns:
        RGB colors, shape [N, 3], values in [0, 1]
    """
    weighted = beliefs ** alpha
    normalization = weighted.sum(axis=1, keepdims=True)
    normalization = np.where(normalization > 0, normalization, 1.0)
    weighted = weighted / normalization

    blended_lab = weighted @ anchors_lab
    return lab_to_rgb(blended_lab)


def draw_triangle_edges(ax, vertices: np.ndarray, color: str = "gray", alpha: float = 0.3, linewidth: float = 1.0):
    """Draw the edges of a triangle in 3D."""
    for i in range(3):
        j = (i + 1) % 3
        p1, p2 = vertices[i], vertices[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, alpha=alpha, linewidth=linewidth, zorder=1)


# ---------------------------------------------------------------------------
# Original constants and code
# ---------------------------------------------------------------------------

SEED = 7
BATCH_SIZE = 3000
SEQ_LEN = 64
MAX_POINTS = 3500
MAX_POINTS_FACTOR = 1500  # Fewer points for factor plot
PLANE_OFFSET = 0.45
ELEV = 30
AZIM = -60
BASE_POINT_SIZE = 6.0
BASE_POINT_SIZE_FACTOR = 10.0  # Larger points for factor plot
BASE_ALPHA = 0.55
BASE_ALPHA_JOINT = 0.55
FACTOR_LAYOUT = "hinged"  # "orthogonal" or "hinged"
SURFACE_ALPHA = 0.51
HINGE_ANGLE_DEG = 45.0
TILT_TOWARD_CAMERA_DEG = 30.0
BOOK_YAW_DEG = 20.0
DEPTH_BRIGHTNESS = 0.4

COLORS = {
    "factor1": "#C9A227",  # warm gold
    "factor2": "#4C78A8",  # muted blue
    "joint": "#D04E4E",    # warm red
}


def build_process():
    return build_factored_process_from_spec(
        structure_type="independent",
        spec=[
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
            {
                "component_type": "hmm",
                "variants": [{"process_name": "mess3", "process_params": {"x": 0.15, "a": 0.6}}],
            },
        ],
    )


def generate_beliefs(process, batch_size: int, seq_len: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    key = jax.random.PRNGKey(seed)

    initial_states = process.initial_states
    gen_states = tuple(jnp.broadcast_to(s, (batch_size, s.shape[0])) for s in initial_states)

    result = generate_data_batch_with_full_history(
        gen_states=gen_states,
        data_generator=process,
        batch_size=batch_size,
        sequence_len=seq_len,
        key=key,
    )

    belief_states = result["belief_states"]
    belief_states = tuple(np.array(bs) for bs in belief_states)

    b0 = belief_states[0].reshape(-1, belief_states[0].shape[-1])
    b1 = belief_states[1].reshape(-1, belief_states[1].shape[-1])
    return b0, b1


def subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def simplex_to_triangle(beliefs: np.ndarray) -> np.ndarray:
    """Map 3-simplex coordinates to an equilateral triangle in 2D."""
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3) / 2.0) * beliefs[:, 2]
    return np.stack([x, y], axis=1)


def simplex_vertices() -> np.ndarray:
    return simplex_to_triangle(np.eye(3))


def embed_factors_orthogonal_4d(
    b0: np.ndarray, b1: np.ndarray, projection_type: str = "random", seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA | None]:
    """
    Embed two 2-simplexes in truly orthogonal subspaces of 4D, then project to 3D.

    Factor 1's simplex lives in the (x, y, 0, 0) plane.
    Factor 2's simplex lives in the (0, 0, z, w) plane.
    These planes are orthogonal in 4D.

    Args:
        projection_type: "random" for random orthogonal, "custom" for hand-tuned, "pca" for PCA
        seed: random seed for random orthogonal projection

    Returns:
        coords0: 3D coordinates for factor 1 points
        coords1: 3D coordinates for factor 2 points
        verts0: 3D coordinates for factor 1 simplex vertices
        verts1: 3D coordinates for factor 2 simplex vertices
        pca: fitted PCA model (or None if not using PCA)
    """
    tri0 = simplex_to_triangle(b0)  # [N, 2]
    tri1 = simplex_to_triangle(b1)  # [N, 2]
    verts2d = simplex_vertices()    # [3, 2]

    # Embed in 4D: factor 1 in (x,y,0,0), factor 2 in (0,0,z,w)
    n0 = tri0.shape[0]
    n1 = tri1.shape[0]

    coords0_4d = np.column_stack([tri0, np.zeros((n0, 2))])  # [N, 4]
    coords1_4d = np.column_stack([np.zeros((n1, 2)), tri1])  # [N, 4]

    verts0_4d = np.column_stack([verts2d, np.zeros((3, 2))])  # [3, 4]
    verts1_4d = np.column_stack([np.zeros((3, 2)), verts2d])  # [3, 4]

    # Combine all points
    all_points_4d = np.vstack([coords0_4d, coords1_4d, verts0_4d, verts1_4d])

    if projection_type == "random":
        # Random orthogonal projection from 4D to 3D
        rng = np.random.default_rng(seed)
        random_matrix = rng.standard_normal((4, 4))
        Q, _ = np.linalg.qr(random_matrix)
        projection = Q[:, :3]  # Take first 3 columns for 4D -> 3D
        all_points_3d = all_points_4d @ projection
        pca = None
    elif projection_type == "custom":
        # Custom 4D->3D projection matrix designed to show both triangles at nice angles
        angle1 = np.deg2rad(35)
        angle2 = np.deg2rad(55)
        projection = np.array([
            [np.cos(angle1), 0, np.sin(angle2), 0],
            [0, np.cos(angle1), 0, np.cos(angle2)],
            [np.sin(angle1), np.sin(angle1), -np.cos(angle2) * 0.5, np.sin(angle2)],
        ]).T
        all_points_3d = all_points_4d @ projection
        pca = None
    elif projection_type == "hinged":
        # Hinged "book" projection - two triangles meeting at an angle
        hinge_angle = np.deg2rad(45)  # Angle between the two planes
        tilt = np.deg2rad(30)  # Overall tilt toward viewer

        c_h = np.cos(hinge_angle)
        s_h = np.sin(hinge_angle)
        c_t = np.cos(tilt)
        s_t = np.sin(tilt)

        projection = np.array([
            [c_h, 0, -c_h, 0],
            [0, c_t, 0, c_t],
            [s_h, s_t, s_h, s_t],
        ]).T
        all_points_3d = all_points_4d @ projection
        pca = None
    elif projection_type == "orthogonal_book":
        # True 90° orthogonal projection - two planes meeting at right angles
        # Factor 1 (x,y,0,0) → XY plane (Z=0)
        # Factor 2 (0,0,z,w) → YZ plane (X=0)
        # They share the Y axis, showing true orthogonality

        # Center the 2D triangles first so COMs coincide at origin
        centroid_2d = verts2d.mean(axis=0)
        tri0_centered = tri0 - centroid_2d
        tri1_centered = tri1 - centroid_2d
        verts_centered = verts2d - centroid_2d

        # Map to orthogonal planes
        # Factor 1: (x, y) → (x, y, 0)
        # Factor 2: (z, w) → (0, z, w)
        coords0 = np.column_stack([tri0_centered[:, 0], tri0_centered[:, 1], np.zeros(n0)])
        coords1 = np.column_stack([np.zeros(n1), tri1_centered[:, 0], tri1_centered[:, 1]])
        verts0 = np.column_stack([verts_centered[:, 0], verts_centered[:, 1], np.zeros(3)])
        verts1 = np.column_stack([np.zeros(3), verts_centered[:, 0], verts_centered[:, 1]])

        # Skip the normal projection path - we've already computed 3D coords
        combined_center = np.vstack([coords0, coords1]).mean(axis=0)
        coords0 = coords0 - combined_center
        coords1 = coords1 - combined_center
        verts0 = verts0 - combined_center
        verts1 = verts1 - combined_center
        return coords0, coords1, verts0, verts1, None
    else:  # pca
        pca = PCA(n_components=3)
        all_points_3d = pca.fit_transform(all_points_4d)

    # Split back
    coords0 = all_points_3d[:n0]
    coords1 = all_points_3d[n0:n0+n1]
    verts0 = all_points_3d[n0+n1:n0+n1+3]
    verts1 = all_points_3d[n0+n1+3:]

    # Center the combined point cloud
    combined_center = np.vstack([coords0, coords1]).mean(axis=0)
    coords0 = coords0 - combined_center
    coords1 = coords1 - combined_center
    verts0 = verts0 - combined_center
    verts1 = verts1 - combined_center

    return coords0, coords1, verts0, verts1, pca


def camera_basis(elev: float, azim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elev_r = np.deg2rad(elev)
    azim_r = np.deg2rad(azim)
    view_dir = np.array([np.cos(elev_r) * np.cos(azim_r), np.cos(elev_r) * np.sin(azim_r), np.sin(elev_r)])
    forward = -view_dir / np.linalg.norm(view_dir)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, forward)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    return forward, right, up


def rotate_about_axis(points: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    cross = np.cross(axis, points)
    dot = np.dot(points, axis)
    return points * cos_a + cross * sin_a + axis * dot[:, None] * (1.0 - cos_a)


def embed_factors(
    b0: np.ndarray, b1: np.ndarray, layout: str = FACTOR_LAYOUT
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tri0 = simplex_to_triangle(b0)
    tri1 = simplex_to_triangle(b1)
    verts2d = simplex_vertices()

    if layout == "orthogonal":
        coords0 = np.column_stack([tri0[:, 0], tri0[:, 1], np.zeros(tri0.shape[0])])
        coords1 = np.column_stack([tri1[:, 0], np.full(tri1.shape[0], PLANE_OFFSET), tri1[:, 1]])
        verts0 = np.column_stack([verts2d[:, 0], verts2d[:, 1], np.zeros(verts2d.shape[0])])
        verts1 = np.column_stack([verts2d[:, 0], np.full(verts2d.shape[0], PLANE_OFFSET), verts2d[:, 1]])
    elif layout == "hinged":
        forward, right, up = camera_basis(ELEV, AZIM)
        tip = verts2d[2]
        angle = np.deg2rad(HINGE_ANGLE_DEG)
        yaw = np.deg2rad(BOOK_YAW_DEG)
        tilt = np.deg2rad(TILT_TOWARD_CAMERA_DEG)

        tri0_local = tri0 - tip
        tri1_local = tri1 - tip
        verts_local = verts2d - tip

        coords0 = tri0_local[:, 0][:, None] * forward + tri0_local[:, 1][:, None] * up
        coords1 = tri1_local[:, 0][:, None] * forward + tri1_local[:, 1][:, None] * up
        verts0 = verts_local[:, 0][:, None] * forward + verts_local[:, 1][:, None] * up
        verts1 = verts0.copy()

        coords0 = rotate_about_axis(coords0, up, angle + yaw)
        coords1 = rotate_about_axis(coords1, up, -angle + yaw)
        verts0 = rotate_about_axis(verts0, up, angle + yaw)
        verts1 = rotate_about_axis(verts1, up, -angle + yaw)

        if tilt != 0.0:
            view_dir = -forward
            base_local = 0.5 * (verts_local[0] + verts_local[1])
            base_point = base_local[0] * forward + base_local[1] * up
            base_depth_pos = rotate_about_axis(base_point[None, :], right, tilt)[0] @ view_dir
            base_depth_neg = rotate_about_axis(base_point[None, :], right, -tilt)[0] @ view_dir
            if base_depth_neg < base_depth_pos:
                tilt = -tilt
            coords0 = rotate_about_axis(coords0, right, tilt)
            coords1 = rotate_about_axis(coords1, right, tilt)
            verts0 = rotate_about_axis(verts0, right, tilt)
            verts1 = rotate_about_axis(verts1, right, tilt)
    else:
        raise ValueError(f"Unknown factor layout: {layout}")

    if layout == "hinged":
        combined_center = np.vstack([coords0, coords1]).mean(axis=0)
        shift = -combined_center
        coords0 = coords0 + shift
        coords1 = coords1 + shift
        verts0 = verts0 + shift
        verts1 = verts1 + shift
    else:
        center0 = coords0.mean(axis=0)
        center1 = coords1.mean(axis=0)
        target = 0.5 * (center0 + center1)
        shift0 = target - center0
        shift1 = target - center1
        coords0 = coords0 + shift0
        coords1 = coords1 + shift1
        verts0 = verts0 + shift0
        verts1 = verts1 + shift1
    return coords0, coords1, verts0, verts1


def compute_joint_pca(b0: np.ndarray, b1: np.ndarray, seed: int) -> tuple[np.ndarray, PCA]:
    """Compute PCA of joint belief space, returning projected points and fitted PCA model."""
    joint = np.einsum("ni,nj->nij", b0, b1).reshape(-1, 9)
    pca = PCA(n_components=3, random_state=seed)
    projected = pca.fit_transform(joint)
    return projected, pca


def get_simplex_vertices_and_edges() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Get the 9 vertices of the 8-simplex (one-hot vectors) and edge list."""
    # 9 vertices: one-hot vectors for each joint state (i, j) flattened
    vertices = np.eye(9)  # Shape [9, 9]

    # Edges of the 8-simplex connect ALL pairs of vertices
    edges = []
    for i in range(9):
        for j in range(i + 1, 9):
            edges.append((i, j))
    return vertices, edges


def draw_simplex_edges(ax, pca: PCA, color: str = "gray", alpha: float = 0.4, linewidth: float = 0.8):
    """Project 8-simplex edges into PCA space and draw them."""
    vertices, edges = get_simplex_vertices_and_edges()
    projected_verts = pca.transform(vertices)  # Shape [9, 3]

    for i, j in edges:
        p1, p2 = projected_verts[i], projected_verts[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=color, alpha=alpha, linewidth=linewidth, zorder=1)


def ad_space_colors(b0: np.ndarray, b1: np.ndarray) -> np.ndarray:
    """Blend colors by the A (state 0 of factor 1) and D (state 0 of factor 2) marginals."""
    a = b0[:, 0]
    d = b1[:, 0]
    c00 = np.array([0.95, 0.95, 0.95])
    c10 = np.array(mcolors.to_rgb(COLORS["factor1"]))
    c01 = np.array(mcolors.to_rgb(COLORS["factor2"]))
    c11 = 0.5 * (c10 + c01)

    w00 = (1.0 - a) * (1.0 - d)
    w10 = a * (1.0 - d)
    w01 = (1.0 - a) * d
    w11 = a * d
    return w00[:, None] * c00 + w10[:, None] * c10 + w01[:, None] * c01 + w11[:, None] * c11


def view_depth(coords: np.ndarray, elev: float, azim: float) -> np.ndarray:
    elev_r = np.deg2rad(elev)
    azim_r = np.deg2rad(azim)
    view_dir = np.array([np.cos(elev_r) * np.cos(azim_r), np.cos(elev_r) * np.sin(azim_r), np.sin(elev_r)])
    depth = coords @ view_dir
    span = depth.max() - depth.min()
    if span == 0:
        return np.zeros_like(depth)
    return (depth - depth.min()) / span


def depth_scaled_sizes(base_size: float, depth: np.ndarray, min_scale: float = 0.6, max_scale: float = 1.8) -> np.ndarray:
    return base_size * (min_scale + (max_scale - min_scale) * depth)


def depth_scaled_colors(base_rgb: np.ndarray, depth: np.ndarray, base_alpha: float) -> np.ndarray:
    # Constant alpha, slight brightness scaling
    alpha = np.full(depth.shape[0], base_alpha)
    if base_rgb.ndim == 1:
        rgb = np.repeat(base_rgb[None, :], depth.shape[0], axis=0)
    else:
        rgb = base_rgb.copy()
    # Very subtle brightness scaling (0.9 to 1.0)
    brightness = 0.9 + 0.1 * depth
    rgb = rgb * brightness[:, None]
    return np.column_stack([rgb, alpha])


def add_simplex_surface(ax, vertices: np.ndarray, color: str, alpha: float):
    poly = Poly3DCollection([vertices], facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(poly)


def style_axes(ax, elev: float, azim: float):
    ax.view_init(elev=elev, azim=azim)
    if hasattr(ax, "set_proj_type"):
        ax.set_proj_type("persp")
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
        }
    )

    process = build_process()
    b0_full, b1_full = generate_beliefs(process, BATCH_SIZE, SEQ_LEN, SEED)

    # Subsample for joint plot
    idx_joint = subsample_indices(b0_full.shape[0], MAX_POINTS, SEED)
    b0_joint = b0_full[idx_joint]
    b1_joint = b1_full[idx_joint]

    # Subsample separately for factor plot (fewer points)
    idx_factor = subsample_indices(b0_full.shape[0], MAX_POINTS_FACTOR, SEED + 100)
    b0_factor = b0_full[idx_factor]
    b1_factor = b1_full[idx_factor]

    coords0, coords1, verts0, verts1, _ = embed_factors_orthogonal_4d(b0_factor, b1_factor, projection_type="orthogonal_book")
    joint_pca, pca_model = compute_joint_pca(b0_joint, b1_joint, SEED + 1)
    joint_base_colors = nine_vertex_colors(b0_joint, b1_joint, alpha=1.5)

    # Compute colors for each factor using 3-vertex interpolation
    factor1_base_colors = three_vertex_colors(b0_factor, _FACTOR1_ANCHORS_LAB, alpha=1.5)
    factor2_base_colors = three_vertex_colors(b1_factor, _FACTOR2_ANCHORS_LAB, alpha=1.5)

    factor_points = np.vstack([coords0, coords1])
    factor_base_colors = np.vstack([factor1_base_colors, factor2_base_colors])

    depth_factor = view_depth(factor_points, ELEV, AZIM)
    depth_joint = view_depth(joint_pca, ELEV, AZIM)

    sizes_factor = depth_scaled_sizes(BASE_POINT_SIZE_FACTOR, depth_factor)
    sizes_joint = depth_scaled_sizes(BASE_POINT_SIZE, depth_joint)

    colors_factor = depth_scaled_colors(factor_base_colors, depth_factor, BASE_ALPHA)
    colors_joint = depth_scaled_colors(joint_base_colors, depth_joint, BASE_ALPHA_JOINT)

    fig = plt.figure(figsize=(12.5, 8.5))
    gs = GridSpec(2, 2, height_ratios=[2.5, 1], hspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    ax3 = fig.add_subplot(gs[1, 0])  # 2D view of factor 1
    ax4 = fig.add_subplot(gs[1, 1])  # 2D view of factor 2

    # Compute axis bounds
    factor_bounds = np.vstack([coords0, coords1, verts0, verts1])
    mins = factor_bounds.min(axis=0)
    maxs = factor_bounds.max(axis=0)
    max_span = np.max(maxs - mins)
    half_span = 0.45 * max_span
    center = 0.5 * (mins + maxs)
    x_min, x_max = center[0] - half_span, center[0] + half_span
    y_min, y_max = center[1] - half_span, center[1] + half_span
    z_min, z_max = center[2] - half_span, center[2] + half_span

    ax1.scatter(
        factor_points[:, 0],
        factor_points[:, 1],
        factor_points[:, 2],
        s=sizes_factor,
        c=colors_factor,
        linewidths=0.0,
        depthshade=True,
    )

    style_axes(ax1, elev=ELEV, azim=AZIM)
    ax1.set_title("Factor Belief Simplexes\n(Orthogonal in 4D, Projected to 3D)", fontsize=14, fontweight="medium", pad=10)

    # Compute joint axis bounds
    lim = np.max(np.abs(joint_pca))
    jx_min, jx_max = -lim, lim
    jy_min, jy_max = -lim, lim
    jz_min, jz_max = -lim, lim

    ax2.scatter(
        joint_pca[:, 0],
        joint_pca[:, 1],
        joint_pca[:, 2],
        s=sizes_joint,
        c=colors_joint,
        linewidths=0.0,
        depthshade=True,
    )
    style_axes(ax2, elev=ELEV, azim=AZIM)
    ax2.set_title("Joint Belief Space\n(8-Simplex, PCA Projection)", fontsize=14, fontweight="medium", pad=10)

    # Set axis limits (bounds already computed above for shadows)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_zlim(z_min, z_max)

    ax2.set_xlim(jx_min, jx_max)
    ax2.set_ylim(jy_min, jy_max)
    ax2.set_zlim(jz_min, jz_max)

    # 2D head-on views of each simplex
    tri0_2d = simplex_to_triangle(b0_factor)
    tri1_2d = simplex_to_triangle(b1_factor)

    # Factor 1 head-on
    ax3.scatter(
        tri0_2d[:, 0],
        tri0_2d[:, 1],
        s=BASE_POINT_SIZE_FACTOR * 1.2,
        c=factor1_base_colors,
        alpha=BASE_ALPHA,
        linewidths=0.0,
    )
    ax3.set_aspect("equal")
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Factor 1 Belief Simplex\n(Head-on View)", fontsize=14, fontweight="medium", pad=10)
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # Factor 2 head-on
    ax4.scatter(
        tri1_2d[:, 0],
        tri1_2d[:, 1],
        s=BASE_POINT_SIZE_FACTOR * 1.2,
        c=factor2_base_colors,
        alpha=BASE_ALPHA,
        linewidths=0.0,
    )
    ax4.set_aspect("equal")
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title("Factor 2 Belief Simplex\n(Head-on View)", fontsize=14, fontweight="medium", pad=10)
    for spine in ax4.spines.values():
        spine.set_visible(False)

    output_base = Path(__file__).parent.parent
    png_path = output_base / "png" / "mess3_beliefs.png"
    pdf_path = output_base / "pdf" / "mess3_beliefs.pdf"
    svg_path = output_base / "svg" / "mess3_beliefs.svg"

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {svg_path}")


if __name__ == "__main__":
    main()
