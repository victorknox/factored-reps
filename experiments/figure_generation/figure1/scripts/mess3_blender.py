"""
Mess3 belief geometry plots in Blender style.

Renders:
- Factor belief simplexes (orthogonal in 4D, projected to 3D)
- Joint belief space (8-simplex, PCA projection)

Run with: /Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/mess3_blender/mess3_3d_blender.py
"""

import argparse
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

# Add the project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# =============================================================================
# Settings
# =============================================================================

# Low res for testing
PREVIEW_RESOLUTION = (400, 400)
CYCLES_SAMPLES = 8

# Data generation
SEED = 7
BATCH_SIZE = 3000
SEQ_LEN = 64
MAX_POINTS_FACTOR = 1500
MAX_POINTS_JOINT = 3500

# =============================================================================
# Color anchors (matching the original)
# =============================================================================

# Factor 1: warm colors (red → orange → yellow)
FACTOR1_ANCHORS = np.array([
    [0.89, 0.10, 0.11],  # State 0: Crimson red
    [0.99, 0.55, 0.00],  # State 1: Vivid orange
    [0.99, 0.91, 0.15],  # State 2: Bright yellow
])

# Factor 2: cool colors (blue → teal → green)
FACTOR2_ANCHORS = np.array([
    [0.12, 0.47, 0.71],  # State 0: Steel blue
    [0.00, 0.75, 0.75],  # State 1: Teal/cyan
    [0.17, 0.63, 0.17],  # State 2: Forest green
])

# 9 anchor colors for joint states [factor1_state, factor2_state]
NINE_VERTEX_ANCHORS = np.array([
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
])


def three_vertex_colors(beliefs: np.ndarray, anchors: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """Compute colors using 3-vertex interpolation with power weighting."""
    weighted = beliefs ** alpha
    normalization = weighted.sum(axis=1, keepdims=True)
    normalization = np.where(normalization > 0, normalization, 1.0)
    weighted = weighted / normalization
    return weighted @ anchors


def nine_vertex_colors(b0: np.ndarray, b1: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """Compute colors for joint belief states using 9-vertex interpolation."""
    joint_probs = np.einsum("ni,nj->nij", b0, b1)
    weighted_probs = joint_probs ** alpha
    normalization = weighted_probs.sum(axis=(1, 2), keepdims=True)
    normalization = np.where(normalization > 0, normalization, 1.0)
    weighted_probs = weighted_probs / normalization
    return np.einsum("nij,ijk->nk", weighted_probs, NINE_VERTEX_ANCHORS)


# =============================================================================
# Geometry
# =============================================================================

def simplex_to_triangle(beliefs: np.ndarray) -> np.ndarray:
    """Map 3-simplex coordinates to an equilateral triangle in 2D."""
    x = beliefs[:, 1] + 0.5 * beliefs[:, 2]
    y = (np.sqrt(3) / 2.0) * beliefs[:, 2]
    return np.stack([x, y], axis=1)


def simplex_vertices_2d() -> np.ndarray:
    """Get 2D coordinates of the simplex vertices."""
    return simplex_to_triangle(np.eye(3))


def embed_factors_orthogonal(b0: np.ndarray, b1: np.ndarray):
    """
    Embed two 2-simplexes in orthogonal subspaces of 4D, then project to 3D.
    Uses "orthogonal_book" layout: planes meeting at right angles.
    """
    tri0 = simplex_to_triangle(b0)
    tri1 = simplex_to_triangle(b1)
    verts2d = simplex_vertices_2d()

    # Center the triangles
    centroid_2d = verts2d.mean(axis=0)
    tri0_centered = tri0 - centroid_2d
    tri1_centered = tri1 - centroid_2d
    verts_centered = verts2d - centroid_2d

    n0 = tri0.shape[0]
    n1 = tri1.shape[0]

    # Map to orthogonal planes
    # Factor 1: (x, y) → (x, y, 0)
    # Factor 2: (z, w) → (0, z, w)
    coords0 = np.column_stack([tri0_centered[:, 0], tri0_centered[:, 1], np.zeros(n0)])
    coords1 = np.column_stack([np.zeros(n1), tri1_centered[:, 0], tri1_centered[:, 1]])
    verts0 = np.column_stack([verts_centered[:, 0], verts_centered[:, 1], np.zeros(3)])
    verts1 = np.column_stack([np.zeros(3), verts_centered[:, 0], verts_centered[:, 1]])

    # Center combined
    combined_center = np.vstack([coords0, coords1]).mean(axis=0)
    coords0 = coords0 - combined_center
    coords1 = coords1 - combined_center
    verts0 = verts0 - combined_center
    verts1 = verts1 - combined_center

    return coords0, coords1, verts0, verts1


def compute_joint_pca(b0: np.ndarray, b1: np.ndarray, seed: int):
    """Compute PCA of joint belief space using numpy (no sklearn dependency)."""
    joint = np.einsum("ni,nj->nij", b0, b1).reshape(-1, 9)

    # Center the data
    mean = joint.mean(axis=0)
    centered = joint - mean

    # Compute covariance matrix
    cov = np.cov(centered, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project onto top 3 components
    projection_matrix = eigenvectors[:, :3]
    projected = centered @ projection_matrix

    return projected, projection_matrix


# =============================================================================
# Data generation
# =============================================================================

def generate_beliefs():
    """Load pre-generated mess3 beliefs from npz file."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mess3_beliefs.npz')

    if os.path.exists(data_path):
        print(f"Loading pre-generated beliefs from {data_path}")
        data = np.load(data_path)
        return data['b0'], data['b1']
    else:
        print(f"Data file not found: {data_path}")
        print("Please run: uv run python experiments/mess3_blender/generate_mess3_data.py")
        print("Using synthetic data as fallback...")
        return generate_synthetic_beliefs()


def generate_synthetic_beliefs():
    """Generate synthetic belief data (fallback if real data not available)."""
    rng = np.random.default_rng(SEED)
    n_points = BATCH_SIZE * SEQ_LEN

    alpha = 0.3
    b0 = rng.dirichlet([alpha, alpha, alpha], size=n_points)
    b1 = rng.dirichlet([alpha, alpha, alpha], size=n_points)

    return b0, b1


def subsample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


# =============================================================================
# Blender helpers
# =============================================================================

def clear_scene():
    """Clear all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def create_ground_plane(z_pos=-0.5, size=10.0):
    """Create a simple ground plane for shadows."""
    # Light gray matte material
    mat = bpy.data.materials.new(name="GroundMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.97, 0.97, 0.97, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.95

    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, z_pos))
    floor = bpy.context.active_object
    floor.name = "Ground"
    floor.data.materials.append(mat)


def create_sphere(center, radius=0.02, color=(0.5, 0.5, 0.8, 1.0)):
    """Create a sphere with matte material (not glowing emission)."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=center,
        segments=8,
        ring_count=6
    )
    obj = bpy.context.active_object

    mat = bpy.data.materials.new(name="SphereMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Use principled BSDF for more natural matte look
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs['Roughness'].default_value = 0.8  # Matte finish
    bsdf.inputs['Specular IOR Level'].default_value = 0.0  # No specular

    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    obj.data.materials.append(mat)
    return obj


def create_tube(start, end, radius=0.003, color=(0.5, 0.5, 0.5, 1.0)):
    """Create a tube between two points."""
    start = np.array(start)
    end = np.array(end)

    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return None

    mid = (start + end) / 2

    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=mid)
    obj = bpy.context.active_object

    direction_normalized = direction / length
    up = Vector((0, 0, 1))

    if abs(direction_normalized[2]) > 0.999:
        if direction_normalized[2] < 0:
            obj.rotation_euler = (np.pi, 0, 0)
    else:
        rot_axis = up.cross(Vector(direction_normalized))
        rot_axis.normalize()
        rot_angle = np.arccos(np.clip(up.dot(Vector(direction_normalized)), -1, 1))
        obj.rotation_mode = 'AXIS_ANGLE'
        obj.rotation_axis_angle = (rot_angle, rot_axis.x, rot_axis.y, rot_axis.z)

    mat = bpy.data.materials.new(name="TubeMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.5
    obj.data.materials.append(mat)

    return obj


def create_triangle_edges(vertices, radius=0.004, color=(0.3, 0.3, 0.3, 1.0)):
    """Create triangle edges."""
    for i in range(3):
        j = (i + 1) % 3
        create_tube(vertices[i], vertices[j], radius=radius, color=color)


def setup_scene(elev=30, azim=-60, resolution=PREVIEW_RESOLUTION, ortho_scale=1.8):
    """Setup scene with camera and lights."""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = CYCLES_SAMPLES

    # World background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = None
    for node in world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            bg_node = node
            break
    if bg_node:
        bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

    # Lights
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (np.radians(30), 0, 0)

    bpy.ops.object.light_add(type='SUN', location=(3, 3, 2))
    fill = bpy.context.active_object
    fill.data.energy = 1.0
    fill.rotation_euler = (np.radians(60), np.radians(45), 0)

    # Camera
    r = 3.5
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    cam_x = r * np.cos(elev_rad) * np.cos(azim_rad)
    cam_y = r * np.cos(elev_rad) * np.sin(azim_rad)
    cam_z = r * np.sin(elev_rad)

    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = ortho_scale

    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = camera

    # Render settings
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.view_settings.view_transform = 'Standard'

    return camera


def render_factor_simplexes(b0, b1, output_path):
    """Render the factor belief simplexes (orthogonal in 4D, projected to 3D)."""
    clear_scene()

    # Subsample
    idx = subsample_indices(b0.shape[0], MAX_POINTS_FACTOR, SEED + 100)
    b0_sub = b0[idx]
    b1_sub = b1[idx]

    # Embed in 3D
    coords0, coords1, verts0, verts1 = embed_factors_orthogonal(b0_sub, b1_sub)

    # Compute colors (desaturate slightly for less garish look)
    colors0_raw = three_vertex_colors(b0_sub, FACTOR1_ANCHORS, alpha=1.5)
    colors1_raw = three_vertex_colors(b1_sub, FACTOR2_ANCHORS, alpha=1.5)
    # Desaturate: blend toward gray
    desaturate = 0.15
    colors0 = colors0_raw * (1 - desaturate) + 0.5 * desaturate
    colors1 = colors1_raw * (1 - desaturate) + 0.5 * desaturate

    # Scale for Blender
    scale = 0.8
    coords0 = coords0 * scale
    coords1 = coords1 * scale
    verts0 = verts0 * scale
    verts1 = verts1 * scale

    setup_scene(elev=30, azim=-60, ortho_scale=1.8)

    # Add ground plane for shadows
    create_ground_plane(z_pos=-0.55, size=8.0)

    # Create triangle edges
    create_triangle_edges(verts0, radius=0.004, color=(0.4, 0.3, 0.2, 1.0))  # Warm color for factor 1
    create_triangle_edges(verts1, radius=0.004, color=(0.2, 0.3, 0.4, 1.0))  # Cool color for factor 2

    # Create spheres for factor 1 - smaller, more of them
    for i, (pos, color) in enumerate(zip(coords0, colors0)):
        if i % 2 == 0:  # Show every 2nd point
            create_sphere(pos, radius=0.006, color=(color[0], color[1], color[2], 1.0))

    # Create spheres for factor 2
    for i, (pos, color) in enumerate(zip(coords1, colors1)):
        if i % 2 == 0:
            create_sphere(pos, radius=0.006, color=(color[0], color[1], color[2], 1.0))

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


def render_joint_belief_space(b0, b1, output_path):
    """Render the joint belief space (8-simplex, PCA projection)."""
    clear_scene()

    # Subsample - use more points
    idx = subsample_indices(b0.shape[0], MAX_POINTS_JOINT, SEED)
    b0_sub = b0[idx]
    b1_sub = b1[idx]

    # Compute PCA projection
    joint_coords, pca = compute_joint_pca(b0_sub, b1_sub, SEED + 1)

    # Compute colors (desaturate slightly)
    colors_raw = nine_vertex_colors(b0_sub, b1_sub, alpha=1.5)
    desaturate = 0.15
    colors = colors_raw * (1 - desaturate) + 0.5 * desaturate

    # Scale for Blender - normalize to reasonable range
    coord_range = np.abs(joint_coords).max()
    scale = 0.6 / coord_range if coord_range > 0 else 1.0
    joint_coords = joint_coords * scale

    setup_scene(elev=30, azim=-60, ortho_scale=1.8)

    # Add ground plane for shadows
    create_ground_plane(z_pos=-0.55, size=8.0)

    # Create spheres - visible size, show more points
    for i, (pos, color) in enumerate(zip(joint_coords, colors)):
        if i % 2 == 0:  # Show every 2nd point
            create_sphere(pos, radius=0.008, color=(color[0], color[1], color[2], 1.0))

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command-line arguments (passed after '--' when running via Blender)."""
    parser = argparse.ArgumentParser(description='Render mess3 belief geometry')
    parser.add_argument('--output-dir', type=str, default='png',
                        help='Output subdirectory name (default: png)')
    # Blender passes args after '--'
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return parser.parse_args(argv)


def main():
    args = parse_args()

    # Output directory: base_dir / output_dir_name
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Generating beliefs...")
    b0, b1 = generate_beliefs()
    print(f"Generated {b0.shape[0]} belief states")

    print("\n=== Rendering Factor Simplexes ===")
    render_factor_simplexes(b0, b1, os.path.join(output_dir, 'mess3_factor_simplexes.png'))

    print("\n=== Rendering Joint Belief Space ===")
    render_joint_belief_space(b0, b1, os.path.join(output_dir, 'mess3_joint_beliefs.png'))

    print(f"\n=== All renders saved to {output_dir} ===")


if __name__ == '__main__':
    main()
