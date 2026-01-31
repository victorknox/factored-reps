"""
Figure: Freeze-and-Vary Visualization

Demonstrates the key insight of the factored world hypothesis:
- Freeze one factor at a value v
- Vary the other factor from 0 to 1
- In 2D (factored space): all trajectories are PARALLEL lines
- In 3D (joint space): trajectories have DIFFERENT tangent directions

This difference explains why factored representations are more efficient.

Run with: /Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/freeze_vary.py
"""

import argparse
import json
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector

# =============================================================================
# Geometry
# =============================================================================

TETRA_VERTICES = np.array([
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
], dtype=float) * 0.65

TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def belief_to_simplex(a1, a2):
    """Convert factor beliefs (a1, a2) to 4-simplex coordinates."""
    return np.array([a1 * a2, a1 * (1 - a2), (1 - a1) * a2, (1 - a1) * (1 - a2)])


def simplex_to_3d(probs):
    """Convert 4-simplex to 3D tetrahedron coordinates."""
    return probs @ TETRA_VERTICES


def factored_to_3d(a1, a2):
    """Convert factored coordinates directly to 3D."""
    return simplex_to_3d(belief_to_simplex(a1, a2))


# =============================================================================
# Render Settings
# =============================================================================

RENDER_RESOLUTION = (1200, 1200)  # Square aspect ratio for all figures
RENDER_SAMPLES = 64  # Cycles samples for high quality

# =============================================================================
# Color Configuration
# =============================================================================

# Default colors (used as fallback if config file not found)
DEFAULT_COLORS = {
    # Surface corner colors (for pos_to_color bilinear interpolation)
    'surface_c00': (0.25, 0.70, 0.75),  # Teal (a1=0, a2=0)
    'surface_c10': (0.95, 0.45, 0.35),  # Coral (a1=1, a2=0)
    'surface_c01': (0.40, 0.65, 0.95),  # Blue (a1=0, a2=1)
    'surface_c11': (1.00, 0.85, 0.40),  # Gold (a1=1, a2=1)

    # Trajectory colors (colorblind-friendly)
    'trajectory': [
        (0.85, 0.37, 0.31),  # Red
        (0.30, 0.69, 0.29),  # Green
        (0.22, 0.47, 0.79),  # Blue
        (0.89, 0.60, 0.20),  # Orange
        (0.58, 0.36, 0.68),  # Purple
    ],

    # Structural elements
    'tetra_edges': (0.75, 0.75, 0.75),      # Light gray
    'grid_lines': (0.5, 0.5, 0.5),          # Medium gray
    'grid_lines_2d': (0.7, 0.7, 0.7),       # Lighter gray for 2D
    'boundary': (0.15, 0.15, 0.15),         # Dark gray
    'outline': (0.25, 0.25, 0.25),          # Freestyle outlines
    'axis': (0.6, 0.6, 0.6),                # Axis lines
    'origin_marker': (0.3, 0.3, 0.3),       # Origin point
}

# Active colors (can be overridden by load_colors())
COLORS = DEFAULT_COLORS.copy()

# Legacy alias for backward compatibility
TRAJECTORY_COLORS = COLORS['trajectory']


def load_colors(palette_name='default'):
    """Load colors from config file, falling back to hardcoded defaults.

    Args:
        palette_name: Name of the color palette to load from configs/colors.json

    Returns:
        dict: Color configuration dictionary
    """
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'configs', 'colors.json'
    )

    if os.path.exists(config_path):
        with open(config_path) as f:
            all_palettes = json.load(f)
        if palette_name in all_palettes:
            # Convert lists to tuples for consistency
            palette = all_palettes[palette_name]
            for key, value in palette.items():
                if isinstance(value, list):
                    if key == 'trajectory':
                        palette[key] = [tuple(c) for c in value]
                    else:
                        palette[key] = tuple(value)
            print(f"Loaded color palette: {palette_name}")
            return palette
        else:
            print(f"Warning: Palette '{palette_name}' not found, using default")

    return DEFAULT_COLORS.copy()


def pos_to_color(a1, a2):
    """Map (a1, a2) to RGB using bilinear interpolation of 4 corner colors."""
    c00 = np.array(COLORS['surface_c00'])
    c10 = np.array(COLORS['surface_c10'])
    c01 = np.array(COLORS['surface_c01'])
    c11 = np.array(COLORS['surface_c11'])

    c_bottom = (1 - a1) * c00 + a1 * c10
    c_top = (1 - a1) * c01 + a1 * c11
    return (1 - a2) * c_bottom + a2 * c_top


# =============================================================================
# Blender Helpers
# =============================================================================

def clear_scene():
    """Clear all objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear orphan data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def create_tube(start, end, radius=0.004, color=(0.8, 0.3, 0.3, 1.0), collection=None, use_emission=False):
    """Create a tube (cylinder) between two points."""
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
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    if use_emission:
        nodes.clear()
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs[0].default_value = color
        emission.inputs[1].default_value = 1.0
        output = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(emission.outputs[0], output.inputs[0])
    else:
        bsdf = nodes["Principled BSDF"]
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = 0.5

    obj.data.materials.append(mat)

    if collection is not None:
        bpy.context.collection.objects.unlink(obj)
        collection.objects.link(obj)

    return obj


def create_sphere(center, radius=0.035, color=(0.5, 0.5, 0.8, 1.0)):
    """Create a sphere with emission material for flat color."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center, segments=24, ring_count=16)
    obj = bpy.context.active_object

    mat = bpy.data.materials.new(name="SphereMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[0].default_value = (color[0], color[1], color[2], 1.0)
    emission.inputs[1].default_value = 1.0

    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs[0], output.inputs[0])

    obj.data.materials.append(mat)
    return obj


def create_segre_surface(res=50, opacity=0.9):
    """Create the Segre surface mesh with vertex colors."""
    verts = []
    faces = []

    for i in range(res):
        for j in range(res):
            a1 = i / (res - 1)
            a2 = j / (res - 1)
            pt = factored_to_3d(a1, a2)
            verts.append(pt)

    for i in range(res - 1):
        for j in range(res - 1):
            v0 = i * res + j
            v1 = v0 + 1
            v2 = v0 + res
            v3 = v2 + 1
            faces.append((v0, v2, v3, v1))

    mesh = bpy.data.meshes.new("SegreMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("SegreSurface", mesh)
    bpy.context.collection.objects.link(obj)

    # Vertex colors
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    color_layer = mesh.vertex_colors.active

    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            i = vert_idx // res
            j = vert_idx % res
            a1 = i / (res - 1)
            a2 = j / (res - 1)
            rgb = pos_to_color(a1, a2)
            color_layer.data[loop_idx].color = (rgb[0], rgb[1], rgb[2], 1.0)

    # Material with vertex colors
    mat = bpy.data.materials.new(name="SegreMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)

    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Alpha'].default_value = opacity
    bsdf.inputs['Roughness'].default_value = 0.4

    vertex_color = nodes.new(type='ShaderNodeVertexColor')
    vertex_color.location = (-300, 0)
    vertex_color.layer_name = color_layer.name

    links.new(vertex_color.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    obj.data.materials.append(mat)
    return obj


def create_tetra_edges(radius=0.001, collection=None):
    """Create tetrahedron edge tubes."""
    for i, j in TETRA_EDGES:
        create_tube(TETRA_VERTICES[i], TETRA_VERTICES[j],
                    radius=radius, color=(*COLORS['tetra_edges'], 1.0), collection=collection)


def create_grid_lines(n_lines=5, radius=0.0015, collection=None):
    """Create grid lines on the Segre surface."""
    res = 30
    grid_color = (*COLORS['grid_lines'], 1.0)
    for i in range(n_lines + 1):
        a = i / n_lines
        # Lines along a1 direction (fixed a2)
        pts = [factored_to_3d(t, a) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k + 1], radius=radius, color=grid_color, collection=collection)
        # Lines along a2 direction (fixed a1)
        pts = [factored_to_3d(a, t) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k + 1], radius=radius, color=grid_color, collection=collection)


def create_surface_boundary(radius=0.006, collection=None):
    """Create dark boundary edges for the Segre surface."""
    res = 40
    boundary_color = (*COLORS['boundary'], 1.0)

    for fixed_val in [0, 1]:
        pts = [factored_to_3d(fixed_val, t) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k + 1], radius=radius, color=boundary_color, collection=collection)
        pts = [factored_to_3d(t, fixed_val) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k + 1], radius=radius, color=boundary_color, collection=collection)


# =============================================================================
# Scene Setup
# =============================================================================

def setup_scene_3d(elev=20, azim=160, resolution=None):
    if resolution is None:
        resolution = RENDER_RESOLUTION
    """Setup 3D scene with camera and lights."""
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64

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
    sun.data.energy = 2.0
    sun.rotation_euler = (np.radians(30), 0, 0)

    bpy.ops.object.light_add(type='SUN', location=(3, 3, 2))
    fill = bpy.context.active_object
    fill.data.energy = 0.7
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
    camera.data.ortho_scale = 2.4

    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = camera

    # Render settings
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.film_transparent = False
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'

    # Freestyle for outlines
    bpy.context.scene.render.use_freestyle = True
    bpy.context.scene.view_layers["ViewLayer"].use_freestyle = True

    freestyle = bpy.context.scene.view_layers["ViewLayer"].freestyle_settings
    lineset = freestyle.linesets.new("OutlineSet")
    lineset.select_silhouette = True
    lineset.select_border = False
    lineset.select_crease = False
    lineset.linestyle.color = COLORS['outline']
    lineset.linestyle.thickness = 1.0

    # Ground plane
    z_min = TETRA_VERTICES[:, 2].min()
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, z_min - 0.02))
    ground = bpy.context.active_object
    ground.name = "Ground"

    ground_mat = bpy.data.materials.new(name="GroundMaterial")
    ground_mat.use_nodes = True
    nodes = ground_mat.node_tree.nodes
    links = ground_mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    mix = nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = 0.05
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy = nodes.new('ShaderNodeBsdfGlossy')
    glossy.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs['Roughness'].default_value = 0.3

    links.new(diffuse.outputs[0], mix.inputs[1])
    links.new(glossy.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs[0])
    ground.data.materials.append(ground_mat)

    return camera


# =============================================================================
# Trajectory Generation and Rendering
# =============================================================================

def generate_freeze_vary_trajectories(frozen_values, n_samples=50):
    """
    Generate trajectories where factor 2 is frozen and factor 1 varies.

    Returns list of trajectories, each trajectory is array of (a1, a2, pos3d) tuples.
    """
    trajectories = []
    alpha1_range = np.linspace(0, 1, n_samples)

    for alpha2_frozen in frozen_values:
        traj = []
        for a1 in alpha1_range:
            pos3d = factored_to_3d(a1, alpha2_frozen)
            traj.append({
                'alpha1': a1,
                'alpha2': alpha2_frozen,
                'pos3d': pos3d
            })
        trajectories.append(traj)

    return trajectories


def create_trajectory_curve_3d(trajectory, color, tube_radius=0.012, collection=None):
    """Create a 3D curve for a trajectory on the Segre surface."""
    points = [t['pos3d'] for t in trajectory]

    for i in range(len(points) - 1):
        create_tube(points[i], points[i + 1], radius=tube_radius,
                   color=(color[0], color[1], color[2], 1.0),
                   collection=collection, use_emission=True)


def create_trajectory_endpoints_3d(trajectory, color, sphere_radius=0.045):
    """Create endpoint spheres for a trajectory."""
    # Start point
    start = trajectory[0]
    create_sphere(start['pos3d'], radius=sphere_radius,
                 color=(color[0], color[1], color[2], 1.0))

    # End point
    end = trajectory[-1]
    create_sphere(end['pos3d'], radius=sphere_radius,
                 color=(color[0], color[1], color[2], 1.0))


def create_arrow_head_3d(position, direction, color, size=0.06):
    """Create an arrow head (cone) at the end of a trajectory."""
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)

    bpy.ops.mesh.primitive_cone_add(
        radius1=size * 0.6,
        radius2=0,
        depth=size,
        location=position
    )
    cone = bpy.context.active_object

    # Orient cone along direction
    up = Vector((0, 0, 1))
    dir_vec = Vector(direction)

    if abs(dir_vec.dot(up)) < 0.999:
        rot_axis = up.cross(dir_vec)
        rot_axis.normalize()
        rot_angle = np.arccos(np.clip(up.dot(dir_vec), -1, 1))
        cone.rotation_mode = 'AXIS_ANGLE'
        cone.rotation_axis_angle = (rot_angle, rot_axis.x, rot_axis.y, rot_axis.z)
    elif dir_vec.dot(up) < 0:
        cone.rotation_euler = (np.pi, 0, 0)

    # Material
    mat = bpy.data.materials.new(name="ArrowMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[0].default_value = (color[0], color[1], color[2], 1.0)
    emission.inputs[1].default_value = 1.0
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs[0], output.inputs[0])

    cone.data.materials.append(mat)
    return cone


# =============================================================================
# Main Rendering Functions
# =============================================================================

def render_3d_freeze_vary(frozen_values, output_path, n_samples=50):
    """Render 3D view showing freeze-and-vary trajectories."""
    clear_scene()
    setup_scene_3d()

    # Create collection for tubes (excluded from Freestyle)
    tubes_collection = bpy.data.collections.new("Tubes")
    bpy.context.scene.collection.children.link(tubes_collection)

    # Create Segre surface (semi-transparent)
    create_segre_surface(res=50, opacity=0.7)

    # Create tetrahedron edges
    create_tetra_edges(radius=0.001, collection=tubes_collection)

    # Create grid lines (lighter)
    create_grid_lines(n_lines=5, radius=0.001, collection=tubes_collection)

    # Create surface boundary
    create_surface_boundary(radius=0.004, collection=tubes_collection)

    # Exclude tubes from Freestyle
    view_layer = bpy.context.scene.view_layers["ViewLayer"]
    freestyle = view_layer.freestyle_settings
    if freestyle.linesets:
        lineset = freestyle.linesets[0]
        lineset.select_by_collection = True
        lineset.collection = tubes_collection
        lineset.collection_negation = 'EXCLUSIVE'

    # Generate and render trajectories with random circle positions
    trajectories = generate_freeze_vary_trajectories(frozen_values, n_samples)
    n_circles = 15
    rng = np.random.default_rng(seed=123)  # Different seed from centered views

    for i, (traj, color) in enumerate(zip(trajectories, TRAJECTORY_COLORS)):
        # Sample random points along trajectory
        indices = np.sort(rng.choice(len(traj), size=n_circles, replace=False))

        for idx in indices:
            pos = traj[idx]['pos3d']
            create_sphere(pos, radius=0.04, color=(color[0], color[1], color[2], 1.0))

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


def render_2d_freeze_vary(frozen_values, output_path, n_samples=50):
    """Render 2D factored square view showing freeze-and-vary trajectories."""
    clear_scene()

    # Create colored square texture
    img_size = 512
    img = bpy.data.images.new("SquareTexture", width=img_size, height=img_size)
    pixels = np.zeros((img_size, img_size, 4))

    for i in range(img_size):
        for j in range(img_size):
            # Coordinate system: a1 maps to Y (bottom=0, top=1), a2 maps to -X (left=1, right=0)
            a1 = i / (img_size - 1)
            a2 = 1 - j / (img_size - 1)
            rgb = np.array(pos_to_color(a1, a2))
            rgb_lightened = rgb * 0.45 + 0.55
            pixels[i, j] = [rgb_lightened[0], rgb_lightened[1], rgb_lightened[2], 1.0]

    img.pixels = pixels.flatten().tolist()

    # Create plane
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.active_object

    mat = bpy.data.materials.new(name="SquareMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs[1].default_value = 1.0
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = img

    links.new(tex_node.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs[0], output.inputs[0])
    plane.data.materials.append(mat)

    # Grid lines
    grid_color = (*COLORS['grid_lines_2d'], 1.0)
    for t in [0.25, 0.5, 0.75]:
        # Horizontal grid lines
        x = -1 + t * 2
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0015, depth=2, location=(x, 0, 0.001))
        hline = bpy.context.active_object
        hline.rotation_euler = (np.pi / 2, 0, 0)
        grid_mat = bpy.data.materials.new(name="GridMat")
        grid_mat.use_nodes = True
        grid_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = grid_color
        hline.data.materials.append(grid_mat)

        # Vertical grid lines
        y = -1 + t * 2
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0015, depth=2, location=(0, y, 0.001))
        vline = bpy.context.active_object
        vline.rotation_euler = (0, np.pi / 2, 0)
        vline.data.materials.append(grid_mat)

    # Border
    border_color = (*COLORS['boundary'], 1.0)
    border_mat = bpy.data.materials.new(name="BorderMat")
    border_mat.use_nodes = True
    border_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = border_color

    for pos, rot in [
        ((0, -1, 0.002), (0, np.pi / 2, 0)),
        ((0, 1, 0.002), (0, np.pi / 2, 0)),
        ((-1, 0, 0.002), (np.pi / 2, 0, 0)),
        ((1, 0, 0.002), (np.pi / 2, 0, 0)),
    ]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=2.02, location=pos)
        border = bpy.context.active_object
        border.rotation_euler = rot
        border.data.materials.append(border_mat)

    # Create trajectories with random circle positions (varying a1, fixed a2)
    # In 2D: x = 1 - a2*2, y = -1 + a1*2
    n_circles = 15
    rng = np.random.default_rng(seed=123)  # Same seed as 3D non-centered

    for i, (alpha2_frozen, color) in enumerate(zip(frozen_values, TRAJECTORY_COLORS)):
        x_pos = 1 - alpha2_frozen * 2  # Fixed x position for frozen a2

        # Sample random alpha1 values along trajectory
        alpha1_samples = np.sort(rng.uniform(0, 1, size=n_circles))

        for a1 in alpha1_samples:
            y_pos = -1 + a1 * 2  # Map alpha1 to y coordinate
            create_flat_disk((x_pos, y_pos, 0.01 + i * 0.005), radius=0.055, color=color, z_offset=0)

    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 3))
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 2.4
    camera.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = camera

    # World
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

    # Render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = RENDER_SAMPLES
    bpy.context.scene.render.resolution_x = RENDER_RESOLUTION[0]
    bpy.context.scene.render.resolution_y = RENDER_RESOLUTION[1]
    bpy.context.scene.view_settings.view_transform = 'Standard'

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


# =============================================================================
# Mean-Centering Functions
# =============================================================================

def mean_center_trajectory_2d(trajectory):
    """Mean-center a 2D trajectory (in factored space)."""
    # Extract (alpha1, alpha2) coordinates
    coords = np.array([[t['alpha1'], t['alpha2']] for t in trajectory])
    mean = coords.mean(axis=0)
    centered = coords - mean
    return centered, mean


def mean_center_trajectory_3d(trajectory):
    """Mean-center a 3D trajectory (in joint space)."""
    coords = np.array([t['pos3d'] for t in trajectory])
    mean = coords.mean(axis=0)
    centered = coords - mean
    return centered, mean


# =============================================================================
# Mean-Centered Rendering Functions
# =============================================================================

def create_flat_disk(location, radius, color, outline_radius=None, z_offset=0):
    """Create a flat disk (circle) with optional dark outline ring."""
    if outline_radius is None:
        outline_radius = radius * 1.15

    # Dark outline disk (slightly larger, below)
    bpy.ops.mesh.primitive_cylinder_add(radius=outline_radius, depth=0.008,
                                        location=(location[0], location[1], location[2] + z_offset))
    outline = bpy.context.active_object
    outline_mat = bpy.data.materials.new(name="OutlineMat")
    outline_mat.use_nodes = True
    outline_nodes = outline_mat.node_tree.nodes
    outline_links = outline_mat.node_tree.links
    outline_nodes.clear()
    outline_emission = outline_nodes.new('ShaderNodeEmission')
    outline_emission.inputs[0].default_value = (0.25, 0.25, 0.25, 1.0)
    outline_emission.inputs[1].default_value = 1.0
    outline_output = outline_nodes.new('ShaderNodeOutputMaterial')
    outline_links.new(outline_emission.outputs[0], outline_output.inputs[0])
    outline.data.materials.append(outline_mat)

    # Colored disk (on top)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=0.008,
                                        location=(location[0], location[1], location[2] + z_offset + 0.005))
    disk = bpy.context.active_object
    disk_mat = bpy.data.materials.new(name="DiskMat")
    disk_mat.use_nodes = True
    disk_nodes = disk_mat.node_tree.nodes
    disk_links = disk_mat.node_tree.links
    disk_nodes.clear()
    disk_emission = disk_nodes.new('ShaderNodeEmission')
    disk_emission.inputs[0].default_value = (color[0], color[1], color[2], 1.0)
    disk_emission.inputs[1].default_value = 1.0
    disk_output = disk_nodes.new('ShaderNodeOutputMaterial')
    disk_links.new(disk_emission.outputs[0], disk_output.inputs[0])
    disk.data.materials.append(disk_mat)

    return outline, disk


def render_2d_mean_centered(frozen_values, output_path, n_samples=50):
    """Render 2D view showing mean-centered trajectories (all should align!)."""
    clear_scene()

    # Generate trajectories and mean-center them
    trajectories = generate_freeze_vary_trajectories(frozen_values, n_samples)
    centered_trajs = []
    for traj in trajectories:
        centered, _ = mean_center_trajectory_2d(traj)
        centered_trajs.append(centered)

    # Find bounds for display
    all_centered = np.vstack(centered_trajs)
    max_extent = np.abs(all_centered).max() * 1.3

    # Create a light gray background plane
    bpy.ops.mesh.primitive_plane_add(size=max_extent * 2.5, location=(0, 0, -0.01))
    bg_plane = bpy.context.active_object
    bg_mat = bpy.data.materials.new(name="BGMat")
    bg_mat.use_nodes = True
    bg_nodes = bg_mat.node_tree.nodes
    bg_links = bg_mat.node_tree.links
    bg_nodes.clear()
    bg_emission = bg_nodes.new('ShaderNodeEmission')
    bg_emission.inputs[0].default_value = (0.97, 0.97, 0.97, 1.0)
    bg_emission.inputs[1].default_value = 1.0
    bg_output = bg_nodes.new('ShaderNodeOutputMaterial')
    bg_links.new(bg_emission.outputs[0], bg_output.inputs[0])
    bg_plane.data.materials.append(bg_mat)

    # Draw axes through origin
    axis_color = (*COLORS['axis'], 1.0)
    axis_length = max_extent * 1.1

    # Horizontal axis (alpha2 direction after centering)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.008, depth=axis_length * 2, location=(0, 0, 0.001))
    h_axis = bpy.context.active_object
    h_axis.rotation_euler = (0, np.pi / 2, 0)
    axis_mat = bpy.data.materials.new(name="AxisMat")
    axis_mat.use_nodes = True
    axis_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = axis_color
    h_axis.data.materials.append(axis_mat)

    # Vertical axis (alpha1 direction after centering)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.008, depth=axis_length * 2, location=(0, 0, 0.001))
    v_axis = bpy.context.active_object
    v_axis.rotation_euler = (np.pi / 2, 0, 0)
    v_axis.data.materials.append(axis_mat)

    # Origin marker (flat disk)
    create_flat_disk((0, 0, 0.01), radius=0.03, color=COLORS['origin_marker'], z_offset=0)

    # Draw centered trajectories - in factored space they should ALL align!
    # Coordinate mapping: x = centered_alpha2, y = centered_alpha1
    # Use circles at RANDOM positions along the trajectory
    n_circles = 15  # Number of circles per trajectory
    rng = np.random.default_rng(seed=42)  # For reproducibility

    for i, (centered, color) in enumerate(zip(centered_trajs, TRAJECTORY_COLORS)):
        # Scale for visibility
        scale = 1.5

        # Sample points RANDOMLY along trajectory for circles (different for each trajectory)
        indices = np.sort(rng.choice(len(centered), size=n_circles, replace=False))

        for j, idx in enumerate(indices):
            c = centered[idx]
            # For factored space, centered[:,0] is alpha1, centered[:,1] is alpha2
            x = c[1] * scale
            y = c[0] * scale
            z = 0.01 + i * 0.008

            create_flat_disk((x, y, z), radius=0.045, color=color, z_offset=0)

    # Camera
    bpy.ops.object.camera_add(location=(0, 0, 3))
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max_extent * 2.8
    camera.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = camera

    # World
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

    # Render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = RENDER_SAMPLES
    bpy.context.scene.render.resolution_x = RENDER_RESOLUTION[0]
    bpy.context.scene.render.resolution_y = RENDER_RESOLUTION[1]
    bpy.context.scene.view_settings.view_transform = 'Standard'

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


def create_camera_facing_disk(location, radius, color, camera_location, outline_ratio=1.15):
    """Create a flat disk that faces the camera with a dark outline ring."""
    location = np.array(location)
    camera_location = np.array(camera_location)

    # Calculate direction from disk to camera
    direction = camera_location - location
    direction = direction / np.linalg.norm(direction)

    # Create outline disk (slightly larger)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius * outline_ratio, depth=0.005, location=location)
    outline = bpy.context.active_object

    # Orient to face camera
    up = Vector((0, 0, 1))
    dir_vec = Vector(direction)
    if abs(dir_vec.dot(up)) < 0.999:
        rot_axis = up.cross(dir_vec)
        rot_axis.normalize()
        rot_angle = np.arccos(np.clip(up.dot(dir_vec), -1, 1))
        outline.rotation_mode = 'AXIS_ANGLE'
        outline.rotation_axis_angle = (rot_angle, rot_axis.x, rot_axis.y, rot_axis.z)
    elif dir_vec.dot(up) < 0:
        outline.rotation_euler = (np.pi, 0, 0)

    outline_mat = bpy.data.materials.new(name="OutlineMat3D")
    outline_mat.use_nodes = True
    outline_nodes = outline_mat.node_tree.nodes
    outline_links = outline_mat.node_tree.links
    outline_nodes.clear()
    outline_emission = outline_nodes.new('ShaderNodeEmission')
    outline_emission.inputs[0].default_value = (0.25, 0.25, 0.25, 1.0)
    outline_emission.inputs[1].default_value = 1.0
    outline_output = outline_nodes.new('ShaderNodeOutputMaterial')
    outline_links.new(outline_emission.outputs[0], outline_output.inputs[0])
    outline.data.materials.append(outline_mat)

    # Create colored disk (slightly in front)
    disk_location = location + direction * 0.003
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=0.005, location=disk_location)
    disk = bpy.context.active_object

    # Same orientation
    if abs(dir_vec.dot(up)) < 0.999:
        disk.rotation_mode = 'AXIS_ANGLE'
        disk.rotation_axis_angle = outline.rotation_axis_angle
    elif dir_vec.dot(up) < 0:
        disk.rotation_euler = (np.pi, 0, 0)

    disk_mat = bpy.data.materials.new(name="DiskMat3D")
    disk_mat.use_nodes = True
    disk_nodes = disk_mat.node_tree.nodes
    disk_links = disk_mat.node_tree.links
    disk_nodes.clear()
    disk_emission = disk_nodes.new('ShaderNodeEmission')
    disk_emission.inputs[0].default_value = (color[0], color[1], color[2], 1.0)
    disk_emission.inputs[1].default_value = 1.0
    disk_output = disk_nodes.new('ShaderNodeOutputMaterial')
    disk_links.new(disk_emission.outputs[0], disk_output.inputs[0])
    disk.data.materials.append(disk_mat)

    return outline, disk


def render_3d_mean_centered(frozen_values, output_path, n_samples=50):
    """Render 3D view showing mean-centered trajectories (different directions!)."""
    clear_scene()

    # Generate trajectories and mean-center them
    trajectories = generate_freeze_vary_trajectories(frozen_values, n_samples)
    centered_trajs = []
    for traj in trajectories:
        centered, _ = mean_center_trajectory_3d(traj)
        centered_trajs.append(centered)

    # Setup scene without Segre surface (just show centered trajectories)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64

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
    sun.data.energy = 2.0
    sun.rotation_euler = (np.radians(30), 0, 0)

    bpy.ops.object.light_add(type='SUN', location=(3, 3, 2))
    fill = bpy.context.active_object
    fill.data.energy = 0.7

    # Camera position
    elev, azim = 25, 150
    r = 2.0
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    cam_x = r * np.cos(elev_rad) * np.cos(azim_rad)
    cam_y = r * np.cos(elev_rad) * np.sin(azim_rad)
    cam_z = r * np.sin(elev_rad)
    camera_location = (cam_x, cam_y, cam_z)

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 1.4

    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = camera

    # Render settings
    bpy.context.scene.render.resolution_x = RENDER_RESOLUTION[0]
    bpy.context.scene.render.resolution_y = RENDER_RESOLUTION[1]
    bpy.context.scene.view_settings.view_transform = 'Standard'

    # Draw axes through origin
    axis_length = 0.5
    axis_radius = 0.008
    axis_colors = [
        (0.8, 0.3, 0.3, 1.0),  # X - red
        (0.3, 0.7, 0.3, 1.0),  # Y - green
        (0.3, 0.3, 0.8, 1.0),  # Z - blue
    ]
    axis_directions = [
        np.array([1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
    ]

    for axis_dir, axis_color in zip(axis_directions, axis_colors):
        start = -axis_dir * axis_length
        end = axis_dir * axis_length
        create_tube(start, end, radius=axis_radius, color=axis_color, use_emission=False)

    # Origin disk (camera-facing)
    create_camera_facing_disk((0, 0, 0), radius=0.04, color=COLORS['origin_marker'], camera_location=camera_location)

    # Draw centered trajectories - in 3D they have DIFFERENT directions!
    # Use circles at RANDOM positions along the trajectory
    n_circles = 15  # Number of circles per trajectory
    rng = np.random.default_rng(seed=42)  # Same seed as 2D for consistency

    for i, (centered, color) in enumerate(zip(centered_trajs, TRAJECTORY_COLORS)):
        # Sample points RANDOMLY along trajectory for circles (different for each trajectory)
        indices = np.sort(rng.choice(len(centered), size=n_circles, replace=False))

        for idx in indices:
            pos = centered[idx]
            create_camera_facing_disk(pos, radius=0.035, color=color, camera_location=camera_location)

    # Add a small semi-transparent sphere at origin to emphasize centering
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.15, location=(0, 0, 0))
    center_sphere = bpy.context.active_object
    center_mat = bpy.data.materials.new(name="CenterMat")
    center_mat.use_nodes = True
    center_nodes = center_mat.node_tree.nodes
    center_links = center_mat.node_tree.links
    center_nodes.clear()
    bsdf = center_nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.9, 1.0)
    bsdf.inputs['Alpha'].default_value = 0.15
    bsdf.inputs['Roughness'].default_value = 0.8
    output = center_nodes.new('ShaderNodeOutputMaterial')
    center_links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    center_sphere.data.materials.append(center_mat)

    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -0.5))
    ground = bpy.context.active_object
    ground_mat = bpy.data.materials.new(name="GroundMat")
    ground_mat.use_nodes = True
    ground_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.95, 0.95, 0.95, 1.0)
    ground.data.materials.append(ground_mat)

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command-line arguments (passed after '--' when running via Blender)."""
    parser = argparse.ArgumentParser(description='Render freeze-and-vary visualizations')
    parser.add_argument('--output-dir', type=str, default='png',
                        help='Output subdirectory name (default: png)')
    parser.add_argument('--color-palette', type=str, default='default',
                        help='Color palette name from configs/colors.yaml (default: default)')
    # Blender passes args after '--'
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return parser.parse_args(argv)


def main():
    """Main function to render freeze-and-vary visualizations."""
    global COLORS, TRAJECTORY_COLORS

    args = parse_args()

    # Load color palette
    COLORS = load_colors(args.color_palette)
    TRAJECTORY_COLORS = COLORS['trajectory']

    # Output directory: base_dir / output_dir_name
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Frozen values for factor 2 (alpha2)
    frozen_values = [0.2, 0.4, 0.6, 0.8]

    print("=" * 60)
    print("Freeze-and-Vary Visualization")
    print("=" * 60)
    print(f"Frozen alpha2 values: {frozen_values}")
    print("Varying alpha1 from 0 to 1 for each frozen value")
    print()

    # Render original 3D view
    print("Rendering 3D view (Segre surface with trajectories)...")
    render_3d_freeze_vary(
        frozen_values,
        os.path.join(output_dir, 'freeze_vary_3d.png'),
        n_samples=50
    )

    # Render original 2D view
    print("Rendering 2D view (factored square with parallel lines)...")
    render_2d_freeze_vary(
        frozen_values,
        os.path.join(output_dir, 'freeze_vary_2d.png'),
        n_samples=50
    )

    # Render mean-centered 2D view
    print()
    print("Rendering MEAN-CENTERED 2D view (all lines should align!)...")
    render_2d_mean_centered(
        frozen_values,
        os.path.join(output_dir, 'freeze_vary_2d_centered.png'),
        n_samples=50
    )

    # Render mean-centered 3D view
    print("Rendering MEAN-CENTERED 3D view (different directions!)...")
    render_3d_mean_centered(
        frozen_values,
        os.path.join(output_dir, 'freeze_vary_3d_centered.png'),
        n_samples=50
    )

    print()
    print("=" * 60)
    print("Key Insight (BEFORE mean-centering):")
    print("- In 2D (factored space): All trajectories are PARALLEL vertical lines")
    print("- In 3D (joint space): Trajectories have DIFFERENT tangent directions")
    print()
    print("Key Insight (AFTER mean-centering):")
    print("- In 2D: All trajectories OVERLAP (same direction = spans 1D)")
    print("- In 3D: Trajectories DIVERGE from origin (different directions = spans >1D)")
    print()
    print("This explains why factored representations are more efficient!")
    print("=" * 60)


if __name__ == '__main__':
    main()
