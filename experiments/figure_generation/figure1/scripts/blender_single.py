"""
Figure 1: Indecomposable case - Blender version

Run with: /Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/blender_single.py
"""

import argparse
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix


def parse_args():
    """Parse command-line arguments (passed after '--' when running via Blender)."""
    parser = argparse.ArgumentParser(description='Render indecomposable belief geometry')
    parser.add_argument('--output-dir', type=str, default='png',
                        help='Output subdirectory name (default: png)')
    # Blender passes args after '--'
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return parser.parse_args(argv)

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# =============================================================================
# Color Mapping
# =============================================================================

def pos_to_color(a1, a2):
    """Map (α₁, α₂) to RGB using bilinear interpolation of 4 corner colors."""
    c00 = np.array([0.25, 0.70, 0.75])  # vibrant teal
    c10 = np.array([0.95, 0.45, 0.35])  # vibrant coral
    c01 = np.array([0.40, 0.65, 0.95])  # vibrant blue
    c11 = np.array([1.00, 0.85, 0.40])  # vibrant gold

    c_bottom = (1 - a1) * c00 + a1 * c10
    c_top = (1 - a1) * c01 + a1 * c11
    rgb = (1 - a2) * c_bottom + a2 * c_top
    return rgb


# =============================================================================
# Geometry
# =============================================================================

TETRA_VERTICES = np.array([
    [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
], dtype=float) * 0.65

TETRA_EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]


def belief_to_simplex(a1, a2):
    return np.array([a1*a2, a1*(1-a2), (1-a1)*a2, (1-a1)*(1-a2)])


def simplex_to_3d(probs):
    return probs @ TETRA_VERTICES


def compute_segre_normal(a1, a2, eps=1e-5):
    """Compute the unit normal to the Segre surface at (α₁, α₂)."""
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
# Indecomposable beliefs
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
# Blender mesh creation
# =============================================================================

def create_segre_surface(res=50, opacity=0.9):
    """Create Segre surface mesh with vertex colors."""
    a1_arr = np.linspace(0, 1, res)
    a2_arr = np.linspace(0, 1, res)

    # Create mesh
    mesh = bpy.data.meshes.new("SegreSurface")
    obj = bpy.data.objects.new("SegreSurface", mesh)
    bpy.context.collection.objects.link(obj)

    # Build mesh with bmesh
    bm = bmesh.new()

    # Create vertices
    vert_map = {}
    for i in range(res):
        for j in range(res):
            a1, a2 = a1_arr[i], a2_arr[j]
            pt = simplex_to_3d(belief_to_simplex(a1, a2))
            v = bm.verts.new(pt)
            vert_map[(i, j)] = v

    bm.verts.ensure_lookup_table()

    # Create faces
    for i in range(res - 1):
        for j in range(res - 1):
            v00 = vert_map[(i, j)]
            v10 = vert_map[(i + 1, j)]
            v11 = vert_map[(i + 1, j + 1)]
            v01 = vert_map[(i, j + 1)]
            bm.faces.new([v00, v10, v11, v01])

    bm.to_mesh(mesh)
    bm.free()

    # Create vertex color layer
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name="Col")

    color_layer = mesh.vertex_colors["Col"]

    # Assign colors per loop (corner of each face)
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            loop = mesh.loops[loop_idx]
            vert_idx = loop.vertex_index
            vert = mesh.vertices[vert_idx]

            # Find (i, j) from vertex position
            # This is approximate - find closest grid point
            pt = np.array(vert.co)
            best_a1, best_a2 = 0.5, 0.5
            best_dist = float('inf')
            for i in range(res):
                for j in range(res):
                    a1, a2 = a1_arr[i], a2_arr[j]
                    test_pt = simplex_to_3d(belief_to_simplex(a1, a2))
                    dist = np.linalg.norm(pt - test_pt)
                    if dist < best_dist:
                        best_dist = dist
                        best_a1, best_a2 = a1, a2

            rgb = pos_to_color(best_a1, best_a2)
            color_layer.data[loop_idx].color = (rgb[0], rgb[1], rgb[2], 1.0)

    # Create material with vertex colors
    mat = bpy.data.materials.new(name="SegreMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create nodes for vertex color material
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Alpha'].default_value = opacity

    vertex_color = nodes.new('ShaderNodeVertexColor')
    vertex_color.location = (-300, 0)
    vertex_color.layer_name = "Col"

    # Connect nodes
    links.new(vertex_color.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    obj.data.materials.append(mat)

    return obj


def create_tube(start, end, radius=0.004, color=(0.8, 0.3, 0.3, 1.0), collection=None):
    """Create a tube (cylinder) between two points."""
    start = np.array(start)
    end = np.array(end)

    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return None

    mid = (start + end) / 2

    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=length,
        location=mid
    )
    obj = bpy.context.active_object

    # Rotate to align with direction
    direction_normalized = direction / length
    up = Vector((0, 0, 1))

    if abs(direction_normalized[2]) > 0.999:
        # Nearly vertical - use simple rotation
        if direction_normalized[2] < 0:
            obj.rotation_euler = (np.pi, 0, 0)
    else:
        # General case
        rot_axis = up.cross(Vector(direction_normalized))
        rot_axis.normalize()
        rot_angle = np.arccos(np.clip(up.dot(Vector(direction_normalized)), -1, 1))
        obj.rotation_mode = 'AXIS_ANGLE'
        obj.rotation_axis_angle = (rot_angle, rot_axis.x, rot_axis.y, rot_axis.z)

    # Create material
    mat = bpy.data.materials.new(name="TubeMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.5
    obj.data.materials.append(mat)

    # Move to collection if specified
    if collection is not None:
        bpy.context.collection.objects.unlink(obj)
        collection.objects.link(obj)

    return obj


def create_sphere(center, radius=0.035, color=(0.5, 0.5, 0.8, 1.0)):
    """Create a flat filled circle (disk) that faces the camera."""
    # Create a UV sphere but flatten it
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        location=center,
        segments=32,
        ring_count=16
    )
    obj = bpy.context.active_object

    # Create flat vibrant material using pure emission shader
    mat = bpy.data.materials.new(name="CircleMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Pure emission shader - flat color, no shading
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[0].default_value = (color[0], color[1], color[2], 1.0)  # Color
    emission.inputs[1].default_value = 1.0  # Standard strength

    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs[0], output.inputs[0])

    obj.data.materials.append(mat)

    return obj


def create_tetra_edges(radius=0.001, collection=None):
    """Create tetrahedron edge tubes."""
    for i, j in TETRA_EDGES:
        create_tube(TETRA_VERTICES[i], TETRA_VERTICES[j],
                   radius=radius, color=(0.75, 0.75, 0.75, 1.0), collection=collection)


def create_grid_lines(n_lines=5, radius=0.002, collection=None):
    """Create grid lines on the Segre surface."""
    res = 30
    for i in range(n_lines + 1):
        a = i / n_lines

        # Lines along a1 direction
        pts = [simplex_to_3d(belief_to_simplex(t, a)) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k+1], radius=radius, color=(0.5, 0.5, 0.5, 1.0), collection=collection)

        # Lines along a2 direction
        pts = [simplex_to_3d(belief_to_simplex(a, t)) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k+1], radius=radius, color=(0.5, 0.5, 0.5, 1.0), collection=collection)


def create_surface_boundary(radius=0.006, collection=None):
    """Create dark boundary edges for the Segre surface."""
    res = 40
    boundary_color = (0.15, 0.15, 0.15, 1.0)  # Darker grey

    # Edge where a1 = 0
    pts = [simplex_to_3d(belief_to_simplex(0, t)) for t in np.linspace(0, 1, res)]
    for k in range(len(pts) - 1):
        create_tube(pts[k], pts[k+1], radius=radius, color=boundary_color, collection=collection)

    # Edge where a1 = 1
    pts = [simplex_to_3d(belief_to_simplex(1, t)) for t in np.linspace(0, 1, res)]
    for k in range(len(pts) - 1):
        create_tube(pts[k], pts[k+1], radius=radius, color=boundary_color, collection=collection)

    # Edge where a2 = 0
    pts = [simplex_to_3d(belief_to_simplex(t, 0)) for t in np.linspace(0, 1, res)]
    for k in range(len(pts) - 1):
        create_tube(pts[k], pts[k+1], radius=radius, color=boundary_color, collection=collection)

    # Edge where a2 = 1
    pts = [simplex_to_3d(belief_to_simplex(t, 1)) for t in np.linspace(0, 1, res)]
    for k in range(len(pts) - 1):
        create_tube(pts[k], pts[k+1], radius=radius, color=boundary_color, collection=collection)


# =============================================================================
# Scene setup
# =============================================================================

def setup_scene(elev=20, azim=160, resolution=(1800, 1600), ortho_scale=2.4):
    # Set render engine to Cycles for proper emission rendering
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64

    # Set background to pure white using existing world
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes

    # Find or create Background node
    bg_node = None
    for node in nodes:
        if node.type == 'BACKGROUND':
            bg_node = node
            break

    if bg_node is None:
        nodes.clear()
        bg_node = nodes.new(type='ShaderNodeBackground')
        output = nodes.new(type='ShaderNodeOutputWorld')
        world.node_tree.links.new(bg_node.outputs[0], output.inputs[0])

    # Set to pure white
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs['Strength'].default_value = 1.0

    # Also set world color directly
    world.color = (1.0, 1.0, 1.0)

    # Add sun light from above
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 3.0
    sun.rotation_euler = (np.radians(30), 0, 0)

    # Add fill light
    bpy.ops.object.light_add(type='SUN', location=(3, 3, 2))
    fill = bpy.context.active_object
    fill.data.energy = 1.0
    fill.rotation_euler = (np.radians(60), np.radians(45), 0)

    # Setup camera
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

    # Point camera at origin
    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = camera

    # No ground plane - cleaner look without shadows

    # Render settings
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.film_transparent = False

    # KEY: Use Standard view transform for accurate colors (not Filmic)
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'

    # Enable Freestyle for ball outlines only (silhouette)
    bpy.context.scene.render.use_freestyle = True
    bpy.context.scene.view_layers["ViewLayer"].use_freestyle = True

    freestyle = bpy.context.scene.view_layers["ViewLayer"].freestyle_settings
    lineset = freestyle.linesets.new("OutlineSet")
    lineset.select_silhouette = True
    lineset.select_border = False
    lineset.select_crease = False

    linestyle = lineset.linestyle
    linestyle.color = (0.25, 0.25, 0.25)
    linestyle.thickness = 1.0

    # Add subtle reflective ground plane for soft ambient effect
    z_min = TETRA_VERTICES[:, 2].min()
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, z_min - 0.02))
    ground = bpy.context.active_object
    ground.name = "Ground"

    # Create a white material with very subtle glossy reflection
    ground_mat = bpy.data.materials.new(name="GroundMaterial")
    ground_mat.use_nodes = True
    nodes = ground_mat.node_tree.nodes
    links = ground_mat.node_tree.links
    nodes.clear()

    # Mix between white diffuse and subtle reflection
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (200, 0)
    mix.inputs[0].default_value = 0.05  # Very subtle reflection (5%)

    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.location = (0, 100)
    diffuse.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)

    glossy = nodes.new('ShaderNodeBsdfGlossy')
    glossy.location = (0, -100)
    glossy.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs['Roughness'].default_value = 0.3

    links.new(diffuse.outputs[0], mix.inputs[1])
    links.new(glossy.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs[0])

    ground.data.materials.append(ground_mat)


def create_figure(output_dir):
    # Generate beliefs
    beliefs = compute_beliefs_indecomposable_random(n_points=25, offset_strength=0.25, seed=303)

    # Setup scene FIRST (creates camera needed for circle constraints)
    setup_scene()

    # Create collection for tubes (will be excluded from Freestyle)
    tubes_collection = bpy.data.collections.new("Tubes")
    bpy.context.scene.collection.children.link(tubes_collection)

    # Create Segre surface with slight transparency
    create_segre_surface(res=50, opacity=0.9)

    # Create tetrahedron edges (very thin and light) - in tubes collection
    create_tetra_edges(radius=0.001, collection=tubes_collection)

    # Create grid lines - in tubes collection
    create_grid_lines(n_lines=5, radius=0.0015, collection=tubes_collection)

    # Create dark boundary edges for the surface - in tubes collection
    create_surface_boundary(radius=0.006, collection=tubes_collection)

    # Configure Freestyle to exclude tubes collection
    view_layer = bpy.context.scene.view_layers["ViewLayer"]
    freestyle = view_layer.freestyle_settings
    if freestyle.linesets:
        lineset = freestyle.linesets[0]
        lineset.select_by_collection = True
        lineset.collection = tubes_collection
        lineset.collection_negation = 'EXCLUSIVE'  # Exclude this collection

    # Create belief points and projection lines
    for i, b in enumerate(beliefs):
        p = b['pos3d']
        sp = b['surface_pos']
        rgb = pos_to_color(b['alpha1'], b['alpha2'])

        if i < 3:  # Debug first few
            print(f"Ball {i}: alpha1={b['alpha1']:.2f}, alpha2={b['alpha2']:.2f}, rgb={rgb}")

        # Projection line (red tube) - in tubes collection
        create_tube(p, sp, radius=0.004, color=(0.85, 0.3, 0.3, 1.0), collection=tubes_collection)

        # Boost saturation for more vibrant ball colors
        gray = np.mean(rgb)
        rgb_boosted = np.clip(gray + (rgb - gray) * 2.5, 0, 1)  # Strong saturation boost
        create_sphere(p, radius=0.04, color=(float(rgb_boosted[0]), float(rgb_boosted[1]), float(rgb_boosted[2]), 1.0))

    # Render main version
    output_path = os.path.join(output_dir, 'blender_indecomposable_3d.png')
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")

    # Now render 2D factored square version
    render_factored_square(beliefs, output_dir)


def render_factored_square(beliefs, output_dir):
    """Render the 2D factored square view in Blender."""
    import tempfile

    # Clear scene for 2D render
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Create colored square texture - lighten to match 3D transparency effect
    img_size = 512
    img = bpy.data.images.new("SquareTexture", width=img_size, height=img_size)
    pixels = np.zeros((img_size, img_size, 4))

    for i in range(img_size):
        for j in range(img_size):
            # Transform so yellow (a1=1,a2=1) is top-left, orange (a1=1,a2=0) is top-right
            # Blender images have origin at bottom-left, so i=0 is bottom
            a1 = i / (img_size - 1)        # bottom=0, top=1
            a2 = 1 - j / (img_size - 1)    # left=1, right=0
            rgb = np.array(pos_to_color(a1, a2))
            # Lighten colors significantly to match 3D surface brightness (~230-250)
            rgb_lightened = rgb * 0.45 + 0.55  # Much more white blend
            pixels[i, j] = [rgb_lightened[0], rgb_lightened[1], rgb_lightened[2], 1.0]

    img.pixels = pixels.flatten().tolist()

    # Create plane for the square
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "FactoredSquare"

    # Create material with texture
    mat = bpy.data.materials.new(name="SquareMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Use emission for flat colors
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (200, 0)
    emission.inputs[1].default_value = 1.0

    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.location = (0, 0)
    tex_node.image = img

    links.new(tex_node.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs[0], output.inputs[0])

    plane.data.materials.append(mat)

    # Add grid lines - very thin and light grey to match 3D view
    grid_color = (0.7, 0.7, 0.7, 1.0)  # Much lighter grey
    for t in [0.25, 0.5, 0.75]:
        # Horizontal line
        x = -1 + t * 2
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0015, depth=2, location=(x, 0, 0.001))
        hline = bpy.context.active_object
        hline.rotation_euler = (np.pi/2, 0, 0)
        grid_mat = bpy.data.materials.new(name="GridMat")
        grid_mat.use_nodes = True
        grid_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = grid_color
        hline.data.materials.append(grid_mat)

        # Vertical line
        y = -1 + t * 2
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0015, depth=2, location=(0, y, 0.001))
        vline = bpy.context.active_object
        vline.rotation_euler = (0, np.pi/2, 0)
        vline.data.materials.append(grid_mat)

    # Add border - thinner, dark like 3D boundary
    border_color = (0.15, 0.15, 0.15, 1.0)  # Match 3D boundary color
    border_mat = bpy.data.materials.new(name="BorderMat")
    border_mat.use_nodes = True
    border_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = border_color

    for pos, rot in [
        ((0, -1, 0.002), (0, np.pi/2, 0)),  # bottom
        ((0, 1, 0.002), (0, np.pi/2, 0)),   # top
        ((-1, 0, 0.002), (np.pi/2, 0, 0)),  # left
        ((1, 0, 0.002), (np.pi/2, 0, 0)),   # right
    ]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=2.02, location=pos)
        border = bpy.context.active_object
        border.rotation_euler = rot
        border.data.materials.append(border_mat)

    # Add belief points as flat disks with dark outline rings
    for b in beliefs:
        a1 = b['alpha1']
        a2 = b['alpha2']

        # Transform to match 2D square: yellow (a1=1,a2=1) top-left, orange (a1=1,a2=0) top-right
        x = 1 - a2 * 2   # a2=1 → left (-1), a2=0 → right (1)
        y = -1 + a1 * 2  # a1=0 → bottom (-1), a1=1 → top (1)

        rgb = pos_to_color(a1, a2)
        gray = np.mean(rgb)
        rgb_boosted = np.clip(gray + (rgb - gray) * 2.5, 0, 1)  # Match 3D saturation boost

        # Create dark outline ring first (slightly larger) - thinner outline
        bpy.ops.mesh.primitive_cylinder_add(radius=0.062, depth=0.008, location=(x, y, 0.004))
        outline = bpy.context.active_object
        outline_mat = bpy.data.materials.new(name="OutlineMat")
        outline_mat.use_nodes = True
        outline_nodes = outline_mat.node_tree.nodes
        outline_links = outline_mat.node_tree.links
        outline_nodes.clear()
        outline_emission = outline_nodes.new('ShaderNodeEmission')
        outline_emission.inputs[0].default_value = (0.25, 0.25, 0.25, 1.0)  # Dark grey outline
        outline_emission.inputs[1].default_value = 1.0
        outline_output = outline_nodes.new('ShaderNodeOutputMaterial')
        outline_links.new(outline_emission.outputs[0], outline_output.inputs[0])
        outline.data.materials.append(outline_mat)

        # Create colored disk on top (slightly smaller)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.055, depth=0.008, location=(x, y, 0.006))
        disk = bpy.context.active_object

        # Emission material for flat color
        disk_mat = bpy.data.materials.new(name="DiskMat")
        disk_mat.use_nodes = True
        disk_nodes = disk_mat.node_tree.nodes
        disk_links = disk_mat.node_tree.links
        disk_nodes.clear()

        disk_emission = disk_nodes.new('ShaderNodeEmission')
        disk_emission.inputs[0].default_value = (float(rgb_boosted[0]), float(rgb_boosted[1]), float(rgb_boosted[2]), 1.0)
        disk_emission.inputs[1].default_value = 1.0

        disk_output = disk_nodes.new('ShaderNodeOutputMaterial')
        disk_links.new(disk_emission.outputs[0], disk_output.inputs[0])

        disk.data.materials.append(disk_mat)

    # Setup camera (top-down ortho)
    bpy.ops.object.camera_add(location=(0, 0, 3))
    camera = bpy.context.active_object
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = 2.4
    camera.rotation_euler = (0, 0, 0)
    bpy.context.scene.camera = camera

    # Setup world
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
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.render.resolution_x = 1200
    bpy.context.scene.render.resolution_y = 1200
    bpy.context.scene.view_settings.view_transform = 'Standard'

    # Add corner labels (as text objects would be complex, skip for now)

    # Render
    output_path = os.path.join(output_dir, 'blender_indecomposable_2d.png')
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    create_figure(output_dir)
