"""
Figure 1: All Blender renders for factored belief geometry.

Renders:
- (a) Independent: 3D + 2D square
- (b) Dependent: 3D + 2D square
- (c) Indecomposable: 3D + 2D square

Output folder: experiments/figure_generation/figure1/
"""

import argparse
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

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
    """Compute the unit normal to the Segre surface at (a1, a2)."""
    p_a1_plus = simplex_to_3d(belief_to_simplex(min(a1 + eps, 1), a2))
    p_a1_minus = simplex_to_3d(belief_to_simplex(max(a1 - eps, 0), a2))
    dp_da1 = (p_a1_plus - p_a1_minus) / (2 * eps)

    p_a2_plus = simplex_to_3d(belief_to_simplex(a1, min(a2 + eps, 1)))
    p_a2_minus = simplex_to_3d(belief_to_simplex(a1, max(a2 - eps, 0)))
    dp_da2 = (p_a2_plus - p_a2_minus) / (2 * eps)

    normal = np.cross(dp_da1, dp_da2)
    norm = np.linalg.norm(normal)
    if norm > 1e-10:
        normal = normal / norm
    return normal


# =============================================================================
# Color mapping
# =============================================================================

def pos_to_color(a1, a2):
    """Map (a1, a2) to RGB using bilinear interpolation."""
    c00 = np.array([0.25, 0.70, 0.75])
    c10 = np.array([0.95, 0.45, 0.35])
    c01 = np.array([0.40, 0.65, 0.95])
    c11 = np.array([1.00, 0.85, 0.40])

    c_bottom = (1 - a1) * c00 + a1 * c10
    c_top = (1 - a1) * c01 + a1 * c11
    return (1 - a2) * c_bottom + a2 * c_top


# =============================================================================
# HMM for belief computation
# =============================================================================

class TwoStateHMM:
    def __init__(self, t_AA, t_BA):
        self.t_AA = t_AA
        self.t_BA = t_BA

    def steady_state(self):
        denom = self.t_BA + (1 - self.t_AA)
        if abs(denom) < 1e-10:
            return 0.5
        return self.t_BA / denom

    def update_belief(self, alpha, y):
        if y == 0:
            return 1.0
        else:
            p_y1 = 1 - (1 - alpha) * self.t_BA
            if p_y1 < 1e-10:
                return alpha
            return alpha * self.t_AA / p_y1


def compute_beliefs_independent(hmm1, hmm2, max_len, init_belief=None):
    """Compute independent beliefs - stays on surface."""
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
    """Coupled beliefs - still on surface."""
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
                mod_t_AA = hmm2.t_AA + coupling * (1 if y1 == 0 else -1) * 0.2
                mod_t_BA = hmm2.t_BA + coupling * (1 if y1 == 0 else -1) * 0.2
                mod_t_AA = np.clip(mod_t_AA, 0.05, 0.95)
                mod_t_BA = np.clip(mod_t_BA, 0.05, 0.95)
                mod_hmm2 = TwoStateHMM(mod_t_AA, mod_t_BA)
                new_a2 = mod_hmm2.update_belief(a2, y2)
                recurse(seq + [(y1,y2)], new_a1, new_a2)

    recurse([], a1_init, a2_init)
    return beliefs


def compute_beliefs_indecomposable_random(n_points=25, offset_strength=0.25, seed=303):
    """Indecomposable beliefs - OFF surface."""
    rng = np.random.default_rng(seed)

    z_min = TETRA_VERTICES[:, 2].min()
    z_max = TETRA_VERTICES[:, 2].max()

    beliefs = []
    attempts = 0
    center_normal = compute_segre_normal(0.5, 0.5)
    while len(beliefs) < n_points and attempts < n_points * 10:
        attempts += 1
        a1 = rng.uniform(0.05, 0.95)
        a2 = rng.uniform(0.05, 0.95)

        factored_simplex = belief_to_simplex(a1, a2)
        factored_pos = simplex_to_3d(factored_simplex)

        magnitude = rng.standard_normal() * offset_strength
        true_pos = factored_pos + magnitude * center_normal

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
# Blender helpers
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


def create_segre_surface(res=50, opacity=0.9):
    """Create the Segre surface mesh with vertex colors."""
    verts = []
    faces = []
    colors = []

    for i in range(res):
        for j in range(res):
            a1 = i / (res - 1)
            a2 = j / (res - 1)
            pt = simplex_to_3d(belief_to_simplex(a1, a2))
            verts.append(pt)
            colors.append(pos_to_color(a1, a2))

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
        for idx, loop_idx in enumerate(poly.loop_indices):
            vert_idx = mesh.loops[loop_idx].vertex_index
            i = vert_idx // res
            j = vert_idx % res
            a1 = i / (res - 1)
            a2 = j / (res - 1)
            rgb = pos_to_color(a1, a2)
            color_layer.data[loop_idx].color = (rgb[0], rgb[1], rgb[2], 1.0)

    # Material
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


def create_tube(start, end, radius=0.004, color=(0.8, 0.3, 0.3, 1.0), collection=None):
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

    if collection is not None:
        bpy.context.collection.objects.unlink(obj)
        collection.objects.link(obj)

    return obj


def create_sphere(center, radius=0.035, color=(0.5, 0.5, 0.8, 1.0)):
    """Create a sphere with emission material."""
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


def create_intersection_ring(center, a1, a2, radius=0.04, ring_radius=0.003):
    """Create a dark ring at the intersection of ball and surface."""
    # Get surface normal to orient the ring
    normal = compute_segre_normal(a1, a2)

    # Create torus for the intersection ring
    bpy.ops.mesh.primitive_torus_add(
        major_radius=radius,
        minor_radius=ring_radius,
        location=center,
        major_segments=32,
        minor_segments=8
    )
    ring = bpy.context.active_object

    # Orient ring to match surface normal
    up = Vector((0, 0, 1))
    normal_vec = Vector(normal)

    if abs(normal_vec.dot(up)) < 0.999:
        rot_axis = up.cross(normal_vec)
        rot_axis.normalize()
        rot_angle = np.arccos(np.clip(up.dot(normal_vec), -1, 1))
        ring.rotation_mode = 'AXIS_ANGLE'
        ring.rotation_axis_angle = (rot_angle, rot_axis.x, rot_axis.y, rot_axis.z)

    # Dark material
    mat = bpy.data.materials.new(name="RingMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs[0].default_value = (0.2, 0.2, 0.2, 1.0)
    emission.inputs[1].default_value = 1.0

    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs[0], output.inputs[0])

    ring.data.materials.append(mat)
    return ring


def create_tetra_edges(radius=0.001, collection=None):
    """Create tetrahedron edges."""
    for i, j in TETRA_EDGES:
        create_tube(TETRA_VERTICES[i], TETRA_VERTICES[j],
                   radius=radius, color=(0.75, 0.75, 0.75, 1.0), collection=collection)


def create_grid_lines(n_lines=5, radius=0.0015, collection=None):
    """Create grid lines on the surface."""
    res = 30
    for i in range(n_lines + 1):
        a = i / n_lines
        pts = [simplex_to_3d(belief_to_simplex(t, a)) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k+1], radius=radius, color=(0.5, 0.5, 0.5, 1.0), collection=collection)

        pts = [simplex_to_3d(belief_to_simplex(a, t)) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k+1], radius=radius, color=(0.5, 0.5, 0.5, 1.0), collection=collection)


def create_surface_boundary(radius=0.006, collection=None):
    """Create boundary edges."""
    res = 40
    boundary_color = (0.15, 0.15, 0.15, 1.0)

    for fixed_val in [0, 1]:
        pts = [simplex_to_3d(belief_to_simplex(fixed_val, t)) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k+1], radius=radius, color=boundary_color, collection=collection)

        pts = [simplex_to_3d(belief_to_simplex(t, fixed_val)) for t in np.linspace(0, 1, res)]
        for k in range(len(pts) - 1):
            create_tube(pts[k], pts[k+1], radius=radius, color=boundary_color, collection=collection)


def setup_scene_3d(elev=20, azim=160):
    """Setup 3D scene with camera and lights."""
    # Render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64  # High quality

    # World
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
    camera.data.ortho_scale = 2.4

    direction = Vector((0, 0, 0)) - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    bpy.context.scene.camera = camera

    # Render resolution (high quality)
    bpy.context.scene.render.resolution_x = 1800
    bpy.context.scene.render.resolution_y = 1600
    bpy.context.scene.render.film_transparent = False

    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'

    # Freestyle for ball outlines
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


def render_3d_view(beliefs, output_path, show_projection_lines=False):
    """Render 3D view of beliefs."""
    clear_scene()
    setup_scene_3d()

    # Tubes collection (excluded from Freestyle)
    tubes_collection = bpy.data.collections.new("Tubes")
    bpy.context.scene.collection.children.link(tubes_collection)

    # Create geometry
    create_segre_surface(res=50, opacity=0.9)
    create_tetra_edges(radius=0.001, collection=tubes_collection)
    create_grid_lines(n_lines=5, radius=0.0015, collection=tubes_collection)
    create_surface_boundary(radius=0.006, collection=tubes_collection)

    # Exclude tubes from Freestyle
    view_layer = bpy.context.scene.view_layers["ViewLayer"]
    freestyle = view_layer.freestyle_settings
    if freestyle.linesets:
        lineset = freestyle.linesets[0]
        lineset.select_by_collection = True
        lineset.collection = tubes_collection
        lineset.collection_negation = 'EXCLUSIVE'

    # Create beliefs
    for b in beliefs:
        p = b['pos3d']
        rgb = pos_to_color(b['alpha1'], b['alpha2'])
        gray = np.mean(rgb)
        rgb_boosted = np.clip(gray + (rgb - gray) * 2.5, 0, 1)

        # Projection line if needed (for off-surface points)
        if show_projection_lines and 'surface_pos' in b:
            sp = b['surface_pos']
            create_tube(p, sp, radius=0.004, color=(0.85, 0.3, 0.3, 1.0), collection=tubes_collection)
        else:
            # For on-surface points, add intersection ring
            create_intersection_ring(p, b['alpha1'], b['alpha2'], radius=0.04, ring_radius=0.004)

        create_sphere(p, radius=0.04, color=(float(rgb_boosted[0]), float(rgb_boosted[1]), float(rgb_boosted[2]), 1.0))

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


def render_2d_square(beliefs, output_path):
    """Render 2D factored square view."""
    clear_scene()

    # Create colored texture
    # Target: yellow (1,1) top-left, orange (1,0) top-right, blue (0,1) bottom-left, teal (0,0) bottom-right
    img_size = 512
    img = bpy.data.images.new("SquareTexture", width=img_size, height=img_size)
    pixels = np.zeros((img_size, img_size, 4))

    for i in range(img_size):
        for j in range(img_size):
            # Blender images have origin at bottom-left, so i=0 is bottom row
            # a1 = 1 at top, 0 at bottom (maps to Y)
            # a2 = 1 at left, 0 at right (maps to -X)
            a1 = i / (img_size - 1)        # bottom=0, top=1
            a2 = 1 - j / (img_size - 1)    # left=1, right=0
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
    grid_color = (0.7, 0.7, 0.7, 1.0)
    for t in [0.25, 0.5, 0.75]:
        x = -1 + t * 2
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0015, depth=2, location=(x, 0, 0.001))
        hline = bpy.context.active_object
        hline.rotation_euler = (np.pi/2, 0, 0)
        grid_mat = bpy.data.materials.new(name="GridMat")
        grid_mat.use_nodes = True
        grid_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = grid_color
        hline.data.materials.append(grid_mat)

        y = -1 + t * 2
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0015, depth=2, location=(0, y, 0.001))
        vline = bpy.context.active_object
        vline.rotation_euler = (0, np.pi/2, 0)
        vline.data.materials.append(grid_mat)

    # Border
    border_color = (0.15, 0.15, 0.15, 1.0)
    border_mat = bpy.data.materials.new(name="BorderMat")
    border_mat.use_nodes = True
    border_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = border_color

    for pos, rot in [
        ((0, -1, 0.002), (0, np.pi/2, 0)),
        ((0, 1, 0.002), (0, np.pi/2, 0)),
        ((-1, 0, 0.002), (np.pi/2, 0, 0)),
        ((1, 0, 0.002), (np.pi/2, 0, 0)),
    ]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=2.02, location=pos)
        border = bpy.context.active_object
        border.rotation_euler = rot
        border.data.materials.append(border_mat)

    # Belief disks
    # Match texture: a1 maps to Y (top=1), a2 maps to -X (left=1)
    for b in beliefs:
        a1 = b['alpha1']
        a2 = b['alpha2']
        x = 1 - a2 * 2  # a2=1 -> x=-1 (left), a2=0 -> x=1 (right)
        y = -1 + a1 * 2  # a1=1 -> y=1 (top), a1=0 -> y=-1 (bottom)

        rgb = pos_to_color(a1, a2)
        gray = np.mean(rgb)
        rgb_boosted = np.clip(gray + (rgb - gray) * 2.5, 0, 1)

        # Outline
        bpy.ops.mesh.primitive_cylinder_add(radius=0.062, depth=0.008, location=(x, y, 0.004))
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

        # Disk
        bpy.ops.mesh.primitive_cylinder_add(radius=0.055, depth=0.008, location=(x, y, 0.006))
        disk = bpy.context.active_object
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

    # Camera (no rotation - orientation handled by coordinate mapping)
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

    # Render settings (high quality)
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 32
    bpy.context.scene.render.resolution_x = 1200
    bpy.context.scene.render.resolution_y = 1200
    bpy.context.scene.view_settings.view_transform = 'Standard'

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command-line arguments (passed after '--' when running via Blender)."""
    parser = argparse.ArgumentParser(description='Render factored belief geometry figures')
    parser.add_argument('--output-dir', type=str, default='png',
                        help='Output subdirectory name (default: png)')
    # Blender passes args after '--'
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return parser.parse_args(argv)


def main():
    args = parse_args()

    # Output directory: base_dir / output_dir_name
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate beliefs
    hmm1 = TwoStateHMM(t_AA=0.7, t_BA=0.3)
    hmm2 = TwoStateHMM(t_AA=0.6, t_BA=0.35)
    max_len = 3
    init_belief = (0.5, 0.5)

    beliefs_ind = compute_beliefs_independent(hmm1, hmm2, max_len, init_belief=init_belief)
    beliefs_dep = compute_beliefs_coupled(hmm1, hmm2, max_len, coupling=-1.5, init_belief=init_belief)
    beliefs_decomp = compute_beliefs_indecomposable_random(n_points=25, offset_strength=0.25, seed=303)

    print(f"Independent: {len(beliefs_ind)} points")
    print(f"Dependent: {len(beliefs_dep)} points")
    print(f"Indecomposable: {len(beliefs_decomp)} points")

    # Render all views
    print("\n=== Rendering Independent ===")
    render_3d_view(beliefs_ind, os.path.join(output_dir, 'blender_independent_3d.png'), show_projection_lines=False)
    render_2d_square(beliefs_ind, os.path.join(output_dir, 'blender_independent_2d.png'))

    print("\n=== Rendering Dependent ===")
    render_3d_view(beliefs_dep, os.path.join(output_dir, 'blender_dependent_3d.png'), show_projection_lines=False)
    render_2d_square(beliefs_dep, os.path.join(output_dir, 'blender_dependent_2d.png'))

    print("\n=== Rendering Indecomposable ===")
    render_3d_view(beliefs_decomp, os.path.join(output_dir, 'blender_indecomposable_3d.png'), show_projection_lines=True)
    render_2d_square(beliefs_decomp, os.path.join(output_dir, 'blender_indecomposable_2d.png'))

    print(f"\n=== All renders saved to {output_dir} ===")


if __name__ == '__main__':
    main()
