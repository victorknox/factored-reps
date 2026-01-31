"""
Minimal Blender script to render a white background with a colored emission sphere.
Run with: /Applications/Blender.app/Contents/MacOS/Blender --background --python experiments/figure_generation/figure1/scripts/blender_test.py
"""
import argparse
import os
import sys

import bpy


def parse_args():
    """Parse command-line arguments (passed after '--' when running via Blender)."""
    parser = argparse.ArgumentParser(description='Render test sphere')
    parser.add_argument('--output-dir', type=str, default='png',
                        help='Output subdirectory name (default: png)')
    # Blender passes args after '--'
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return parser.parse_args(argv)


args = parse_args()

# Get output path (absolute)
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
output_dir = os.path.join(base_dir, args.output_dir)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "blender_test.png")

print(f"Output will be saved to: {output_path}")

# Clear default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Get scene
scene = bpy.context.scene

# ============================================
# 1. SET UP WHITE BACKGROUND
# ============================================
world = scene.world
if world is None:
    world = bpy.data.worlds.new("World")
    scene.world = world

world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links

# Clear existing nodes
nodes.clear()

# Create Background node - pure white
bg_node = nodes.new(type='ShaderNodeBackground')
bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
bg_node.inputs['Strength'].default_value = 1.0
bg_node.location = (0, 0)

# Create World Output node
output_node = nodes.new(type='ShaderNodeOutputWorld')
output_node.location = (200, 0)

# Link Background to World Output
links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

# ============================================
# 2. CREATE VIBRANT COLORED SPHERE
# ============================================
bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0), segments=32, ring_count=16)
sphere = bpy.context.active_object

# Create emission material (vibrant bright red)
mat = bpy.data.materials.new(name="EmissionRed")
mat.use_nodes = True
mat_nodes = mat.node_tree.nodes
mat_links = mat.node_tree.links

# Clear default nodes
mat_nodes.clear()

# Add emission shader
emission = mat_nodes.new('ShaderNodeEmission')
emission.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1.0)  # Bright red
emission.inputs['Strength'].default_value = 1.0
emission.location = (0, 0)

# Add output node
mat_output = mat_nodes.new('ShaderNodeOutputMaterial')
mat_output.location = (200, 0)

# Connect emission to output
mat_links.new(emission.outputs['Emission'], mat_output.inputs['Surface'])

# Assign material to sphere
sphere.data.materials.append(mat)

# ============================================
# 3. SET UP CAMERA
# ============================================
bpy.ops.object.camera_add(location=(4, -4, 3))
camera = bpy.context.active_object
camera.rotation_euler = (1.1, 0, 0.8)
scene.camera = camera

# ============================================
# 4. RENDER SETTINGS
# ============================================
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32
scene.render.resolution_x = 800
scene.render.resolution_y = 600
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.filepath = output_path

# Color management - use Standard view transform for accurate colors
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'

# ============================================
# 5. RENDER
# ============================================
bpy.ops.render.render(write_still=True)

print(f"Render complete! Output saved to: {output_path}")
