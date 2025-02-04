import bpy
import sys
from pathlib import Path
import os

def clear_scene():
    """Clear existing scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Clear textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)

def setup_scene():
    """Setup basic scene with good lighting"""
    # Add environment light
    bpy.ops.world.new()
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[1].default_value = 1.0  # Strength
    
    # Add camera
    bpy.ops.object.camera_add(location=(0, -5, 2), rotation=(1.1, 0, 0))
    camera = bpy.context.active_object
    bpy.context.scene.camera = camera
    
    # Add lights
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10), rotation=(0.5, -0.5, 0))
    sun = bpy.context.active_object
    sun.data.energy = 5.0
    
    # Add ground plane
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = 'Ground'

def ensure_addon_enabled():
    """Ensure the OBJ importer addon is enabled"""
    print("\nChecking OBJ importer addon...")
    
    # Check if addon is already registered
    if hasattr(bpy.ops, 'import_scene') and hasattr(bpy.ops.import_scene, 'obj'):
        print("OBJ importer addon already enabled")
        return True
        
    print("OBJ importer not found, attempting to use built-in import...")
    return False

def check_blender_version():
    """Check Blender version and available import methods"""
    print(f"\nBlender Version: {bpy.app.version_string}")
    print(f"Blender API (bpy) Version: {'.'.join(str(v) for v in bpy.app.version)}")
    
    # Check available import methods
    has_legacy_import = hasattr(bpy.ops, 'import_scene') and hasattr(bpy.ops.import_scene, 'obj')
    has_new_import = hasattr(bpy.ops, 'wm') and hasattr(bpy.ops.wm, 'obj_import')
    
    print(f"Legacy OBJ import available: {has_legacy_import}")
    print(f"New OBJ import available: {has_new_import}")
    
    return has_legacy_import, has_new_import

def setup_materials(obj, generation_dir):
    """Ensure materials are properly set up with textures"""
    for mat in obj.data.materials:
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Clear existing nodes
        nodes.clear()
        
        # Create basic nodes
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        output = nodes.new('ShaderNodeOutputMaterial')
        tex_image = nodes.new('ShaderNodeTexImage')
        
        # Link nodes
        mat.node_tree.links.new(tex_image.outputs[0], principled.inputs[0])  # Base Color
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        
        # Load texture
        if 'upscaled' in obj.name:
            tex_path = generation_dir / 'upscaled.png'
        else:
            tex_path = generation_dir / 'original_texture.png'
            
        if tex_path.exists():
            tex_image.image = bpy.data.images.load(str(tex_path))
            print(f"Loaded texture: {tex_path}")

def load_model(generation_dir):
    """Load both original and upscaled versions for comparison"""
    generation_dir = Path(generation_dir)
    
    # Load upscaled version
    upscaled_path = generation_dir / "textured.obj"
    if upscaled_path.exists():
        bpy.ops.wm.obj_import(
            filepath=str(upscaled_path),
            forward_axis='Y',
            up_axis='Z'
        )
        obj = bpy.context.selected_objects[0]
        obj.name = 'upscaled'
        setup_materials(obj, generation_dir)
        print(f"Loaded upscaled model from {upscaled_path}")
    
    # Load original version offset to the side
    original_path = generation_dir / "textured_original.obj"
    if original_path.exists():
        bpy.ops.wm.obj_import(
            filepath=str(original_path),
            forward_axis='Y',
            up_axis='Z'
        )
        # Move the second model to the side
        bpy.context.selected_objects[0].location.x += 3
        print(f"Loaded original model from {original_path}")
    else:
        print(f"Warning: Original model not found at {original_path}")

def main():
    # Get script directory
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    
    # Get generation directory from command line or use latest
    if len(sys.argv) > 4:  # Blender adds its own args, so check for more than default
        generation_dir = Path(sys.argv[-1]).resolve()  # Use last argument
    else:
        output_dir = project_dir / 'output'
        print(f"\nSearching for generations in: {output_dir}")
        
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found at {output_dir}")
            
        generations = sorted(output_dir.glob('generation_*'))
        print(f"Found generations: {[g.name for g in generations]}")
        
        if not generations:
            raise FileNotFoundError("No generations found in output directory")
            
        generation_dir = generations[-1]
    
    print(f"\nLoading from: {generation_dir}")
    
    # Verify it's a directory and exists
    if not generation_dir.is_dir():
        raise NotADirectoryError(f"Invalid generation directory: {generation_dir}")
    
    # Clear existing scene
    clear_scene()
    
    # Setup scene
    setup_scene()
    
    # Load model
    obj = load_model(generation_dir)
    
    # Setup viewport shading
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
    
    # Save blend file
    blend_path = generation_dir / 'viewer.blend'
    print(f"\nSaving Blender file to: {blend_path}")
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path.resolve()))
    print("Save complete!")

if __name__ == "__main__":
    main() 