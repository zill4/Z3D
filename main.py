import os
import base64
import requests
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import trimesh
import torch
from pathlib import Path
from glob import glob
from shutil import copy
from PIL import Image

def get_next_number():
    """Get the next available number for file naming"""
    output_dir = Path('output')
    # Get all numbered folders
    existing_folders = [f for f in output_dir.glob('generation_*') if f.is_dir()]
    if not existing_folders:
        return 1
    # Extract numbers from folder names and find max
    numbers = [int(f.name.split('_')[1]) for f in existing_folders]
    return max(numbers) + 1

# Test image path
TEST_IMAGE = "./images/Heisei_godzilla.png"

# Create output directory if it doesn't exist
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Get next generation number and create its directory
gen_num = get_next_number()
gen_dir = output_dir / f'generation_{gen_num}'
gen_dir.mkdir(exist_ok=True)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and process input image
print("Loading and processing input image...")
image = Image.open(TEST_IMAGE).convert('RGBA')  # Force RGBA format

# Create white background for transparent areas
bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
image = Image.alpha_composite(bg, image)
image = image.convert('RGB')  # Convert back to RGB for processing

# Generate shape first
print("Initializing shape pipeline...")
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')

print("Generating 3D shape...")
mesh = shape_pipeline(image=TEST_IMAGE)[0]

# Modify the mesh processing section
print(f"Original mesh faces: {len(mesh.faces)}")

# Calculate target reduction ratio based on desired face count
target_faces = 80000
reduction_ratio = target_faces / len(mesh.faces)

# Apply decimation with correct parameters
decimated_mesh = mesh.simplify_quadric_decimation(
    face_count=target_faces
)

# Add quality preservation steps (fixed parameters)
decimated_mesh = decimated_mesh.process(validate=True)

# Additional repair steps
decimated_mesh.fill_holes()  # Fill any holes in the mesh
decimated_mesh.remove_degenerate_faces()  # Remove bad triangles
decimated_mesh.remove_duplicate_faces()  # Remove duplicate faces

print(f"Processed mesh faces: {len(decimated_mesh.faces)}")

# Force OBJ export even if texturing fails
print("Saving base mesh...")
decimated_mesh.export(str(gen_dir / "base_mesh.obj"))

# Generate texture using the texture-optimized mesh
print("Initializing texture pipeline...")
paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    texture_resolution=1024,  # Increased from default 512
    texture_quality='high'     # Enable higher quality settings
)

print("Generating texture...")
try:
    textured_mesh = paint_pipeline(decimated_mesh, image=TEST_IMAGE)

    # Transfer texture to high-poly mesh
    print("Transferring texture to high-poly mesh...")
    decimated_mesh.visual = textured_mesh.visual  # Copy texture data

    # Save both versions
    print(f"Saving models to {gen_dir}...")
    decimated_mesh.export(str(gen_dir / "high_poly.glb"))  # 80k faces with texture
    textured_mesh.export(str(gen_dir / "textured.glb"))    # 48k faces with texture

    # Save texture separately if available
    if hasattr(textured_mesh, 'visual') and hasattr(textured_mesh.visual, 'material'):
        if hasattr(textured_mesh.visual.material, 'image'):
            textured_mesh.visual.material.image.save(str(gen_dir / 'texture.png'))
            print("Texture saved separately as texture.png")
        if hasattr(textured_mesh.visual.material, 'baseColorTexture'):
            with open(str(gen_dir / 'texture.png'), 'wb') as f:
                f.write(textured_mesh.visual.material.baseColorTexture)
            print("Base color texture saved as texture.png")

    # Create a copy of the input image for reference
    copy(TEST_IMAGE, str(gen_dir / Path(TEST_IMAGE).name))

    print(f"Done! Generation {gen_num} saved in {gen_dir}")
    print("Files generated:")
    for file in gen_dir.glob('*'):
        print(f"- {file.name}")

except Exception as e:
    print(f"Texture generation failed: {e}")
    decimated_mesh.export(str(gen_dir / "untextured_mesh.obj"))
    print("Saved untextured mesh as fallback")