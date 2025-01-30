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
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
import numpy as np

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

# Modified mesh processing section
print(f"Original mesh faces: {len(mesh.faces)}")

# Apply the same processing as the API server
processed_mesh = FloaterRemover()(mesh)                # Remove floating artifacts
processed_mesh = DegenerateFaceRemover()(processed_mesh)  # Remove bad triangles
processed_mesh = FaceReducer()(processed_mesh, max_facenum=218824)  # Reduce to 40k faces, changed to 218824 (base model size)

# Additional quality checks
processed_mesh = processed_mesh.process(validate=True)  # Validate mesh
processed_mesh.fill_holes()                           # Fill any remaining holes

print(f"Processed mesh faces: {len(processed_mesh.faces)}")

# Save processed base mesh
print("Saving processed base mesh...")
# Export OBJ with UV coordinates
processed_mesh.export(
    str(gen_dir / "base_mesh.obj"),
    include_normals=True,
    include_texture=True,  # For UV coordinates
    resolver=None,  # For material handling
    mtl_name=str(gen_dir / 'material.mtl')  # Material file name
)

# Generate texture using the processed mesh
print("Initializing texture pipeline...")
paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

print("Generating texture...")
try:
    # Apply texture to processed mesh
    textured_mesh = paint_pipeline(processed_mesh, image=TEST_IMAGE)
    
    # Save textured model
    print(f"Saving models to {gen_dir}...")
    # Export GLB with embedded textures
    textured_mesh.export(
        str(gen_dir / "textured.glb"),
        file_type='glb'
    )
    
    # Export material information
    if hasattr(textured_mesh.visual, 'material'):
        # Save material properties with standard values
        material_props = {
            'Ka': [0.2, 0.2, 0.2],  # ambient
            'Kd': [0.8, 0.8, 0.8],  # diffuse
            'Ks': [0.0, 0.0, 0.0],  # specular
            'Ns': 1.0,              # specular coefficient
            'd': 1.0,               # transparency
            'map_Kd': 'texture.png' # diffuse texture map
        }
        # Save as MTL file
        with open(str(gen_dir / 'material.mtl'), 'w') as f:
            f.write(f"newmtl material0\n")
            for key, value in material_props.items():
                if isinstance(value, list):
                    value = ' '.join(map(str, value))
                f.write(f"{key} {value}\n")
        
        # Save texture with specific format
        if hasattr(textured_mesh.visual.material, 'image'):
            img = textured_mesh.visual.material.image
            # Check if img is a PIL Image and convert to numpy array if needed
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            # Convert to RGBA
            if len(img.shape) == 3 and img.shape[2] == 3:  # If RGB
                alpha = np.ones((img.shape[0], img.shape[1], 1)) * 255
                img = np.concatenate([img, alpha], axis=2)
            elif len(img.shape) == 2:  # If grayscale
                img = np.stack([img, img, img, np.ones_like(img) * 255], axis=-1)
            
            # Convert back to PIL and save
            Image.fromarray(img.astype(np.uint8)).save(str(gen_dir / 'texture.png'))
            print("Texture saved with material properties")
    
    # Also save an OBJ version of the textured mesh
    textured_mesh.export(
        str(gen_dir / "textured.obj"),
        include_normals=True,
        include_texture=True,
        resolver=None,
        mtl_name='textured_material.mtl'  # Changed to relative path
    )
    print("Saved both GLB and OBJ versions of textured mesh")
            
    # Create input image reference
    copy(TEST_IMAGE, str(gen_dir / Path(TEST_IMAGE).name))

except Exception as e:
    print(f"Texture generation failed: {e}")
    processed_mesh.export(str(gen_dir / "untextured_mesh.glb"))
    print("Saved untextured mesh as fallback")