import torch
from pathlib import Path
from glob import glob
from shutil import copy
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.texgen import Hunyuan3DPaintPipeline
import numpy as np
import subprocess
import sys
import os
import time

def upscale_texture(texture_path: Path, output_path: Path) -> bool:
    """Upscale texture using RealESRGAN with proper synchronization"""
    print(f"Starting texture upscaling from {texture_path} to {output_path}")
    
    try:
        # Get conda executable reliably across platforms
        conda_executable = None
        try:
            if sys.platform == "win32":
                conda_executable = Path(subprocess.check_output(["where", "conda"], shell=True).decode().split()[0])
            else:
                conda_executable = Path(subprocess.check_output(["which", "conda"]).decode().strip())
        except (subprocess.CalledProcessError, IndexError):
            conda_executable = Path(sys.executable).parent.parent / "Scripts" / "conda.exe"

        if not conda_executable.exists():
            raise FileNotFoundError(f"Conda executable not found at {conda_executable}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Directly call the upscaler script with proper paths
        result = subprocess.run([
            str(conda_executable), 'run', '-n', 'realesrgan',
            'python', 'scripts/upscale_texture.py',
            '--input', str(texture_path.resolve()),
            '--output', str(output_path.resolve())
        ], check=True, capture_output=True, text=True, timeout=300)

        print(f"Upscaling completed with status: {result.returncode}")
        
        if output_path.exists():
            print(f"Successfully upscaled texture to {output_path}")
            return True
            
        print(f"Upscaling completed but output missing at {output_path}")
        return False
        
    except subprocess.CalledProcessError as e:
        print(f"Upscaling failed with error code {e.returncode}")
        print(f"Error output:\n{e.stderr}")
        print(f"Standard output:\n{e.stdout}")
        return False
    except Exception as e:
        print(f"Unexpected error during upscaling: {str(e)}")
        return False

def ensure_model_downloaded():
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'RealESRGAN_x4plus.pth'
    if not model_path.exists():
        print("Please Download RealESRGAN model")
        # import gdown
        # gdown.download(
        #     'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        #     str(model_path)
        # )
    return model_path

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
processed_mesh = FaceReducer()(processed_mesh, max_facenum=251436)  # Reduce to 40k faces, changed to 218824 (base model size)

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

# Configure pipeline parameters through the modified class
paint_pipeline.view_size = 1024
paint_pipeline.texture_size = 4096

# Configure intermediate saves
intermediate_dir = gen_dir / 'intermediate_textures'
intermediate_dir.mkdir(exist_ok=True)
paint_pipeline.set_intermediate_dir(str(intermediate_dir))

print("Generating texture...")
try:
    # Apply texture to processed mesh
    print("Starting texture generation with paint_pipeline...")
    textured_mesh = paint_pipeline(processed_mesh, image=TEST_IMAGE)
    print("Texture generation completed successfully")
    
    # Save original texture
    texture_path = gen_dir / 'original_texture.png'
    print(f"Saving original texture to {texture_path}")
    
    # Handle texture saving based on type
    if isinstance(textured_mesh.visual.material.image, Image.Image):
        print("Texture is already a PIL Image, saving directly")
        textured_mesh.visual.material.image.save(texture_path)
    elif isinstance(textured_mesh.visual.material.image, np.ndarray):
        print("Texture is numpy array, converting to PIL Image")
        Image.fromarray(textured_mesh.visual.material.image).save(texture_path)
    else:
        print(f"Unexpected texture type: {type(textured_mesh.visual.material.image)}")
        raise TypeError(f"Unexpected texture type: {type(textured_mesh.visual.material.image)}")

    # Verify texture was saved
    if not texture_path.exists():
        raise FileNotFoundError(f"Failed to save texture to {texture_path}")
    print(f"Successfully saved texture to {texture_path}")

    # Attempt to upscale texture
    print("Attempting to upscale texture...")
    upscaled_path = gen_dir / 'upscaled_texture.png'

    # Add synchronization point
    if texture_path.exists():
        print(f"Found original texture at {texture_path}")
        
        # Wait for file to be fully written
        for _ in range(10):
            try:
                with texture_path.open('rb') as f:
                    f.seek(-2, 2)
                    break
            except IOError:
                time.sleep(0.5)
        else:
            print("Timeout waiting for texture file to be ready")
        
        # Verify file can be opened
        try:
            Image.open(texture_path).verify()
        except Exception as e:
            print(f"Invalid texture file: {e}")
            texture_path.unlink(missing_ok=True)
        
        if upscale_texture(texture_path, upscaled_path):
            print("Upscale successful!")
        else:
            print("Using original texture due to upscaling failure")
    else:
        print(f"Critical error: Original texture missing at {texture_path}")
    
    # Save textured model with upscaled texture
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

    # Add this before and after upscaling to monitor VRAM
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")

except Exception as e:
    print(f"Texture generation failed with error: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Full traceback:\n{traceback.format_exc()}")
    processed_mesh.export(str(gen_dir / "untextured_mesh.glb"))
    print("Saved untextured mesh as fallback")