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
import stat

def ensure_write_permissions(path):
    """Ensure write permissions on Windows"""
    if os.name == 'nt':  # Windows
        try:
            os.chmod(path, stat.S_IWRITE)
        except OSError:
            pass

def upscale_texture(texture_path: Path, output_path: Path, original_size: int, target_size: int) -> bool:
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

        # Directly call the upscaler script with proper paths (fixed argument names)
        result = subprocess.run([
            str(conda_executable), 'run', '-n', 'realesrgan',
            'python', 'scripts/upscale_texture.py',
            '--input', str(texture_path.resolve()),
            '--output', str(output_path.resolve()),
            '--original-size', str(original_size),  # Changed from original_size
            '--target-size', str(target_size)      # Changed from target_size
        ], check=True, capture_output=True, text=True, timeout=300)

        # Print output for debugging
        if result.stdout:
            print("Upscaler output:")
            print(result.stdout)
        if result.stderr:
            print("Upscaler errors:")
            print(result.stderr)

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
    """Ensure RealESRGAN model exists"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'RealESRGAN_x4plus.pth'
    
    # Check if model exists in repo first
    repo_model = Path('models/RealESRGAN_x4plus.pth')
    if repo_model.exists():
        print(f"Using RealESRGAN model from repo")
        if not model_path.exists():
            shutil.copy2(repo_model, model_path)
        return model_path
    
    # If not in repo, download only if needed
    if not model_path.exists():
        print("Downloading RealESRGAN model...")
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        torch.hub.download_url_to_file(url, str(model_path))
        print(f"Model downloaded to {model_path}")
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
TEST_IMAGE = "./images/cyber_kunoichi.png"

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
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    cache_dir='models/hunyuan/shape_pipeline'
)

print("Generating 3D shape...")
mesh = shape_pipeline(image=TEST_IMAGE)[0]

# Before reduction, print original face count
print(f"Original mesh faces: {len(mesh.faces)}")

# Apply the same processing as the API server
processed_mesh = FloaterRemover()(mesh)                # Remove floating artifacts
processed_mesh = DegenerateFaceRemover()(processed_mesh)  # Remove bad triangles
# processed_mesh = FaceReducer()(processed_mesh)  # Remove max_facenum parameter to keep all valid faces

# Additional quality checks
processed_mesh = processed_mesh.process(validate=True)  # Validate mesh
processed_mesh.fill_holes()                           # Fill any remaining holes

print(f"Processed mesh faces: {len(processed_mesh.faces)}")

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
    # Generate and save base mesh
    processed_mesh.export(
        str(gen_dir / "base_mesh.obj"),
        include_normals=True,
        include_texture=True
    )

    # Generate texture
    textured_mesh = paint_pipeline(processed_mesh, image=TEST_IMAGE)
    
    # Save original texture
    texture_path = gen_dir / 'original_texture.png'
    if isinstance(textured_mesh.visual.material.image, (Image.Image, np.ndarray)):
        img = textured_mesh.visual.material.image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img.save(texture_path)
    
    # Upscale texture
    upscaled_path = gen_dir / 'upscaled.png'
    if upscale_texture(
        texture_path=texture_path,
        output_path=upscaled_path,
        original_size=paint_pipeline.view_size,
        target_size=paint_pipeline.texture_size
    ):
        print("Upscale successful!")
        # Only try to load upscaled texture if upscaling succeeded
        if upscaled_path.exists():  # Double check file exists
            try:
                textured_mesh.visual.material.image = Image.open(upscaled_path)
                print(f"Successfully loaded upscaled texture from {upscaled_path}")
            except Exception as e:
                print(f"Failed to load upscaled texture: {e}")
                print("Falling back to original texture")
    else:
        print("Upscaling failed, using original texture")

    # Save final textured mesh with whatever texture was loaded
    textured_mesh.export(
        str(gen_dir / "textured.obj"),
        include_normals=True,
        include_texture=True,
        resolver=None,
        mtl_name='textured.mtl'
    )

    # Save an additional OBJ/MTL pair with original texture
    textured_mesh.visual.material.image = Image.open(texture_path)  # Load original texture
    textured_mesh.export(
        str(gen_dir / "textured_original.obj"),
        include_normals=True,
        include_texture=True,
        resolver=None,
        mtl_name='textured_original.mtl'
    )

    # Clean up any unnecessary files
    for file in gen_dir.glob('*'):
        if file.name not in [
            'base_mesh.obj',
            'textured.obj',
            'textured.mtl',
            'textured_original.obj',
            'textured_original.mtl',
            'original_texture.png',
            'upscaled.png'
        ]:
            ensure_write_permissions(file)
            try:
                file.unlink()
            except PermissionError as e:
                print(f"Warning: Could not remove {file}: {e}")

except Exception as e:
    print(f"Texture generation failed with error: {str(e)}")
    print(f"Error type: {type(e)}")
    import traceback
    print(f"Full traceback:\n{traceback.format_exc()}")
    processed_mesh.export(str(gen_dir / "untextured_mesh.glb"))
    print("Saved untextured mesh as fallback")

# Add this near the start of your script, before any processing
ensure_model_downloaded()

def ensure_hunyuan_models():
    """Ensure Hunyuan3D models are downloaded only once"""
    models_dir = Path('models/hunyuan')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Define model paths
    shape_model = models_dir / 'shape_pipeline'
    
    if not shape_model.exists():
        print("Downloading Hunyuan3D shape model (this may take a while)...")
        # Only use cache_dir for shape pipeline
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2',
            cache_dir=str(shape_model)
        )
        print("Shape model downloaded successfully!")
    else:
        print("Using existing shape model")

# Add this near the start of your script
ensure_hunyuan_models()