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
import subprocess

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
    textured_mesh = paint_pipeline(processed_mesh, image=TEST_IMAGE)
    
    # Save original texture
    original_texture_path = gen_dir / 'original_texture.png'
    Image.fromarray(textured_mesh.visual.material.image).save(original_texture_path)

    # Upscale using external process
    try:
        upscaled_path = gen_dir / 'texture_4k.png'
        subprocess.run([
            'conda', 'run', '-n', 'realesrgan',
            'python', 'src/upscaler/test_upscaler.py',
            '--input', str(original_texture_path),
            '--output_dir', str(gen_dir),
            '--output_filename', 'texture_4k.png'
        ], check=True)
        
        # Load upscaled texture
        textured_mesh.visual.material.image = Image.open(upscaled_path)
    except subprocess.CalledProcessError as e:
        print(f"Upscaling failed, using original texture: {e}")
    
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
    print(f"Texture generation failed: {e}")
    processed_mesh.export(str(gen_dir / "untextured_mesh.glb"))
    print("Saved untextured mesh as fallback")

def upscale_texture(img):
    """Custom SwinIR-based upscaler"""
    import torch
    from torchvision.transforms import ToTensor
    from swinir import SwinIR  # Changed from relative import
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path('models/SwinIR/SwinIR_4x.pth')
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"SwinIR model not found at {model_path}. "
            "Run download_swinir_model.py first!"
        )
    
    # Load model
    model = SwinIR(
        upscale=4,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Convert input to tensor
    if isinstance(img, Image.Image):
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    else:
        img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device).float()/255.0
    
    # Process
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert back to numpy array
    upscaled = output.squeeze().permute(1,2,0).clamp(0,1).cpu().numpy()
    return (upscaled * 255).astype('uint8')