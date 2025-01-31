import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image

# Add Real_ESRGAN to path
repo_path = os.path.join(os.path.dirname(__file__), 'Real_ESRGAN')
sys.path.insert(0, repo_path)

# Import from the realesrgan package
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def upscale_texture(img):
    """
    Upscale an image using RealESRGAN.
    Args:
        img: PIL Image to upscale
    Returns:
        Upscaled image as numpy array
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    
    upsampler = RealESRGANer(
        scale=4,
        model_path='Real_ESRGAN/experiments/pretrained_models/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )
    
    # Convert PIL to numpy array
    input_array = np.array(img)
    
    # Upscale
    output, _ = upsampler.enhance(input_array)
    
    return output

def test_upscaling(input_path, output_dir, output_filename):
    """Test the upscaler with a real texture file"""
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test texture
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input texture not found at: {input_path}")
            
        print(f"Upscaling texture: {input_path}")
        test_img = Image.open(input_path)
        
        # Upscale
        result = upscale_texture(test_img)
        
        # Save results
        output_path = output_dir / output_filename
        Image.fromarray(result).save(output_path)
        print(f"Saved upscaled texture to: {output_path}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error during upscaling test: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upscale textures using RealESRGAN')
    parser.add_argument('--input', type=str, default='texture.png', 
                       help='Path to input texture file')
    parser.add_argument('--output_dir', type=str, default='upscaled_outputs',
                       help='Directory to save upscaled textures')
    parser.add_argument('--output_filename', type=str, default='texture_4x.png',
                       help='Filename for upscaled texture')
    
    args = parser.parse_args()
    
    try:
        test_upscaling(
            input_path=args.input,
            output_dir=args.output_dir,
            output_filename=args.output_filename
        )
    except Exception as e:
        print(f"Texture upscaling failed: {str(e)}")
        sys.exit(1)