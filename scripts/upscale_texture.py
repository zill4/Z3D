import argparse
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import torch
import torchvision
from torchvision.transforms import functional as F  # Updated import path
import os
import subprocess
from pathlib import Path
import shutil
import sys

torchvision.disable_beta_transforms_warning()  # Suppress version warnings

# Global model path
MODELS_DIR = Path('models')
ESRGAN_MODEL = 'RealESRGAN_x4plus.pth'

def get_model_path():
    """Get the path to the RealESRGAN model"""
    model_path = MODELS_DIR / ESRGAN_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"RealESRGAN model not found at {model_path}. Please run main.py first.")
    return model_path

def ensure_model(cleanup_old=True):
    """Ensure model exists and cleanup old versions if needed"""
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / ESRGAN_MODEL
    
    if model_path.exists():
        print(f"Using existing model at {model_path}")
        return model_path
        
    print("Downloading RealESRGAN model...")
    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    torch.hub.download_url_to_file(url, str(model_path))
    
    if cleanup_old:
        # Remove any other .pth files in the models directory
        for file in MODELS_DIR.glob('*.pth'):
            if file.name != ESRGAN_MODEL:
                print(f"Removing old model: {file}")
                file.unlink()
    
    return model_path

def upscale_texture(texture_path: Path, output_path: Path, original_size: int, target_size: int) -> bool:
    """Upscale texture using RealESRGAN directly"""
    print(f"Upscaling: {texture_path} → {output_path}")
    print(f"Scaling from {original_size}x{original_size} to {target_size}x{target_size}")
    
    if not texture_path.exists():
        print(f"Error: Input file {texture_path} not found!")
        return False

    try:
        # Get model path
        model_path = ensure_model()
        
        # Calculate scale factor
        scale_factor = target_size // original_size
        print(f"Using scale factor: {scale_factor}x")
        
        # Initialize upsampler
        upsampler = RealESRGANer(
            scale=scale_factor,
            model_path=str(model_path),
            model=RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=23, 
                num_grow_ch=32, 
                scale=scale_factor
            ),
            tile=original_size,
            tile_pad=10,
            pre_pad=0,
            half=False
        )
        
        # Load and process image
        img = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {texture_path}")
        print(f"Loaded image shape: {img.shape}")
        
        # Upscale
        output, _ = upsampler.enhance(img, outscale=scale_factor)
        print(f"Output image shape: {output.shape}")
        
        # Save result
        cv2.imwrite(str(output_path), output)
        print(f"Saved upscaled texture to {output_path}")
        return True

    except Exception as e:
        print(f"Upscaling failed: {str(e)}")
        return False

def main():
    """Main entry point for upscaling script"""
    parser = argparse.ArgumentParser(description='Upscale texture using RealESRGAN')
    parser.add_argument('--input', type=str, required=True, help='Input texture path')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    parser.add_argument('--original-size', type=int, default=1024, help='Original texture size')
    parser.add_argument('--target-size', type=int, default=4096, help='Target texture size')
    args = parser.parse_args()

    try:
        # Get model path (downloads only if needed)
        model_path = ensure_model()
        
        # Calculate scale factor
        scale_factor = args.target_size // args.original_size
        
        # Initialize upsampler with explicit device management
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=scale_factor
        ).to(device)
        
        upsampler = RealESRGANer(
            scale=scale_factor,
            model_path=str(model_path),
            model=model,
            tile=args.original_size,
            tile_pad=10,
            pre_pad=0,
            half=True  # Use half precision to reduce memory usage
        )
        
        # Load and process image
        img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image from {args.input}")
        
        # Upscale
        output, _ = upsampler.enhance(img, outscale=scale_factor)
        
        # Save result
        cv2.imwrite(args.output, output)
        print(f"Successfully saved upscaled texture to {args.output}")
        
        # Explicit cleanup
        del upsampler
        del model
        torch.cuda.empty_cache()
        
        # Exit immediately after successful upscale
        sys.exit(0)
        
    except Exception as e:
        print(f"Upscaling failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 