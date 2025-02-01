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

torchvision.disable_beta_transforms_warning()  # Suppress version warnings

def upscale_texture(texture_path: Path, output_path: Path) -> bool:
    """Upscale texture using RealESRGAN with absolute paths"""
    print(f"Upscaling: {texture_path} → {output_path}")
    
    if not texture_path.exists():
        print(f"Error: Input file {texture_path} not found!")
        return False

    try:
        # Use absolute paths to avoid directory confusion
        conda_exe = Path(subprocess.check_output(["conda", "info", "--base"]).decode().strip()) / "Scripts/conda.exe"
        
        result = subprocess.run([
            str(conda_exe), "run", "-n", "realesrgan",
            "python", "scripts/upscale_texture.py",
            "--input", str(texture_path.resolve()),
            "--output", str(output_path.resolve())
        ], capture_output=True, text=True, check=True)

        print("Upscaling output:", result.stdout)
        return output_path.exists()

    except subprocess.CalledProcessError as e:
        print(f"Upscaling failed: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input texture path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Initialize ESRGAN upsampler
    model_path = os.path.join('Real_ESRGAN', 'weights', 'RealESRGAN_x4plus.pth')
    
    # Check if model exists, if not download it
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        print(f'Downloading model to {model_path}...')
        torch.hub.download_url_to_file(url, model_path)

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False  # Disable half precision for compatibility
    )
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Torch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    
    # Load and process image
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {args.input}")
    print(f"Loaded image shape: {img.shape}")
    
    output, _ = upsampler.enhance(img, outscale=4)
    print(f"Output image shape: {output.shape}")
    
    # Save result
    output_path = os.path.join(args.output, os.path.basename(args.input))
    cv2.imwrite(output_path, output)
    print(f"Saved output to {output_path}")

    # Verify texture exists before upscaling
    texture_path = Path(args.input)
    upscaled_path = Path(args.output) / f"{texture_path.stem}_upscaled{texture_path.suffix}"
    if texture_path.exists():
        if upscale_texture(texture_path, upscaled_path):
            print("Upscale successful!")
        else:
            print("Using original texture due to upscaling failure")
    else:
        print(f"Critical error: Original texture missing at {texture_path}")

if __name__ == '__main__':
    main() 