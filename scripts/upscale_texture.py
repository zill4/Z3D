import argparse
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import cv2
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    # Initialize model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    upsampler = RealESRGANer(
        scale=4,
        model_path='models/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True
    )
    
    # Load and process image
    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=4)
    
    # Save result
    cv2.imwrite(args.output, output)

if __name__ == '__main__':
    main() 