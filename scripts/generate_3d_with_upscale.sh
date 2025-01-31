#!/bin/bash
set -e  # Exit immediately if any command fails

# 1. Generate initial 3D model and texture using z3d_env
echo "Step 1/3: Generating 3D model..."
conda run -n z3d_env python src/hunyuan3d/main.py \
    --input_text "$1" \
    --output_dir ./intermediate_models \
    --texture_size 1024

# 2. Upscale texture using realesrgan environment
echo "Step 2/3: Upscaling texture..."
conda run -n realesrgan python src/upscaler/test_upscaler.py \
    --input ./intermediate_models/texture.png \
    --output_dir ./intermediate_models \
    --output_filename texture_4k.png

# 3. Update 3D model with upscaled texture using z3d_env
echo "Step 3/3: Updating 3D model material..."
conda run -n z3d_env python src/utils/update_material.py \
    --model ./intermediate_models/model.glb \
    --texture ./intermediate_models/texture_4k.png \
    --output ./final_models

echo "Pipeline complete! Final model in ./final_models"