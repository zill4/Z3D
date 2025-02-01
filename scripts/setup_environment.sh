# TODO: Just provide steps for doing this correctly in order.

#!/bin/bash
echo "Creating z3d environment with dependencies..."

# Remove existing environments
conda env remove --name z3d_env
# conda env remove --name realesrgan

# Recreate core environment
conda env create -f environments/z3d_env.yml

# Activate and rebuild custom packages
conda activate z3d_env
pip install --force-reinstall --no-cache-dir -e ./hy3dgen/texgen/custom_rasterizer
pip install --force-reinstall --no-cache-dir -e ./hy3dgen/texgen/differentiable_renderer

# Verify CUDA alignment
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
nvidia-smi  # Should show CUDA 12.7

echo "Verifying installations..."
python -c "import custom_rasterizer; print('custom_rasterizer installed successfully')"
python -c "import differentiable_renderer; print('differentiable_renderer installed successfully')"

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate z3d_env" 