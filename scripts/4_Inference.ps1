param(
    [string]$Prompt = "A futuristic combat drone with rotating thrusters"
)

# Activate Python environment
C:\Local3DGenProject\.venv\Scripts\Activate.ps1

# Run Point-E inference
python -c "
from point_e.diffusion.configs import DIFFUSION_CONFIGS
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

# Load models
model = load_checkpoint('base40M', device=device)
diffusion = DIFFUSION_CONFIGS['base40M']().to(device)
sampler = PointCloudSampler(model=model, diffusion=diffusion)

# Generate samples
samples = sampler.sample_batch_progressive(
    batch_size=1,
    model_kwargs=dict(texts=[$Prompt])
)

# Save output
output_path = 'C:\Local3DGenProject\assets\raw3D\output.ply'
for sample in samples:
    pc = sampler.output_to_point_clouds(sample)[0]
    pc.write_ply(output_path)
" 