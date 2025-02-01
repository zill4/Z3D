# Z3D - Advanced Image-to-3D Generation Pipeline 🎮

<div align="center">

A complete local pipeline for generating rigged 3D assets from text prompts. Powered by DeepSeek-R1.

```
__________________  ________   
\____    /\_____  \ \______ \  
  /     /   _(__  <  |    |  \ 
 /     /_  /       \ |    `   \
/_______ \/______  //_______  /
        \/       \/         \/
```

## 🛠️ Technical Stack

- **3D Generation**: Hunyuan3D-2
- **Texture Upscaling**: RealESRGAN
- **Visualization**: Ursina Engine
- **Core Dependencies**: PyTorch, Trimesh

## 💻 System Requirements

- NVIDIA GPU with 24.5GB+ VRAM (for full pipeline)
- CUDA 11.8+
- Windows 10/11
- 32GB RAM recommended

## 🚀 Quick Start

1. Install prerequisites listed above
2. Clone repository:

```
realesrgan install steps
# 1. Clean start
conda deactivate
conda env remove -n realesrgan

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

```

## 🛠️ Validation Suite
```powershell
# Run comprehensive system check
.\Validate-3DWorkflow.ps1 -Verbose

# Successful output shows:
[PASS] Core dependencies verified
[PASS] Repository structure valid  
[PASS] Basic generation successful
```

## 📂 Project Anatomy
```bash
z3d-pipeline/
├── pipelines/       # Modular workflow stages
├── asset_library/   # Prebuilt templates
├── runtime/         # Engine integration
├── docs/           # Technical documentation
└── tools/          # Conversion utilities
```

## 🚨 Common Solutions
| Issue                  | Resolution                         |
|------------------------|------------------------------------|
| CUDA_PATH not found    | Re-run setup script                |
| Python import errors   | Verify virtual environment         |
| Partial generations    | Check VRAM allocation (>10GB free)|

## 📅 Roadmap
```asciidoc
Q3 2024 :: Basic rigging system
Q4 2024 :: Blender integration
Q1 2025 :: Unreal Engine plugin
```

Licensed under MIT :: Contributions welcome
Maintained by zill4 | Powered by DeepSeek-R1

> **Note** This is Phase I focusing on core generation capabilities. Phase II will introduce animation systems and game engine integration.

## 📊 Generation Pipeline

1. **Mesh Generation**
   - Input: Reference image
   - Output: High-poly mesh (250k+ faces)
   - Format: OBJ/GLB

2. **Texture Processing**
   - Base texture generation (1024x1024)
   - AI upscaling to 4K resolution
   - Multiple format support (PNG/RGBA)

3. **Visualization**
   - Interactive gallery viewer
   - Multiple rendering techniques
   - Real-time comparison tools

## 🎯 Current Progress

- [x] Base mesh generation pipeline
- [x] Texture synthesis and upscaling
- [x] Interactive viewer implementation
- [x] Multiple rendering methods
- [ ] Automated rigging system (Planned)
- [ ] Animation support (Planned)
- [ ] Game engine export pipeline (Planned)

## 🔍 Debug Tools

- Real-time position tracking
- Texture loading verification
- Model comparison views
- Performance monitoring

## 📝 License

MIT License - See LICENSE file for details

---

<div align="center">

*Created with ❤️ by zill4*

</div>
