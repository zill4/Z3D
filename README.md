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

---

<div align="center">

*Created with ❤️ by zill4*

</div>
