FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app:$PYTHONPATH \
    FORCE_CUDA=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    ninja-build \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install core build dependencies first
RUN pip3 install --no-cache-dir ninja setuptools wheel pybind11

# Copy hy3dgen files first
COPY hy3dgen/ ./hy3dgen/
COPY setup.py .

# Set CUDA paths and architecture flags
ENV PATH="/usr/local/cuda/bin:${PATH}" \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

# Install torch and CUDA dependencies first
RUN pip3 install --no-cache-dir torch torchvision

# Install hy3dgen and its custom components
RUN pip3 install -e . && \
    cd /app/hy3dgen/texgen/custom_rasterizer && \
    python3 setup.py install && \
    cd /app && \
    cd /app/hy3dgen/texgen/differentiable_renderer && \
    python3 setup.py install && \
    cd /app

# Install Azure specific requirements
RUN pip3 install --no-cache-dir azure-storage-blob

# Now install remaining requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the image processing code
COPY image_processing.py .

# Create necessary directories
RUN mkdir -p uploads prepped models

# Run the image processing service
CMD ["python3", "image_processing.py"]
