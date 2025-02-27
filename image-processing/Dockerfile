FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app:$PYTHONPATH \
    FORCE_CUDA=1

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

# Install Python packages first (excluding hy3dgen)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Explicitly install torch extension dependencies
RUN pip3 install --no-cache-dir ninja setuptools wheel

# Copy the application code
COPY . .

# Verify CUDA setup
RUN ls -la /usr/local/cuda/bin/nvcc || echo "nvcc not found"
RUN which nvcc || echo "nvcc not in PATH"
RUN echo "CUDA_HOME=$CUDA_HOME"

# Make sure nvcc is in PATH
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install the custom components of hy3dgen as per the README
RUN cd /app/hy3dgen/texgen/custom_rasterizer && \
    python3 setup.py install && \
    cd /app && \
    cd /app/hy3dgen/texgen/differentiable_renderer && \
    python3 setup.py install && \
    cd /app

# Create necessary directories
RUN mkdir -p uploads prepped models

# Run the image processing service
CMD ["python3", "image_processing.py"]
