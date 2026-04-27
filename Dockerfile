# Use PyTorch official image as base
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic tools
    wget \
    git \
    vim \
    curl \
    cmake \
    # OpenCV required system libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxcb-shm0 \
    # Image format libraries
    libpng16-16 \
    libjpeg-turbo8 \
    libjpeg8 \
    libtiff5 \
    # Build tools
    build-essential \
    # Zstd for data decompression
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to use Tsinghua mirror
RUN mkdir -p /root/.pip && \
    echo "[global]\n\
index-url = https://pypi.tuna.tsinghua.edu.cn/simple\n\
trusted-host = pypi.tuna.tsinghua.edu.cn" > /root/.pip/pip.conf

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Downgrade setuptools to fix gym 0.21 compatibility issue
RUN pip install --no-cache-dir setuptools==65.5.0 pip==21

# Install Python packages for LeWorldModel
RUN pip install --no-cache-dir \
    stable-worldmodel[train,env] \
    hydra-core \
    wandb \
    h5py \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    jupyterlab \
    notebook \
    tensorboard \
    tqdm \
    pillow \
    requests \
    einops \
    timm \
    opencv-python-headless

# Upgrade pip back for future use
RUN pip install --no-cache-dir --upgrade pip

# Set default STABLEWM_HOME (will be overridden by docker run)
ENV STABLEWM_HOME=/workspace/.stable-wm

# Create stable-wm directory (as fallback)
RUN mkdir -p $STABLEWM_HOME

# Create mount point for host data
RUN mkdir -p /host_data

# Verify CUDA environment
RUN nvcc --version && \
    echo "CUDA_HOME: ${CUDA_HOME}" && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Set default working directory to project
WORKDIR /workspace/le-wm
