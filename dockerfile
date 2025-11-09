FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_INSTALL="https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
ARG TORCHVISION_TAG="v0.20.0"
ARG CUDA_VERSION=12.2
ARG OPENCV_VERSION=4.12.0
ARG MEDIAMTX_VERSION=1.9.8

# Core dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        libopenblas-dev \
        wget \
        ca-certificates \
        git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Remove any existing OpenCV packages
RUN apt-get update && \
    apt-get remove --purge -y \
        libopencv-dev \
        libopencv-core-dev \
        libopencv-imgproc-dev \
        python3-opencv || true && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Install build/runtime dependencies for OpenCV and media IO
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        pv \
        cmake \
        ccache \
        git \
        unzip \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        v4l-utils \
        libxvidcore-dev \
        libx264-dev \
        libgtk-3-dev \
        libcanberra-gtk3-dev \
        libtbb2 \
        libtbb-dev \
        libdc1394-dev \
        python3-dev \
        python3-numpy \
        libopenjp2-7-dev \
        liblapack-dev \
        gfortran \
        libhdf5-dev \
        libcudnn8-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-libav \
        gstreamer1.0-tools \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-bad1.0-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit matching JetPack
RUN apt-get update && \
    apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA_VERSION/./-}" && \
    rm -rf /var/lib/apt/lists/*

# Ensure CUDA environment
ENV CUDA_VERSION=${CUDA_VERSION}
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION} \
    CUDA_PATH=/usr/local/cuda-${CUDA_VERSION} \
    PATH=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:/usr/local/cuda-${CUDA_VERSION}/lib64/stubs

RUN ln -sf /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda

# Install PyTorch and TorchVision (Jetson wheels already include dependencies)
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy==1.26.4 && \
    python3 -m pip install --no-cache-dir "${TORCH_INSTALL}" && \
    python3 -m pip install --no-cache-dir "git+https://github.com/pytorch/vision.git@${TORCHVISION_TAG}"

# Mosquitto MQTT broker and client utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        mosquitto \
        mosquitto-clients && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone OpenCV and contrib repositories
RUN git clone --branch "${OPENCV_VERSION}" https://github.com/opencv/opencv.git && \
    git clone --branch "${OPENCV_VERSION}" https://github.com/opencv/opencv_contrib.git

WORKDIR /opt/opencv/build

# Configure and build OpenCV with CUDA support
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_CUDA=ON \
          -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
          -D CUDA_ARCH_BIN=8.7 \
          -D CUDA_ARCH_PTX="" \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D ENABLE_FAST_MATH=ON \
          -D CUDA_FAST_MATH=ON \
          -D WITH_CUBLAS=ON \
          -D WITH_V4L=ON \
          -D WITH_LIBV4L=ON \
          -D WITH_OPENGL=ON \
          -D BUILD_OPENCV_PYTHON3=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D CMAKE_C_COMPILER_LAUNCHER=ccache \
          -D CMAKE_CXX_COMPILER_LAUNCHER=ccache .. && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig && \
    rm -rf /opt/opencv /opt/opencv_contrib

# Verify installation
RUN python3 -c "import cv2; print(cv2.getBuildInformation())"

ENV HUGGINGFACE_HUB_CACHE=/opt/hf-cache \
    HF_HOME=/opt/hf-cache \
    TRANSFORMERS_CACHE=/opt/hf-cache \
    SAM2_BUILD_CUDA=0

RUN mkdir -p "${HUGGINGFACE_HUB_CACHE}"

# Install MediaMTX RTSP server
RUN wget -O /tmp/mediamtx.tar.gz \
        "https://github.com/bluenviron/mediamtx/releases/download/v${MEDIAMTX_VERSION}/mediamtx_v${MEDIAMTX_VERSION}_linux_arm64.tar.gz" && \
    tar -xzf /tmp/mediamtx.tar.gz -C /tmp && \
    install -m 0755 /tmp/mediamtx /usr/local/bin/mediamtx && \
    mkdir -p /etc/mediamtx && \
    mv /tmp/mediamtx.yml /etc/mediamtx/mediamtx.yml && \
    rm -rf /tmp/mediamtx /tmp/mediamtx.yml /tmp/mediamtx.tar.gz

WORKDIR /workspace

COPY . /workspace

# Python dependencies for application components
RUN python3 -m pip install --no-cache-dir \
        accelerate==1.11.0 \
        huggingface-hub==0.36.0 \
        transformers==4.57.1 \
        timm==1.0.22 \
        ultralytics==8.3.65 \
        supervision==0.26.1 \
        paho-mqtt==1.6.1 \
        python-dotenv==1.0.1 \
        tqdm==4.67.1 \
        hydra-core==1.3.2 \
        iopath==0.1.10 \
        pillow==11.1.0 \
        einops==0.8.1 \
        decord==0.6.0 \
        safetensors==0.6.2 \
        sentencepiece==0.2.0

# Install EdgeTAM in editable mode (CUDA extension optional)
RUN python3 -m pip install --no-cache-dir -e model_inference/EdgeTAM

ENV PYTHONPATH="/workspace"