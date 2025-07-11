# Use NVIDIA CUDA base image for GPU support - using a valid tag
FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Update pip to latest version
RUN python3.10 -m pip install --upgrade pip

# Copy the requirements file
COPY docker_requirements.txt .

# Install TensorFlow GPU version first
RUN python3.10 -m pip install --no-cache-dir tensorflow[and-cuda]==2.15.1

# Install remaining packages
RUN python3.10 -m pip install --no-cache-dir \
    --use-deprecated=legacy-resolver \
    -r docker_requirements.txt

# Clean up build dependencies
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*