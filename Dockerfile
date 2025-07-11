# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy the requirements file
COPY docker_requirements.txt .

# Upgrade pip and install packages
RUN pip install --no-cache-dir --upgrade pip

# Install TensorFlow GPU version first
RUN pip install --no-cache-dir tensorflow[and-cuda]==2.15.1

# Install remaining packages
RUN pip install --no-cache-dir \
    --use-deprecated=legacy-resolver \
    -r docker_requirements.txt

# Clean up build dependencies
RUN apt-get purge -y --auto-remove build-essential git