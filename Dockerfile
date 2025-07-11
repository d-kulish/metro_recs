# Alternative: Use Ubuntu base image and install CUDA manually
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
# Force TFX to use Vertex AI Training instead of AI Platform Training
ENV TFX_USE_VERTEX_AI_TRAINING=true
# Fix Keras version conflicts
ENV KERAS_VERSION=2.15.0

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    gnupg2 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-venv \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Install pip for Python 3.10 directly using get-pip.py
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Copy the requirements file
COPY docker_requirements.txt .

# Install TensorFlow GPU version that matches TFX 1.16.0 with specific Keras version
RUN python3.10 -m pip install --no-cache-dir tensorflow[and-cuda]==2.15.1
RUN python3.10 -m pip install --no-cache-dir keras==2.15.0

# Install remaining packages
RUN python3.10 -m pip install --no-cache-dir \
    --use-deprecated=legacy-resolver \
    -r docker_requirements.txt

# Clean up build dependencies
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*