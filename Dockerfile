# Use Ubuntu 20.04 with Python 3.10 for TensorFlow 2.13.1 compatibility
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
# Use TFX 1.14.0 with Vertex AI Training (no longer AI Platform)
ENV TFX_USE_VERTEX_AI_TRAINING=true
# Force TensorFlow to use Keras 2.13.1 (stable combination)
ENV TF_USE_LEGACY_KERAS=1
# Fix protobuf compatibility issues
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

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

# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Copy requirements and install packages
COPY docker_requirements.txt .

# Install stable TensorFlow 2.13.1 with GPU support
RUN python3.10 -m pip install --no-cache-dir tensorflow[and-cuda]==2.13.1

# Install protobuf first with compatible version
RUN python3.10 -m pip install --no-cache-dir protobuf==3.20.3

# Install compatible jsonschema stack to avoid version conflicts
RUN python3.10 -m pip install --no-cache-dir \
    attrs==22.2.0 \
    referencing==0.28.4 \
    jsonschema==4.17.3

# Install remaining packages with legacy resolver for compatibility
RUN python3.10 -m pip install --no-cache-dir \
    --use-deprecated=legacy-resolver \
    -r docker_requirements.txt

# Clean up
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*