# Use an official Python 3.10 image. TFX 1.15.0 requires Python >=3.9 and <3.11.
FROM python:3.10-slim

# Copy the requirements file that specifies our additional libraries.
COPY docker_requirements.txt .

# Install system dependencies and Python packages in separate steps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install packages with more lenient resolver
RUN pip install --no-cache-dir --upgrade pip

# Install packages in order of dependency complexity
RUN pip install --no-cache-dir \
    --no-deps \
    tensorflow==2.15.1

RUN pip install --no-cache-dir \
    --no-deps \
    tensorflow-transform==1.15.0

RUN pip install --no-cache-dir \
    --no-deps \
    tensorflow-recommenders==0.7.3

# Install remaining packages
RUN pip install --no-cache-dir \
    --use-deprecated=legacy-resolver \
    -r docker_requirements.txt

# Clean up build dependencies
RUN apt-get purge -y --auto-remove build-essential git