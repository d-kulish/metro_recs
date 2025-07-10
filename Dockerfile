# Use an official Python 3.10 image. TFX 1.15.0 requires Python >=3.9 and <3.11.
FROM python:3.10-slim

# Upgrade pip and install TensorFlow and TFX in a single layer.
# The standard 'tensorflow' package now includes GPU support and will work on
# Vertex AI GPU nodes.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.15.0 tfx==1.15.0

# Copy the requirements file that specifies our additional libraries.
COPY docker_requirements.txt .

# Install the additional packages using pip.
RUN pip install --no-cache-dir -r docker_requirements.txt