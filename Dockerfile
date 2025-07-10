# Use the official TensorFlow 2.15.0 GPU image as a base. This is the version
# compatible with TFX 1.15.0 and ensures CUDA drivers are available.
FROM tensorflow/tensorflow:2.15.0-gpu

# Install the TFX library itself. This will pull in all required dependencies
# like Apache Beam and MLMD.
RUN pip install --no-cache-dir tfx==1.15.0

# Copy the requirements file that specifies our additional libraries.
COPY docker_requirements.txt .

# Install the additional packages using pip.
RUN pip install --no-cache-dir -r docker_requirements.txt