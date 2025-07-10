# Use the official TFX image as the base. This ensures all core TFX
# and TensorFlow dependencies are correctly installed.
FROM gcr.io/tfx-oss-public/tfx:1.15.0-gpu

# Copy the requirements file that specifies our additional libraries.
COPY docker_requirements.txt .

# Install the additional packages using pip.
RUN pip install --no-cache-dir -r docker_requirements.txt