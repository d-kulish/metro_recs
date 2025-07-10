# Use an official Python 3.10 image. TFX 1.15.0 requires Python >=3.9 and <3.11.
FROM python:3.10-slim

# Copy the requirements file that specifies our additional libraries.
COPY docker_requirements.txt .

# Install build-essential for compiling C++ extensions like pyfarmhash,
# then install all Python packages in a single layer, and finally clean up
# the build tools to keep the final image slim.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tensorflow==2.15.1 tfx==1.15.0 && \
    pip install --no-cache-dir -r docker_requirements.txt && \
    apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*