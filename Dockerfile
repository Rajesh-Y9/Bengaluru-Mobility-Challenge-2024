# Use PyTorch with CUDA support as the base image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project directory into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make sure the script is executable
RUN chmod +x app.py

# Set the entrypoint to run app.py
ENTRYPOINT ["python3", "app.py"]