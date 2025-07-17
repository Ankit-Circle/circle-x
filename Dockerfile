# Use official Python base image
FROM python:3.11-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip

# Install system dependencies for rembg
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download u2net.onnx model for rembg
# RUN mkdir -p /root/.u2net && \
#     curl -L -o /root/.u2net/u2net.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

# RUN mkdir -p /root/.u2net && \
    # curl -L -o /root/.u2net/u2netp.onnx https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx

# Copy application files
COPY . .

# Expose port 8080 as required by Digital Ocean App Platform
EXPOSE 8080

# Run the app with gunicorn on port 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "main:app"]
