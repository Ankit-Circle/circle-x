# Use official Python base image
FROM python:3.11-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip

# Install system dependencies for rembg
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Flask default is 5000)
EXPOSE 5000

# Run the app with gunicorn (production WSGI server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
