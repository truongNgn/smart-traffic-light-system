# Stage 1: Base image with Python and OpenCV dependencies
FROM python:3.11-slim AS base

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU PyTorch first so Ultralytics does not pull a full CUDA stack.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.2.2" \
    "torchvision==0.17.2"

RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "ultralytics==8.1.47" \
    "opencv-python==4.9.0.80" \
    "redis==5.0.8" \
    "pydantic==2.8.2" \
    "pydantic-settings==2.4.0" \
    "structlog==24.4.0" \
    "lapx==0.5.11"

# Stage 2: Application
FROM base

COPY . /app/
ENV PYTHONPATH=/app

CMD ["python", "-m", "vision.producer.smoke_test"]
