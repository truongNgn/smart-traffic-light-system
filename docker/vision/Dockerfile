# Stage 1: Base image with Python and OpenCV dependencies
FROM python:3.11-slim AS base

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies directly
# (In a real project, we'd use Poetry or uv, but for now we'll install via pip)
RUN pip install --no-cache-dir \
    "ultralytics>=8.1" \
    "opencv-python>=4.9" \
    "redis>=5.0" \
    "pydantic>=2.7" \
    "pydantic-settings>=2.3" \
    "structlog>=24.1" \
    "lapx>=0.5"

# Stage 2: Application
FROM base

COPY . /app/
ENV PYTHONPATH=/app

CMD ["python", "-m", "vision.producer.smoke_test"]
