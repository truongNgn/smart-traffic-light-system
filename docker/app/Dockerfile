FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV SUMO_HOME=/usr/share/sumo

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    sumo \
    sumo-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.2.2"

RUN pip install --no-cache-dir \
    "fastapi==0.111.1" \
    "uvicorn[standard]==0.30.6" \
    "streamlit==1.37.1" \
    "redis==5.0.8" \
    "pydantic==2.8.2" \
    "pydantic-settings==2.4.0" \
    "structlog==24.4.0" \
    "traci==1.20.0" \
    "sumolib==1.20.0" \
    "gymnasium==0.29.1" \
    "numpy==1.26.4" \
    "pandas==2.2.2" \
    "plotly==5.23.0" \
    "websocket-client==1.8.0" \
    "tqdm==4.66.5"

COPY . /app/

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
