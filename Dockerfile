# Pinned to same PyTorch/CUDA version as Dockerfile.pipeline for consistency.
# If the webapp does not actually import torch, swap this for a plain
# python:3.11-slim image to reduce image size significantly.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip setuptools wheel

# Install app dependencies (torch already in base image — do not add to requirements).
COPY requirements-webapp.txt .
RUN pip install --no-cache-dir -r requirements-webapp.txt

# Copy source code only.
# corpus/, data/, and cache/ are mounted at runtime via docker-compose.
COPY anispeakarchive/ anispeakarchive/
COPY scripts/         scripts/
COPY configs/         configs/
COPY app.py wsgi.py   ./

EXPOSE 5696

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5696", \
     "--workers", "1", \
     "--threads", "5", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "wsgi:app"]