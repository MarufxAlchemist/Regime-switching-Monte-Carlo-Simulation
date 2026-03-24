# ─────────────────────────────────────────────────────────────────────────────
# Regime-Switching Monte Carlo — Docker Image
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — dependency builder (keeps final image lean)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (e.g. hmmlearn, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        liblapack-dev \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage layer caching
COPY requirements.txt .

# Install into a prefix directory so we can copy it cleanly into the runtime stage
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Maruf" \
      description="Regime-switching Monte Carlo systemic risk engine" \
      version="0.1.0"

# Runtime system libraries (no build tools — keeps image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy the entire project
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Hugging Face model cache — writable by appuser
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: run the full pipeline
CMD ["python", "main.py"]
