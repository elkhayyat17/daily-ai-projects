# ---- Stage 1: builder ----------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for librosa / soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: runtime ----------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime audio libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8004/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "2"]
