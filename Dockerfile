# ============================================================
# BraTS AI Backend — Production Dockerfile
# Base: python:3.10-slim (Railway-compatible)
# Model: model/model.keras bundled inside image (~68MB)
# Server: gunicorn gthread (TF-safe, no async)
# ============================================================

FROM python:3.11-slim

# ── System dependencies ──────────────────────────────────────
# libgomp1  → OpenMP required by TensorFlow/numpy
# libgl1    → nibabel/PIL image I/O
# curl      → health-check probe during build (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (cached layer) ───────────────
# Copy requirements first so Docker cache is reused on code-only changes
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ──────────────────────────────────────
# .dockerignore excludes: .venv, __pycache__, .git, db.sqlite3, .env, node_modules
COPY . .

# ── Ensure required directories exist ───────────────────────
# /app/model/model.keras  → model copied by COPY . . above
# /app/media/             → ephemeral uploads (Railway ephemeral disk)
# /app/staticfiles/       → whitenoise serves from here
RUN mkdir -p /app/media /app/media/previews /app/staticfiles

# ── Collect static files ─────────────────────────────────────
# Must run with a valid SECRET_KEY; use a throwaway one for build only
RUN SECRET_KEY="build-time-throwaway-key-do-not-use" \
    DEBUG=False \
    DATABASE_URL="sqlite:////tmp/build.db" \
    python manage.py collectstatic --noinput --clear 2>/dev/null || true

# ── Expose port ──────────────────────────────────────────────
EXPOSE 8000

# ── Production entrypoint ────────────────────────────────────
# --workers 2      → 2 processes, each loads the Keras model once
# --threads 4      → 4 OS threads per worker (handles concurrent requests)
# --timeout 300    → 5 min timeout (model inference can take ~60-120s)
# --worker-class gthread → thread-based, safe for TensorFlow (no async)
# --bind 0.0.0.0:$PORT → Railway injects $PORT dynamically
CMD ["sh", "-c", "gunicorn config.wsgi:application \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers 2 \
  --threads 4 \
  --timeout 300 \
  --worker-class gthread \
  --log-level info \
  --access-logfile - \
  --error-logfile -"]
