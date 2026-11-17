# =========================================================================
# Production Dockerfile for AgroVision Agricultural Intelligence Platform
# =========================================================================
FROM python:3.11-slim

LABEL maintainer="AgroVision Analytics Team"
LABEL description="AI Agricultural Economist: Bangalore Grocery Price Prediction & Verification Platform"

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5001
ENV HOST=0.0.0.0
ENV FLASK_ENV=production

WORKDIR /app

# Install system dependencies (build-essential and libpq for PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy project files
COPY data/ ./data/
COPY models/ ./models/
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY scripts/ ./scripts/
COPY app.py .
COPY wsgi.py .
COPY README.md .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:5001", "--workers", "2", "--threads", "4", "--timeout", "120"]
