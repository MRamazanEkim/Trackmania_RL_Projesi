# Trackmania RL Training Environment
# ===================================
# Multi-stage build for optimized image size

FROM python:3.11-slim as base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (with retry logic for transient network failures)
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80retries && \
    echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80retries && \
    echo 'Acquire::ftp::Timeout "30";' >> /etc/apt/apt.conf.d/80retries && \
    for i in 1 2 3; do \
        apt-get update && \
        apt-get install -y --no-install-recommends --fix-missing \
            build-essential \
            libgl1 \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            libevdev2 \
            git \
        && break || { \
            echo "apt-get attempt $i failed"; \
            [ "$i" -lt 3 ] && sleep 10 || { echo "All apt-get attempts failed"; exit 1; }; \
        }; \
    done && \
    rm -rf /var/lib/apt/lists/*

# ── Dependencies Stage ────────────────────────────────────────────────────────
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional ML/RL dependencies
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    opencv-python-headless>=4.8.0 \
    gymnasium>=0.29.0 \
    stable-baselines3>=2.0.0 \
    tensorboard>=2.14.0 \
    matplotlib>=3.7.0 \
    pandas>=2.0.0

# ── Production Stage ──────────────────────────────────────────────────────────
FROM dependencies as production

# Copy application code
COPY . .

# Create directories for logs and data
RUN mkdir -p /app/telemetry_logs /app/trajectories /app/checkpoints

# Set default environment variables
ENV TMRL_CONFIG_PATH=/app/config
ENV LOG_DIR=/app/telemetry_logs
ENV CHECKPOINT_DIR=/app/checkpoints

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tmrl; print('OK')" || exit 1

# Default command
CMD ["python", "telemetry_monitor.py"]

# ── Development Stage ─────────────────────────────────────────────────────────
FROM dependencies as development

# Install development tools
RUN pip install --no-cache-dir \
    pytest>=7.0.0 \
    pytest-cov>=4.0.0 \
    black>=23.0.0 \
    flake8>=6.0.0 \
    mypy>=1.0.0 \
    ipython>=8.0.0

# Copy application code
COPY . .

# Create directories
RUN mkdir -p /app/telemetry_logs /app/trajectories /app/checkpoints

CMD ["bash"]
