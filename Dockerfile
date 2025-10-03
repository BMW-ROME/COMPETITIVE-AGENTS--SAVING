# 🚀 PROFIT-OPTIMIZED COMPETITIVE TRADING AGENTS
# Docker container for maximum profit generation systems
FROM python:3.11-slim

# Set metadata for optimized trading system
LABEL maintainer="BiggRee007"
LABEL description="Profit-Maximized Trading Agents - 50X-500X Position Boosts"
LABEL version="2.0-optimized"

# Set working directory
WORKDIR /app

# Install system dependencies for high-performance trading
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libblas-dev \
    liblapack-dev \
    ca-certificates \
    curl \
    wget \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimization flags
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy optimized application code
COPY . .

# Create comprehensive directory structure for profit tracking
RUN mkdir -p logs reports data models cache backups && \
    chmod -R 755 logs reports data models cache backups

# Expose all dashboard ports for multi-system deployment
EXPOSE 5000 5001 5002 5003 8080

# Environment variables for maximum performance
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV TRADING_MODE=PROFIT_MAXIMIZED
ENV LOG_LEVEL=INFO

# Health check for system monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default to maximal profit system (can be overridden)
CMD ["python", "alpaca_paper_trading_maximal.py"]
