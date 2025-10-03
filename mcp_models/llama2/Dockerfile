FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.100.0 \
    uvicorn==0.22.0 \
    pydantic==2.0.0 \
    requests==2.31.0 \
    transformers==4.30.2 \
    torch==2.0.1 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    sentence-transformers==2.2.2

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
