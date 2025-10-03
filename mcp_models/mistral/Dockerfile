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
    scikit-learn==1.3.0

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
