#!/bin/bash
echo "🚀 LAUNCHING MAXIMAL ALPACA DOCKER SYSTEM"
echo "========================================"

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker compose -f docker-compose-maximal.yml down 2>/dev/null || true

# Remove old containers and images
echo "🧹 Cleaning up..."
docker container prune -f
docker system prune -f

# Build fresh images
echo "🔨 Building maximal Docker image..."
docker compose -f docker-compose-maximal.yml build --no-cache

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Start the services
    echo "🚀 Starting maximal services..."
    docker compose -f docker-compose-maximal.yml up -d
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 MAXIMAL SYSTEM DEPLOYED SUCCESSFULLY!"
        echo "========================================"
        echo "📊 Dashboard: http://localhost:8000"
        echo "📈 Analytics: http://localhost:8001" 
        echo "🤖 ML API: http://localhost:8002"
        echo "📡 WebSocket: http://localhost:8003"
        echo ""
        echo "📋 View logs with:"
        echo "   docker compose -f docker-compose-maximal.yml logs -f"
        echo ""
        echo "⏹️ Stop system with:"
        echo "   docker compose -f docker-compose-maximal.yml down"
    else
        echo "❌ Failed to start services"
        exit 1
    fi
else
    echo "❌ Build failed"
    exit 1
fi