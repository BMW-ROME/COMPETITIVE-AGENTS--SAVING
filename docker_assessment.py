#!/usr/bin/env python3
"""
Docker Readiness Assessment for Trading System
Analyzes system resources and provides Docker deployment recommendations
"""

import os
import psutil
import json
from datetime import datetime

def assess_system_resources():
    """Assess current system resources"""
    print("🐳 DOCKER DEPLOYMENT READINESS ASSESSMENT")
    print("=" * 60)
    
    # CPU Assessment
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent(interval=1)
    load_avg = os.getloadavg()
    
    print(f"🖥️  CPU ANALYSIS:")
    print(f"   Logical Cores: {cpu_count}")
    print(f"   Physical Cores: {cpu_physical}")
    print(f"   Current Usage: {cpu_percent:.1f}%")
    print(f"   Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
    
    # Memory Assessment
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print(f"\n💾 MEMORY ANALYSIS:")
    print(f"   Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"   Available: {memory.available / (1024**3):.1f} GB")
    print(f"   Used: {memory.percent:.1f}%")
    print(f"   Swap: {swap.total / (1024**3):.1f} GB")
    
    # Disk Assessment
    disk = psutil.disk_usage('/')
    
    print(f"\n💿 DISK ANALYSIS:")
    print(f"   Total: {disk.total / (1024**3):.1f} GB")
    print(f"   Free: {disk.free / (1024**3):.1f} GB")
    print(f"   Used: {disk.percent:.1f}%")
    
    # Network Assessment (basic)
    network = psutil.net_io_counters()
    
    print(f"\n🌐 NETWORK:")
    print(f"   Bytes Sent: {network.bytes_sent / (1024**2):.1f} MB")
    print(f"   Bytes Received: {network.bytes_recv / (1024**2):.1f} MB")
    
    return {
        'cpu_cores': cpu_count,
        'cpu_physical': cpu_physical,
        'cpu_usage': cpu_percent,
        'memory_gb': memory.total / (1024**3),
        'memory_available_gb': memory.available / (1024**3),
        'memory_percent': memory.percent,
        'disk_gb': disk.total / (1024**3),
        'disk_free_gb': disk.free / (1024**3)
    }

def generate_docker_recommendations(system_info):
    """Generate Docker deployment recommendations"""
    print("\n🐳 DOCKER CONTAINER RECOMMENDATIONS")
    print("=" * 60)
    
    cpu_cores = system_info['cpu_cores']
    memory_gb = system_info['memory_gb']
    
    # Trading System Container
    print("📈 TRADING SYSTEM CONTAINER:")
    trading_cpu = min(4, max(2, cpu_cores // 2))
    trading_memory = min(2.0, max(1.0, memory_gb * 0.3))
    
    print(f"   CPU Limit: {trading_cpu} cores")
    print(f"   Memory Limit: {trading_memory:.1f} GB")
    print(f"   Recommended: python:3.12-slim base image")
    print(f"   Volumes: ./data:/app/data, ./logs:/app/logs")
    
    # Database Container (if using)
    print(f"\n🗄️  DATABASE CONTAINER (PostgreSQL/TimescaleDB):")
    db_cpu = min(2, max(1, cpu_cores // 4))
    db_memory = min(1.0, max(0.5, memory_gb * 0.15))
    
    print(f"   CPU Limit: {db_cpu} cores")
    print(f"   Memory Limit: {db_memory:.1f} GB")
    print(f"   Recommended: timescale/timescaledb:latest-pg15")
    print(f"   Volumes: ./postgres_data:/var/lib/postgresql/data")
    
    # Redis Container (for caching)
    print(f"\n🔴 REDIS CONTAINER (Caching):")
    redis_memory = min(0.5, max(0.1, memory_gb * 0.05))
    
    print(f"   CPU Limit: 1 core")
    print(f"   Memory Limit: {redis_memory:.1f} GB")
    print(f"   Recommended: redis:7-alpine")
    print(f"   Volumes: ./redis_data:/data")
    
    # Monitoring Container
    print(f"\n📊 MONITORING CONTAINER (Optional):")
    monitor_memory = min(0.5, max(0.2, memory_gb * 0.1))
    
    print(f"   CPU Limit: 1 core")
    print(f"   Memory Limit: {monitor_memory:.1f} GB")
    print(f"   Recommended: grafana/grafana:latest")
    
    return {
        'trading_system': {'cpu': trading_cpu, 'memory': trading_memory},
        'database': {'cpu': db_cpu, 'memory': db_memory},
        'redis': {'cpu': 1, 'memory': redis_memory},
        'monitoring': {'cpu': 1, 'memory': monitor_memory}
    }

def assess_current_containers():
    """Assess what containers might already be running"""
    print("\n📦 EXISTING CONTAINER ANALYSIS")
    print("=" * 60)
    
    # Check for common container processes
    container_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                
                # Look for container-related processes
                if any(keyword in cmdline.lower() for keyword in [
                    'docker', 'containerd', 'postgres', 'redis', 'nginx',
                    'grafana', 'prometheus', 'mysql', 'mongodb'
                ]):
                    container_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cmdline': cmdline[:80] + ('...' if len(cmdline) > 80 else '')
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if container_processes:
        print(f"🔍 Found {len(container_processes)} container-related processes:")
        for proc in container_processes:
            print(f"   PID {proc['pid']}: {proc['name']} - {proc['cmdline']}")
    else:
        print("ℹ️  No obvious container processes detected")
        print("   (This is normal in a dev container environment)")
    
    return container_processes

def generate_docker_compose():
    """Generate a Docker Compose configuration"""
    print("\n🐳 DOCKER COMPOSE TEMPLATE")
    print("=" * 60)
    
    compose_template = """
version: '3.8'

services:
  trading-system:
    build: .
    container_name: competitive-trading-agents
    environment:
      - APCA_API_KEY_ID=${APCA_API_KEY_ID}
      - APCA_API_SECRET_KEY=${APCA_API_SECRET_KEY}
      - APCA_API_BASE_URL=https://paper-api.alpaca.markets
      - TRADING_MODE=PAPER
      - PYTHONUNBUFFERED=1
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    cpus: '4'
    mem_limit: 2g
    networks:
      - trading-net

  redis:
    image: redis:7-alpine
    container_name: trading-redis
    volumes:
      - redis_data:/data
    restart: unless-stopped
    cpus: '1'
    mem_limit: 512m
    networks:
      - trading-net

  timescaledb:
    image: timescale/timescaledb:latest-pg15
    container_name: trading-db
    environment:
      - POSTGRES_DB=trading_data
      - POSTGRES_USER=trading_user
      - POSTGRES_PASSWORD=secure_password_123
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    cpus: '2'
    mem_limit: 1g
    networks:
      - trading-net

volumes:
  redis_data:
  postgres_data:

networks:
  trading-net:
    driver: bridge
"""
    
    # Save to file
    with open('/workspaces/competitive-trading-agents/docker-compose-optimized.yml', 'w') as f:
        f.write(compose_template.strip())
    
    print("✅ Generated: docker-compose-optimized.yml")
    print("   Configured for PAPER trading with resource limits")

def main():
    """Main assessment function"""
    print(f"🔍 System Assessment - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Assess system resources
    system_info = assess_system_resources()
    
    # Generate recommendations
    recommendations = generate_docker_recommendations(system_info)
    
    # Check existing containers
    assess_current_containers()
    
    # Generate Docker Compose
    generate_docker_compose()
    
    # Overall assessment
    print(f"\n📋 OVERALL ASSESSMENT")
    print("=" * 60)
    
    cpu_sufficient = system_info['cpu_cores'] >= 8
    memory_sufficient = system_info['memory_gb'] >= 4
    disk_sufficient = system_info['disk_free_gb'] >= 10
    
    print(f"CPU Capacity: {'✅' if cpu_sufficient else '⚠️'} {system_info['cpu_cores']} cores")
    print(f"Memory Capacity: {'✅' if memory_sufficient else '⚠️'} {system_info['memory_gb']:.1f} GB")
    print(f"Disk Space: {'✅' if disk_sufficient else '⚠️'} {system_info['disk_free_gb']:.1f} GB free")
    
    if cpu_sufficient and memory_sufficient and disk_sufficient:
        print("\n🎉 EXCELLENT: System is well-suited for Docker deployment")
        print("   Recommended: Use all containers with suggested limits")
    elif system_info['cpu_cores'] >= 4 and system_info['memory_gb'] >= 2:
        print("\n✅ GOOD: System can handle Docker deployment")
        print("   Recommended: Use trading system + Redis containers")
    else:
        print("\n⚠️ LIMITED: System has constraints")
        print("   Recommended: Single container deployment only")
    
    # Save assessment
    assessment = {
        'timestamp': datetime.now().isoformat(),
        'system_info': system_info,
        'recommendations': recommendations,
        'docker_ready': cpu_sufficient and memory_sufficient and disk_sufficient
    }
    
    with open('/workspaces/competitive-trading-agents/docker_readiness_assessment.json', 'w') as f:
        json.dump(assessment, f, indent=2)
    
    print(f"\n💾 Assessment saved: docker_readiness_assessment.json")

if __name__ == "__main__":
    main()