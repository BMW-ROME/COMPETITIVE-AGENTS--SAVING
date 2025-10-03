
#!/usr/bin/env python3
import asyncio
import logging
import docker
import psycopg2
import redis
from datetime import datetime

async def health_check():
    logger = logging.getLogger("HealthCheck")
    
    # Check Docker containers
    docker_client = docker.from_env()
    containers = docker_client.containers.list()
    
    for container in containers:
        if 'tyree' in container.name or 'trading' in container.name:
            status = container.status
            if status != 'running':
                logger.warning(f"Container {container.name} is {status}")
    
    # Check PostgreSQL
    try:
        conn = psycopg2.connect(
            host="localhost", port=5432, database="trading_agents",
            user="trading_user", password="trading_password"
        )
        conn.close()
        logger.info("PostgreSQL connection healthy")
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
    
    # Check Redis
    try:
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        logger.info("Redis connection healthy")
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")

if __name__ == "__main__":
    asyncio.run(health_check())
