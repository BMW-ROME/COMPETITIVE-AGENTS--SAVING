# 🗄️ Database Setup Guide - Competitive Trading Agents

## 📊 **Current Situation Analysis**

### **Existing Containers:**
- **PostgreSQL**: `tyree-postgres` (port 5432)
- **Redis**: `tyree-redis` (port 6379)
- **Database**: `trading_agents`
- **User**: `trading_user`
- **Password**: `trading_password`

### **Can They Work for Separate Repos?**

**YES, but with limitations:**

✅ **PostgreSQL**: Multiple projects can share the same database with different schemas
✅ **Redis**: Multiple projects can use different Redis databases (0-15)
⚠️ **Conflicts**: Port conflicts, data mixing, dependency issues

## 🚀 **RECOMMENDED: Dedicated Containers**

For **complete isolation** and **no conflicts**, use dedicated containers.

## 📋 **Setup Options**

### **Option 1: Use Existing Containers (Quick Start)**

```bash
# Check if containers are accessible
docker exec tyree-postgres psql -U trading_user -d trading_agents -c "SELECT version();"
docker exec tyree-redis redis-cli ping
```

**Pros:**
- ✅ Quick setup
- ✅ No additional resources
- ✅ Shared infrastructure

**Cons:**
- ⚠️ Potential conflicts
- ⚠️ Data mixing
- ⚠️ Port conflicts

### **Option 2: Dedicated Containers (Recommended)**

```bash
# Windows
launch-dedicated-services.bat

# Linux/Mac
chmod +x launch-dedicated-services.sh
./launch-dedicated-services.sh
```

**Pros:**
- ✅ Complete isolation
- ✅ No conflicts
- ✅ Dedicated resources
- ✅ Easy management

**Cons:**
- ⚠️ Additional resources
- ⚠️ More containers

## 🛠️ **Manual Setup Commands**

### **PostgreSQL Setup:**

```bash
# Create dedicated PostgreSQL container
docker run -d \
  --name competitive-trading-postgres \
  --network competitive-trading-network \
  -e POSTGRES_DB=competitive_trading_agents \
  -e POSTGRES_USER=trading_user \
  -e POSTGRES_PASSWORD=trading_password \
  -p 5433:5432 \
  -v competitive_postgres_data:/var/lib/postgresql/data \
  -v $(pwd)/init.sql:/docker-entrypoint-initdb.d/init.sql \
  --restart unless-stopped \
  postgres:15-alpine
```

### **Redis Setup:**

```bash
# Create dedicated Redis container
docker run -d \
  --name competitive-trading-redis \
  --network competitive-trading-network \
  -p 6380:6379 \
  -v competitive_redis_data:/data \
  --restart unless-stopped \
  redis:7-alpine
```

### **Network Setup:**

```bash
# Create dedicated network
docker network create competitive-trading-network
```

## 🔧 **Docker Compose Setup**

### **Dedicated Services:**

```bash
# Start dedicated services
docker-compose -f docker-compose-dedicated.yml up --build -d

# Check status
docker-compose -f docker-compose-dedicated.yml ps

# View logs
docker-compose -f docker-compose-dedicated.yml logs -f
```

### **Using Existing Services:**

```bash
# Start with existing services
docker-compose -f docker-compose-trading-only.yml up --build -d
```

## 📊 **Port Configuration**

| Service | Existing | Dedicated | Purpose |
|---------|----------|-----------|---------|
| PostgreSQL | 5432 | 5433 | Database |
| Redis | 6379 | 6380 | Cache |
| Trading System | 8000 | 8000 | Web Interface |

## 🔍 **Connection Testing**

### **Test PostgreSQL:**

```bash
# Test connection
docker exec competitive-trading-postgres psql -U trading_user -d competitive_trading_agents -c "SELECT version();"

# Test from host
psql -h localhost -p 5433 -U trading_user -d competitive_trading_agents
```

### **Test Redis:**

```bash
# Test connection
docker exec competitive-trading-redis redis-cli ping

# Test from host
redis-cli -h localhost -p 6380 ping
```

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Port Conflicts:**
   ```bash
   # Check port usage
   netstat -an | findstr :5432
   netstat -an | findstr :6379
   ```

2. **Container Conflicts:**
   ```bash
   # Stop conflicting containers
   docker stop tyree-postgres tyree-redis
   ```

3. **Network Issues:**
   ```bash
   # Recreate network
   docker network rm competitive-trading-network
   docker network create competitive-trading-network
   ```

### **Reset Everything:**

```bash
# Stop all containers
docker-compose -f docker-compose-dedicated.yml down

# Remove volumes (WARNING: Data loss)
docker volume rm competitive_postgres_data competitive_redis_data

# Start fresh
docker-compose -f docker-compose-dedicated.yml up --build -d
```

## 📈 **Performance Considerations**

### **Resource Allocation:**

```yaml
# PostgreSQL
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 1G
      cpus: '0.5'

# Redis
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
    reservations:
      memory: 512M
      cpus: '0.25'
```

## 🔐 **Security Considerations**

### **Database Security:**

```sql
-- Create dedicated user
CREATE USER competitive_trading_user WITH PASSWORD 'secure_password';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE competitive_trading_agents TO competitive_trading_user;
```

### **Network Security:**

```bash
# Create isolated network
docker network create --driver bridge --internal competitive-trading-network
```

## 📋 **Quick Start Commands**

### **Windows:**

```cmd
# Launch dedicated services
launch-dedicated-services.bat

# Start trading system
docker-compose -f docker-compose-dedicated.yml up --build -d
```

### **Linux/Mac:**

```bash
# Launch dedicated services
chmod +x launch-dedicated-services.sh
./launch-dedicated-services.sh

# Start trading system
docker-compose -f docker-compose-dedicated.yml up --build -d
```

## 🎯 **Recommendation**

**Use dedicated containers** for:
- ✅ Complete isolation
- ✅ No conflicts with other projects
- ✅ Easy management
- ✅ Dedicated resources

**Use existing containers** for:
- ✅ Quick testing
- ✅ Resource conservation
- ✅ Shared infrastructure

---

## 🚀 **Next Steps**

1. **Choose your approach** (dedicated vs existing)
2. **Run the setup commands**
3. **Test connections**
4. **Start the trading system**
5. **Monitor performance**

The system is ready to go with either approach!

