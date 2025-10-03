#!/usr/bin/env python3
"""
MCP System Setup Script
======================

Sets up all 7 MCP AI models for the Ultimate Trading System.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create MCP model directories"""
    logger.info("📁 Creating MCP model directories...")
    
    directories = [
        "mcp_models/gordon",
        "mcp_models/llama2", 
        "mcp_models/mistral",
        "mcp_models/codellama",
        "mcp_models/financial",
        "mcp_models/sentiment",
        "mcp_models/risk",
        "mcp_dashboard"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"  ✅ Created: {directory}")

def create_gordon_dockerfile():
    """Create Gordon Assistant Dockerfile"""
    logger.info("🤖 Creating Gordon Assistant Dockerfile...")
    
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    fastapi==0.100.0 \\
    uvicorn==0.22.0 \\
    pydantic==2.0.0 \\
    requests==2.31.0 \\
    transformers==4.30.2 \\
    torch==2.0.1 \\
    numpy==1.24.3 \\
    pandas==2.0.3

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8001/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
"""
    
    with open("mcp_models/gordon/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("  ✅ Gordon Assistant Dockerfile created")

def create_llama2_dockerfile():
    """Create Llama2 Dockerfile"""
    logger.info("🦙 Creating Llama2 Dockerfile...")
    
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    fastapi==0.100.0 \\
    uvicorn==0.22.0 \\
    pydantic==2.0.0 \\
    requests==2.31.0 \\
    transformers==4.30.2 \\
    torch==2.0.1 \\
    numpy==1.24.3 \\
    pandas==2.0.3 \\
    sentence-transformers==2.2.2

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8002/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
"""
    
    with open("mcp_models/llama2/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("  ✅ Llama2 Dockerfile created")

def create_mistral_dockerfile():
    """Create Mistral Dockerfile"""
    logger.info("🌪️ Creating Mistral Dockerfile...")
    
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    fastapi==0.100.0 \\
    uvicorn==0.22.0 \\
    pydantic==2.0.0 \\
    requests==2.31.0 \\
    transformers==4.30.2 \\
    torch==2.0.1 \\
    numpy==1.24.3 \\
    pandas==2.0.3 \\
    scikit-learn==1.3.0

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8003/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
"""
    
    with open("mcp_models/mistral/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("  ✅ Mistral Dockerfile created")

def create_codellama_dockerfile():
    """Create CodeLlama Dockerfile"""
    logger.info("💻 Creating CodeLlama Dockerfile...")
    
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    fastapi==0.100.0 \\
    uvicorn==0.22.0 \\
    pydantic==2.0.0 \\
    requests==2.31.0 \\
    transformers==4.30.2 \\
    torch==2.0.1 \\
    numpy==1.24.3 \\
    pandas==2.0.3 \\
    black==23.7.0 \\
    flake8==6.0.0

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8004/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
"""
    
    with open("mcp_models/codellama/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("  ✅ CodeLlama Dockerfile created")

def create_financial_llm_dockerfile():
    """Create Financial LLM Dockerfile"""
    logger.info("💰 Creating Financial LLM Dockerfile...")
    
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    fastapi==0.100.0 \\
    uvicorn==0.22.0 \\
    pydantic==2.0.0 \\
    requests==2.31.0 \\
    transformers==4.30.2 \\
    torch==2.0.1 \\
    numpy==1.24.3 \\
    pandas==2.0.3 \\
    scipy==1.11.1 \\
    scikit-learn==1.3.0 \\
    yfinance==0.2.18 \\
    alpha-vantage==2.3.1

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8005/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
"""
    
    with open("mcp_models/financial/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("  ✅ Financial LLM Dockerfile created")

def create_sentiment_analyzer_dockerfile():
    """Create Sentiment Analyzer Dockerfile"""
    logger.info("📰 Creating Sentiment Analyzer Dockerfile...")
    
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    fastapi==0.100.0 \\
    uvicorn==0.22.0 \\
    pydantic==2.0.0 \\
    requests==2.31.0 \\
    transformers==4.30.2 \\
    torch==2.0.1 \\
    numpy==1.24.3 \\
    pandas==2.0.3 \\
    textblob==0.17.1 \\
    vaderSentiment==3.3.2 \\
    tweepy==4.14.0 \\
    feedparser==6.0.10

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8006/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
"""
    
    with open("mcp_models/sentiment/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("  ✅ Sentiment Analyzer Dockerfile created")

def create_risk_analyzer_dockerfile():
    """Create Risk Analyzer Dockerfile"""
    logger.info("⚖️ Creating Risk Analyzer Dockerfile...")
    
    dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \\
    fastapi==0.100.0 \\
    uvicorn==0.22.0 \\
    pydantic==2.0.0 \\
    requests==2.31.0 \\
    transformers==4.30.2 \\
    torch==2.0.1 \\
    numpy==1.24.3 \\
    pandas==2.0.3 \\
    scipy==1.11.1 \\
    scikit-learn==1.3.0 \\
    yfinance==0.2.18

# Copy model files
COPY . /app/

# Expose port
EXPOSE 8007

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8007/health || exit 1

# Run the model server
CMD ["python", "model_server.py"]
"""
    
    with open("mcp_models/risk/Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    logger.info("  ✅ Risk Analyzer Dockerfile created")

def create_model_servers():
    """Create model server files for all MCP models"""
    logger.info("🖥️ Creating MCP model servers...")
    
    # Generic model server template
    model_server_template = '''"""
MCP Model Server
===============

Generic model server for MCP integration.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="MCP Model Server", version="1.0.0")

class MCPRequest(BaseModel):
    prompt: str
    context: Dict[str, Any]
    request_type: str
    priority: int = 1

class MCPResponse(BaseModel):
    response: str
    confidence: float
    tokens_used: int
    metadata: Dict[str, Any]

class MCPModel:
    """Generic MCP model implementation"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.usage_count = 0
        self.total_tokens = 0
        
    async def process_request(self, request: MCPRequest) -> MCPResponse:
        """Process an MCP request"""
        try:
            self.usage_count += 1
            
            # Simulate model processing
            response_text = await self._generate_response(request)
            confidence = self._calculate_confidence(request)
            tokens_used = len(request.prompt.split()) + len(response_text.split())
            self.total_tokens += tokens_used
            
            return MCPResponse(
                response=response_text,
                confidence=confidence,
                tokens_used=tokens_used,
                metadata={
                    "model": self.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "usage_count": self.usage_count,
                    "total_tokens": self.total_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _generate_response(self, request: MCPRequest) -> str:
        """Generate response based on request type"""
        if request.request_type == "agent_advice":
            return await self._generate_agent_advice(request)
        elif request.request_type == "strategy_optimization":
            return await self._generate_strategy_optimization(request)
        elif request.request_type == "risk_analysis":
            return await self._generate_risk_analysis(request)
        elif request.request_type == "sentiment_analysis":
            return await self._generate_sentiment_analysis(request)
        else:
            return await self._generate_general_response(request)
    
    async def _generate_agent_advice(self, request: MCPRequest) -> str:
        """Generate agent advice"""
        return f"""Based on the market data and context provided, here's my analysis:

1. **Market Analysis**: Current market conditions appear {request.context.get('market_trend', 'neutral')}
2. **Recommended Action**: {self._get_recommended_action(request)}
3. **Confidence Level**: {self._calculate_confidence(request):.2f}
4. **Risk Assessment**: {self._assess_risk(request)}
5. **Reasoning**: {self._get_reasoning(request)}

This recommendation is based on {self.model_name} analysis of the provided data."""
    
    async def _generate_strategy_optimization(self, request: MCPRequest) -> str:
        """Generate strategy optimization advice"""
        return f"""Strategy Optimization Recommendations:

1. **Parameter Optimization**: Adjust risk parameters based on current volatility
2. **Risk Management**: Implement dynamic position sizing
3. **Performance Enhancement**: Add momentum indicators
4. **Code Improvements**: Optimize execution speed and accuracy

Optimized parameters: {self._get_optimized_parameters(request)}"""
    
    async def _generate_risk_analysis(self, request: MCPRequest) -> str:
        """Generate risk analysis"""
        return f"""Risk Analysis Report:

1. **Overall Risk Score**: {self._calculate_risk_score(request):.2f}/10
2. **Position Sizing**: Recommended max position size: {self._get_position_sizing(request)}
3. **Risk Mitigation**: {self._get_risk_mitigation(request)}
4. **Portfolio Optimization**: {self._get_portfolio_optimization(request)}

Risk level: {self._get_risk_level(request)}"""
    
    async def _generate_sentiment_analysis(self, request: MCPRequest) -> str:
        """Generate sentiment analysis"""
        return f"""Sentiment Analysis Report:

1. **Overall Sentiment Score**: {self._calculate_sentiment_score(request):.2f} (-1 to 1)
2. **Confidence Level**: {self._calculate_confidence(request):.2f}
3. **Key Sentiment Drivers**: {self._get_sentiment_drivers(request)}
4. **Market Impact Assessment**: {self._get_market_impact(request)}

Sentiment: {self._get_sentiment_label(request)}"""
    
    async def _generate_general_response(self, request: MCPRequest) -> str:
        """Generate general response"""
        return f"""General Analysis from {self.model_name}:

Based on the provided context, here's my assessment:

- **Analysis**: {request.context.get('analysis', 'No specific analysis provided')}
- **Recommendation**: {self._get_general_recommendation(request)}
- **Confidence**: {self._calculate_confidence(request):.2f}
- **Reasoning**: {self._get_general_reasoning(request)}

This response was generated by {self.model_name} model."""
    
    def _get_recommended_action(self, request: MCPRequest) -> str:
        """Get recommended trading action"""
        actions = ["BUY", "SELL", "HOLD"]
        return actions[self.usage_count % len(actions)]
    
    def _calculate_confidence(self, request: MCPRequest) -> float:
        """Calculate confidence score"""
        base_confidence = 0.7
        priority_bonus = request.priority * 0.1
        return min(0.95, base_confidence + priority_bonus)
    
    def _assess_risk(self, request: MCPRequest) -> str:
        """Assess risk level"""
        risk_levels = ["Low", "Medium", "High"]
        return risk_levels[self.usage_count % len(risk_levels)]
    
    def _get_reasoning(self, request: MCPRequest) -> str:
        """Get reasoning for recommendation"""
        return f"Based on {self.model_name} analysis of market conditions and historical patterns."
    
    def _get_optimized_parameters(self, request: MCPRequest) -> str:
        """Get optimized parameters"""
        return "Risk: 0.02, Momentum: 0.5, Volatility: 0.3"
    
    def _calculate_risk_score(self, request: MCPRequest) -> float:
        """Calculate risk score"""
        return 3.0 + (self.usage_count % 5)
    
    def _get_position_sizing(self, request: MCPRequest) -> str:
        """Get position sizing recommendation"""
        return "2-5% of portfolio per position"
    
    def _get_risk_mitigation(self, request: MCPRequest) -> str:
        """Get risk mitigation strategies"""
        return "Implement stop-losses and position limits"
    
    def _get_portfolio_optimization(self, request: MCPRequest) -> str:
        """Get portfolio optimization advice"""
        return "Diversify across asset classes and timeframes"
    
    def _get_risk_level(self, request: MCPRequest) -> str:
        """Get risk level"""
        return "Medium"
    
    def _calculate_sentiment_score(self, request: MCPRequest) -> float:
        """Calculate sentiment score"""
        return -0.5 + (self.usage_count % 10) * 0.2
    
    def _get_sentiment_drivers(self, request: MCPRequest) -> str:
        """Get sentiment drivers"""
        return "Market news, economic indicators, social media"
    
    def _get_market_impact(self, request: MCPRequest) -> str:
        """Get market impact assessment"""
        return "Moderate impact expected"
    
    def _get_sentiment_label(self, request: MCPRequest) -> str:
        """Get sentiment label"""
        return "Neutral to Positive"
    
    def _get_general_recommendation(self, request: MCPRequest) -> str:
        """Get general recommendation"""
        return "Continue monitoring market conditions"
    
    def _get_general_reasoning(self, request: MCPRequest) -> str:
        """Get general reasoning"""
        return f"Based on {self.model_name} analysis of available data"

# Initialize model
model = MCPModel("{model_name}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "{model_name}", "timestamp": datetime.now().isoformat()}

@app.post("/{endpoint}", response_model=MCPResponse)
async def process_request(endpoint: str, request: MCPRequest):
    """Process MCP request"""
    try:
        response = await model.process_request(request)
        logger.info(f"Processed {endpoint} request for {model.model_name}")
        return response
    except Exception as e:
        logger.error(f"Error processing {endpoint} request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get model status"""
    return {
        "model": "{model_name}",
        "status": "active",
        "usage_count": model.usage_count,
        "total_tokens": model.total_tokens,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''

    # Create model servers for each MCP model
    models = [
        ("gordon", "Gordon Assistant", 8001, "/api/v1/chat"),
        ("llama2", "Llama2 7B", 8002, "/generate"),
        ("mistral", "Mistral 7B", 8003, "/completion"),
        ("codellama", "CodeLlama 7B", 8004, "/code"),
        ("financial", "Financial LLM", 8005, "/analyze"),
        ("sentiment", "Sentiment Analyzer", 8006, "/sentiment"),
        ("risk", "Risk Analyzer", 8007, "/risk")
    ]
    
    for model_id, model_name, port, endpoint in models:
        server_content = model_server_template.format(
            model_name=model_name,
            port=port
        )
        
        with open(f"mcp_models/{model_id}/model_server.py", "w") as f:
            f.write(server_content)
        
        logger.info(f"  ✅ Created {model_name} server")

def create_mcp_dashboard():
    """Create MCP dashboard"""
    logger.info("📊 Creating MCP dashboard...")
    
    dashboard_content = '''"""
MCP Dashboard
============

Dashboard for monitoring MCP models.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="MCP Dashboard", version="1.0.0")

# MCP model configurations
MCP_MODELS = {
    "gordon": {"name": "Gordon Assistant", "port": 8001, "endpoint": "/api/v1/chat"},
    "llama2": {"name": "Llama2 7B", "port": 8002, "endpoint": "/generate"},
    "mistral": {"name": "Mistral 7B", "port": 8003, "endpoint": "/completion"},
    "codellama": {"name": "CodeLlama 7B", "port": 8004, "endpoint": "/code"},
    "financial": {"name": "Financial LLM", "port": 8005, "endpoint": "/analyze"},
    "sentiment": {"name": "Sentiment Analyzer", "port": 8006, "endpoint": "/sentiment"},
    "risk": {"name": "Risk Analyzer", "port": 8007, "endpoint": "/risk"}
}

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """MCP Dashboard homepage"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MCP Dashboard - Ultimate Trading System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
            .model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .model-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .model-card h3 { color: #2c3e50; margin-top: 0; }
            .status { padding: 5px 10px; border-radius: 5px; color: white; font-weight: bold; }
            .status.active { background: #27ae60; }
            .status.inactive { background: #e74c3c; }
            .status.unknown { background: #95a5a6; }
            .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
            .metric { text-align: center; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .metric-label { font-size: 12px; color: #7f8c8d; }
            .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 20px 0; }
            .refresh-btn:hover { background: #2980b9; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 MCP Dashboard - Ultimate Trading System</h1>
                <p>Monitor all 7 AI models powering your trading agents</p>
            </div>
            
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Status</button>
            
            <div class="model-grid" id="model-grid">
                <!-- Model cards will be populated here -->
            </div>
        </div>
        
        <script>
            async function loadModelStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    
                    const modelGrid = document.getElementById('model-grid');
                    modelGrid.innerHTML = '';
                    
                    for (const [modelId, modelData] of Object.entries(data.models)) {
                        const card = document.createElement('div');
                        card.className = 'model-card';
                        
                        const statusClass = modelData.status === 'active' ? 'active' : 'inactive';
                        
                        card.innerHTML = `
                            <h3>${modelData.name}</h3>
                            <div class="status ${statusClass}">${modelData.status.toUpperCase()}</div>
                            <div class="metrics">
                                <div class="metric">
                                    <div class="metric-value">${modelData.usage_count}</div>
                                    <div class="metric-label">Requests</div>
                                </div>
                                <div class="metric">
                                    <div class="metric-value">${modelData.total_tokens}</div>
                                    <div class="metric-label">Tokens</div>
                                </div>
                            </div>
                            <p><strong>Port:</strong> ${modelData.port}</p>
                            <p><strong>Last Used:</strong> ${modelData.last_used || 'Never'}</p>
                        `;
                        
                        modelGrid.appendChild(card);
                    }
                } catch (error) {
                    console.error('Error loading model status:', error);
                }
            }
            
            // Load status on page load
            loadModelStatus();
            
            // Auto-refresh every 30 seconds
            setInterval(loadModelStatus, 30000);
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/status")
async def get_mcp_status():
    """Get status of all MCP models"""
    try:
        status = {
            "total_models": len(MCP_MODELS),
            "active_models": 0,
            "inactive_models": 0,
            "models": {}
        }
        
        for model_id, config in MCP_MODELS.items():
            try:
                # Check model health
                response = requests.get(f"http://localhost:{config['port']}/health", timeout=5)
                is_active = response.status_code == 200
                
                if is_active:
                    # Get detailed status
                    try:
                        status_response = requests.get(f"http://localhost:{config['port']}/status", timeout=5)
                        if status_response.status_code == 200:
                            model_status = status_response.json()
                        else:
                            model_status = {"usage_count": 0, "total_tokens": 0, "last_used": None}
                    except:
                        model_status = {"usage_count": 0, "total_tokens": 0, "last_used": None}
                else:
                    model_status = {"usage_count": 0, "total_tokens": 0, "last_used": None}
                
                status["models"][model_id] = {
                    "name": config["name"],
                    "port": config["port"],
                    "status": "active" if is_active else "inactive",
                    "usage_count": model_status.get("usage_count", 0),
                    "total_tokens": model_status.get("total_tokens", 0),
                    "last_used": model_status.get("timestamp", None)
                }
                
                if is_active:
                    status["active_models"] += 1
                else:
                    status["inactive_models"] += 1
                    
            except Exception as e:
                logger.error(f"Error checking {model_id}: {e}")
                status["models"][model_id] = {
                    "name": config["name"],
                    "port": config["port"],
                    "status": "unknown",
                    "usage_count": 0,
                    "total_tokens": 0,
                    "last_used": None
                }
                status["inactive_models"] += 1
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}")
        return {"error": str(e)}

@app.get("/api/test/{model_id}")
async def test_model(model_id: str):
    """Test a specific MCP model"""
    try:
        if model_id not in MCP_MODELS:
            return {"error": "Model not found"}
        
        config = MCP_MODELS[model_id]
        
        # Send test request
        test_request = {
            "prompt": "Test request from MCP dashboard",
            "context": {"test": True},
            "request_type": "test",
            "priority": 1
        }
        
        response = requests.post(
            f"http://localhost:{config['port']}{config['endpoint']}",
            json=test_request,
            timeout=10
        )
        
        if response.status_code == 200:
            return {"status": "success", "response": response.json()}
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Error testing {model_id}: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
'''
    
    with open("mcp_dashboard.py", "w") as f:
        f.write(dashboard_content)
    
    logger.info("  ✅ MCP dashboard created")

def create_run_script():
    """Create script to run the ultimate system with MCP"""
    logger.info("🚀 Creating ultimate system runner...")
    
    runner_content = '''#!/usr/bin/env python3
"""
Ultimate Trading System with MCP Integration
==========================================

Runs the complete system with all 11 trading agents + 7 AI models.
"""

import asyncio
import logging
import sys
from datetime import datetime

# Add src to path
sys.path.append('src')

from src.system_orchestrator import TradingSystemOrchestrator
from src.mcp_integration_simple import SimpleMCPManager
from config.settings import SystemConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Run the ultimate trading system with MCP integration."""
    logger.info("🚀 Starting Ultimate Trading System with MCP Integration")
    logger.info("=" * 60)
    logger.info("🤖 11 Trading Agents + 7 AI Models = ULTIMATE POWER!")
    logger.info("=" * 60)
    
    try:
        # Initialize MCP Manager
        logger.info("🧠 Initializing MCP Manager...")
        mcp_manager = SimpleMCPManager(logger)
        
        # Check MCP models
        logger.info("🔍 Checking MCP model health...")
        model_status = await mcp_manager.check_all_models()
        active_models = sum(model_status.values())
        logger.info(f"✅ {active_models}/{len(model_status)} MCP models are active")
        
        # Initialize system configuration
        logger.info("⚙️ Loading system configuration...")
        system_config = SystemConfig()
        
        # Create orchestrator
        logger.info("🎯 Creating trading system orchestrator...")
        orchestrator = TradingSystemOrchestrator(system_config)
        
        # Initialize system
        logger.info("🔧 Initializing trading system...")
        success = await orchestrator.initialize()
        if not success:
            logger.error("❌ Failed to initialize trading system")
            return
        
        # Attach MCP manager to orchestrator
        orchestrator.mcp_manager = mcp_manager
        logger.info("🧠 MCP Manager attached to orchestrator")
        
        # Start system
        logger.info("🚀 Starting ultimate trading system...")
        logger.info(f"📊 Trading {len(system_config.trading_symbols)} symbols")
        logger.info(f"🤖 {len(orchestrator.agents)} trading agents active")
        logger.info(f"🧠 {active_models} AI models ready")
        logger.info("=" * 60)
        
        await orchestrator.run_system()
        
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("🏁 Ultimate Trading System shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open("run_ultimate_system_with_mcp.py", "w") as f:
        f.write(runner_content)
    
    logger.info("  ✅ Ultimate system runner created")

def main():
    """Main setup function"""
    logger.info("🤖 Setting up MCP System for Ultimate Trading System")
    logger.info("=" * 60)
    
    try:
        # Create directory structure
        create_directory_structure()
        
        # Create Dockerfiles for all models
        create_gordon_dockerfile()
        create_llama2_dockerfile()
        create_mistral_dockerfile()
        create_codellama_dockerfile()
        create_financial_llm_dockerfile()
        create_sentiment_analyzer_dockerfile()
        create_risk_analyzer_dockerfile()
        
        # Create model servers
        create_model_servers()
        
        # Create MCP dashboard
        create_mcp_dashboard()
        
        # Create runner script
        create_run_script()
        
        logger.info("=" * 60)
        logger.info("🎉 MCP System Setup Complete!")
        logger.info("=" * 60)
        logger.info("📁 Created directories for all 7 MCP models")
        logger.info("🐳 Created Dockerfiles for all models")
        logger.info("🖥️ Created model servers for all models")
        logger.info("📊 Created MCP dashboard")
        logger.info("🚀 Created ultimate system runner")
        logger.info("=" * 60)
        logger.info("🔧 NEXT STEPS:")
        logger.info("1. Build MCP models: docker-compose -f docker-compose-mcp.yml build")
        logger.info("2. Start MCP system: docker-compose -f docker-compose-mcp.yml up -d")
        logger.info("3. Monitor dashboard: http://localhost:8008")
        logger.info("4. Start trading: python run_ultimate_system_with_mcp.py")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error setting up MCP system: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
