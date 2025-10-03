"""
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
