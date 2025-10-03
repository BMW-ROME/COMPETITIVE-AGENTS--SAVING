"""
MCP (Model Context Protocol) Integration for Ultimate Trading System
"""
import asyncio
import logging
import json
import docker
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import requests
import subprocess
import os

class MCPModelType(Enum):
    """Types of MCP models available"""
    GORDON_ASSISTANT = "gordon_assistant"
    LLAMA2 = "llama2"
    MISTRAL = "mistral"
    CODELLAMA = "codellama"
    FINANCIAL_LLM = "financial_llm"
    SENTIMENT_ANALYZER = "sentiment_analyzer"
    RISK_ANALYZER = "risk_analyzer"

@dataclass
class MCPModel:
    """MCP model configuration"""
    name: str
    model_type: MCPModelType
    docker_image: str
    port: int
    endpoint: str
    capabilities: List[str]
    status: str = "inactive"
    last_used: Optional[datetime] = None
    usage_count: int = 0

@dataclass
class MCPRequest:
    """MCP request structure"""
    model: str
    prompt: str
    context: Dict[str, Any]
    request_type: str
    priority: int = 1
    timeout: int = 30

@dataclass
class MCPResponse:
    """MCP response structure"""
    model: str
    response: str
    confidence: float
    processing_time: float
    tokens_used: int
    metadata: Dict[str, Any]

class MCPManager:
    """MCP Manager for Ultimate Trading System"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.docker_client = docker.from_env()
        self.models: Dict[str, MCPModel] = {}
        self.active_connections: Dict[str, Any] = {}
        self.request_queue: List[MCPRequest] = []
        self.response_cache: Dict[str, MCPResponse] = {}
        
        # Initialize available models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available MCP models"""
        self.models = {
            "gordon": MCPModel(
                name="gordon_assistant",
                model_type=MCPModelType.GORDON_ASSISTANT,
                docker_image="gordon-assistant:latest",
                port=8001,
                endpoint="/api/v1/chat",
                capabilities=[
                    "general_conversation",
                    "code_analysis",
                    "strategy_optimization",
                    "risk_assessment",
                    "market_analysis"
                ]
            ),
            "llama2": MCPModel(
                name="llama2_7b",
                model_type=MCPModelType.LLAMA2,
                docker_image="llama2:7b",
                port=8002,
                endpoint="/generate",
                capabilities=[
                    "text_generation",
                    "market_analysis",
                    "sentiment_analysis",
                    "strategy_suggestions"
                ]
            ),
            "mistral": MCPModel(
                name="mistral_7b",
                model_type=MCPModelType.MISTRAL,
                docker_image="mistral:7b",
                port=8003,
                endpoint="/completion",
                capabilities=[
                    "reasoning",
                    "financial_analysis",
                    "risk_calculation",
                    "portfolio_optimization"
                ]
            ),
            "codellama": MCPModel(
                name="codellama_7b",
                model_type=MCPModelType.CODELLAMA,
                docker_image="codellama:7b",
                port=8004,
                endpoint="/code",
                capabilities=[
                    "code_generation",
                    "strategy_implementation",
                    "backtesting_code",
                    "optimization_algorithms"
                ]
            ),
            "financial_llm": MCPModel(
                name="financial_llm",
                model_type=MCPModelType.FINANCIAL_LLM,
                docker_image="financial-llm:latest",
                port=8005,
                endpoint="/analyze",
                capabilities=[
                    "financial_analysis",
                    "market_prediction",
                    "risk_assessment",
                    "portfolio_management"
                ]
            ),
            "sentiment_analyzer": MCPModel(
                name="sentiment_analyzer",
                model_type=MCPModelType.SENTIMENT_ANALYZER,
                docker_image="sentiment-analyzer:latest",
                port=8006,
                endpoint="/sentiment",
                capabilities=[
                    "news_sentiment",
                    "social_media_analysis",
                    "market_sentiment",
                    "sentiment_scoring"
                ]
            ),
            "risk_analyzer": MCPModel(
                name="risk_analyzer",
                model_type=MCPModelType.RISK_ANALYZER,
                docker_image="risk-analyzer:latest",
                port=8007,
                endpoint="/risk",
                capabilities=[
                    "risk_calculation",
                    "portfolio_risk",
                    "position_sizing",
                    "risk_management"
                ]
            )
        }
        
        self.logger.info(f"Initialized {len(self.models)} MCP models")
    
    async def start_model(self, model_name: str) -> bool:
        """Start an MCP model container"""
        try:
            if model_name not in self.models:
                self.logger.error(f"Model {model_name} not found")
                return False
            
            model = self.models[model_name]
            
            # Check if container is already running
            try:
                container = self.docker_client.containers.get(f"mcp-{model_name}")
                if container.status == "running":
                    self.logger.info(f"Model {model_name} is already running")
                    model.status = "active"
                    return True
            except docker.errors.NotFound:
                pass
            
            # Start the container
            container = self.docker_client.containers.run(
                model.docker_image,
                name=f"mcp-{model_name}",
                ports={f"{model.port}/tcp": model.port},
                detach=True,
                restart_policy={"Name": "unless-stopped"},
                environment={
                    "MODEL_NAME": model.name,
                    "PORT": str(model.port),
                    "LOG_LEVEL": "INFO"
                }
            )
            
            # Wait for container to be ready
            await asyncio.sleep(5)
            
            # Test the connection
            if await self._test_model_connection(model_name):
                model.status = "active"
                self.logger.info(f"✅ Model {model_name} started successfully")
                return True
            else:
                self.logger.error(f"❌ Model {model_name} failed to start properly")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting model {model_name}: {e}")
            return False
    
    async def stop_model(self, model_name: str) -> bool:
        """Stop an MCP model container"""
        try:
            if model_name not in self.models:
                self.logger.error(f"Model {model_name} not found")
                return False
            
            model = self.models[model_name]
            
            try:
                container = self.docker_client.containers.get(f"mcp-{model_name}")
                container.stop()
                container.remove()
                model.status = "inactive"
                self.logger.info(f"✅ Model {model_name} stopped successfully")
                return True
            except docker.errors.NotFound:
                self.logger.info(f"Model {model_name} container not found")
                model.status = "inactive"
                return True
                
        except Exception as e:
            self.logger.error(f"Error stopping model {model_name}: {e}")
            return False
    
    async def _test_model_connection(self, model_name: str) -> bool:
        """Test connection to an MCP model"""
        try:
            model = self.models[model_name]
            
            # Simple health check
            response = requests.get(
                f"http://localhost:{model.port}/health",
                timeout=5
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Health check failed for {model_name}: {e}")
            return False
    
    async def send_request(self, request: MCPRequest) -> Optional[MCPResponse]:
        """Send a request to an MCP model"""
        try:
            if request.model not in self.models:
                self.logger.error(f"Model {request.model} not found")
                return None
            
            model = self.models[request.model]
            
            if model.status != "active":
                self.logger.warning(f"Model {request.model} is not active, starting it...")
                if not await self.start_model(request.model):
                    return None
            
            # Prepare request data
            request_data = {
                "prompt": request.prompt,
                "context": request.context,
                "request_type": request.request_type,
                "priority": request.priority
            }
            
            # Send request to model
            start_time = datetime.now()
            
            response = requests.post(
                f"http://localhost:{model.port}{model.endpoint}",
                json=request_data,
                timeout=request.timeout
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if response.status_code == 200:
                response_data = response.json()
                
                mcp_response = MCPResponse(
                    model=request.model,
                    response=response_data.get("response", ""),
                    confidence=response_data.get("confidence", 0.0),
                    processing_time=processing_time,
                    tokens_used=response_data.get("tokens_used", 0),
                    metadata=response_data.get("metadata", {})
                )
                
                # Update model usage
                model.last_used = datetime.now()
                model.usage_count += 1
                
                # Cache response
                cache_key = f"{request.model}:{hash(request.prompt)}"
                self.response_cache[cache_key] = mcp_response
                
                self.logger.info(f"✅ MCP request completed for {request.model} in {processing_time:.2f}s")
                return mcp_response
            else:
                self.logger.error(f"MCP request failed for {request.model}: {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error sending MCP request to {request.model}: {e}")
            return None
    
    async def get_agent_advice(self, agent_id: str, market_data: Dict[str, Any], 
                             decision_context: Dict[str, Any]) -> Optional[MCPResponse]:
        """Get AI advice for a trading agent"""
        try:
            # Determine best model for the request
            model_name = self._select_best_model(decision_context)
            
            # Create request
            request = MCPRequest(
                model=model_name,
                prompt=self._create_agent_prompt(agent_id, market_data, decision_context),
                context={
                    "agent_id": agent_id,
                    "market_data": market_data,
                    "decision_context": decision_context,
                    "timestamp": datetime.now().isoformat()
                },
                request_type="agent_advice",
                priority=1
            )
            
            return await self.send_request(request)
            
        except Exception as e:
            self.logger.error(f"Error getting agent advice: {e}")
            return None
    
    def _select_best_model(self, context: Dict[str, Any]) -> str:
        """Select the best model for the given context"""
        # Simple model selection logic
        if "risk" in context.get("decision_type", "").lower():
            return "risk_analyzer"
        elif "sentiment" in context.get("decision_type", "").lower():
            return "sentiment_analyzer"
        elif "financial" in context.get("decision_type", "").lower():
            return "financial_llm"
        elif "code" in context.get("decision_type", "").lower():
            return "codellama"
        else:
            return "gordon"  # Default to Gordon assistant
    
    def _create_agent_prompt(self, agent_id: str, market_data: Dict[str, Any], 
                           decision_context: Dict[str, Any]) -> str:
        """Create a prompt for the agent"""
        prompt = f"""
        You are an AI assistant helping trading agent {agent_id} make decisions.
        
        Current Market Data:
        {json.dumps(market_data, indent=2)}
        
        Decision Context:
        {json.dumps(decision_context, indent=2)}
        
        Please provide:
        1. Analysis of current market conditions
        2. Recommended action (BUY/SELL/HOLD)
        3. Confidence level (0-1)
        4. Risk assessment
        5. Reasoning for your recommendation
        
        Be concise but thorough in your analysis.
        """
        
        return prompt
    
    async def get_strategy_optimization(self, strategy_data: Dict[str, Any]) -> Optional[MCPResponse]:
        """Get strategy optimization advice"""
        try:
            request = MCPRequest(
                model="codellama",
                prompt=f"""
                Optimize this trading strategy:
                {json.dumps(strategy_data, indent=2)}
                
                Provide:
                1. Optimized parameters
                2. Risk management improvements
                3. Performance enhancements
                4. Code improvements
                """,
                context=strategy_data,
                request_type="strategy_optimization",
                priority=2
            )
            
            return await self.send_request(request)
            
        except Exception as e:
            self.logger.error(f"Error getting strategy optimization: {e}")
            return None
    
    async def get_risk_analysis(self, portfolio_data: Dict[str, Any]) -> Optional[MCPResponse]:
        """Get risk analysis from MCP models"""
        try:
            request = MCPRequest(
                model="risk_analyzer",
                prompt=f"""
                Analyze the risk of this portfolio:
                {json.dumps(portfolio_data, indent=2)}
                
                Provide:
                1. Overall risk score
                2. Position sizing recommendations
                3. Risk mitigation strategies
                4. Portfolio optimization suggestions
                """,
                context=portfolio_data,
                request_type="risk_analysis",
                priority=1
            )
            
            return await self.send_request(request)
            
        except Exception as e:
            self.logger.error(f"Error getting risk analysis: {e}")
            return None
    
    async def get_sentiment_analysis(self, news_data: List[str]) -> Optional[MCPResponse]:
        """Get sentiment analysis from MCP models"""
        try:
            request = MCPRequest(
                model="sentiment_analyzer",
                prompt=f"""
                Analyze the sentiment of these news items:
                {json.dumps(news_data, indent=2)}
                
                Provide:
                1. Overall sentiment score (-1 to 1)
                2. Confidence level
                3. Key sentiment drivers
                4. Market impact assessment
                """,
                context={"news_data": news_data},
                request_type="sentiment_analysis",
                priority=1
            )
            
            return await self.send_request(request)
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment analysis: {e}")
            return None
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all MCP models"""
        try:
            status = {
                "total_models": len(self.models),
                "active_models": 0,
                "inactive_models": 0,
                "models": {}
            }
            
            for name, model in self.models.items():
                model_status = {
                    "name": model.name,
                    "type": model.model_type.value,
                    "status": model.status,
                    "port": model.port,
                    "capabilities": model.capabilities,
                    "usage_count": model.usage_count,
                    "last_used": model.last_used.isoformat() if model.last_used else None
                }
                
                status["models"][name] = model_status
                
                if model.status == "active":
                    status["active_models"] += 1
                else:
                    status["inactive_models"] += 1
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting model status: {e}")
            return {}
    
    async def start_all_models(self) -> Dict[str, bool]:
        """Start all available MCP models"""
        try:
            results = {}
            
            for model_name in self.models.keys():
                self.logger.info(f"Starting model: {model_name}")
                results[model_name] = await self.start_model(model_name)
                await asyncio.sleep(2)  # Stagger startup
            
            active_count = sum(results.values())
            self.logger.info(f"✅ Started {active_count}/{len(self.models)} MCP models")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error starting all models: {e}")
            return {}
    
    async def stop_all_models(self) -> Dict[str, bool]:
        """Stop all MCP models"""
        try:
            results = {}
            
            for model_name in self.models.keys():
                results[model_name] = await self.stop_model(model_name)
            
            self.logger.info("✅ Stopped all MCP models")
            return results
            
        except Exception as e:
            self.logger.error(f"Error stopping all models: {e}")
            return {}
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def get_model_capabilities(self, model_name: str) -> List[str]:
        """Get capabilities of a specific model"""
        if model_name in self.models:
            return self.models[model_name].capabilities
        return []


