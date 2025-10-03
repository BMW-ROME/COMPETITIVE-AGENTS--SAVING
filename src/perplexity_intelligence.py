"""
Perplexity AI Integration for Real-Time Market Intelligence
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

class PerplexityIntelligence:
    """Perplexity AI integration for market intelligence and sentiment analysis"""
    
    def __init__(self, api_key: str, logger: logging.Logger):
        self.api_key = api_key
        self.logger = logger
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.session = None
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_market_intelligence(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive market intelligence for symbols"""
        try:
            intelligence = {}
            
            for symbol in symbols:
                # Check cache first
                cache_key = f"intelligence_{symbol}"
                if cache_key in self.cache:
                    cached_data, timestamp = self.cache[cache_key]
                    if datetime.now() - timestamp < timedelta(seconds=self.cache_duration):
                        intelligence[symbol] = cached_data
                        continue
                
                # Get fresh intelligence
                symbol_intelligence = await self._analyze_symbol(symbol)
                intelligence[symbol] = symbol_intelligence
                
                # Cache the result
                self.cache[cache_key] = (symbol_intelligence, datetime.now())
            
            return intelligence
            
        except Exception as e:
            self.logger.error(f"Error getting market intelligence: {e}")
            return {}
    
    async def _analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a specific symbol for market intelligence"""
        try:
            # Create comprehensive analysis prompt
            prompt = f"""
            Analyze the current market situation for {symbol}. Provide:
            1. Recent news sentiment (bullish/bearish/neutral)
            2. Key market drivers and catalysts
            3. Technical analysis insights
            4. Risk factors and concerns
            5. Short-term price outlook (1-3 days)
            6. Key support and resistance levels
            7. Volume analysis
            8. Market sentiment score (0-100)
            
            Format as JSON with these keys: sentiment, news_summary, catalysts, technical_insights, risk_factors, outlook, support_resistance, volume_analysis, sentiment_score
            """
            
            response = await self._query_perplexity(prompt)
            
            if response:
                return self._parse_intelligence_response(response, symbol)
            else:
                return self._get_fallback_intelligence(symbol)
                
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return self._get_fallback_intelligence(symbol)
    
    async def _query_perplexity(self, prompt: str) -> Optional[str]:
        """Query Perplexity API"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            }
            
            async with self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    self.logger.warning(f"Perplexity API error: {response.status}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error querying Perplexity: {e}")
            return None
    
    def _parse_intelligence_response(self, response: str, symbol: str) -> Dict[str, Any]:
        """Parse Perplexity response into structured intelligence"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response
            
            intelligence = json.loads(json_str)
            
            # Ensure all required keys exist
            required_keys = [
                "sentiment", "news_summary", "catalysts", "technical_insights",
                "risk_factors", "outlook", "support_resistance", "volume_analysis", "sentiment_score"
            ]
            
            for key in required_keys:
                if key not in intelligence:
                    intelligence[key] = "Not available"
            
            return intelligence
            
        except Exception as e:
            self.logger.warning(f"Error parsing intelligence response for {symbol}: {e}")
            return self._get_fallback_intelligence(symbol)
    
    def _get_fallback_intelligence(self, symbol: str) -> Dict[str, Any]:
        """Fallback intelligence when API fails"""
        return {
            "sentiment": "neutral",
            "news_summary": f"Limited data available for {symbol}",
            "catalysts": "Market conditions normal",
            "technical_insights": "Standard technical analysis recommended",
            "risk_factors": "General market risks apply",
            "outlook": "Neutral outlook",
            "support_resistance": "Standard levels apply",
            "volume_analysis": "Normal volume patterns",
            "sentiment_score": 50
        }
    
    async def get_sector_analysis(self, sectors: List[str]) -> Dict[str, Any]:
        """Get sector-wide market intelligence"""
        try:
            sector_intelligence = {}
            
            for sector in sectors:
                prompt = f"""
                Analyze the current market conditions for the {sector} sector. Provide:
                1. Sector performance trends
                2. Key sector drivers
                3. Top performing stocks
                4. Sector risks and opportunities
                5. Overall sector sentiment (0-100)
                
                Format as JSON with: performance_trend, drivers, top_stocks, risks_opportunities, sentiment_score
                """
                
                response = await self._query_perplexity(prompt)
                
                if response:
                    try:
                        if "```json" in response:
                            json_start = response.find("```json") + 7
                            json_end = response.find("```", json_start)
                            json_str = response[json_start:json_end].strip()
                        else:
                            json_str = response
                        
                        sector_data = json.loads(json_str)
                        sector_intelligence[sector] = sector_data
                    except:
                        sector_intelligence[sector] = self._get_fallback_sector_analysis(sector)
                else:
                    sector_intelligence[sector] = self._get_fallback_sector_analysis(sector)
            
            return sector_intelligence
            
        except Exception as e:
            self.logger.error(f"Error getting sector analysis: {e}")
            return {}
    
    def _get_fallback_sector_analysis(self, sector: str) -> Dict[str, Any]:
        """Fallback sector analysis"""
        return {
            "performance_trend": "Neutral",
            "drivers": "Standard market factors",
            "top_stocks": "Sector leaders performing normally",
            "risks_opportunities": "Standard sector risks and opportunities",
            "sentiment_score": 50
        }
    
    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment"""
        try:
            prompt = """
            Analyze the current overall market sentiment. Provide:
            1. Market mood (bullish/bearish/neutral)
            2. Key market themes
            3. Risk appetite level (0-100)
            4. Market volatility outlook
            5. Key economic indicators to watch
            
            Format as JSON with: mood, themes, risk_appetite, volatility_outlook, key_indicators
            """
            
            response = await self._query_perplexity(prompt)
            
            if response:
                try:
                    if "```json" in response:
                        json_start = response.find("```json") + 7
                        json_end = response.find("```", json_start)
                        json_str = response[json_start:json_end].strip()
                    else:
                        json_str = response
                    
                    return json.loads(json_str)
                except:
                    return self._get_fallback_market_sentiment()
            else:
                return self._get_fallback_market_sentiment()
                
        except Exception as e:
            self.logger.error(f"Error getting market sentiment: {e}")
            return self._get_fallback_market_sentiment()
    
    def _get_fallback_market_sentiment(self) -> Dict[str, Any]:
        """Fallback market sentiment"""
        return {
            "mood": "neutral",
            "themes": "Standard market conditions",
            "risk_appetite": 50,
            "volatility_outlook": "Normal volatility expected",
            "key_indicators": "Standard economic indicators"
        }
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of cached intelligence"""
        return {
            "cached_symbols": list(self.cache.keys()),
            "cache_size": len(self.cache),
            "api_available": bool(self.api_key)
        }



