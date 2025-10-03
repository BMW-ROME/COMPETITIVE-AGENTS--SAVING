"""
Cross-Asset Correlation Analysis and Portfolio Optimization
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

class AssetClass(Enum):
    """Asset class categories"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"

@dataclass
class CorrelationData:
    """Correlation data between two assets"""
    asset1: str
    asset2: str
    correlation: float
    p_value: float
    confidence: float
    period: str
    timestamp: datetime

@dataclass
class PortfolioOptimization:
    """Portfolio optimization result"""
    symbol: str
    recommended_weight: float
    expected_return: float
    risk: float
    sharpe_ratio: float
    correlation_penalty: float
    diversification_benefit: float

@dataclass
class CrossAssetAnalysis:
    """Cross-asset analysis result"""
    timestamp: datetime
    correlations: Dict[str, CorrelationData]
    portfolio_optimization: Dict[str, PortfolioOptimization]
    sector_correlations: Dict[str, float]
    risk_clusters: List[List[str]]
    diversification_score: float
    recommendations: List[str]

class CorrelationAnalyzer:
    """Cross-asset correlation analyzer and portfolio optimizer"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.correlation_cache = {}
        self.price_history = {}
        self.asset_classes = {
            # Stocks
            'AAPL': AssetClass.STOCK, 'MSFT': AssetClass.STOCK, 'GOOGL': AssetClass.STOCK,
            'TSLA': AssetClass.STOCK, 'AMZN': AssetClass.STOCK, 'NVDA': AssetClass.STOCK,
            'META': AssetClass.STOCK, 'NFLX': AssetClass.STOCK, 'ADBE': AssetClass.STOCK,
            'CRM': AssetClass.STOCK, 'JPM': AssetClass.STOCK, 'BAC': AssetClass.STOCK,
            'WFC': AssetClass.STOCK, 'GS': AssetClass.STOCK, 'MS': AssetClass.STOCK,
            'C': AssetClass.STOCK, 'AXP': AssetClass.STOCK, 'BLK': AssetClass.STOCK,
            'SPGI': AssetClass.STOCK, 'V': AssetClass.STOCK,
            
            # Crypto
            'BTC': AssetClass.CRYPTO, 'ETH': AssetClass.CRYPTO, 'ADA': AssetClass.CRYPTO,
            'DOT': AssetClass.CRYPTO, 'LINK': AssetClass.CRYPTO, 'UNI': AssetClass.CRYPTO,
            'AAVE': AssetClass.CRYPTO, 'MATIC': AssetClass.CRYPTO, 'SOL': AssetClass.CRYPTO,
            
            # Forex
            'USDJPY': AssetClass.FOREX, 'EURUSD': AssetClass.FOREX, 'GBPUSD': AssetClass.FOREX,
            'USDCHF': AssetClass.FOREX, 'USDCAD': AssetClass.FOREX, 'AUDUSD': AssetClass.FOREX,
            'NZDUSD': AssetClass.FOREX, 'EURJPY': AssetClass.FOREX, 'GBPJPY': AssetClass.FOREX,
            'EURGBP': AssetClass.FOREX, 'CHFJPY': AssetClass.FOREX, 'AUDJPY': AssetClass.FOREX,
            'CADJPY': AssetClass.FOREX
        }
        
        # Sector classifications for stocks
        self.sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'TSLA': 'Automotive', 'AMZN': 'Consumer Discretionary', 'NVDA': 'Technology',
            'META': 'Technology', 'NFLX': 'Communication Services', 'ADBE': 'Technology',
            'CRM': 'Technology', 'JPM': 'Financial Services', 'BAC': 'Financial Services',
            'WFC': 'Financial Services', 'GS': 'Financial Services', 'MS': 'Financial Services',
            'C': 'Financial Services', 'AXP': 'Financial Services', 'BLK': 'Financial Services',
            'SPGI': 'Financial Services', 'V': 'Financial Services'
        }
    
    async def analyze_correlations(self, market_data: Dict[str, Any], lookback_days: int = 30) -> CrossAssetAnalysis:
        """Perform comprehensive cross-asset correlation analysis"""
        try:
            self.logger.info(f"Analyzing correlations for {len(market_data)} assets")
            
            # Update price history
            await self._update_price_history(market_data)
            
            # Calculate correlations
            correlations = await self._calculate_correlations(lookback_days)
            
            # Portfolio optimization
            portfolio_optimization = await self._optimize_portfolio(market_data)
            
            # Sector analysis
            sector_correlations = await self._analyze_sector_correlations()
            
            # Risk clustering
            risk_clusters = await self._identify_risk_clusters(correlations)
            
            # Diversification analysis
            diversification_score = await self._calculate_diversification_score(correlations)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                correlations, portfolio_optimization, risk_clusters, diversification_score
            )
            
            analysis = CrossAssetAnalysis(
                timestamp=datetime.now(),
                correlations=correlations,
                portfolio_optimization=portfolio_optimization,
                sector_correlations=sector_correlations,
                risk_clusters=risk_clusters,
                diversification_score=diversification_score,
                recommendations=recommendations
            )
            
            # Cache analysis
            self.correlation_cache = correlations
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return None
    
    async def _update_price_history(self, market_data: Dict[str, Any]):
        """Update price history for correlation calculations"""
        try:
            for symbol, data in market_data.items():
                if 'bars' in data and data['bars']:
                    # Extract price history
                    prices = []
                    timestamps = []
                    
                    for bar in data['bars'][-100:]:  # Last 100 bars
                        prices.append(bar.close)
                        timestamps.append(bar.timestamp)
                    
                    if len(prices) > 10:
                        self.price_history[symbol] = {
                            'prices': prices,
                            'timestamps': timestamps,
                            'returns': self._calculate_returns(prices)
                        }
            
            self.logger.info(f"Updated price history for {len(self.price_history)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error updating price history: {e}")
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price series"""
        try:
            if len(prices) < 2:
                return []
            
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] != 0:
                    returns.append((prices[i] - prices[i-1]) / prices[i-1])
                else:
                    returns.append(0)
            
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return []
    
    async def _calculate_correlations(self, lookback_days: int) -> Dict[str, CorrelationData]:
        """Calculate correlations between all asset pairs"""
        try:
            correlations = {}
            symbols = list(self.price_history.keys())
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    try:
                        # Get returns for both symbols
                        returns1 = self.price_history[symbol1]['returns']
                        returns2 = self.price_history[symbol2]['returns']
                        
                        if len(returns1) < 10 or len(returns2) < 10:
                            continue
                        
                        # Align returns (take minimum length)
                        min_length = min(len(returns1), len(returns2))
                        returns1_aligned = returns1[-min_length:]
                        returns2_aligned = returns2[-min_length:]
                        
                        # Calculate correlation
                        correlation_matrix = np.corrcoef(returns1_aligned, returns2_aligned)
                        correlation = correlation_matrix[0, 1]
                        
                        # Calculate p-value (simplified)
                        n = len(returns1_aligned)
                        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2)) if correlation != 1 else 0
                        p_value = 2 * (1 - self._t_cdf(abs(t_stat), n - 2)) if n > 2 else 0.5
                        
                        # Calculate confidence
                        confidence = 1 - p_value if not np.isnan(p_value) else 0.5
                        
                        # Create correlation data
                        pair_key = f"{symbol1}_{symbol2}"
                        correlations[pair_key] = CorrelationData(
                            asset1=symbol1,
                            asset2=symbol2,
                            correlation=correlation,
                            p_value=p_value,
                            confidence=confidence,
                            period=f"{lookback_days}d",
                            timestamp=datetime.now()
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Error calculating correlation between {symbol1} and {symbol2}: {e}")
                        continue
            
            self.logger.info(f"Calculated {len(correlations)} correlations")
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
            return {}
    
    def _t_cdf(self, t: float, df: int) -> float:
        """Simplified t-distribution CDF"""
        try:
            # This is a simplified approximation
            # In production, you'd use scipy.stats.t.cdf
            if df <= 0:
                return 0.5
            
            # Simple approximation for t-distribution
            if abs(t) < 1:
                return 0.5 + t * 0.3
            elif abs(t) < 2:
                return 0.5 + np.sign(t) * (0.3 + (abs(t) - 1) * 0.2)
            else:
                return 0.5 + np.sign(t) * 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating t-cdf: {e}")
            return 0.5
    
    async def _optimize_portfolio(self, market_data: Dict[str, Any]) -> Dict[str, PortfolioOptimization]:
        """Optimize portfolio weights using correlation analysis"""
        try:
            optimization = {}
            symbols = list(self.price_history.keys())
            
            if len(symbols) < 2:
                return optimization
            
            # Calculate expected returns and risks
            expected_returns = {}
            risks = {}
            
            for symbol in symbols:
                returns = self.price_history[symbol]['returns']
                if len(returns) > 5:
                    expected_returns[symbol] = np.mean(returns) * 252  # Annualized
                    risks[symbol] = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate correlation matrix
            correlation_matrix = np.eye(len(symbols))
            symbol_list = list(symbols)
            
            for i, symbol1 in enumerate(symbol_list):
                for j, symbol2 in enumerate(symbol_list):
                    if i != j:
                        pair_key = f"{symbol1}_{symbol2}" if symbol1 < symbol2 else f"{symbol2}_{symbol1}"
                        if pair_key in self.correlation_cache:
                            correlation_matrix[i, j] = self.correlation_cache[pair_key].correlation
                        else:
                            correlation_matrix[i, j] = 0.3  # Default correlation
            
            # Portfolio optimization using simplified Markowitz
            for i, symbol in enumerate(symbol_list):
                try:
                    # Calculate individual metrics
                    expected_return = expected_returns.get(symbol, 0.05)
                    risk = risks.get(symbol, 0.2)
                    
                    # Calculate correlation penalty
                    correlation_penalty = 0
                    for j, other_symbol in enumerate(symbol_list):
                        if i != j:
                            correlation_penalty += abs(correlation_matrix[i, j]) * 0.1
                    
                    # Calculate diversification benefit
                    diversification_benefit = 1 - correlation_penalty
                    
                    # Calculate Sharpe ratio (simplified)
                    risk_free_rate = 0.02
                    sharpe_ratio = (expected_return - risk_free_rate) / risk if risk > 0 else 0
                    
                    # Calculate recommended weight (simplified)
                    # Higher Sharpe ratio and lower correlation = higher weight
                    base_weight = 1.0 / len(symbols)  # Equal weight base
                    sharpe_adjustment = max(0.5, min(2.0, sharpe_ratio + 1))  # Scale Sharpe ratio
                    correlation_adjustment = diversification_benefit
                    
                    recommended_weight = base_weight * sharpe_adjustment * correlation_adjustment
                    
                    # Normalize weight
                    recommended_weight = max(0.01, min(0.3, recommended_weight))  # Cap between 1% and 30%
                    
                    optimization[symbol] = PortfolioOptimization(
                        symbol=symbol,
                        recommended_weight=recommended_weight,
                        expected_return=expected_return,
                        risk=risk,
                        sharpe_ratio=sharpe_ratio,
                        correlation_penalty=correlation_penalty,
                        diversification_benefit=diversification_benefit
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing portfolio for {symbol}: {e}")
                    continue
            
            # Normalize weights to sum to 1
            total_weight = sum(opt.recommended_weight for opt in optimization.values())
            if total_weight > 0:
                for opt in optimization.values():
                    opt.recommended_weight /= total_weight
            
            self.logger.info(f"Optimized portfolio for {len(optimization)} assets")
            return optimization
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            return {}
    
    async def _analyze_sector_correlations(self) -> Dict[str, float]:
        """Analyze correlations within and between sectors"""
        try:
            sector_correlations = {}
            
            # Group symbols by sector
            sector_groups = {}
            for symbol, sector in self.sectors.items():
                if symbol in self.price_history:
                    if sector not in sector_groups:
                        sector_groups[sector] = []
                    sector_groups[sector].append(symbol)
            
            # Calculate intra-sector correlations
            for sector, symbols in sector_groups.items():
                if len(symbols) >= 2:
                    correlations = []
                    for i, symbol1 in enumerate(symbols):
                        for symbol2 in symbols[i+1:]:
                            pair_key = f"{symbol1}_{symbol2}" if symbol1 < symbol2 else f"{symbol2}_{symbol1}"
                            if pair_key in self.correlation_cache:
                                correlations.append(self.correlation_cache[pair_key].correlation)
                    
                    if correlations:
                        sector_correlations[f"{sector}_intra"] = np.mean(correlations)
            
            # Calculate inter-sector correlations
            sectors = list(sector_groups.keys())
            for i, sector1 in enumerate(sectors):
                for sector2 in sectors[i+1:]:
                    correlations = []
                    for symbol1 in sector_groups[sector1]:
                        for symbol2 in sector_groups[sector2]:
                            pair_key = f"{symbol1}_{symbol2}" if symbol1 < symbol2 else f"{symbol2}_{symbol1}"
                            if pair_key in self.correlation_cache:
                                correlations.append(self.correlation_cache[pair_key].correlation)
                    
                    if correlations:
                        sector_correlations[f"{sector1}_{sector2}"] = np.mean(correlations)
            
            return sector_correlations
            
        except Exception as e:
            self.logger.error(f"Error analyzing sector correlations: {e}")
            return {}
    
    async def _identify_risk_clusters(self, correlations: Dict[str, CorrelationData]) -> List[List[str]]:
        """Identify clusters of highly correlated assets"""
        try:
            # Build correlation matrix
            symbols = set()
            for corr in correlations.values():
                symbols.add(corr.asset1)
                symbols.add(corr.asset2)
            
            symbols = list(symbols)
            n = len(symbols)
            correlation_matrix = np.eye(n)
            
            # Fill correlation matrix
            for corr in correlations.values():
                i = symbols.index(corr.asset1)
                j = symbols.index(corr.asset2)
                correlation_matrix[i, j] = corr.correlation
                correlation_matrix[j, i] = corr.correlation
            
            # Simple clustering based on high correlation threshold
            threshold = 0.7
            clusters = []
            visited = set()
            
            for i, symbol1 in enumerate(symbols):
                if symbol1 in visited:
                    continue
                
                cluster = [symbol1]
                visited.add(symbol1)
                
                for j, symbol2 in enumerate(symbols):
                    if symbol2 not in visited and correlation_matrix[i, j] > threshold:
                        cluster.append(symbol2)
                        visited.add(symbol2)
                
                if len(cluster) > 1:  # Only include clusters with multiple assets
                    clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error identifying risk clusters: {e}")
            return []
    
    async def _calculate_diversification_score(self, correlations: Dict[str, CorrelationData]) -> float:
        """Calculate portfolio diversification score"""
        try:
            if not correlations:
                return 0.0
            
            # Calculate average correlation
            correlations_list = [abs(corr.correlation) for corr in correlations.values()]
            avg_correlation = np.mean(correlations_list)
            
            # Diversification score (lower correlation = higher diversification)
            diversification_score = 1 - avg_correlation
            
            return max(0.0, min(1.0, diversification_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification score: {e}")
            return 0.5
    
    async def _generate_recommendations(self, correlations: Dict[str, CorrelationData], 
                                      portfolio_optimization: Dict[str, PortfolioOptimization],
                                      risk_clusters: List[List[str]], 
                                      diversification_score: float) -> List[str]:
        """Generate trading and portfolio recommendations"""
        try:
            recommendations = []
            
            # Diversification recommendations
            if diversification_score < 0.3:
                recommendations.append("LOW_DIVERSIFICATION: Consider adding uncorrelated assets")
            elif diversification_score > 0.7:
                recommendations.append("HIGH_DIVERSIFICATION: Portfolio is well diversified")
            
            # Risk cluster recommendations
            for cluster in risk_clusters:
                if len(cluster) > 2:
                    recommendations.append(f"RISK_CLUSTER: {', '.join(cluster)} are highly correlated - consider reducing exposure")
            
            # High correlation warnings
            high_correlations = [corr for corr in correlations.values() if abs(corr.correlation) > 0.8]
            if high_correlations:
                recommendations.append(f"HIGH_CORRELATION: {len(high_correlations)} asset pairs have correlation > 0.8")
            
            # Portfolio optimization recommendations
            high_weight_assets = [opt for opt in portfolio_optimization.values() if opt.recommended_weight > 0.2]
            if high_weight_assets:
                recommendations.append("CONCENTRATION_RISK: Some assets have high recommended weights")
            
            # Sector concentration
            sector_weights = {}
            for opt in portfolio_optimization.values():
                sector = self.sectors.get(opt.symbol, 'Other')
                sector_weights[sector] = sector_weights.get(sector, 0) + opt.recommended_weight
            
            max_sector_weight = max(sector_weights.values()) if sector_weights else 0
            if max_sector_weight > 0.4:
                recommendations.append("SECTOR_CONCENTRATION: Consider diversifying across sectors")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def get_correlation_summary(self) -> Dict[str, Any]:
        """Get summary of correlation analysis"""
        try:
            if not self.correlation_cache:
                return {}
            
            correlations = list(self.correlation_cache.values())
            
            # Calculate statistics
            correlation_values = [corr.correlation for corr in correlations]
            
            summary = {
                'total_pairs': len(correlations),
                'avg_correlation': np.mean(correlation_values),
                'max_correlation': max(correlation_values),
                'min_correlation': min(correlation_values),
                'high_correlations': len([c for c in correlation_values if abs(c) > 0.7]),
                'low_correlations': len([c for c in correlation_values if abs(c) < 0.3]),
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting correlation summary: {e}")
            return {}


