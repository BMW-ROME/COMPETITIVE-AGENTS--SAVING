"""
Advanced Risk Management System with Dynamic Position Sizing
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    symbol: str
    position_size: float
    position_value: float
    portfolio_percentage: float
    risk_per_trade: float
    max_risk: float
    current_drawdown: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    risk_score: float  # Overall risk score 0-100

@dataclass
class PositionSizing:
    """Position sizing recommendation"""
    symbol: str
    recommended_size: float
    max_size: float
    risk_amount: float
    confidence: float
    reasoning: str
    stop_loss: float
    take_profit: float

class AdvancedRiskManager:
    """Advanced risk management system with dynamic position sizing"""
    
    def __init__(self, logger: logging.Logger, initial_capital: float = 100000):
        self.logger = logger
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.risk_metrics = {}
        self.portfolio_history = []
        self.risk_limits = {
            'max_portfolio_risk': 0.02,  # 2% max portfolio risk per trade
            'max_position_size': 0.10,  # 10% max position size
            'max_correlation': 0.7,     # 70% max correlation between positions
            'max_drawdown': 0.15,       # 15% max drawdown
            'max_volatility': 0.5,      # 50% max volatility
            'min_sharpe': 0.5,          # Minimum Sharpe ratio
            'max_positions': 20         # Maximum number of positions
        }
        
        # Risk calculation parameters
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.confidence_level = 0.95  # 95% confidence for VaR
        self.lookback_period = 252  # 1 year of daily data
        
    async def calculate_position_risk(self, symbol: str, position_size: float, 
                                    current_price: float, market_data: Dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a position"""
        try:
            position_value = position_size * current_price
            portfolio_percentage = position_value / self.current_capital
            
            # Calculate volatility
            volatility = await self._calculate_volatility(symbol, market_data)
            
            # Calculate beta (market correlation)
            beta = await self._calculate_beta(symbol, market_data)
            
            # Calculate correlation with existing positions
            correlation = await self._calculate_correlation(symbol, market_data)
            
            # Calculate Value at Risk (VaR)
            var_95 = await self._calculate_var(symbol, position_value, market_data)
            
            # Calculate Expected Shortfall
            expected_shortfall = await self._calculate_expected_shortfall(symbol, position_value, market_data)
            
            # Calculate risk-adjusted returns
            sharpe_ratio = await self._calculate_sharpe_ratio(symbol, market_data)
            sortino_ratio = await self._calculate_sortino_ratio(symbol, market_data)
            calmar_ratio = await self._calculate_calmar_ratio(symbol, market_data)
            
            # Calculate current drawdown
            current_drawdown = await self._calculate_current_drawdown()
            max_drawdown = await self._calculate_max_drawdown()
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(
                portfolio_percentage, volatility, correlation, current_drawdown,
                sharpe_ratio, var_95
            )
            
            # Calculate risk per trade
            risk_per_trade = position_value * volatility
            max_risk = self.current_capital * self.risk_limits['max_portfolio_risk']
            
            return RiskMetrics(
                symbol=symbol,
                position_size=position_size,
                position_value=position_value,
                portfolio_percentage=portfolio_percentage,
                risk_per_trade=risk_per_trade,
                max_risk=max_risk,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                correlation=correlation,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                risk_score=risk_score
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk for {symbol}: {e}")
            return None
    
    async def calculate_optimal_position_size(self, symbol: str, current_price: float,
                                            market_data: Dict, agent_confidence: float,
                                            expected_return: float) -> PositionSizing:
        """Calculate optimal position size using Kelly Criterion and risk management"""
        try:
            # Get market data for the symbol
            symbol_data = market_data.get(symbol, {})
            if not symbol_data:
                return PositionSizing(
                    symbol=symbol,
                    recommended_size=0,
                    max_size=0,
                    risk_amount=0,
                    confidence=0,
                    reasoning="No market data available",
                    stop_loss=0,
                    take_profit=0
                )
            
            # Calculate volatility
            volatility = await self._calculate_volatility(symbol, market_data)
            
            # Calculate expected return and win probability
            win_probability = max(0.5, min(0.9, agent_confidence))
            expected_return = max(0.01, expected_return)  # Minimum 1% expected return
            
            # Kelly Criterion calculation
            kelly_fraction = (win_probability * expected_return - (1 - win_probability)) / expected_return
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
            
            # Risk-based position sizing
            risk_amount = self.current_capital * self.risk_limits['max_portfolio_risk']
            risk_based_size = risk_amount / (volatility * current_price)
            
            # Volatility-based position sizing
            volatility_based_size = (self.current_capital * 0.02) / (volatility * current_price)
            
            # Correlation-based adjustment
            correlation = await self._calculate_correlation(symbol, market_data)
            correlation_adjustment = 1 - (correlation * 0.5)  # Reduce size for high correlation
            
            # Portfolio concentration adjustment
            current_positions = len(self.positions)
            concentration_adjustment = max(0.5, 1 - (current_positions / self.risk_limits['max_positions']))
            
            # Calculate final position size
            kelly_size = kelly_fraction * self.current_capital / current_price
            risk_size = risk_based_size * correlation_adjustment * concentration_adjustment
            volatility_size = volatility_based_size * correlation_adjustment
            
            # Take the minimum of all sizing methods
            recommended_size = min(kelly_size, risk_size, volatility_size)
            
            # Apply maximum position size limit
            max_size = self.current_capital * self.risk_limits['max_position_size'] / current_price
            recommended_size = min(recommended_size, max_size)
            
            # Calculate stop loss and take profit
            stop_loss = current_price * (1 - (volatility * 2))  # 2x volatility stop loss
            take_profit = current_price * (1 + (expected_return * 2))  # 2x expected return target
            
            # Calculate confidence score
            confidence = self._calculate_position_confidence(
                win_probability, expected_return, volatility, correlation
            )
            
            # Generate reasoning
            reasoning = self._generate_position_reasoning(
                kelly_fraction, volatility, correlation, current_positions,
                recommended_size, max_size
            )
            
            return PositionSizing(
                symbol=symbol,
                recommended_size=recommended_size,
                max_size=max_size,
                risk_amount=risk_amount,
                confidence=confidence,
                reasoning=reasoning,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size for {symbol}: {e}")
            return PositionSizing(
                symbol=symbol,
                recommended_size=0,
                max_size=0,
                risk_amount=0,
                confidence=0,
                reasoning=f"Error: {str(e)}",
                stop_loss=0,
                take_profit=0
            )
    
    async def validate_trade(self, symbol: str, action: str, quantity: float,
                           current_price: float, market_data: Dict) -> Tuple[bool, str]:
        """Validate if a trade meets risk management criteria"""
        try:
            # Check if we have enough capital for buy orders
            if action == 'BUY':
                required_capital = quantity * current_price
                if required_capital > self.current_capital * 0.95:  # Leave 5% cash buffer
                    return False, f"Insufficient capital. Required: ${required_capital:.2f}, Available: ${self.current_capital * 0.95:.2f}"
            
            # Check if we have the position for sell orders
            if action == 'SELL':
                if symbol not in self.positions or self.positions[symbol] < quantity:
                    return False, f"Insufficient position. Required: {quantity}, Available: {self.positions.get(symbol, 0)}"
            
            # Calculate position size percentage
            position_value = quantity * current_price
            position_percentage = position_value / self.current_capital
            
            # Check maximum position size
            if position_percentage > self.risk_limits['max_position_size']:
                return False, f"Position size too large. {position_percentage:.2%} exceeds {self.risk_limits['max_position_size']:.2%} limit"
            
            # Check maximum number of positions
            if action == 'BUY' and symbol not in self.positions:
                if len(self.positions) >= self.risk_limits['max_positions']:
                    return False, f"Maximum positions reached. {len(self.positions)}/{self.risk_limits['max_positions']}"
            
            # Calculate risk metrics
            risk_metrics = await self.calculate_position_risk(symbol, quantity, current_price, market_data)
            if risk_metrics:
                # Check risk score
                if risk_metrics.risk_score > 80:  # High risk
                    return False, f"Risk score too high: {risk_metrics.risk_score:.1f}/100"
                
                # Check volatility
                if risk_metrics.volatility > self.risk_limits['max_volatility']:
                    return False, f"Volatility too high: {risk_metrics.volatility:.2%} exceeds {self.risk_limits['max_volatility']:.2%}"
                
                # Check correlation
                if risk_metrics.correlation > self.risk_limits['max_correlation']:
                    return False, f"Correlation too high: {risk_metrics.correlation:.2%} exceeds {self.risk_limits['max_correlation']:.2%}"
            
            # Check portfolio drawdown
            current_drawdown = await self._calculate_current_drawdown()
            if current_drawdown > self.risk_limits['max_drawdown']:
                return False, f"Portfolio drawdown too high: {current_drawdown:.2%} exceeds {self.risk_limits['max_drawdown']:.2%}"
            
            return True, "Trade validated"
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {str(e)}"
    
    async def update_portfolio(self, symbol: str, action: str, quantity: float, price: float):
        """Update portfolio after trade execution"""
        try:
            if action == 'BUY':
                if symbol in self.positions:
                    self.positions[symbol] += quantity
                else:
                    self.positions[symbol] = quantity
                self.current_capital -= quantity * price
                
            elif action == 'SELL':
                if symbol in self.positions:
                    self.positions[symbol] -= quantity
                    if self.positions[symbol] <= 0:
                        del self.positions[symbol]
                self.current_capital += quantity * price
            
            # Update portfolio history
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'portfolio_value': self.current_capital + sum(
                    pos * price for pos, price in self.positions.items()
                ),
                'positions': self.positions.copy()
            })
            
            self.logger.info(f"Portfolio updated: {action} {quantity} {symbol} at ${price:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    async def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary"""
        try:
            total_portfolio_value = self.current_capital + sum(
                pos * 100 for pos in self.positions.values()  # Assuming $100 average price
            )
            
            # Calculate portfolio metrics
            current_drawdown = await self._calculate_current_drawdown()
            max_drawdown = await self._calculate_max_drawdown()
            
            # Calculate portfolio volatility
            portfolio_volatility = await self._calculate_portfolio_volatility()
            
            # Calculate portfolio beta
            portfolio_beta = await self._calculate_portfolio_beta()
            
            # Calculate portfolio VaR
            portfolio_var = await self._calculate_portfolio_var()
            
            # Calculate Sharpe ratio
            portfolio_sharpe = await self._calculate_portfolio_sharpe()
            
            # Calculate concentration risk
            concentration_risk = self._calculate_concentration_risk()
            
            # Calculate correlation risk
            correlation_risk = await self._calculate_correlation_risk()
            
            return {
                'timestamp': datetime.now(),
                'portfolio_value': total_portfolio_value,
                'cash': self.current_capital,
                'positions': self.positions.copy(),
                'position_count': len(self.positions),
                'current_drawdown': current_drawdown,
                'max_drawdown': max_drawdown,
                'portfolio_volatility': portfolio_volatility,
                'portfolio_beta': portfolio_beta,
                'portfolio_var': portfolio_var,
                'portfolio_sharpe': portfolio_sharpe,
                'concentration_risk': concentration_risk,
                'correlation_risk': correlation_risk,
                'risk_limits': self.risk_limits.copy(),
                'risk_status': self._get_risk_status()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio risk summary: {e}")
            return {}
    
    def _calculate_risk_score(self, portfolio_percentage: float, volatility: float,
                            correlation: float, drawdown: float, sharpe: float, var: float) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            # Weighted risk factors
            size_risk = min(100, portfolio_percentage * 1000)  # 10% = 100 points
            volatility_risk = min(100, volatility * 200)  # 50% = 100 points
            correlation_risk = min(100, correlation * 100)  # 100% = 100 points
            drawdown_risk = min(100, drawdown * 500)  # 20% = 100 points
            var_risk = min(100, var / self.current_capital * 1000)  # 10% = 100 points
            
            # Sharpe ratio adjustment (lower Sharpe = higher risk)
            sharpe_risk = max(0, 50 - sharpe * 25)  # Sharpe 2 = 0 risk, Sharpe 0 = 50 risk
            
            # Calculate weighted average
            risk_score = (
                size_risk * 0.2 +
                volatility_risk * 0.25 +
                correlation_risk * 0.15 +
                drawdown_risk * 0.2 +
                var_risk * 0.15 +
                sharpe_risk * 0.05
            )
            
            return min(100, max(0, risk_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 50  # Default medium risk
    
    def _calculate_position_confidence(self, win_prob: float, expected_return: float,
                                     volatility: float, correlation: float) -> float:
        """Calculate position confidence score"""
        try:
            # Base confidence from win probability
            base_confidence = win_prob
            
            # Adjust for expected return
            return_adjustment = min(0.2, expected_return * 10)  # 2% return = 0.2 adjustment
            
            # Adjust for volatility (lower volatility = higher confidence)
            volatility_adjustment = max(-0.2, -volatility * 0.5)  # 40% volatility = -0.2 adjustment
            
            # Adjust for correlation (lower correlation = higher confidence)
            correlation_adjustment = max(-0.1, -correlation * 0.2)  # 50% correlation = -0.1 adjustment
            
            final_confidence = base_confidence + return_adjustment + volatility_adjustment + correlation_adjustment
            return max(0, min(1, final_confidence))
            
        except Exception as e:
            self.logger.error(f"Error calculating position confidence: {e}")
            return 0.5  # Default medium confidence
    
    def _generate_position_reasoning(self, kelly_fraction: float, volatility: float,
                                   correlation: float, position_count: int,
                                   recommended_size: float, max_size: float) -> str:
        """Generate reasoning for position sizing recommendation"""
        try:
            reasoning_parts = []
            
            # Kelly Criterion reasoning
            if kelly_fraction > 0.1:
                reasoning_parts.append(f"Kelly Criterion suggests {kelly_fraction:.1%} allocation")
            else:
                reasoning_parts.append("Kelly Criterion suggests conservative allocation")
            
            # Volatility reasoning
            if volatility > 0.3:
                reasoning_parts.append(f"High volatility ({volatility:.1%}) requires smaller position")
            elif volatility < 0.1:
                reasoning_parts.append(f"Low volatility ({volatility:.1%}) allows larger position")
            
            # Correlation reasoning
            if correlation > 0.5:
                reasoning_parts.append(f"High correlation ({correlation:.1%}) reduces position size")
            
            # Portfolio concentration reasoning
            if position_count > 15:
                reasoning_parts.append("Portfolio is well diversified")
            elif position_count < 5:
                reasoning_parts.append("Portfolio needs more diversification")
            
            # Size reasoning
            size_utilization = recommended_size / max_size if max_size > 0 else 0
            if size_utilization > 0.8:
                reasoning_parts.append("Position size near maximum allowed")
            elif size_utilization < 0.3:
                reasoning_parts.append("Conservative position sizing applied")
            
            return "; ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating position reasoning: {e}")
            return "Position sizing based on risk management rules"
    
    def _get_risk_status(self) -> str:
        """Get overall portfolio risk status"""
        try:
            # This would be calculated based on current portfolio metrics
            # For now, return a simple status
            if len(self.positions) < 5:
                return "UNDERDIVERSIFIED"
            elif len(self.positions) > 15:
                return "OVERDIVERSIFIED"
            else:
                return "BALANCED"
                
        except Exception as e:
            self.logger.error(f"Error getting risk status: {e}")
            return "UNKNOWN"
    
    # Placeholder methods for risk calculations
    # These would be implemented with actual market data and statistical calculations
    
    async def _calculate_volatility(self, symbol: str, market_data: Dict) -> float:
        """Calculate volatility for a symbol"""
        # Simplified implementation - would use actual price data
        return 0.2  # 20% default volatility
    
    async def _calculate_beta(self, symbol: str, market_data: Dict) -> float:
        """Calculate beta for a symbol"""
        # Simplified implementation - would use actual market correlation
        return 1.0  # Default beta
    
    async def _calculate_correlation(self, symbol: str, market_data: Dict) -> float:
        """Calculate correlation with existing positions"""
        # Simplified implementation - would use actual correlation analysis
        return 0.3  # Default correlation
    
    async def _calculate_var(self, symbol: str, position_value: float, market_data: Dict) -> float:
        """Calculate Value at Risk"""
        # Simplified implementation - would use actual VaR calculation
        return position_value * 0.05  # 5% VaR
    
    async def _calculate_expected_shortfall(self, symbol: str, position_value: float, market_data: Dict) -> float:
        """Calculate Expected Shortfall"""
        # Simplified implementation - would use actual ES calculation
        return position_value * 0.08  # 8% ES
    
    async def _calculate_sharpe_ratio(self, symbol: str, market_data: Dict) -> float:
        """Calculate Sharpe ratio"""
        # Simplified implementation - would use actual return data
        return 1.0  # Default Sharpe ratio
    
    async def _calculate_sortino_ratio(self, symbol: str, market_data: Dict) -> float:
        """Calculate Sortino ratio"""
        # Simplified implementation - would use actual return data
        return 1.2  # Default Sortino ratio
    
    async def _calculate_calmar_ratio(self, symbol: str, market_data: Dict) -> float:
        """Calculate Calmar ratio"""
        # Simplified implementation - would use actual return data
        return 0.8  # Default Calmar ratio
    
    async def _calculate_current_drawdown(self) -> float:
        """Calculate current portfolio drawdown"""
        try:
            if not self.portfolio_history:
                return 0.0
            
            # Get current portfolio value
            current_value = self.current_capital
            
            # Calculate peak value from history
            peak_value = max([entry.get('portfolio_value', self.initial_capital) for entry in self.portfolio_history])
            peak_value = max(peak_value, self.initial_capital)
            
            # Calculate current drawdown
            if peak_value > 0:
                current_drawdown = (peak_value - current_value) / peak_value
                return max(0.0, current_drawdown)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating current drawdown: {e}")
            return 0.0
    
    async def _calculate_max_drawdown(self) -> float:
        """Calculate maximum portfolio drawdown"""
        try:
            if not self.portfolio_history:
                return 0.0
            
            # Calculate running maximum and drawdowns
            peak = self.initial_capital
            max_drawdown = 0.0
            
            for entry in self.portfolio_history:
                portfolio_value = entry.get('portfolio_value', self.initial_capital)
                peak = max(peak, portfolio_value)
                drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    async def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        # Simplified implementation - would use actual portfolio data
        return 0.15  # 15% portfolio volatility
    
    async def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta"""
        # Simplified implementation - would use actual market data
        return 0.8  # Default portfolio beta
    
    async def _calculate_portfolio_var(self) -> float:
        """Calculate portfolio VaR"""
        # Simplified implementation - would use actual portfolio data
        return self.current_capital * 0.03  # 3% portfolio VaR
    
    async def _calculate_portfolio_sharpe(self) -> float:
        """Calculate portfolio Sharpe ratio"""
        # Simplified implementation - would use actual portfolio returns
        return 1.2  # Default portfolio Sharpe ratio
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk"""
        # Simplified implementation - would use actual position analysis
        return 0.3  # 30% concentration risk
    
    async def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk"""
        # Simplified implementation - would use actual correlation analysis
        return 0.4  # 40% correlation risk


