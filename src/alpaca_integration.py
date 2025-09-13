"""
Simplified Alpaca trading platform integration for demo purposes.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config.settings import AlpacaConfig, SystemConfig, TradingMode

class AlpacaTradingInterface:
    """Simplified interface for Alpaca trading platform (demo mode)."""
    
    def __init__(self, config: AlpacaConfig, system_config: SystemConfig):
        self.config = config
        self.system_config = system_config
        self.logger = logging.getLogger("AlpacaTradingInterface")
        
        # Trading state
        self.positions = {}
        self.orders = {}
        self.account_info = {
            "account_number": "DEMO123456",
            "buying_power": 100000.0,
            "cash": 100000.0,
            "portfolio_value": 100000.0,
            "equity": 100000.0,
            "day_trade_count": 0,
            "pattern_day_trader": False,
            "trading_blocked": False,
            "account_blocked": False
        }
        
        # Risk management
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.stop_loss_pct = 0.03  # 3% stop loss
        
    async def initialize(self) -> bool:
        """Initialize the trading interface."""
        try:
            self.logger.info("Alpaca interface initialized (DEMO MODE)")
            self.logger.info(f"Demo account: {self.account_info['account_number']}")
            self.logger.info(f"Initial capital: ${self.account_info['portfolio_value']:,.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Alpaca interface: {e}")
            return False
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        return self.account_info.copy()
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions."""
        return self.positions.copy()
    
    async def get_market_data(self, symbols: List[str], timeframe: str = "1Min", limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """Get mock market data for symbols."""
        market_data = {}
        
        for symbol in symbols:
            # Generate mock market data
            base_price = 100.0 + hash(symbol) % 100
            data = []
            
            for i in range(limit):
                price_change = np.random.normal(0, 0.01)
                base_price *= (1 + price_change)
                
                data.append({
                    "timestamp": (datetime.now() - timedelta(minutes=limit-i)).isoformat(),
                    "open": base_price * 0.999,
                    "high": base_price * 1.001,
                    "low": base_price * 0.998,
                    "close": base_price,
                    "volume": int(np.random.uniform(100000, 1000000))
                })
            
            market_data[symbol] = data
        
        return market_data
    
    async def place_order(self, symbol: str, qty: float, side: str, order_type: str = "market", 
                         time_in_force: str = "day", limit_price: Optional[float] = None,
                         stop_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Place a mock trading order."""
        try:
            # Validate order parameters
            if not self._validate_order(symbol, qty, side):
                return None
            
            # Calculate position size based on risk management
            adjusted_qty = self._calculate_position_size(symbol, qty, side)
            if adjusted_qty <= 0:
                self.logger.warning(f"Position size too small for {symbol}: {adjusted_qty}")
                return None
            
            # Create mock order
            order_id = f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"
            
            order_info = {
                "id": order_id,
                "symbol": symbol,
                "qty": float(adjusted_qty),
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
                "status": "filled",  # Mock orders are always filled
                "submitted_at": datetime.now().isoformat(),
                "filled_at": datetime.now().isoformat(),
                "limit_price": limit_price,
                "stop_price": stop_price,
                "filled_avg_price": self._get_current_price(symbol)
            }
            
            self.orders[order_id] = order_info
            self.logger.info(f"Mock order placed: {order_info}")
            
            return order_info
            
        except Exception as e:
            self.logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    def _validate_order(self, symbol: str, qty: float, side: str) -> bool:
        """Validate order parameters."""
        # Check if symbol is tradeable
        if symbol not in self.system_config.trading_symbols:
            self.logger.warning(f"Symbol {symbol} not in trading list")
            return False
        
        # Check quantity
        if qty <= 0:
            self.logger.warning(f"Invalid quantity: {qty}")
            return False
        
        # Check if we have enough buying power for buy orders
        if side == "buy":
            buying_power = self.account_info["buying_power"]
            estimated_cost = qty * 100  # Rough estimate
            
            if estimated_cost > buying_power:
                self.logger.warning(f"Insufficient buying power: {estimated_cost} > {buying_power}")
                return False
        
        # Check if we have enough shares for sell orders
        elif side == "sell":
            if symbol in self.positions:
                available_qty = self.positions[symbol]["qty"]
                if qty > available_qty:
                    self.logger.warning(f"Insufficient shares: {qty} > {available_qty}")
                    return False
            else:
                self.logger.warning(f"No position in {symbol} to sell")
                return False
        
        return True
    
    def _calculate_position_size(self, symbol: str, requested_qty: float, side: str) -> float:
        """Calculate appropriate position size based on risk management."""
        try:
            portfolio_value = self.account_info["portfolio_value"]
            
            # Calculate maximum position value
            max_position_value = portfolio_value * self.max_position_size
            
            # Get current price
            current_price = self._get_current_price(symbol)
            if not current_price:
                return 0
            
            # Calculate maximum quantity based on position size limit
            max_qty = max_position_value / current_price
            
            # Apply risk management
            if side == "buy":
                # For buy orders, limit to maximum position size
                adjusted_qty = min(requested_qty, max_qty)
            else:
                # For sell orders, use requested quantity (already validated)
                adjusted_qty = requested_qty
            
            # Round to whole shares
            adjusted_qty = int(adjusted_qty)
            
            return adjusted_qty
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            # Try to get from positions first
            if symbol in self.positions:
                return self.positions[symbol]["current_price"]
            
            # Generate mock price
            base_price = 100.0 + hash(symbol) % 100
            return base_price * (1 + np.random.normal(0, 0.01))
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    async def execute_trade_decision(self, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a trade decision from an agent."""
        try:
            symbol = decision.get("symbol")
            action = decision.get("action")
            quantity = decision.get("quantity", 0)
            price = decision.get("price", 0)
            confidence = decision.get("confidence", 0)
            
            if not symbol or not action or quantity <= 0:
                self.logger.warning(f"Invalid trade decision: {decision}")
                return None
            
            # Map action to Alpaca side
            side = "buy" if action.upper() == "BUY" else "sell"
            
            # Check risk limits before executing
            risk_status = await self.check_risk_limits()
            if risk_status.get("max_daily_loss_exceeded", False):
                self.logger.warning("Daily loss limit exceeded, not executing trade")
                return None
            
            # Place order
            order_result = await self.place_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type="market"
            )
            
            if order_result:
                # Add decision metadata to order result
                order_result["decision"] = decision
                order_result["executed_at"] = datetime.now().isoformat()
                
                # Update positions
                if side == "buy":
                    if symbol in self.positions:
                        self.positions[symbol]["qty"] += quantity
                    else:
                        self.positions[symbol] = {
                            "symbol": symbol,
                            "qty": quantity,
                            "side": "long",
                            "current_price": price,
                            "market_value": quantity * price
                        }
                elif side == "sell":
                    if symbol in self.positions:
                        self.positions[symbol]["qty"] -= quantity
                
                self.logger.info(f"Mock trade executed: {symbol} {action} {quantity} shares")
                return order_result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade decision: {e}")
            return None
    
    async def check_risk_limits(self) -> Dict[str, Any]:
        """Check if risk limits are being exceeded."""
        try:
            portfolio_value = self.account_info["portfolio_value"]
            equity = self.account_info["equity"]
            
            # Calculate daily P&L
            daily_pl = equity - portfolio_value
            daily_pl_pct = daily_pl / portfolio_value if portfolio_value > 0 else 0
            
            risk_status = {
                "daily_pl": daily_pl,
                "daily_pl_pct": daily_pl_pct,
                "max_daily_loss_exceeded": daily_pl_pct < -self.max_daily_loss,
                "portfolio_value": portfolio_value,
                "equity": equity,
                "buying_power": self.account_info["buying_power"],
                "risk_level": "low"
            }
            
            # Determine risk level
            if abs(daily_pl_pct) > 0.03:
                risk_status["risk_level"] = "high"
            elif abs(daily_pl_pct) > 0.01:
                risk_status["risk_level"] = "medium"
            
            return risk_status
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return {"error": str(e)}
    
    async def get_portfolio_performance(self) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        try:
            portfolio_value = self.account_info["portfolio_value"]
            equity = self.account_info["equity"]
            cash = self.account_info["cash"]
            
            # Calculate position values
            total_position_value = sum(pos["market_value"] for pos in self.positions.values())
            
            # Calculate unrealized P&L
            total_unrealized_pl = sum(pos.get("unrealized_pl", 0) for pos in self.positions.values())
            total_unrealized_pl_pct = total_unrealized_pl / portfolio_value if portfolio_value > 0 else 0
            
            performance = {
                "portfolio_value": portfolio_value,
                "equity": equity,
                "cash": cash,
                "total_position_value": total_position_value,
                "total_unrealized_pl": total_unrealized_pl,
                "total_unrealized_pl_pct": total_unrealized_pl_pct,
                "num_positions": len(self.positions),
                "positions": self.positions,
                "buying_power": self.account_info["buying_power"],
                "day_trade_count": self.account_info["day_trade_count"]
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio performance: {e}")
            return {}
    
    async def close_all_positions(self) -> List[Dict[str, Any]]:
        """Close all open positions."""
        try:
            close_orders = []
            
            for symbol, position in self.positions.items():
                if position["qty"] > 0:
                    # Close position
                    order_result = await self.place_order(
                        symbol=symbol,
                        qty=position["qty"],
                        side="sell",
                        order_type="market"
                    )
                    
                    if order_result:
                        close_orders.append(order_result)
            
            self.logger.info(f"Closed {len(close_orders)} positions")
            return close_orders
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return []
