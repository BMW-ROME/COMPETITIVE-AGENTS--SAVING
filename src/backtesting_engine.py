"""
Real-time Backtesting Engine for Continuous Strategy Optimization
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    agent_id: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_hold_time: float
    volatility: float
    alpha: float
    beta: float
    information_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    trades: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]

class RealTimeBacktestingEngine:
    """Real-time backtesting engine that continuously optimizes strategies"""
    
    def __init__(self, data_provider, logger: logging.Logger):
        self.data_provider = data_provider
        self.logger = logger
        self.historical_data = {}
        self.backtest_results = {}
        self.optimization_history = []
        
    async def initialize_historical_data(self, symbols: List[str], days_back: int = 30):
        """Initialize historical data for backtesting"""
        try:
            self.logger.info(f"Initializing {days_back} days of historical data for {len(symbols)} symbols")
            
            for symbol in symbols:
                try:
                    # Try multiple intervals to get more data
                    historical_bars = []
                    
                    # Try 1-hour data first
                    try:
                        bars_1h = await self.data_provider.get_price_data(
                            [symbol], period="30d", interval="1h"
                        )
                        bars_1h = bars_1h.get(symbol, [])
                        if bars_1h:
                            historical_bars.extend(bars_1h)
                    except Exception as e:
                        self.logger.debug(f"1h data failed for {symbol}: {e}")
                    
                    # Try 15-minute data for more granular data
                    try:
                        bars_15m = await self.data_provider.get_price_data(
                            [symbol], period="7d", interval="15m"
                        )
                        bars_15m = bars_15m.get(symbol, [])
                        if bars_15m:
                            historical_bars.extend(bars_15m)
                    except Exception as e:
                        self.logger.debug(f"15m data failed for {symbol}: {e}")
                    
                    # Try 1-minute data for recent data
                    try:
                        bars_1m = await self.data_provider.get_price_data(
                            [symbol], period="1d", interval="1m"
                        )
                        bars_1m = bars_1m.get(symbol, [])
                        if bars_1m:
                            historical_bars.extend(bars_1m)
                    except Exception as e:
                        self.logger.debug(f"1m data failed for {symbol}: {e}")
                    
                    # Remove duplicates and sort by timestamp
                    if historical_bars:
                        # Remove duplicates based on timestamp
                        seen_timestamps = set()
                        unique_bars = []
                        for bar in historical_bars:
                            timestamp = bar.get('timestamp')
                            if timestamp not in seen_timestamps:
                                seen_timestamps.add(timestamp)
                                unique_bars.append(bar)
                        
                        # Sort by timestamp
                        unique_bars.sort(key=lambda x: x.get('timestamp', ''))
                        
                        self.historical_data[symbol] = unique_bars
                        self.logger.info(f"Loaded {len(unique_bars)} bars for {symbol}")
                    else:
                        self.logger.warning(f"No historical data for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading historical data for {symbol}: {e}")
                    
            self.logger.info(f"Historical data initialization complete. {len(self.historical_data)} symbols loaded")
            
        except Exception as e:
            self.logger.error(f"Error initializing historical data: {e}")
    
    async def run_agent_backtest(self, agent, symbol: str, days_back: int = 7) -> Optional[BacktestResult]:
        """Run backtest for a specific agent and symbol"""
        try:
            if symbol not in self.historical_data:
                self.logger.warning(f"No historical data for {symbol}")
                return None
                
            # Get recent data for backtest
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Filter historical data to backtest period
            backtest_data = []
            
            # Create timezone-naive versions for comparison
            start_date_naive = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
            end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
            
            for bar in self.historical_data[symbol]:
                try:
                    # Convert timestamp to datetime if it's a string
                    timestamp = bar.get('timestamp', start_date_naive)
                    if isinstance(timestamp, str):
                        # Handle both ISO format and simple format
                        if 'T' in timestamp:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.fromisoformat(timestamp)
                    elif not isinstance(timestamp, datetime):
                        timestamp = start_date_naive
                    
                    # Ensure timestamp is timezone-naive for comparison
                    if timestamp.tzinfo is not None:
                        timestamp = timestamp.replace(tzinfo=None)
                    
                    if start_date_naive <= timestamp <= end_date_naive:
                        backtest_data.append(bar)
                except Exception as e:
                    self.logger.warning(f"Error parsing timestamp for {symbol}: {e}")
                    continue
            
            if len(backtest_data) < 5:
                self.logger.warning(f"Insufficient data for {symbol} backtest (only {len(backtest_data)} bars)")
                return None
            
            # Simulate agent decisions
            trades = []
            portfolio_value = 10000  # Starting portfolio
            positions = {}
            cash = portfolio_value
            
            for i, bar in enumerate(backtest_data):
                try:
                    # Create market data for agent
                    market_data = {
                        symbol: {
                            'bars': backtest_data[:i+1],
                            'current_price': bar.get('close', 0),
                            'volume': bar.get('volume', 0),
                            'timestamp': bar.get('timestamp', datetime.now())
                        }
                    }
                    
                    # Get agent decision
                    decision = await agent.make_trading_decision(market_data)
                    
                    if decision and decision.action in ['BUY', 'SELL']:
                        # Execute trade
                        trade_result = self._execute_backtest_trade(
                            decision, bar, positions, cash, portfolio_value
                        )
                        
                        if trade_result:
                            trades.append(trade_result)
                            positions = trade_result['positions']
                            cash = trade_result['cash']
                            portfolio_value = trade_result['portfolio_value']
                            
                except Exception as e:
                    self.logger.error(f"Error in backtest simulation for {symbol}: {e}")
                    continue
            
            # Calculate performance metrics
            if len(trades) > 0:
                result = self._calculate_backtest_metrics(
                    agent.agent_id, symbol, trades, portfolio_value, backtest_data
                )
                return result
            else:
                self.logger.info(f"No trades generated in backtest for {agent.agent_id} on {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error running backtest for {agent.agent_id} on {symbol}: {e}")
            return None
    
    def _execute_backtest_trade(self, decision, bar, positions: Dict, cash: float, portfolio_value: float) -> Optional[Dict]:
        """Execute a trade in the backtest simulation"""
        try:
            symbol = decision.symbol
            action = decision.action
            quantity = decision.quantity
            price = bar.get('close', 0)
            
            if action == 'BUY':
                # Check if we have enough cash
                cost = quantity * price
                if cost <= cash:
                    cash -= cost
                    if symbol in positions:
                        positions[symbol] += quantity
                    else:
                        positions[symbol] = quantity
                        
                    portfolio_value = cash + sum(pos * price for pos in positions.values())
                    
                    return {
                        'timestamp': bar.get('timestamp', datetime.now()),
                        'action': action,
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'cash': cash,
                        'positions': positions.copy(),
                        'portfolio_value': portfolio_value
                    }
                    
            elif action == 'SELL':
                # Check if we have the position
                if symbol in positions and positions[symbol] >= quantity:
                    proceeds = quantity * price
                    cash += proceeds
                    positions[symbol] -= quantity
                    
                    if positions[symbol] <= 0:
                        del positions[symbol]
                        
                    portfolio_value = cash + sum(pos * price for pos in positions.values())
                    
                    return {
                        'timestamp': bar.get('timestamp', datetime.now()),
                        'action': action,
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'cash': cash,
                        'positions': positions.copy(),
                        'portfolio_value': portfolio_value
                    }
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing backtest trade: {e}")
            return None
    
    def _calculate_backtest_metrics(self, agent_id: str, symbol: str, trades: List[Dict], 
                                  final_portfolio_value: float, market_data: List) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""
        try:
            if not trades:
                return None
                
            # Basic metrics
            total_return = (final_portfolio_value - 10000) / 10000
            
            # Calculate returns for each period
            returns = []
            for i in range(1, len(trades)):
                prev_value = trades[i-1]['portfolio_value']
                curr_value = trades[i]['portfolio_value']
                period_return = (curr_value - prev_value) / prev_value
                returns.append(period_return)
            
            if not returns:
                returns = [total_return]
            
            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
            sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility > 0 else 0
            
            # Drawdown calculation
            portfolio_values = [trade['portfolio_value'] for trade in trades]
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Trade metrics
            total_trades = len(trades)
            profitable_trades = sum(1 for trade in trades if trade['action'] == 'SELL')
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # Average hold time
            hold_times = []
            buy_trades = {}
            for trade in trades:
                if trade['action'] == 'BUY':
                    buy_trades[trade['symbol']] = trade['timestamp']
                elif trade['action'] == 'SELL' and trade['symbol'] in buy_trades:
                    try:
                        # Handle timestamp comparison - convert strings to datetime if needed
                        sell_time = trade['timestamp']
                        buy_time = buy_trades[trade['symbol']]
                        
                        if isinstance(sell_time, str):
                            if 'T' in sell_time:
                                sell_time = datetime.fromisoformat(sell_time.replace('Z', '+00:00'))
                            else:
                                sell_time = datetime.fromisoformat(sell_time)
                        if isinstance(buy_time, str):
                            if 'T' in buy_time:
                                buy_time = datetime.fromisoformat(buy_time.replace('Z', '+00:00'))
                            else:
                                buy_time = datetime.fromisoformat(buy_time)
                        
                        # Ensure both are naive for comparison
                        if sell_time.tzinfo is not None:
                            sell_time = sell_time.replace(tzinfo=None)
                        if buy_time.tzinfo is not None:
                            buy_time = buy_time.replace(tzinfo=None)
                            
                        hold_time = (sell_time - buy_time).total_seconds() / 3600
                        hold_times.append(hold_time)
                    except Exception as e:
                        self.logger.warning(f"Error calculating hold time: {e}")
                        continue
                    del buy_trades[trade['symbol']]
            
            avg_hold_time = np.mean(hold_times) if hold_times else 0
            
            # Advanced metrics
            alpha = 0  # Would need benchmark comparison
            beta = 0   # Would need benchmark comparison
            information_ratio = sharpe_ratio  # Simplified
            
            # Calmar ratio
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
            sortino_ratio = (np.mean(returns) * 252) / downside_deviation if downside_deviation > 0 else 0
            
            # Performance metrics dictionary
            performance_metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'volatility': volatility,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio,
                'total_trades': total_trades,
                'avg_hold_time': avg_hold_time
            }
            
            return BacktestResult(
                agent_id=agent_id,
                strategy_name=f"{agent_id}_{symbol}",
                start_date=trades[0]['timestamp'],
                end_date=trades[-1]['timestamp'],
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                avg_hold_time=avg_hold_time,
                volatility=volatility,
                alpha=alpha,
                beta=beta,
                information_ratio=information_ratio,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                trades=trades,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating backtest metrics: {e}")
            return None
    
    async def optimize_agent_strategy(self, agent, symbols: List[str]) -> Dict[str, Any]:
        """Optimize agent strategy based on backtest results"""
        try:
            self.logger.info(f"Optimizing strategy for {agent.agent_id}")
            
            optimization_results = {}
            
            for symbol in symbols:
                # Run backtest
                backtest_result = await self.run_agent_backtest(agent, symbol)
                
                if backtest_result:
                    optimization_results[symbol] = backtest_result
                    
                    # Store results
                    key = f"{agent.agent_id}_{symbol}"
                    self.backtest_results[key] = backtest_result
                    
                    # Log performance
                    self.logger.info(
                        f"{agent.agent_id} on {symbol}: "
                        f"Return: {backtest_result.total_return:.2%}, "
                        f"Sharpe: {backtest_result.sharpe_ratio:.2f}, "
                        f"Max DD: {backtest_result.max_drawdown:.2%}, "
                        f"Win Rate: {backtest_result.win_rate:.2%}"
                    )
            
            # Calculate overall optimization score
            if optimization_results:
                avg_return = np.mean([r.total_return for r in optimization_results.values()])
                avg_sharpe = np.mean([r.sharpe_ratio for r in optimization_results.values()])
                avg_win_rate = np.mean([r.win_rate for r in optimization_results.values()])
                
                optimization_score = (avg_return * 0.4 + avg_sharpe * 0.3 + avg_win_rate * 0.3)
                
                optimization_summary = {
                    'agent_id': agent.agent_id,
                    'optimization_score': optimization_score,
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'avg_win_rate': avg_win_rate,
                    'symbols_tested': len(optimization_results),
                    'results': optimization_results
                }
                
                # Store optimization history
                self.optimization_history.append({
                    'timestamp': datetime.now(),
                    'agent_id': agent.agent_id,
                    'optimization_score': optimization_score,
                    'summary': optimization_summary
                })
                
                self.logger.info(f"Optimization complete for {agent.agent_id}. Score: {optimization_score:.3f}")
                return optimization_summary
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy for {agent.agent_id}: {e}")
            return {}
    
    async def run_continuous_optimization(self, agents: List, symbols: List[str], 
                                        optimization_interval: int = 3600):
        """Run continuous optimization for all agents"""
        try:
            self.logger.info("Starting continuous optimization engine")
            
            while True:
                try:
                    # Run optimization for each agent
                    for agent in agents:
                        if not getattr(agent, 'is_suspended', False):
                            await self.optimize_agent_strategy(agent, symbols)
                    
                    # Wait for next optimization cycle
                    await asyncio.sleep(optimization_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in continuous optimization cycle: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
                    
        except Exception as e:
            self.logger.error(f"Error in continuous optimization: {e}")
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results"""
        try:
            if not self.optimization_history:
                return {}
            
            # Get latest optimization for each agent
            latest_optimizations = {}
            for entry in self.optimization_history:
                agent_id = entry['agent_id']
                if agent_id not in latest_optimizations or \
                   entry['timestamp'] > latest_optimizations[agent_id]['timestamp']:
                    latest_optimizations[agent_id] = entry
            
            # Calculate overall system performance
            if latest_optimizations:
                avg_score = np.mean([opt['optimization_score'] for opt in latest_optimizations.values()])
                best_agent = max(latest_optimizations.items(), key=lambda x: x[1]['optimization_score'])
                
                return {
                    'system_avg_score': avg_score,
                    'best_performing_agent': best_agent[0],
                    'best_score': best_agent[1]['optimization_score'],
                    'total_optimizations': len(self.optimization_history),
                    'agent_optimizations': latest_optimizations
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting optimization summary: {e}")
            return {}
