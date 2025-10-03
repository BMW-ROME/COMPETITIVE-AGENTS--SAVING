#!/usr/bin/env python3
"""
Smart Portfolio Rebalancer - Free up buying power strategically
"""
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
import time

def main():
    # Load environment
    load_dotenv()
    
    api = tradeapi.REST(
        os.getenv('APCA_API_KEY_ID'),
        os.getenv('APCA_API_SECRET_KEY'), 
        'https://paper-api.alpaca.markets'
    )
    
    print("🔄 SMART PORTFOLIO REBALANCER")
    print("=" * 50)
    
    # Get current account status
    account = api.get_account()
    buying_power = float(account.buying_power)
    equity = float(account.equity)
    
    print(f"💰 Current Buying Power: ${buying_power:,.2f}")
    print(f"📊 Total Equity: ${equity:,.2f}")
    print(f"📈 Position Utilization: {((equity - buying_power) / equity) * 100:.1f}%")
    
    # Get positions
    positions = api.list_positions()
    if not positions:
        print("✅ No positions to rebalance")
        return
    
    print(f"\n📋 Current Positions ({len(positions)}):")
    
    # Analyze positions for rebalancing
    position_analysis = []
    total_unrealized_pnl = 0
    
    for pos in positions:
        pnl = float(pos.unrealized_pl or 0.0)
        pnl_pct = float(pos.unrealized_plpc or 0.0) * 100
        market_value = float(pos.market_value)
        qty = float(pos.qty)
        
        total_unrealized_pnl += pnl
        
        # Calculate rebalancing priority
        priority_score = 0
        action = "HOLD"
        reason = ""
        
        # High priority to reduce large losing positions
        if pnl < -200:
            priority_score += 10
            action = "REDUCE_50%"
            reason = f"Large loss ${pnl:.2f}"
        elif pnl < -100:
            priority_score += 5
            action = "REDUCE_25%"
            reason = f"Moderate loss ${pnl:.2f}"
        
        # Take some profits from big winners
        elif pnl > 500:
            priority_score += 8
            action = "REDUCE_30%"
            reason = f"Large profit ${pnl:.2f}"
        
        # Reduce oversized positions (>$50K)
        if market_value > 50000:
            priority_score += 7
            if action == "HOLD":
                action = "REDUCE_40%"
                reason = f"Oversized position ${market_value:,.0f}"
        
        position_analysis.append({
            'symbol': pos.symbol,
            'qty': qty,
            'market_value': market_value,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'priority': priority_score,
            'action': action,
            'reason': reason,
            'side': pos.side
        })
        
        print(f"  {pos.symbol}: {qty:.4f} shares, ${market_value:,.0f}, P&L: ${pnl:.2f} ({pnl_pct:.1f}%) - {action}")
    
    print(f"\n💰 Total Unrealized P&L: ${total_unrealized_pnl:.2f}")
    
    # Sort by priority (highest first)
    position_analysis.sort(key=lambda x: x['priority'], reverse=True)
    
    # Execute rebalancing
    target_buying_power = min(50000, equity * 0.2)  # Target 20% cash or $50K max
    needed_cash = target_buying_power - buying_power
    
    print(f"\n🎯 Target Buying Power: ${target_buying_power:,.2f}")
    print(f"💸 Cash Needed: ${needed_cash:,.2f}")
    
    if needed_cash <= 0:
        print("✅ Sufficient buying power already available")
        return
    
    cash_freed = 0
    executed_trades = 0
    
    for pos in position_analysis:
        if pos['action'] == 'HOLD' or cash_freed >= needed_cash:
            continue
            
        symbol = pos['symbol']
        current_qty = pos['qty']
        
        # Calculate reduction amount
        if "50%" in pos['action']:
            reduce_qty = current_qty * 0.5
        elif "40%" in pos['action']:
            reduce_qty = current_qty * 0.4
        elif "30%" in pos['action']:
            reduce_qty = current_qty * 0.3
        elif "25%" in pos['action']:
            reduce_qty = current_qty * 0.25
        else:
            continue
        
        try:
            print(f"\n🔄 Executing: {pos['action']} for {symbol}")
            print(f"   Selling {reduce_qty:.4f} shares ({pos['reason']})")
            
            # Execute the trade
            order = api.submit_order(
                symbol=symbol,
                qty=reduce_qty,
                side='sell' if pos['side'] == 'long' else 'buy',
                type='market',
                time_in_force='day'
            )
            
            if order and hasattr(order, 'id'):
                estimated_cash = reduce_qty * (pos['market_value'] / current_qty)
                cash_freed += estimated_cash
                executed_trades += 1
                
                print(f"   ✅ Order submitted: {order.id}")
                print(f"   💰 Estimated cash freed: ${estimated_cash:,.2f}")
                
                # Wait briefly between orders
                time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Failed to execute {symbol}: {e}")
    
    print(f"\n📊 REBALANCING COMPLETE")
    print(f"   Executed Trades: {executed_trades}")
    print(f"   Estimated Cash Freed: ${cash_freed:,.2f}")
    
    # Wait and check new buying power
    print("\n⏳ Waiting 10 seconds for settlement...")
    time.sleep(10)
    
    new_account = api.get_account()
    new_buying_power = float(new_account.buying_power)
    improvement = new_buying_power - buying_power
    
    print(f"💰 New Buying Power: ${new_buying_power:,.2f} (+${improvement:,.2f})")
    
    if new_buying_power > 10000:
        print("🚀 SUCCESS: Sufficient buying power restored!")
        print("   Ready to resume competitive trading!")
    else:
        print("⚠️  Additional rebalancing may be needed")

if __name__ == "__main__":
    main()