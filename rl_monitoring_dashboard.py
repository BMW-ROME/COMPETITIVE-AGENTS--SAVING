#!/usr/bin/env python3
"""
RL Performance Monitoring Dashboard
=================================
Real-time monitoring of RL system performance and execution rates
"""

import time
import json
import os
from datetime import datetime
from rl_100_percent_execution import get_100_percent_execution_optimizer

def display_rl_performance():
    """Display current RL performance metrics"""
    try:
        optimizer = get_100_percent_execution_optimizer()
        report = optimizer.get_execution_performance_report()
        
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🧠 RL SYSTEM PERFORMANCE DASHBOARD")
        print("=" * 50)
        print(f"📊 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Execution Rate: {report['overall_execution_rate']:.1%}")
        print(f"📈 Total Attempts: {report['total_attempts']}")
        print(f"✅ Successful Executions: {report['total_successes']}")
        print(f"📋 Barriers Recorded: {report['execution_barriers_recorded']}")
        
        print("\n📊 SYMBOL PERFORMANCE:")
        print("-" * 30)
        for symbol, stats in report['symbol_performance'].items():
            success_rate = stats['success_rate']
            status_emoji = "🟢" if success_rate > 0.8 else "🟡" if success_rate > 0.6 else "🔴"
            print(f"{status_emoji} {symbol}: {success_rate:.1%} ({stats['successful_executions']}/{stats['total_attempts']})")
        
        if report['failure_analysis']:
            print("\n❌ FAILURE ANALYSIS:")
            print("-" * 20)
            for failure_type, count in report['failure_analysis'].items():
                print(f"  {failure_type}: {count}")
        
        print("\n🔄 Press Ctrl+C to exit | Refreshing every 30 seconds...")
        
    except Exception as e:
        print(f"❌ Error displaying RL performance: {e}")

if __name__ == "__main__":
    try:
        while True:
            display_rl_performance()
            time.sleep(30)  # Refresh every 30 seconds
    except KeyboardInterrupt:
        print("\n👋 RL Dashboard stopped by user")
