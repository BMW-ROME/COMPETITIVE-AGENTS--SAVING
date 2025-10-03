#!/usr/bin/env python3
"""
Monitoring Dashboard
===================

Simple Flask dashboard to monitor trading system status.
"""

from flask import Flask, render_template_string, jsonify
import os
import json
from datetime import datetime

app = Flask(__name__)

# Dashboard HTML template
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading System Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
        .status.running { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.stopped { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-value { font-weight: bold; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Competitive Trading Agents Monitor</h1>
            <p>Real-time system status and performance metrics</p>
            <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>📄 Paper Trading Status</h3>
                <div id="paper-status" class="status">Loading...</div>
                <div class="metric">
                    <span>Cycles:</span>
                    <span id="paper-cycles" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Successful Trades:</span>
                    <span id="paper-successful" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Failed Trades:</span>
                    <span id="paper-failed" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Total Return:</span>
                    <span id="paper-return" class="metric-value">-</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🚨 Live Trading Status</h3>
                <div id="live-status" class="status">Loading...</div>
                <div class="metric">
                    <span>Cycles:</span>
                    <span id="live-cycles" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Successful Trades:</span>
                    <span id="live-successful" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Failed Trades:</span>
                    <span id="live-failed" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Total Return:</span>
                    <span id="live-return" class="metric-value">-</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🔧 System Health</h3>
                <div class="metric">
                    <span>Redis:</span>
                    <span id="redis-status" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>PostgreSQL:</span>
                    <span id="postgres-status" class="metric-value">-</span>
                </div>
                <div class="metric">
                    <span>Last Update:</span>
                    <span id="last-update" class="metric-value">-</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Quick Actions</h3>
                <p><a href="/api/status">📡 API Status</a></p>
                <p><a href="/api/logs/paper">📄 Paper Trading Logs</a></p>
                <p><a href="/api/logs/live">🚨 Live Trading Logs</a></p>
                <p><a href="/api/performance">📈 Performance Data</a></p>
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setInterval(function() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    updateStatus('paper', data.paper);
                    updateStatus('live', data.live);
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                })
                .catch(error => console.error('Error:', error));
        }, 30000);
        
        function updateStatus(type, data) {
            const statusEl = document.getElementById(type + '-status');
            const cyclesEl = document.getElementById(type + '-cycles');
            const successfulEl = document.getElementById(type + '-successful');
            const failedEl = document.getElementById(type + '-failed');
            const returnEl = document.getElementById(type + '-return');
            
            if (data.running) {
                statusEl.className = 'status running';
                statusEl.textContent = '🟢 Running';
            } else {
                statusEl.className = 'status stopped';
                statusEl.textContent = '🔴 Stopped';
            }
            
            cyclesEl.textContent = data.cycles || 0;
            successfulEl.textContent = data.successful_trades || 0;
            failedEl.textContent = data.failed_trades || 0;
            returnEl.textContent = (data.total_return || 0).toFixed(4);
        }
        
        // Initial load
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                updateStatus('paper', data.paper);
                updateStatus('live', data.live);
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            });
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    try:
        # Check if containers are running (match docker-compose container_name)
        paper_running = check_container_running('trading-agents-system')
        live_running = check_container_running('trading-agents-live')
        
        # Get performance data
        paper_performance = get_performance_data('paper')
        live_performance = get_performance_data('live')
        
        return jsonify({
            'paper': {
                'running': paper_running,
                'cycles': paper_performance.get('total_cycles', 0),
                'successful_trades': paper_performance.get('successful_trades', 0),
                'failed_trades': paper_performance.get('failed_trades', 0),
                'total_return': paper_performance.get('total_return', 0.0)
            },
            'live': {
                'running': live_running,
                'cycles': live_performance.get('total_cycles', 0),
                'successful_trades': live_performance.get('successful_trades', 0),
                'failed_trades': live_performance.get('failed_trades', 0),
                'total_return': live_performance.get('total_return', 0.0)
            },
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs/<trading_type>')
def api_logs(trading_type):
    """API endpoint for trading logs."""
    try:
        log_file = f'logs/{trading_type}_trading.log'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Return last 100 lines
                return jsonify({'logs': lines[-100:]})
        else:
            return jsonify({'logs': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def api_performance():
    """API endpoint for performance data."""
    try:
        paper_perf = get_performance_data('paper')
        live_perf = get_performance_data('live')
        
        return jsonify({
            'paper': paper_perf,
            'live': live_perf,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

def check_container_running(container_name):
    """Check if a Docker container is running."""
    try:
        import subprocess
        result = subprocess.run(['docker', 'ps', '--filter', f'name={container_name}', '--format', '{{.Names}}'], 
                              capture_output=True, text=True)
        return container_name in result.stdout
    except:
        return False

def get_performance_data(trading_type):
    """Get performance data from logs or files."""
    try:
        # Try to read from a performance file if it exists
        perf_file = f'logs/{trading_type}_performance.json'
        if os.path.exists(perf_file):
            with open(perf_file, 'r') as f:
                return json.load(f)
        else:
            # Return default data
            return {
                'total_cycles': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'total_return': 0.0
            }
    except:
        return {
            'total_cycles': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_return': 0.0
        }

if __name__ == '__main__':
    print("🌐 Starting Trading System Monitor")
    print("Dashboard available at:")
    print("  Paper Trading: http://localhost:8000")
    print("  Live Trading:  http://localhost:8001")
    
    app.run(host='0.0.0.0', port=8000, debug=False)
