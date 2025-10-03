-- Maximal Trading Database Schema 
CREATE TABLE IF NOT EXISTS trades ( 
  id SERIAL PRIMARY KEY, 
  agent_name VARCHAR(100), 
  symbol VARCHAR(10), 
  side VARCHAR(10), 
  quantity DECIMAL(10,4), 
  price DECIMAL(10,2), 
  strategy VARCHAR(50), 
  pnl DECIMAL(10,2), 
  confidence DECIMAL(5,3), 
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
); 
