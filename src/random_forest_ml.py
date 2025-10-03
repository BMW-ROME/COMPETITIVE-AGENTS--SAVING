"""
Random Forest Machine Learning Model for Advanced Trading Predictions
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import joblib
import os
import json

class RandomForestTradingML:
    """Random Forest ML model for trading predictions and pattern recognition"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.training_data = {}
        self.prediction_cache = {}
        
    def prepare_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepare feature matrix from market data"""
        try:
            features = []
            
            for symbol, data in market_data.get('price_data', {}).items():
                if not data or len(data) < 20:
                    continue
                    
                # Convert to DataFrame for easier manipulation
                df = pd.DataFrame(data[-20:])  # Last 20 bars
                
                # Technical indicators as features
                symbol_features = self._extract_technical_features(df)
                features.extend(symbol_features)
            
            # Add market-wide features
            market_features = self._extract_market_features(market_data)
            features.extend(market_features)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return np.array([]).reshape(1, 0)
    
    def _extract_technical_features(self, df: pd.DataFrame) -> List[float]:
        """Extract technical indicators as features"""
        try:
            features = []
            
            # Price-based features
            if 'close' in df.columns:
                closes = pd.to_numeric(df['close'], errors='coerce')
                if len(closes) > 0:
                    # Price momentum
                    features.append(closes.pct_change().mean())
                    features.append(closes.pct_change().std())
                    
                    # Price levels
                    features.append(closes.iloc[-1] / closes.mean())
                    features.append((closes.max() - closes.min()) / closes.mean())
                    
                    # Moving averages
                    if len(closes) >= 5:
                        features.append(closes.iloc[-1] / closes.tail(5).mean())
                    if len(closes) >= 10:
                        features.append(closes.iloc[-1] / closes.tail(10).mean())
                    
                    # Volatility
                    features.append(closes.rolling(5).std().iloc[-1] if len(closes) >= 5 else 0)
                    features.append(closes.rolling(10).std().iloc[-1] if len(closes) >= 10 else 0)
            
            # Volume features
            if 'volume' in df.columns:
                volumes = pd.to_numeric(df['volume'], errors='coerce')
                if len(volumes) > 0:
                    features.append(volumes.pct_change().mean())
                    features.append(volumes.iloc[-1] / volumes.mean())
                    features.append(volumes.rolling(5).mean().iloc[-1] if len(volumes) >= 5 else 0)
            
            # High-Low features
            if 'high' in df.columns and 'low' in df.columns:
                highs = pd.to_numeric(df['high'], errors='coerce')
                lows = pd.to_numeric(df['low'], errors='coerce')
                if len(highs) > 0 and len(lows) > 0:
                    features.append((highs.iloc[-1] - lows.iloc[-1]) / highs.iloc[-1])
                    features.append(highs.pct_change().mean())
                    features.append(lows.pct_change().mean())
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting technical features: {e}")
            return [0.0] * 15  # Return default features
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> List[float]:
        """Extract market-wide features"""
        try:
            features = []
            
            # Market sentiment
            sentiment = market_data.get('sentiment', {})
            features.append(sentiment.get('overall_sentiment', 0.5))
            features.append(sentiment.get('fear_greed_index', 50))
            
            # Technical indicators
            technical = market_data.get('technical_indicators', {})
            features.append(technical.get('rsi', 50))
            features.append(technical.get('macd', 0))
            features.append(technical.get('bollinger_position', 0.5))
            
            # Market volatility
            volatility = market_data.get('volatility', {})
            features.append(volatility.get('current_volatility', 0.2))
            features.append(volatility.get('volatility_trend', 0))
            
            # Economic indicators
            economic = market_data.get('economic_indicators', {})
            features.append(economic.get('vix', 20))
            features.append(economic.get('yield_curve', 0))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting market features: {e}")
            return [0.0] * 8  # Return default features
    
    def train_price_prediction_model(self, symbol: str, historical_data: List[Dict]) -> bool:
        """Train Random Forest model for price prediction"""
        try:
            if len(historical_data) < 50:
                self.logger.warning(f"Insufficient data for {symbol} price prediction model")
                return False
            
            # Prepare training data
            df = pd.DataFrame(historical_data)
            df = df.sort_values('timestamp')
            
            # Create features and targets
            X, y = self._prepare_price_training_data(df)
            
            if len(X) < 20:
                self.logger.warning(f"Insufficient features for {symbol}")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = rf_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2_score = rf_model.score(X_test_scaled, y_test)
            
            # Store model and metrics
            self.models[f"{symbol}_price"] = rf_model
            self.scalers[f"{symbol}_price"] = scaler
            self.model_metrics[f"{symbol}_price"] = {
                'mse': mse,
                'r2_score': r2_score,
                'feature_importance': dict(zip(
                    [f"feature_{i}" for i in range(len(rf_model.feature_importances_))],
                    rf_model.feature_importances_
                ))
            }
            
            self.logger.info(f"Price prediction model trained for {symbol}: R²={r2_score:.3f}, MSE={mse:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training price prediction model for {symbol}: {e}")
            return False
    
    def train_direction_prediction_model(self, symbol: str, historical_data: List[Dict]) -> bool:
        """Train Random Forest model for direction prediction (up/down)"""
        try:
            if len(historical_data) < 50:
                self.logger.warning(f"Insufficient data for {symbol} direction prediction model")
                return False
            
            # Prepare training data
            df = pd.DataFrame(historical_data)
            df = df.sort_values('timestamp')
            
            # Create features and targets
            X, y = self._prepare_direction_training_data(df)
            
            if len(X) < 20:
                self.logger.warning(f"Insufficient features for {symbol}")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            rf_model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = rf_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store model and metrics
            self.models[f"{symbol}_direction"] = rf_model
            self.scalers[f"{symbol}_direction"] = scaler
            self.model_metrics[f"{symbol}_direction"] = {
                'accuracy': accuracy,
                'feature_importance': dict(zip(
                    [f"feature_{i}" for i in range(len(rf_model.feature_importances_))],
                    rf_model.feature_importances_
                ))
            }
            
            self.logger.info(f"Direction prediction model trained for {symbol}: Accuracy={accuracy:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training direction prediction model for {symbol}: {e}")
            return False
    
    def _prepare_price_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for price prediction"""
        try:
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create features
            features = []
            targets = []
            
            for i in range(20, len(df)):  # Need at least 20 bars for features
                # Historical features
                window = df.iloc[i-20:i]
                
                feature_vector = []
                
                # Price features
                closes = window['close'].values
                feature_vector.extend([
                    closes[-1] / closes.mean(),  # Price relative to mean
                    (closes.max() - closes.min()) / closes.mean(),  # Range
                    np.std(closes),  # Volatility
                    closes[-1] / closes[-5:].mean() if len(closes) >= 5 else 1,  # Short-term momentum
                    closes[-1] / closes[-10:].mean() if len(closes) >= 10 else 1,  # Medium-term momentum
                ])
                
                # Volume features
                if 'volume' in window.columns:
                    volumes = window['volume'].values
                    feature_vector.extend([
                        volumes[-1] / volumes.mean(),
                        np.std(volumes),
                        volumes[-1] / volumes[-5:].mean() if len(volumes) >= 5 else 1
                    ])
                else:
                    feature_vector.extend([1, 0, 1])
                
                # Technical indicators
                if len(closes) >= 14:
                    rsi = self._calculate_rsi(closes)
                    feature_vector.append(rsi)
                else:
                    feature_vector.append(50)
                
                if len(closes) >= 12:
                    macd = self._calculate_macd(closes)
                    feature_vector.append(macd)
                else:
                    feature_vector.append(0)
                
                features.append(feature_vector)
                
                # Target: next period return
                if i < len(df) - 1:
                    current_price = closes[-1]
                    next_price = df.iloc[i+1]['close']
                    target = (next_price - current_price) / current_price
                    targets.append(target)
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Error preparing price training data: {e}")
            return np.array([]), np.array([])
    
    def _prepare_direction_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for direction prediction"""
        try:
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Create features
            features = []
            targets = []
            
            for i in range(20, len(df) - 1):  # Need at least 20 bars for features
                # Historical features
                window = df.iloc[i-20:i]
                
                feature_vector = []
                
                # Price features
                closes = window['close'].values
                feature_vector.extend([
                    closes[-1] / closes.mean(),
                    (closes.max() - closes.min()) / closes.mean(),
                    np.std(closes),
                    closes[-1] / closes[-5:].mean() if len(closes) >= 5 else 1,
                    closes[-1] / closes[-10:].mean() if len(closes) >= 10 else 1,
                ])
                
                # Volume features
                if 'volume' in window.columns:
                    volumes = window['volume'].values
                    feature_vector.extend([
                        volumes[-1] / volumes.mean(),
                        np.std(volumes),
                        volumes[-1] / volumes[-5:].mean() if len(volumes) >= 5 else 1
                    ])
                else:
                    feature_vector.extend([1, 0, 1])
                
                # Technical indicators
                if len(closes) >= 14:
                    rsi = self._calculate_rsi(closes)
                    feature_vector.append(rsi)
                else:
                    feature_vector.append(50)
                
                if len(closes) >= 12:
                    macd = self._calculate_macd(closes)
                    feature_vector.append(macd)
                else:
                    feature_vector.append(0)
                
                features.append(feature_vector)
                
                # Target: direction (1 for up, 0 for down)
                current_price = closes[-1]
                next_price = df.iloc[i+1]['close']
                direction = 1 if next_price > current_price else 0
                targets.append(direction)
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Error preparing direction training data: {e}")
            return np.array([]), np.array([])
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return 50
    
    def _calculate_macd(self, prices: np.ndarray) -> float:
        """Calculate MACD indicator"""
        try:
            if len(prices) < 26:
                return 0
            
            # Simple MACD calculation
            ema12 = np.mean(prices[-12:])
            ema26 = np.mean(prices[-26:])
            macd = ema12 - ema26
            
            return macd
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return 0
    
    def predict_price_movement(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict price movement using Random Forest"""
        try:
            model_key = f"{symbol}_price"
            if model_key not in self.models:
                return {"prediction": 0, "confidence": 0, "model_available": False}
            
            # Prepare features
            features = self.prepare_features(market_data)
            if features.size == 0:
                return {"prediction": 0, "confidence": 0, "model_available": True}
            
            # Scale features
            scaler = self.scalers[model_key]
            features_scaled = scaler.transform(features)
            
            # Make prediction
            model = self.models[model_key]
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence based on prediction variance
            predictions = []
            for estimator in model.estimators_:
                pred = estimator.predict(features_scaled)[0]
                predictions.append(pred)
            
            confidence = 1 - np.std(predictions) / (np.abs(prediction) + 1e-6)
            confidence = max(0, min(1, confidence))
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "model_available": True,
                "feature_importance": self.model_metrics[model_key].get('feature_importance', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting price movement for {symbol}: {e}")
            return {"prediction": 0, "confidence": 0, "model_available": False}
    
    def predict_direction(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict direction (up/down) using Random Forest"""
        try:
            model_key = f"{symbol}_direction"
            if model_key not in self.models:
                return {"direction": "neutral", "confidence": 0, "model_available": False}
            
            # Prepare features
            features = self.prepare_features(market_data)
            if features.size == 0:
                return {"direction": "neutral", "confidence": 0, "model_available": True}
            
            # Scale features
            scaler = self.scalers[model_key]
            features_scaled = scaler.transform(features)
            
            # Make prediction
            model = self.models[model_key]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Get direction and confidence
            direction_idx = np.argmax(probabilities)
            direction = "up" if direction_idx == 1 else "down"
            confidence = probabilities[direction_idx]
            
            return {
                "direction": direction,
                "confidence": confidence,
                "model_available": True,
                "probabilities": {
                    "down": probabilities[0],
                    "up": probabilities[1]
                },
                "feature_importance": self.model_metrics[model_key].get('feature_importance', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting direction for {symbol}: {e}")
            return {"direction": "neutral", "confidence": 0, "model_available": False}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        return {
            "models_trained": len(self.models),
            "model_metrics": self.model_metrics,
            "available_models": list(self.models.keys())
        }
    
    def save_models(self, filepath: str) -> bool:
        """Save trained models to disk"""
        try:
            model_data = {
                "models": self.models,
                "scalers": self.scalers,
                "metrics": self.model_metrics
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, filepath: str) -> bool:
        """Load trained models from disk"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Model file not found: {filepath}")
                return False
            
            model_data = joblib.load(filepath)
            
            self.models = model_data.get("models", {})
            self.scalers = model_data.get("scalers", {})
            self.model_metrics = model_data.get("metrics", {})
            
            self.logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False



