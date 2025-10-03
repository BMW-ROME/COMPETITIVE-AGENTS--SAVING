"""
Machine Learning Enhancement Module for Trading Agents
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

@dataclass
class MLPrediction:
    """Machine learning prediction result"""
    symbol: str
    prediction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    expected_return: float
    risk_score: float
    features: Dict[str, float]
    model_name: str
    timestamp: datetime

@dataclass
class MLModel:
    """Machine learning model wrapper"""
    name: str
    model: Any
    scaler: StandardScaler
    feature_columns: List[str]
    accuracy: float
    last_trained: datetime
    prediction_count: int

class MLEnhancementEngine:
    """Machine learning enhancement engine for trading agents"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.models: Dict[str, MLModel] = {}
        self.feature_cache = {}
        self.prediction_history = []
        self.model_directory = "ml_models"
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_directory, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models"""
        try:
            # Price Direction Prediction Model
            self.models['price_direction'] = MLModel(
                name='price_direction',
                model=RandomForestClassifier(n_estimators=100, random_state=42),
                scaler=StandardScaler(),
                feature_columns=[
                    'rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'volume_ratio',
                    'price_change_1h', 'price_change_4h', 'price_change_24h',
                    'volatility', 'momentum', 'trend_strength'
                ],
                accuracy=0.0,
                last_trained=datetime.now(),
                prediction_count=0
            )
            
            # Return Prediction Model
            self.models['return_prediction'] = MLModel(
                name='return_prediction',
                model=GradientBoostingRegressor(n_estimators=100, random_state=42),
                scaler=StandardScaler(),
                feature_columns=[
                    'rsi', 'macd', 'bollinger_position', 'volume_ratio',
                    'price_change_1h', 'price_change_4h', 'volatility',
                    'momentum', 'trend_strength', 'support_resistance'
                ],
                accuracy=0.0,
                last_trained=datetime.now(),
                prediction_count=0
            )
            
            # Risk Assessment Model
            self.models['risk_assessment'] = MLModel(
                name='risk_assessment',
                model=RandomForestClassifier(n_estimators=50, random_state=42),
                scaler=StandardScaler(),
                feature_columns=[
                    'volatility', 'volume_ratio', 'price_change_24h',
                    'rsi', 'bollinger_position', 'trend_strength',
                    'market_correlation', 'liquidity_score'
                ],
                accuracy=0.0,
                last_trained=datetime.now(),
                prediction_count=0
            )
            
            self.logger.info("Machine learning models initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing ML models: {e}")
    
    async def extract_features(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for machine learning models"""
        try:
            symbol_data = market_data.get(symbol, {})
            if not symbol_data or 'bars' not in symbol_data:
                return {}
            
            bars = symbol_data['bars']
            if len(bars) < 20:  # Need minimum data for indicators
                return {}
            
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame([{
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            } for bar in bars[-50:]])  # Use last 50 bars
            
            if len(df) < 20:
                return {}
            
            features = {}
            
            # Technical indicators
            features.update(self._calculate_technical_indicators(df))
            
            # Price change features
            features.update(self._calculate_price_changes(df))
            
            # Volume features
            features.update(self._calculate_volume_features(df))
            
            # Volatility features
            features.update(self._calculate_volatility_features(df))
            
            # Trend features
            features.update(self._calculate_trend_features(df))
            
            # Support/Resistance
            features.update(self._calculate_support_resistance(df))
            
            # Cache features
            self.feature_cache[symbol] = {
                'features': features,
                'timestamp': datetime.now()
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {symbol}: {e}")
            return {}
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        try:
            features = {}
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = (100 - (100 / (1 + rs))).iloc[-1] if not rs.isna().iloc[-1] else 50
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            features['macd'] = macd.iloc[-1] if not macd.isna().iloc[-1] else 0
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            features['bollinger_upper'] = (sma_20 + (std_20 * 2)).iloc[-1] if not sma_20.isna().iloc[-1] else df['close'].iloc[-1]
            features['bollinger_lower'] = (sma_20 - (std_20 * 2)).iloc[-1] if not sma_20.isna().iloc[-1] else df['close'].iloc[-1]
            features['bollinger_position'] = (df['close'].iloc[-1] - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower']) if features['bollinger_upper'] != features['bollinger_lower'] else 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _calculate_price_changes(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate price change features"""
        try:
            features = {}
            current_price = df['close'].iloc[-1]
            
            # Price changes over different timeframes
            if len(df) >= 2:
                features['price_change_1h'] = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]
            else:
                features['price_change_1h'] = 0
            
            if len(df) >= 5:
                features['price_change_4h'] = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            else:
                features['price_change_4h'] = 0
            
            if len(df) >= 25:
                features['price_change_24h'] = (current_price - df['close'].iloc[-25]) / df['close'].iloc[-25]
            else:
                features['price_change_24h'] = 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating price changes: {e}")
            return {}
    
    def _calculate_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume features"""
        try:
            features = {}
            
            # Volume ratio (current vs average)
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            features['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 1
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating volume features: {e}")
            return {}
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility features"""
        try:
            features = {}
            
            # Price volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            features['volatility'] = returns.std() * np.sqrt(24)  # Annualized
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {e}")
            return {}
    
    def _calculate_trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trend features"""
        try:
            features = {}
            
            # Simple trend strength (slope of linear regression)
            if len(df) >= 10:
                x = np.arange(len(df[-10:]))
                y = df['close'].iloc[-10:].values
                slope = np.polyfit(x, y, 1)[0]
                features['trend_strength'] = slope / df['close'].iloc[-1]  # Normalized
            else:
                features['trend_strength'] = 0
            
            # Momentum (rate of change)
            if len(df) >= 5:
                features['momentum'] = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            else:
                features['momentum'] = 0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating trend features: {e}")
            return {}
    
    def _calculate_support_resistance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate support and resistance levels"""
        try:
            features = {}
            
            # Simple support/resistance based on recent highs and lows
            recent_high = df['high'].iloc[-20:].max()
            recent_low = df['low'].iloc[-20:].min()
            current_price = df['close'].iloc[-1]
            
            # Distance to support and resistance
            features['support_resistance'] = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return {}
    
    async def make_prediction(self, symbol: str, market_data: Dict[str, Any]) -> Optional[MLPrediction]:
        """Make ML prediction for a symbol"""
        try:
            # Extract features
            features = await self.extract_features(symbol, market_data)
            if not features:
                return None
            
            # Prepare feature vector
            feature_vector = []
            for model_name, model in self.models.items():
                model_features = []
                for col in model.feature_columns:
                    model_features.append(features.get(col, 0))
                feature_vector.append(model_features)
            
            # Make predictions
            predictions = {}
            
            # Price direction prediction
            if 'price_direction' in self.models and feature_vector[0]:
                direction_model = self.models['price_direction']
                direction_pred = direction_model.model.predict([feature_vector[0]])[0]
                direction_proba = direction_model.model.predict_proba([feature_vector[0]])[0]
                predictions['direction'] = direction_pred
                predictions['direction_confidence'] = max(direction_proba)
            
            # Return prediction
            if 'return_prediction' in self.models and feature_vector[1]:
                return_model = self.models['return_prediction']
                expected_return = return_model.model.predict([feature_vector[1]])[0]
                predictions['expected_return'] = expected_return
            
            # Risk assessment
            if 'risk_assessment' in self.models and feature_vector[2]:
                risk_model = self.models['risk_assessment']
                risk_pred = risk_model.model.predict([feature_vector[2]])[0]
                risk_proba = risk_model.model.predict_proba([feature_vector[2]])[0]
                predictions['risk_score'] = risk_pred
                predictions['risk_confidence'] = max(risk_proba)
            
            # Combine predictions
            if 'direction' in predictions and 'expected_return' in predictions:
                action = 'BUY' if predictions['direction'] == 1 and predictions['expected_return'] > 0.01 else 'SELL' if predictions['direction'] == 0 and predictions['expected_return'] < -0.01 else 'HOLD'
                confidence = predictions.get('direction_confidence', 0.5)
                expected_return = predictions['expected_return']
                risk_score = predictions.get('risk_score', 0.5)
                
                prediction = MLPrediction(
                    symbol=symbol,
                    prediction=action,
                    confidence=confidence,
                    expected_return=expected_return,
                    risk_score=risk_score,
                    features=features,
                    model_name='ensemble',
                    timestamp=datetime.now()
                )
                
                # Update prediction count
                for model in self.models.values():
                    model.prediction_count += 1
                
                # Store prediction
                self.prediction_history.append(prediction)
                
                return prediction
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error making ML prediction for {symbol}: {e}")
            return None
    
    async def train_models(self, training_data: List[Dict[str, Any]]):
        """Train ML models with historical data"""
        try:
            self.logger.info(f"Training ML models with {len(training_data)} samples")
            
            for model_name, model in self.models.items():
                try:
                    # Prepare training data
                    X = []
                    y = []
                    
                    for sample in training_data:
                        features = sample.get('features', {})
                        target = sample.get('target', {})
                        
                        # Extract features for this model
                        model_features = []
                        for col in model.feature_columns:
                            model_features.append(features.get(col, 0))
                        
                        if len(model_features) == len(model.feature_columns):
                            X.append(model_features)
                            
                            # Extract target based on model type
                            if model_name == 'price_direction':
                                y.append(1 if target.get('price_change_1h', 0) > 0.01 else 0)
                            elif model_name == 'return_prediction':
                                y.append(target.get('price_change_1h', 0))
                            elif model_name == 'risk_assessment':
                                y.append(1 if target.get('volatility', 0) > 0.3 else 0)
                    
                    if len(X) > 10:  # Minimum training samples
                        X = np.array(X)
                        y = np.array(y)
                        
                        # Scale features
                        X_scaled = model.scaler.fit_transform(X)
                        
                        # Train model
                        model.model.fit(X_scaled, y)
                        
                        # Calculate accuracy
                        if model_name == 'price_direction' or model_name == 'risk_assessment':
                            y_pred = model.model.predict(X_scaled)
                            model.accuracy = accuracy_score(y, y_pred)
                        else:
                            # For regression, use R² score
                            y_pred = model.model.predict(X_scaled)
                            model.accuracy = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
                        
                        model.last_trained = datetime.now()
                        
                        # Save model
                        self._save_model(model_name, model)
                        
                        self.logger.info(f"Model {model_name} trained with accuracy: {model.accuracy:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training model {model_name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
    
    def _save_model(self, model_name: str, model: MLModel):
        """Save trained model to disk"""
        try:
            model_path = os.path.join(self.model_directory, f"{model_name}_model.joblib")
            joblib.dump(model, model_path)
            self.logger.info(f"Model {model_name} saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
    
    def _load_model(self, model_name: str) -> Optional[MLModel]:
        """Load trained model from disk"""
        try:
            model_path = os.path.join(self.model_directory, f"{model_name}_model.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.logger.info(f"Model {model_name} loaded from {model_path}")
                return model
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
        return None
    
    async def get_ml_insights(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive ML insights for a symbol"""
        try:
            prediction = await self.make_prediction(symbol, market_data)
            if not prediction:
                return {}
            
            # Get feature importance
            feature_importance = {}
            for model_name, model in self.models.items():
                if hasattr(model.model, 'feature_importances_'):
                    importance = model.model.feature_importances_
                    for i, col in enumerate(model.feature_columns):
                        if i < len(importance):
                            feature_importance[col] = importance[i]
            
            return {
                'symbol': symbol,
                'prediction': prediction.prediction,
                'confidence': prediction.confidence,
                'expected_return': prediction.expected_return,
                'risk_score': prediction.risk_score,
                'feature_importance': feature_importance,
                'model_accuracy': {name: model.accuracy for name, model in self.models.items()},
                'timestamp': prediction.timestamp.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting ML insights for {symbol}: {e}")
            return {}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all ML models"""
        try:
            status = {}
            for model_name, model in self.models.items():
                status[model_name] = {
                    'accuracy': model.accuracy,
                    'last_trained': model.last_trained.isoformat(),
                    'prediction_count': model.prediction_count,
                    'feature_count': len(model.feature_columns)
                }
            
            return {
                'models': status,
                'total_predictions': sum(model.prediction_count for model in self.models.values()),
                'average_accuracy': np.mean([model.accuracy for model in self.models.values()]),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting model status: {e}")
            return {}


