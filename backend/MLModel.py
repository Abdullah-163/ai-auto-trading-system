"""
MLModel Module for AI-powered Auto-Trading Application
Implements machine learning capabilities using scikit-learn and TensorFlow
Learns from historical market data to optimize trading strategies and predict price movements
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from loguru import logger
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available - using scikit-learn only")
    TENSORFLOW_AVAILABLE = False

from config import config
from StrategyEngine import strategy_engine, SignalType

class MLTradingModel:
    """
    Machine Learning model for trading signal prediction and strategy optimization
    Supports both traditional ML (scikit-learn) and deep learning (TensorFlow) approaches
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ML Trading Model
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boost', 'logistic', 'svm', 'neural_network', 'lstm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.is_trained = False
        self.feature_names = []
        self.model_metrics = {}
        self.model_path = f"models/{model_type}_trading_model"
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Initialize model based on type
        self._initialize_model()
        
        logger.info(f"MLTradingModel initialized with {model_type}")
    
    def _initialize_model(self):
        """
        Initialize the specific model based on model_type
        """
        try:
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            
            elif self.model_type == 'gradient_boost':
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            
            elif self.model_type == 'logistic':
                self.model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    multi_class='ovr'
                )
            
            elif self.model_type == 'svm':
                self.model = SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42
                )
            
            elif self.model_type in ['neural_network', 'lstm']:
                if not TENSORFLOW_AVAILABLE:
                    logger.error("TensorFlow not available for neural network models")
                    self.model_type = 'random_forest'
                    self._initialize_model()
                    return
                # Neural network models will be created in _create_neural_network
                
            logger.info(f"Model {self.model_type} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model {self.model_type}: {e}")
            # Fallback to random forest
            self.model_type = 'random_forest'
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
    
    def _create_neural_network(self, input_shape: int, num_classes: int) -> Sequential:
        """
        Create neural network model for classification
        
        Args:
            input_shape: Number of input features
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_lstm_model(self, sequence_length: int, num_features: int, num_classes: int) -> Sequential:
        """
        Create LSTM model for time series prediction
        
        Args:
            sequence_length: Length of input sequences
            num_features: Number of features per timestep
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, num_features)),
            Dropout(0.2),
            
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            
            LSTM(50),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Prepare features for machine learning from OHLCV data
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            DataFrame with engineered features
        """
        try:
            logger.info(f"Preparing features for {symbol}")
            
            # Make a copy to avoid modifying original data
            df = data.copy()
            
            # Basic price features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['open_close_ratio'] = df['open'] / df['close']
            df['volume_change'] = df['volume'].pct_change()
            
            # Technical indicators from StrategyEngine
            df['rsi'] = strategy_engine.calculate_rsi(df)
            
            macd = strategy_engine.calculate_macd(df)
            df['macd'] = macd['macd']
            df['macd_signal'] = macd['signal']
            df['macd_histogram'] = macd['histogram']
            
            bb = strategy_engine.calculate_bollinger_bands(df)
            df['bb_upper'] = bb['upper']
            df['bb_middle'] = bb['middle']
            df['bb_lower'] = bb['lower']
            df['bb_width'] = bb['width']
            df['bb_position'] = (df['close'] - bb['lower']) / (bb['upper'] - bb['lower'])
            
            ma = strategy_engine.calculate_moving_averages(df)
            df['sma_20'] = ma['sma_20']
            df['sma_50'] = ma['sma_50']
            df['ema_12'] = ma['ema_12']
            df['ema_26'] = ma['ema_26']
            
            # Price position relative to moving averages
            df['price_vs_sma20'] = df['close'] / df['sma_20'] - 1
            df['price_vs_sma50'] = df['close'] / df['sma_50'] - 1
            df['sma20_vs_sma50'] = df['sma_20'] / df['sma_50'] - 1
            
            # Volatility features
            df['volatility'] = df['price_change'].rolling(window=20).std()
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            
            # Volume features
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Momentum features
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Support and resistance levels
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['price_vs_resistance'] = df['close'] / df['resistance']
            df['price_vs_support'] = df['close'] / df['support']
            
            # Time-based features
            df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
            df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
            df['month'] = df.index.month if hasattr(df.index, 'month') else 1
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            
            # Remove original OHLCV columns and keep only features
            feature_columns = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            features_df = df[feature_columns].copy()
            
            # Remove rows with NaN values
            features_df = features_df.dropna()
            
            logger.info(f"Prepared {len(feature_columns)} features for {symbol}")
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features for {symbol}: {e}")
            return pd.DataFrame()
    
    def create_labels(self, data: pd.DataFrame, lookahead: int = 5) -> pd.Series:
        """
        Create trading labels based on future price movements
        
        Args:
            data: OHLCV DataFrame
            lookahead: Number of periods to look ahead for labeling
            
        Returns:
            Series with trading labels (0=SELL, 1=HOLD, 2=BUY)
        """
        try:
            # Calculate future returns
            future_returns = data['close'].shift(-lookahead) / data['close'] - 1
            
            # Create labels based on return thresholds
            labels = pd.Series(index=data.index, dtype=int)
            
            # Define thresholds
            buy_threshold = 0.02   # 2% gain
            sell_threshold = -0.02  # 2% loss
            
            # Assign labels
            labels[future_returns > buy_threshold] = 2  # BUY
            labels[future_returns < sell_threshold] = 0  # SELL
            labels[(future_returns >= sell_threshold) & (future_returns <= buy_threshold)] = 1  # HOLD
            
            # Remove NaN values
            labels = labels.dropna()
            
            logger.info(f"Created labels: BUY={sum(labels==2)}, HOLD={sum(labels==1)}, SELL={sum(labels==0)}")
            return labels
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return pd.Series()
    
    def train_model(self, data: pd.DataFrame, symbol: str, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the machine learning model
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            logger.info(f"Training {self.model_type} model for {symbol}")
            
            # Prepare features and labels
            features_df = self.prepare_features(data, symbol)
            if features_df.empty:
                return {'error': 'Failed to prepare features'}
            
            labels = self.create_labels(data)
            if labels.empty:
                return {'error': 'Failed to create labels'}
            
            # Align features and labels
            common_index = features_df.index.intersection(labels.index)
            X = features_df.loc[common_index]
            y = labels.loc[common_index]
            
            if len(X) < 100:  # Minimum samples required
                return {'error': f'Insufficient data: {len(X)} samples'}
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Feature selection
            self.feature_selector = SelectKBest(f_classif, k=min(20, len(self.feature_names)))
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            # Train model based on type
            if self.model_type in ['neural_network', 'lstm']:
                return self._train_neural_network(X_train_selected, X_test_selected, y_train, y_test, symbol)
            else:
                return self._train_sklearn_model(X_train_selected, X_test_selected, y_train, y_test, symbol)
                
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {'error': str(e)}
    
    def _train_sklearn_model(self, X_train, X_test, y_train, y_test, symbol: str) -> Dict[str, Any]:
        """
        Train scikit-learn model
        """
        try:
            # Hyperparameter tuning for Random Forest
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
                
                grid_search = GridSearchCV(
                    self.model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                logger.info(f"Best parameters: {grid_search.best_params_}")
            
            else:
                # Train model directly
                self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            
            self.model_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            logger.info(f"Model trained successfully for {symbol}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            
            return {
                'success': True,
                'metrics': self.model_metrics,
                'feature_importance': self._get_feature_importance(),
                'classification_report': classification_report(y_test, y_pred)
            }
            
        except Exception as e:
            logger.error(f"Error training sklearn model: {e}")
            return {'error': str(e)}
    
    def _train_neural_network(self, X_train, X_test, y_train, y_test, symbol: str) -> Dict[str, Any]:
        """
        Train neural network model
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                return {'error': 'TensorFlow not available'}
            
            # Create model
            num_classes = len(np.unique(y_train))
            
            if self.model_type == 'lstm':
                # Reshape data for LSTM (samples, timesteps, features)
                sequence_length = 10
                X_train_lstm = self._create_sequences(X_train, sequence_length)
                X_test_lstm = self._create_sequences(X_test, sequence_length)
                y_train_lstm = y_train[sequence_length-1:]
                y_test_lstm = y_test[sequence_length-1:]
                
                self.model = self._create_lstm_model(sequence_length, X_train.shape[1], num_classes)
                X_train_final, X_test_final = X_train_lstm, X_test_lstm
                y_train_final, y_test_final = y_train_lstm, y_test_lstm
            else:
                self.model = self._create_neural_network(X_train.shape[1], num_classes)
                X_train_final, X_test_final = X_train, X_test
                y_train_final, y_test_final = y_train, y_test
            
            # Callbacks
            early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5)
            
            # Train model
            history = self.model.fit(
                X_train_final, y_train_final,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_final, y_test_final),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(X_test_final, y_test_final, verbose=0)
            y_pred = np.argmax(self.model.predict(X_test_final), axis=1)
            
            precision = precision_score(y_test_final, y_pred, average='weighted')
            recall = recall_score(y_test_final, y_pred, average='weighted')
            f1 = f1_score(y_test_final, y_pred, average='weighted')
            
            self.model_metrics = {
                'accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'test_loss': test_loss,
                'training_samples': len(X_train_final),
                'test_samples': len(X_test_final)
            }
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            logger.info(f"Neural network trained successfully for {symbol}: Accuracy={test_accuracy:.3f}")
            
            return {
                'success': True,
                'metrics': self.model_metrics,
                'training_history': history.history,
                'classification_report': classification_report(y_test_final, y_pred)
            }
            
        except Exception as e:
            logger.error(f"Error training neural network: {e}")
            return {'error': str(e)}
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Create sequences for LSTM training
        """
        sequences = []
        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
        return np.array(sequences)
    
    def predict(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Make predictions using the trained model
        
        Args:
            data: OHLCV DataFrame
            symbol: Trading symbol
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            if not self.is_trained:
                return {'error': 'Model not trained'}
            
            # Prepare features
            features_df = self.prepare_features(data, symbol)
            if features_df.empty:
                return {'error': 'Failed to prepare features'}
            
            # Use only the latest data point
            X = features_df.iloc[-1:][self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Select features
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Make prediction
            if self.model_type in ['neural_network', 'lstm']:
                if self.model_type == 'lstm':
                    # For LSTM, we need sequence data
                    if len(features_df) < 10:
                        return {'error': 'Insufficient data for LSTM prediction'}
                    X_seq = features_df.iloc[-10:][self.feature_names]
                    X_seq_scaled = self.scaler.transform(X_seq)
                    X_seq_selected = self.feature_selector.transform(X_seq_scaled)
                    X_seq_reshaped = X_seq_selected.reshape(1, 10, -1)
                    
                    prediction_proba = self.model.predict(X_seq_reshaped)[0]
                    prediction = np.argmax(prediction_proba)
                else:
                    prediction_proba = self.model.predict(X_selected)[0]
                    prediction = np.argmax(prediction_proba)
            else:
                prediction = self.model.predict(X_selected)[0]
                prediction_proba = self.model.predict_proba(X_selected)[0] if hasattr(self.model, 'predict_proba') else None
            
            # Convert prediction to signal type
            signal_map = {0: SignalType.SELL, 1: SignalType.HOLD, 2: SignalType.BUY}
            predicted_signal = signal_map.get(prediction, SignalType.HOLD)
            
            # Calculate confidence
            confidence = float(np.max(prediction_proba)) if prediction_proba is not None else 0.5
            
            result = {
                'signal': predicted_signal,
                'confidence': confidence,
                'probabilities': {
                    'sell': float(prediction_proba[0]) if prediction_proba is not None else 0.33,
                    'hold': float(prediction_proba[1]) if prediction_proba is not None else 0.33,
                    'buy': float(prediction_proba[2]) if prediction_proba is not None else 0.33
                },
                'model_type': self.model_type,
                'timestamp': data.index[-1]
            }
            
            logger.info(f"ML Prediction for {symbol}: {predicted_signal.value} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {e}")
            return {'error': str(e)}
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model
        """
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Get selected feature names
                selected_features = self.feature_selector.get_support()
                selected_feature_names = [name for i, name in enumerate(self.feature_names) if selected_features[i]]
                
                importance_dict = dict(zip(selected_feature_names, self.model.feature_importances_))
                # Sort by importance
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def save_model(self):
        """
        Save the trained model and preprocessing objects
        """
        try:
            if self.model_type in ['neural_network', 'lstm']:
                # Save TensorFlow model
                self.model.save(f"{self.model_path}.h5")
            else:
                # Save scikit-learn model
                joblib.dump(self.model, f"{self.model_path}.pkl")
            
            # Save preprocessing objects
            joblib.dump(self.scaler, f"{self.model_path}_scaler.pkl")
            joblib.dump(self.feature_selector, f"{self.model_path}_selector.pkl")
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'is_trained': self.is_trained,
                'metrics': self.model_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(f"{self.model_path}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved successfully: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self) -> bool:
        """
        Load a previously trained model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load metadata
            import json
            with open(f"{self.model_path}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.model_type = metadata['model_type']
            self.feature_names = metadata['feature_names']
            self.model_metrics = metadata['metrics']
            
            # Load model
            if self.model_type in ['neural_network', 'lstm']:
                if TENSORFLOW_AVAILABLE:
                    self.model = load_model(f"{self.model_path}.h5")
                else:
                    logger.error("TensorFlow not available for loading neural network")
                    return False
            else:
                self.model = joblib.load(f"{self.model_path}.pkl")
            
            # Load preprocessing objects
            self.scaler = joblib.load(f"{self.model_path}_scaler.pkl")
            self.feature_selector = joblib.load(f"{self.model_path}_selector.pkl")
            
            self.is_trained = True
            logger.info(f"Model loaded successfully: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

class MLModelManager:
    """
    Manager class for handling multiple ML models for different symbols and strategies
    """
    
    def __init__(self):
        """
        Initialize ML Model Manager
        """
        self.models = {}  # symbol -> MLTradingModel
        self.model_types = ['random_forest', 'gradient_boost', 'neural_network']
        self.retrain_interval = timedelta(hours=config.ML_MODEL_RETRAIN_INTERVAL)
        self.last_retrain = {}
        
        logger.info("MLModelManager initialized")
    
    def get_or_create_model(self, symbol: str, model_type: str = 'random_forest') -> MLTradingModel:
        """
        Get existing model or create new one for symbol
        
        Args:
            symbol: Trading symbol
            model_type: Type of ML model
            
        Returns:
            MLTradingModel instance
        """
        model_key = f"{symbol}_{model_type}"
        
        if model_key not in self.models:
            self.models[model_key] = MLTradingModel(model_type)
            # Try to load existing model
            self.models[model_key].load_model()
        
        return self.models[model_key]
    
    def train_model_for_symbol(self, symbol: str, data: pd.DataFrame, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train ML model for specific symbol
        
        Args:
            symbol: Trading symbol
            data: Historical OHLCV data
            model_type: Type of ML model
            
        Returns:
            Training results
        """
        try:
            model = self.get_or_create_model(symbol, model_type)
            results = model.train_model(data, symbol)
            
            if results.get('success'):
                self.last_retrain[f"{symbol}_{model_type}"] = datetime.now()
                logger.info(f"Successfully trained {model_type} model for {symbol}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            return {'error': str(e)}
    
    def get_ml_prediction(self, symbol: str, data: pd.DataFrame, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Get ML prediction for symbol
        
        Args:
            symbol: Trading symbol
            data: Recent OHLCV data
            model_type: Type of ML model
            
        Returns:
            Prediction results
        """
        try:
            model = self.get_or_create_model(symbol, model_type)
            
            if not model.is_trained:
                return {'error': f'Model not trained for {symbol}'}
            
            return model.predict(data, symbol)
            
        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return {'error': str(e)}
    
    def should_retrain(self, symbol: str, model_type: str = 'random_
