"""
Configuration module for AI-powered Auto-Trading Application
Handles environment variables, API keys, and global settings
"""

import os
from dotenv import load_dotenv
from typing import Optional
from loguru import logger

# Load environment variables from .env file
load_dotenv()

class TradingConfig:
    """
    Central configuration class for the trading application
    Manages API keys, trading parameters, and system settings
    """
    
    def __init__(self):
        # API Keys - Retrieved from environment variables for security
        self.BINANCE_API_KEY: Optional[str] = os.getenv("BINANCE_API_KEY")
        self.BINANCE_SECRET_KEY: Optional[str] = os.getenv("BINANCE_SECRET_KEY")
        
        self.BYBIT_API_KEY: Optional[str] = os.getenv("BYBIT_API_KEY")
        self.BYBIT_SECRET_KEY: Optional[str] = os.getenv("BYBIT_SECRET_KEY")
        
        self.METATRADER_LOGIN: Optional[str] = os.getenv("METATRADER_LOGIN")
        self.METATRADER_PASSWORD: Optional[str] = os.getenv("METATRADER_PASSWORD")
        self.METATRADER_SERVER: Optional[str] = os.getenv("METATRADER_SERVER")
        
        # Trading Configuration
        self.AUTO_TRADING_ENABLED: bool = os.getenv("AUTO_TRADING_ENABLED", "False").lower() == "true"
        self.MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))  # 2% default
        self.MAX_DAILY_TRADES: int = int(os.getenv("MAX_DAILY_TRADES", "10"))
        self.MIN_ACCURACY_THRESHOLD: float = float(os.getenv("MIN_ACCURACY_THRESHOLD", "0.75"))  # 75% minimum
        
        # Technical Analysis Parameters
        self.RSI_PERIOD: int = int(os.getenv("RSI_PERIOD", "14"))
        self.RSI_OVERBOUGHT: float = float(os.getenv("RSI_OVERBOUGHT", "70"))
        self.RSI_OVERSOLD: float = float(os.getenv("RSI_OVERSOLD", "30"))
        
        self.MACD_FAST: int = int(os.getenv("MACD_FAST", "12"))
        self.MACD_SLOW: int = int(os.getenv("MACD_SLOW", "26"))
        self.MACD_SIGNAL: int = int(os.getenv("MACD_SIGNAL", "9"))
        
        self.BB_PERIOD: int = int(os.getenv("BB_PERIOD", "20"))
        self.BB_STD: float = float(os.getenv("BB_STD", "2.0"))
        
        # Machine Learning Configuration
        self.ML_MODEL_RETRAIN_INTERVAL: int = int(os.getenv("ML_MODEL_RETRAIN_INTERVAL", "24"))  # hours
        self.ML_LOOKBACK_PERIOD: int = int(os.getenv("ML_LOOKBACK_PERIOD", "100"))  # candles
        self.ML_PREDICTION_CONFIDENCE: float = float(os.getenv("ML_PREDICTION_CONFIDENCE", "0.8"))
        
        # Risk Management
        self.DEFAULT_STOP_LOSS_PCT: float = float(os.getenv("DEFAULT_STOP_LOSS_PCT", "0.02"))  # 2%
        self.DEFAULT_TAKE_PROFIT_PCT: float = float(os.getenv("DEFAULT_TAKE_PROFIT_PCT", "0.04"))  # 4%
        self.MAX_POSITION_SIZE_PCT: float = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.1"))  # 10% of portfolio
        
        # Data Configuration
        self.DATA_REFRESH_INTERVAL: int = int(os.getenv("DATA_REFRESH_INTERVAL", "60"))  # seconds
        self.HISTORICAL_DATA_DAYS: int = int(os.getenv("HISTORICAL_DATA_DAYS", "365"))
        
        # Supported Markets and Symbols
        self.FOREX_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        self.CRYPTO_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        self.STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        self.COMMODITY_SYMBOLS = ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"]
        
        # Logging Configuration
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH", "logs/trading_app.log")
        
        # Validate critical configurations
        self._validate_config()
    
    def _validate_config(self):
        """
        Validate critical configuration parameters
        Logs warnings for missing API keys and invalid parameters
        """
        # Check API keys
        if not self.BINANCE_API_KEY or not self.BINANCE_SECRET_KEY:
            logger.warning("Binance API keys not configured - Binance trading will be disabled")
        
        if not self.BYBIT_API_KEY or not self.BYBIT_SECRET_KEY:
            logger.warning("Bybit API keys not configured - Bybit trading will be disabled")
        
        if not all([self.METATRADER_LOGIN, self.METATRADER_PASSWORD, self.METATRADER_SERVER]):
            logger.warning("MetaTrader5 credentials not configured - MT5 trading will be disabled")
        
        # Validate risk parameters
        if self.MAX_RISK_PER_TRADE > 0.1:  # 10%
            logger.warning(f"High risk per trade configured: {self.MAX_RISK_PER_TRADE*100}%")
        
        if self.MIN_ACCURACY_THRESHOLD < 0.6:  # 60%
            logger.warning(f"Low accuracy threshold: {self.MIN_ACCURACY_THRESHOLD*100}%")
        
        logger.info("Configuration validation completed")
    
    def get_active_symbols(self) -> list:
        """
        Returns list of all active trading symbols across all markets
        """
        return (self.FOREX_SYMBOLS + self.CRYPTO_SYMBOLS + 
                self.STOCK_SYMBOLS + self.COMMODITY_SYMBOLS)
    
    def is_trading_enabled(self) -> bool:
        """
        Check if auto-trading is currently enabled
        """
        return self.AUTO_TRADING_ENABLED
    
    def toggle_trading(self) -> bool:
        """
        Toggle auto-trading on/off
        Returns new state
        """
        self.AUTO_TRADING_ENABLED = not self.AUTO_TRADING_ENABLED
        logger.info(f"Auto-trading {'enabled' if self.AUTO_TRADING_ENABLED else 'disabled'}")
        return self.AUTO_TRADING_ENABLED

# Global configuration instance
config = TradingConfig()

# Environment file template for users
ENV_TEMPLATE = """
# AI Auto-Trading Application Environment Configuration
# Copy this to .env and fill in your actual API keys

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Bybit API Configuration  
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_SECRET_KEY=your_bybit_secret_key_here

# MetaTrader5 Configuration
METATRADER_LOGIN=your_mt5_login_here
METATRADER_PASSWORD=your_mt5_password_here
METATRADER_SERVER=your_mt5_server_here

# Trading Settings
AUTO_TRADING_ENABLED=false
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_TRADES=10
MIN_ACCURACY_THRESHOLD=0.75

# Technical Analysis Settings
RSI_PERIOD=14
RSI_OVERBOUGHT=70
RSI_OVERSOLD=30
MACD_FAST=12
MACD_SLOW=26
MACD_SIGNAL=9
BB_PERIOD=20
BB_STD=2.0

# Machine Learning Settings
ML_MODEL_RETRAIN_INTERVAL=24
ML_LOOKBACK_PERIOD=100
ML_PREDICTION_CONFIDENCE=0.8

# Risk Management
DEFAULT_STOP_LOSS_PCT=0.02
DEFAULT_TAKE_PROFIT_PCT=0.04
MAX_POSITION_SIZE_PCT=0.1

# Data Settings
DATA_REFRESH_INTERVAL=60
HISTORICAL_DATA_DAYS=365

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/trading_app.log
"""

def create_env_template():
    """
    Create a .env.template file for users to configure their API keys
    """
    try:
        with open('.env.template', 'w') as f:
            f.write(ENV_TEMPLATE)
        logger.info("Created .env.template file - please copy to .env and configure your API keys")
    except Exception as e:
        logger.error(f"Failed to create .env.template: {e}")

if __name__ == "__main__":
    # Create environment template when run directly
    create_env_template()
    print("Configuration module loaded successfully")
    print(f"Auto-trading enabled: {config.is_trading_enabled()}")
    print(f"Active symbols: {len(config.get_active_symbols())}")
