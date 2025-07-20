"""
DataFetcher Module for AI-powered Auto-Trading Application
Handles market data retrieval from Binance, Bybit, and MetaTrader5
Implements error handling, retry logic, and data validation
"""

import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import time
import MetaTrader5 as mt5
from binance.client import Client as BinanceClient
from binance.exceptions import BinanceAPIException
import pytz
from config import config

class DataFetcher:
    """
    Comprehensive data fetching class for multiple trading platforms
    Supports Forex, Stocks, Crypto, and Commodities across different exchanges
    """
    
    def __init__(self):
        """
        Initialize DataFetcher with API clients and connection parameters
        """
        self.binance_client = None
        self.bybit_session = None
        self.mt5_initialized = False
        self.last_fetch_time = {}
        self.data_cache = {}
        self.retry_count = 3
        self.retry_delay = 1  # seconds
        
        # Initialize API connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """
        Initialize connections to all trading platforms
        Handles authentication and connection validation
        """
        try:
            # Initialize Binance client
            if config.BINANCE_API_KEY and config.BINANCE_SECRET_KEY:
                self.binance_client = BinanceClient(
                    api_key=config.BINANCE_API_KEY,
                    api_secret=config.BINANCE_SECRET_KEY
                )
                logger.info("Binance client initialized successfully")
            else:
                logger.warning("Binance API keys not configured")
            
            # Initialize MetaTrader5
            if all([config.METATRADER_LOGIN, config.METATRADER_PASSWORD, config.METATRADER_SERVER]):
                if mt5.initialize():
                    if mt5.login(
                        login=int(config.METATRADER_LOGIN),
                        password=config.METATRADER_PASSWORD,
                        server=config.METATRADER_SERVER
                    ):
                        self.mt5_initialized = True
                        logger.info("MetaTrader5 initialized successfully")
                    else:
                        logger.error("MetaTrader5 login failed")
                else:
                    logger.error("MetaTrader5 initialization failed")
            else:
                logger.warning("MetaTrader5 credentials not configured")
                
        except Exception as e:
            logger.error(f"Error initializing connections: {e}")
    
    async def fetch_binance_data(self, symbol: str, interval: str = '1h', limit: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Binance
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max 1000)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.binance_client:
            logger.error("Binance client not initialized")
            return None
        
        for attempt in range(self.retry_count):
            try:
                logger.info(f"Fetching Binance data for {symbol} - Attempt {attempt + 1}")
                
                # Fetch klines data
                klines = self.binance_client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                
                if not klines:
                    logger.warning(f"No data received for {symbol}")
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Data preprocessing
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['open'] = pd.to_numeric(df['open'])
                df['high'] = pd.to_numeric(df['high'])
                df['low'] = pd.to_numeric(df['low'])
                df['close'] = pd.to_numeric(df['close'])
                df['volume'] = pd.to_numeric(df['volume'])
                
                # Set timestamp as index
                df.set_index('timestamp', inplace=True)
                
                # Keep only OHLCV columns
                df = df[['open', 'high', 'low', 'close', 'volume']]
                
                # Validate data quality
                if self._validate_data(df, symbol):
                    logger.info(f"Successfully fetched {len(df)} candles for {symbol} from Binance")
                    return df
                else:
                    logger.error(f"Data validation failed for {symbol}")
                    return None
                    
            except BinanceAPIException as e:
                logger.error(f"Binance API error for {symbol}: {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error fetching Binance data for {symbol}: {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                return None
        
        return None
    
    async def fetch_bybit_data(self, symbol: str, interval: str = '60', limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Bybit
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Timeframe in minutes ('1', '5', '15', '60', '240', 'D')
            limit: Number of candles to fetch (max 200)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        url = "https://api.bybit.com/v2/public/kline/list"
        
        for attempt in range(self.retry_count):
            try:
                logger.info(f"Fetching Bybit data for {symbol} - Attempt {attempt + 1}")
                
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if data['ret_code'] == 0 and data['result']:
                                # Convert to DataFrame
                                df = pd.DataFrame(data['result'])
                                
                                # Rename columns to standard format
                                df.rename(columns={
                                    'open_time': 'timestamp',
                                    'open': 'open',
                                    'high': 'high',
                                    'low': 'low',
                                    'close': 'close',
                                    'volume': 'volume'
                                }, inplace=True)
                                
                                # Data preprocessing
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                                df['open'] = pd.to_numeric(df['open'])
                                df['high'] = pd.to_numeric(df['high'])
                                df['low'] = pd.to_numeric(df['low'])
                                df['close'] = pd.to_numeric(df['close'])
                                df['volume'] = pd.to_numeric(df['volume'])
                                
                                # Set timestamp as index
                                df.set_index('timestamp', inplace=True)
                                
                                # Keep only OHLCV columns
                                df = df[['open', 'high', 'low', 'close', 'volume']]
                                
                                # Sort by timestamp
                                df.sort_index(inplace=True)
                                
                                if self._validate_data(df, symbol):
                                    logger.info(f"Successfully fetched {len(df)} candles for {symbol} from Bybit")
                                    return df
                                else:
                                    logger.error(f"Data validation failed for {symbol}")
                                    return None
                            else:
                                logger.error(f"Bybit API returned error: {data}")
                                return None
                        else:
                            logger.error(f"HTTP error {response.status} from Bybit")
                            
            except Exception as e:
                logger.error(f"Error fetching Bybit data for {symbol}: {e}")
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                return None
        
        return None
    
    def fetch_metatrader_data(self, symbol: str, timeframe: int = mt5.TIMEFRAME_H1, count: int = 500) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from MetaTrader5
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: MT5 timeframe constant
            count: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.mt5_initialized:
            logger.error("MetaTrader5 not initialized")
            return None
        
        try:
            logger.info(f"Fetching MT5 data for {symbol}")
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No data received for {symbol} from MT5")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Rename columns to standard format
            df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            }, inplace=True)
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Keep only OHLCV columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            if self._validate_data(df, symbol):
                logger.info(f"Successfully fetched {len(df)} candles for {symbol} from MT5")
                return df
            else:
                logger.error(f"Data validation failed for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching MT5 data for {symbol}: {e}")
            return None
    
    def _validate_data(self, df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate the quality and integrity of fetched data
        
        Args:
            df: DataFrame to validate
            symbol: Symbol name for logging
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check if DataFrame is empty
            if df.empty:
                logger.error(f"Empty DataFrame for {symbol}")
                return False
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}")
                return False
            
            # Check for null values
            if df.isnull().any().any():
                logger.warning(f"Null values found in data for {symbol}")
                # Fill null values with forward fill
                df.fillna(method='ffill', inplace=True)
            
            # Check for invalid OHLC relationships
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            
            if invalid_ohlc.any():
                logger.error(f"Invalid OHLC relationships found for {symbol}")
                return False
            
            # Check for negative values
            if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
                logger.error(f"Negative values found for {symbol}")
                return False
            
            # Check for extreme price movements (more than 50% in one candle)
            price_change = abs(df['close'] - df['open']) / df['open']
            if (price_change > 0.5).any():
                logger.warning(f"Extreme price movements detected for {symbol}")
            
            logger.info(f"Data validation passed for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {symbol}: {e}")
            return False
    
    async def fetch_multiple_symbols(self, symbols: List[str], exchange: str = 'binance') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols concurrently
        
        Args:
            symbols: List of symbols to fetch
            exchange: Exchange to fetch from ('binance', 'bybit', 'mt5')
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        results = {}
        
        if exchange.lower() == 'binance':
            tasks = [self.fetch_binance_data(symbol) for symbol in symbols]
        elif exchange.lower() == 'bybit':
            tasks = [self.fetch_bybit_data(symbol) for symbol in symbols]
        elif exchange.lower() == 'mt5':
            # MT5 doesn't support async, so we'll run sequentially
            for symbol in symbols:
                data = self.fetch_metatrader_data(symbol)
                if data is not None:
                    results[symbol] = data
            return results
        else:
            logger.error(f"Unsupported exchange: {exchange}")
            return results
        
        # Execute concurrent requests
        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, response in zip(symbols, responses):
                if isinstance(response, Exception):
                    logger.error(f"Error fetching {symbol}: {response}")
                elif response is not None:
                    results[symbol] = response
                    
        except Exception as e:
            logger.error(f"Error in concurrent fetch: {e}")
        
        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results
    
    def get_latest_price(self, symbol: str, exchange: str = 'binance') -> Optional[float]:
        """
        Get the latest price for a symbol
        
        Args:
            symbol: Trading symbol
            exchange: Exchange to fetch from
            
        Returns:
            Latest price or None if failed
        """
        try:
            if exchange.lower() == 'binance' and self.binance_client:
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            
            elif exchange.lower() == 'mt5' and self.mt5_initialized:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    return float(tick.bid)
            
            logger.error(f"Cannot get latest price for {symbol} from {exchange}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    def cleanup(self):
        """
        Clean up connections and resources
        """
        try:
            if self.mt5_initialized:
                mt5.shutdown()
                logger.info("MetaTrader5 connection closed")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global DataFetcher instance
data_fetcher = DataFetcher()

# Example usage and testing
async def test_data_fetcher():
    """
    Test function to verify DataFetcher functionality
    """
    logger.info("Testing DataFetcher...")
    
    # Test Binance data fetch
    btc_data = await data_fetcher.fetch_binance_data('BTCUSDT', '1h', 100)
    if btc_data is not None:
        logger.info(f"BTC data shape: {btc_data.shape}")
        logger.info(f"Latest BTC price: {btc_data['close'].iloc[-1]}")
    
    # Test multiple symbols
    crypto_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    crypto_data = await data_fetcher.fetch_multiple_symbols(crypto_symbols, 'binance')
    logger.info(f"Fetched data for {len(crypto_data)} crypto symbols")
    
    # Test latest price
    latest_btc = data_fetcher.get_latest_price('BTCUSDT', 'binance')
    if latest_btc:
        logger.info(f"Latest BTC price: ${latest_btc}")

if __name__ == "__main__":
    # Run test when module is executed directly
    asyncio.run(test_data_fetcher())
