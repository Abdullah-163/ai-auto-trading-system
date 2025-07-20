"""
StrategyEngine Module for AI-powered Auto-Trading Application
Implements comprehensive trading strategies and technical analysis indicators
Includes RSI, MACD, Bollinger Bands, Moving Averages, Fibonacci, Price Action, and Volume Analysis
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
from dataclasses import dataclass
from enum import Enum
from config import config

class SignalType(Enum):
    """Enumeration for trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    """
    Data class representing a trading signal
    Contains signal type, strength, reasoning, and metadata
    """
    signal: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    indicators: Dict[str, float]
    timestamp: pd.Timestamp
    symbol: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class StrategyEngine:
    """
    Comprehensive trading strategy engine implementing multiple technical analysis methods
    Combines traditional indicators with advanced pattern recognition and volume analysis
    """
    
    def __init__(self):
        """
        Initialize StrategyEngine with configuration parameters
        """
        self.rsi_period = config.RSI_PERIOD
        self.rsi_overbought = config.RSI_OVERBOUGHT
        self.rsi_oversold = config.RSI_OVERSOLD
        
        self.macd_fast = config.MACD_FAST
        self.macd_slow = config.MACD_SLOW
        self.macd_signal = config.MACD_SIGNAL
        
        self.bb_period = config.BB_PERIOD
        self.bb_std = config.BB_STD
        
        # Additional strategy parameters
        self.sma_short = 20
        self.sma_long = 50
        self.ema_short = 12
        self.ema_long = 26
        
        logger.info("StrategyEngine initialized with configured parameters")
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            data: DataFrame with OHLCV data
            period: RSI calculation period (default from config)
            
        Returns:
            Series with RSI values
        """
        if period is None:
            period = self.rsi_period
            
        try:
            rsi = ta.momentum.RSIIndicator(close=data['close'], window=period).rsi()
            logger.debug(f"RSI calculated with period {period}")
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series(index=data.index, dtype=float)
    
    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        try:
            macd_indicator = ta.trend.MACD(
                close=data['close'],
                window_fast=self.macd_fast,
                window_slow=self.macd_slow,
                window_sign=self.macd_signal
            )
            
            result = {
                'macd': macd_indicator.macd(),
                'signal': macd_indicator.macd_signal(),
                'histogram': macd_indicator.macd_diff()
            }
            
            logger.debug("MACD calculated successfully")
            return result
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return {
                'macd': pd.Series(index=data.index, dtype=float),
                'signal': pd.Series(index=data.index, dtype=float),
                'histogram': pd.Series(index=data.index, dtype=float)
            }
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with upper band, middle band (SMA), and lower band
        """
        try:
            bb_indicator = ta.volatility.BollingerBands(
                close=data['close'],
                window=self.bb_period,
                window_dev=self.bb_std
            )
            
            result = {
                'upper': bb_indicator.bollinger_hband(),
                'middle': bb_indicator.bollinger_mavg(),
                'lower': bb_indicator.bollinger_lband(),
                'width': bb_indicator.bollinger_wband(),
                'percent': bb_indicator.bollinger_pband()
            }
            
            logger.debug("Bollinger Bands calculated successfully")
            return result
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return {key: pd.Series(index=data.index, dtype=float) for key in 
                   ['upper', 'middle', 'lower', 'width', 'percent']}
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate various moving averages (SMA and EMA)
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with different moving averages
        """
        try:
            result = {
                'sma_20': ta.trend.SMAIndicator(close=data['close'], window=self.sma_short).sma_indicator(),
                'sma_50': ta.trend.SMAIndicator(close=data['close'], window=self.sma_long).sma_indicator(),
                'ema_12': ta.trend.EMAIndicator(close=data['close'], window=self.ema_short).ema_indicator(),
                'ema_26': ta.trend.EMAIndicator(close=data['close'], window=self.ema_long).ema_indicator(),
                'sma_200': ta.trend.SMAIndicator(close=data['close'], window=200).sma_indicator()
            }
            
            logger.debug("Moving averages calculated successfully")
            return result
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
            return {key: pd.Series(index=data.index, dtype=float) for key in 
                   ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'sma_200']}
    
    def calculate_fibonacci_levels(self, data: pd.DataFrame, lookback: int = 50) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels based on recent high/low
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Number of periods to look back for high/low
            
        Returns:
            Dictionary with Fibonacci levels
        """
        try:
            # Get recent high and low
            recent_data = data.tail(lookback)
            high = recent_data['high'].max()
            low = recent_data['low'].min()
            
            # Calculate Fibonacci levels
            diff = high - low
            levels = {
                'high': high,
                'low': low,
                'fib_23.6': high - (diff * 0.236),
                'fib_38.2': high - (diff * 0.382),
                'fib_50.0': high - (diff * 0.500),
                'fib_61.8': high - (diff * 0.618),
                'fib_78.6': high - (diff * 0.786)
            }
            
            logger.debug("Fibonacci levels calculated successfully")
            return levels
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {e}")
            return {}
    
    def analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive volume analysis
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume analysis results
        """
        try:
            # Volume indicators
            volume_sma = ta.volume.VolumeSMAIndicator(
                close=data['close'], volume=data['volume'], window=20
            ).volume_sma()
            
            # On-Balance Volume
            obv = ta.volume.OnBalanceVolumeIndicator(
                close=data['close'], volume=data['volume']
            ).on_balance_volume()
            
            # Volume Rate of Change
            volume_roc = data['volume'].pct_change(periods=10)
            
            # Price-Volume Trend
            pvt = ta.volume.VolumePriceTrendIndicator(
                close=data['close'], volume=data['volume']
            ).volume_price_trend()
            
            # Current volume vs average
            current_volume = data['volume'].iloc[-1]
            avg_volume = volume_sma.iloc[-1] if not pd.isna(volume_sma.iloc[-1]) else current_volume
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            result = {
                'volume_sma': volume_sma,
                'obv': obv,
                'volume_roc': volume_roc,
                'pvt': pvt,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'high_volume': volume_ratio > 1.5,  # Volume 50% above average
                'low_volume': volume_ratio < 0.5    # Volume 50% below average
            }
            
            logger.debug("Volume analysis completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in volume analysis: {e}")
            return {}
    
    def detect_price_patterns(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Detect common price action patterns
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with detected patterns
        """
        try:
            patterns = {}
            
            # Get recent candles for pattern detection
            recent = data.tail(10)
            
            # Doji pattern (open ≈ close)
            last_candle = recent.iloc[-1]
            body_size = abs(last_candle['close'] - last_candle['open'])
            candle_range = last_candle['high'] - last_candle['low']
            patterns['doji'] = body_size < (candle_range * 0.1) if candle_range > 0 else False
            
            # Hammer pattern (small body, long lower shadow)
            lower_shadow = last_candle['open'] - last_candle['low'] if last_candle['open'] < last_candle['close'] else last_candle['close'] - last_candle['low']
            upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            patterns['hammer'] = (lower_shadow > body_size * 2) and (upper_shadow < body_size * 0.5)
            
            # Shooting star pattern (small body, long upper shadow)
            patterns['shooting_star'] = (upper_shadow > body_size * 2) and (lower_shadow < body_size * 0.5)
            
            # Engulfing patterns
            if len(recent) >= 2:
                prev_candle = recent.iloc[-2]
                curr_candle = recent.iloc[-1]
                
                # Bullish engulfing
                patterns['bullish_engulfing'] = (
                    prev_candle['close'] < prev_candle['open'] and  # Previous red
                    curr_candle['close'] > curr_candle['open'] and  # Current green
                    curr_candle['open'] < prev_candle['close'] and  # Gap down
                    curr_candle['close'] > prev_candle['open']       # Engulfs previous
                )
                
                # Bearish engulfing
                patterns['bearish_engulfing'] = (
                    prev_candle['close'] > prev_candle['open'] and  # Previous green
                    curr_candle['close'] < curr_candle['open'] and  # Current red
                    curr_candle['open'] > prev_candle['close'] and  # Gap up
                    curr_candle['close'] < prev_candle['open']       # Engulfs previous
                )
            
            # Support and resistance levels
            highs = recent['high'].rolling(window=3, center=True).max()
            lows = recent['low'].rolling(window=3, center=True).min()
            
            patterns['at_resistance'] = recent['close'].iloc[-1] >= highs.iloc[-2] * 0.99
            patterns['at_support'] = recent['close'].iloc[-1] <= lows.iloc[-2] * 1.01
            
            logger.debug("Price pattern detection completed")
            return patterns
        except Exception as e:
            logger.error(f"Error detecting price patterns: {e}")
            return {}
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> TradingSignal:
        """
        Generate comprehensive trading signal based on all indicators
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            TradingSignal object with recommendation
        """
        try:
            if len(data) < 50:  # Minimum data required
                logger.warning(f"Insufficient data for {symbol}: {len(data)} candles")
                return TradingSignal(
                    signal=SignalType.HOLD,
                    strength=0.0,
                    confidence=0.0,
                    reasoning=["Insufficient data"],
                    indicators={},
                    timestamp=data.index[-1],
                    symbol=symbol
                )
            
            # Calculate all indicators
            rsi = self.calculate_rsi(data)
            macd = self.calculate_macd(data)
            bb = self.calculate_bollinger_bands(data)
            ma = self.calculate_moving_averages(data)
            fib = self.calculate_fibonacci_levels(data)
            volume_analysis = self.analyze_volume(data)
            patterns = self.detect_price_patterns(data)
            
            # Current values
            current_price = data['close'].iloc[-1]
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            current_macd = macd['macd'].iloc[-1] if not pd.isna(macd['macd'].iloc[-1]) else 0
            current_signal = macd['signal'].iloc[-1] if not pd.isna(macd['signal'].iloc[-1]) else 0
            
            # Initialize signal components
            buy_signals = []
            sell_signals = []
            signal_strength = 0.0
            reasoning = []
            
            # RSI Analysis
            if current_rsi < self.rsi_oversold:
                buy_signals.append(0.3)
                reasoning.append(f"RSI oversold ({current_rsi:.1f})")
            elif current_rsi > self.rsi_overbought:
                sell_signals.append(0.3)
                reasoning.append(f"RSI overbought ({current_rsi:.1f})")
            
            # MACD Analysis
            if current_macd > current_signal and macd['histogram'].iloc[-1] > 0:
                buy_signals.append(0.25)
                reasoning.append("MACD bullish crossover")
            elif current_macd < current_signal and macd['histogram'].iloc[-1] < 0:
                sell_signals.append(0.25)
                reasoning.append("MACD bearish crossover")
            
            # Bollinger Bands Analysis
            if not pd.isna(bb['lower'].iloc[-1]) and current_price < bb['lower'].iloc[-1]:
                buy_signals.append(0.2)
                reasoning.append("Price below lower Bollinger Band")
            elif not pd.isna(bb['upper'].iloc[-1]) and current_price > bb['upper'].iloc[-1]:
                sell_signals.append(0.2)
                reasoning.append("Price above upper Bollinger Band")
            
            # Moving Average Analysis
            if (not pd.isna(ma['sma_20'].iloc[-1]) and not pd.isna(ma['sma_50'].iloc[-1]) and
                ma['sma_20'].iloc[-1] > ma['sma_50'].iloc[-1] and current_price > ma['sma_20'].iloc[-1]):
                buy_signals.append(0.2)
                reasoning.append("Price above rising moving averages")
            elif (not pd.isna(ma['sma_20'].iloc[-1]) and not pd.isna(ma['sma_50'].iloc[-1]) and
                  ma['sma_20'].iloc[-1] < ma['sma_50'].iloc[-1] and current_price < ma['sma_20'].iloc[-1]):
                sell_signals.append(0.2)
                reasoning.append("Price below falling moving averages")
            
            # Volume Analysis
            if volume_analysis.get('high_volume', False):
                # High volume strengthens the signal
                if buy_signals:
                    buy_signals.append(0.1)
                    reasoning.append("High volume confirms trend")
                elif sell_signals:
                    sell_signals.append(0.1)
                    reasoning.append("High volume confirms trend")
            
            # Pattern Analysis
            if patterns.get('bullish_engulfing', False):
                buy_signals.append(0.25)
                reasoning.append("Bullish engulfing pattern")
            elif patterns.get('bearish_engulfing', False):
                sell_signals.append(0.25)
                reasoning.append("Bearish engulfing pattern")
            
            if patterns.get('hammer', False):
                buy_signals.append(0.15)
                reasoning.append("Hammer pattern detected")
            elif patterns.get('shooting_star', False):
                sell_signals.append(0.15)
                reasoning.append("Shooting star pattern detected")
            
            # Calculate final signal
            buy_strength = sum(buy_signals)
            sell_strength = sum(sell_signals)
            
            # Determine signal type and strength
            if buy_strength > sell_strength:
                if buy_strength >= 0.7:
                    signal_type = SignalType.STRONG_BUY
                else:
                    signal_type = SignalType.BUY
                signal_strength = buy_strength
            elif sell_strength > buy_strength:
                if sell_strength >= 0.7:
                    signal_type = SignalType.STRONG_SELL
                else:
                    signal_type = SignalType.SELL
                signal_strength = sell_strength
            else:
                signal_type = SignalType.HOLD
                signal_strength = 0.0
                reasoning.append("Mixed signals - holding position")
            
            # Calculate confidence based on signal consistency
            confidence = min(signal_strength, 1.0)
            if len(reasoning) >= 3:  # Multiple confirming indicators
                confidence = min(confidence + 0.1, 1.0)
            
            # Prepare indicators dictionary
            indicators = {
                'rsi': current_rsi,
                'macd': current_macd,
                'macd_signal': current_signal,
                'bb_position': (current_price - bb['lower'].iloc[-1]) / (bb['upper'].iloc[-1] - bb['lower'].iloc[-1]) if not pd.isna(bb['lower'].iloc[-1]) else 0.5,
                'volume_ratio': volume_analysis.get('volume_ratio', 1.0),
                'price': current_price
            }
            
            # Calculate stop loss and take profit levels
            stop_loss = None
            take_profit = None
            
            if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                stop_loss = current_price * (1 - config.DEFAULT_STOP_LOSS_PCT)
                take_profit = current_price * (1 + config.DEFAULT_TAKE_PROFIT_PCT)
            elif signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                stop_loss = current_price * (1 + config.DEFAULT_STOP_LOSS_PCT)
                take_profit = current_price * (1 - config.DEFAULT_TAKE_PROFIT_PCT)
            
            signal = TradingSignal(
                signal=signal_type,
                strength=signal_strength,
                confidence=confidence,
                reasoning=reasoning,
                indicators=indicators,
                timestamp=data.index[-1],
                symbol=symbol,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            logger.info(f"Generated {signal_type.value} signal for {symbol} with strength {signal_strength:.2f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return TradingSignal(
                signal=SignalType.HOLD,
                strength=0.0,
                confidence=0.0,
                reasoning=[f"Error in analysis: {str(e)}"],
                indicators={},
                timestamp=data.index[-1] if len(data) > 0 else pd.Timestamp.now(),
                symbol=symbol
            )
    
    def backtest_strategy(self, data: pd.DataFrame, symbol: str, initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data
        
        Args:
            data: Historical OHLCV data
            symbol: Trading symbol
            initial_capital: Starting capital for backtest
            
        Returns:
            Dictionary with backtest results
        """
        try:
            logger.info(f"Starting backtest for {symbol} with {len(data)} candles")
            
            trades = []
            capital = initial_capital
            position = None
            position_size = 0
            
            # Generate signals for each candle (using expanding window)
            for i in range(50, len(data)):  # Start after minimum required data
                current_data = data.iloc[:i+1]
                signal = self.generate_signal(current_data, symbol)
                current_price = data['close'].iloc[i]
                
                # Execute trades based on signals
                if position is None:  # No position
                    if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] and signal.confidence > 0.6:
                        # Enter long position
                        position_size = (capital * config.MAX_POSITION_SIZE_PCT) / current_price
                        position = {
                            'type': 'LONG',
                            'entry_price': current_price,
                            'entry_time': data.index[i],
                            'size': position_size,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit
                        }
                        
                    elif signal.signal in [SignalType.SELL, SignalType.STRONG_SELL] and signal.confidence > 0.6:
                        # Enter short position (if supported)
                        position_size = (capital * config.MAX_POSITION_SIZE_PCT) / current_price
                        position = {
                            'type': 'SHORT',
                            'entry_price': current_price,
                            'entry_time': data.index[i],
                            'size': position_size,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit
                        }
                
                elif position is not None:  # Have position
                    exit_trade = False
                    exit_reason = ""
                    
                    # Check stop loss and take profit
                    if position['type'] == 'LONG':
                        if current_price <= position['stop_loss']:
                            exit_trade = True
                            exit_reason = "Stop Loss"
                        elif current_price >= position['take_profit']:
                            exit_trade = True
                            exit_reason = "Take Profit"
                        elif signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                            exit_trade = True
                            exit_reason = "Signal Reversal"
                    
                    elif position['type'] == 'SHORT':
                        if current_price >= position['stop_loss']:
                            exit_trade = True
                            exit_reason = "Stop Loss"
                        elif current_price <= position['take_profit']:
                            exit_trade = True
                            exit_reason = "Take Profit"
                        elif signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                            exit_trade = True
                            exit_reason = "Signal Reversal"
                    
                    if exit_trade:
                        # Calculate P&L
                        if position['type'] == 'LONG':
                            pnl = (current_price - position['entry_price']) * position['size']
                        else:  # SHORT
                            pnl = (position['entry_price'] - current_price) * position['size']
                        
                        capital += pnl
                        
                        # Record trade
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': data.index[i],
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'return_pct': (pnl / (position['entry_price'] * position['size'])) * 100,
                            'exit_reason': exit_reason
                        })
                        
                        position = None
                        position_size = 0
            
            # Calculate backtest metrics
            if trades:
                total_trades = len(trades)
                winning_trades = len([t for t in trades if t['pnl'] > 0])
                losing_trades = total_trades - winning_trades
                win_rate = (winning_trades / total_trades) * 100
                
                total_pnl = sum(t['pnl'] for t in trades)
                total_return = ((capital - initial_capital) / initial_capital) * 100
                
                avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
                avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if losing_trades > 0 else 0
                
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
                
                # Calculate maximum drawdown
                equity_curve = [initial_capital]
                for trade in trades:
                    equity_curve.append(equity_curve[-1] + trade['pnl'])
                
                peak = equity_curve[0]
                max_drawdown = 0
                for equity in equity_curve:
                    if equity > peak:
                        peak = equity
                    drawdown = ((peak - equity) / peak) * 100
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                results = {
                    'initial_capital': initial_capital,
                    'final_capital': capital,
                    'total_return_pct': total_return,
                    'total_pnl': total_pnl,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate_pct': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'max_drawdown_pct': max_drawdown,
                    'trades': trades,
                    'equity_curve': equity_curve
                }
                
                logger.info(f"Backtest completed for {symbol}: {total_trades} trades, {win_rate:.1f}% win rate, {total_return:.2f}% return")
                return results
            
            else:
                logger.warning(f"No trades generated in backtest for {symbol}")
                return {
                    'initial_capital': initial_capital,
                    'final_capital': capital,
                    'total_return_pct': 0,
                    'total_trades': 0,
                    'message': 'No trades generated'
                }
                
        except Exception as e:
            logger.error(f"Error in backtest for {symbol}: {e}")
            return {'error': str(e)}

# Global StrategyEngine instance
strategy_engine = StrategyEngine()

# Example usage and testing
def test_strategy_engine():
    """
    Test function to verify StrategyEngine functionality
    """
    logger.info("Testing StrategyEngine...")
    
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    volumes = np.random.randint(1000, 10000, 100)
    
    test_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Test signal generation
    signal = strategy_engine.generate_signal(test_data, 'TEST')
    logger.info(f"Generated signal: {signal.signal.value} with strength {signal.strength:.2f}")
    
    # Test backtest
    backtest_results = strategy_engine.backtest_strategy(test_data, 'TEST')
    if 'error' not in backtest_results:
        logger.info(f"Backtest results: {backtest_results.get('total_trades', 0)} trades, "
                   f"{backtest_results.get('win_rate_pct', 0):.1f}% win rate")

if __name__ == "__main__":
    test_strategy_engine()
