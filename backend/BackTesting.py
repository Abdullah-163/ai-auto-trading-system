"""
BackTesting Module for AI-powered Auto-Trading Application
Provides comprehensive backtesting capabilities for trading strategies
Includes performance metrics, risk analysis, and detailed reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from loguru import logger
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config import config
from StrategyEngine import strategy_engine, SignalType, TradingSignal
from MLModel import MLModelManager
from RiskManager import risk_manager, RiskLevel
from DataFetcher import data_fetcher

class BacktestMode(Enum):
    """Backtesting mode enumeration"""
    STRATEGY_ONLY = "STRATEGY_ONLY"
    ML_ONLY = "ML_ONLY"
    COMBINED = "COMBINED"
    COMPARISON = "COMPARISON"

@dataclass
class BacktestConfig:
    """
    Configuration for backtesting parameters
    """
    initial_capital: float = 10000.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005   # 0.05%
    max_positions: int = 5
    position_size_pct: float = 0.2  # 20% per position
    stop_loss_pct: float = 0.02     # 2%
    take_profit_pct: float = 0.04   # 4%
    min_trade_interval: int = 1     # Minimum bars between trades
    benchmark_symbol: str = "BTCUSDT"

@dataclass
class Trade:
    """
    Individual trade record for backtesting
    """
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    duration_hours: float
    exit_reason: str
    strategy_signal: Optional[str] = None
    ml_signal: Optional[str] = None
    confidence: float = 0.0

@dataclass
class BacktestResults:
    """
    Comprehensive backtesting results
    """
    # Basic metrics
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_pnl: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    
    # Performance metrics
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Advanced metrics
    var_95_pct: float  # Value at Risk
    expected_shortfall_pct: float
    recovery_factor: float
    
    # Time-based metrics
    total_duration_days: int
    avg_trade_duration_hours: float
    trades_per_month: float
    
    # Detailed records
    trades: List[Trade]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    monthly_returns: pd.Series
    
    # Strategy comparison (if applicable)
    strategy_performance: Optional[Dict[str, float]] = None
    ml_performance: Optional[Dict[str, float]] = None

class BackTester:
    """
    Comprehensive backtesting engine for trading strategies
    Supports multiple strategies, ML models, and detailed performance analysis
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize BackTester
        
        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.ml_manager = MLModelManager()
        
        # Results storage
        self.results_history = []
        self.comparison_results = {}
        
        # Create results directory
        self.results_dir = Path("backtest_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("BackTester initialized")
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    symbol: str,
                    mode: BacktestMode = BacktestMode.STRATEGY_ONLY,
                    strategy_func: Optional[Callable] = None,
                    ml_model_type: str = 'random_forest') -> BacktestResults:
        """
        Run comprehensive backtest
        
        Args:
            data: Historical OHLCV data
            symbol: Trading symbol
            mode: Backtesting mode
            strategy_func: Custom strategy function (optional)
            ml_model_type: ML model type for ML modes
            
        Returns:
            BacktestResults object
        """
        try:
            logger.info(f"Starting backtest for {symbol} in {mode.value} mode")
            
            # Validate data
            if len(data) < 100:
                raise ValueError(f"Insufficient data: {len(data)} bars")
            
            # Filter data by date range if specified
            if self.config.start_date or self.config.end_date:
                data = self._filter_data_by_date(data)
            
            # Initialize backtest state
            capital = self.config.initial_capital
            positions = {}  # symbol -> position info
            trades = []
            equity_curve = []
            timestamps = []
            
            # Prepare ML model if needed
            if mode in [BacktestMode.ML_ONLY, BacktestMode.COMBINED, BacktestMode.COMPARISON]:
                logger.info(f"Training ML model for backtesting...")
                train_data = data.iloc[:int(len(data) * 0.7)]  # Use 70% for training
                ml_results = self.ml_manager.train_model_for_symbol(symbol, train_data, ml_model_type)
                if not ml_results.get('success'):
                    logger.warning(f"ML model training failed: {ml_results.get('error')}")
                    if mode == BacktestMode.ML_ONLY:
                        mode = BacktestMode.STRATEGY_ONLY
            
            # Main backtesting loop
            for i in range(50, len(data)):  # Start after minimum required data
                current_time = data.index[i]
                current_data = data.iloc[:i+1]
                current_price = data['close'].iloc[i]
                
                # Update position P&L
                self._update_positions_pnl(positions, current_price)
                
                # Check for exit conditions
                exits = self._check_exit_conditions(positions, current_price, current_time)
                for exit_info in exits:
                    trade = self._close_position(positions, exit_info, capital)
                    if trade:
                        trades.append(trade)
                        capital += trade.pnl
                
                # Generate signals based on mode
                signals = self._generate_signals(current_data, symbol, mode, strategy_func, ml_model_type)
                
                # Process entry signals
                for signal_info in signals:
                    if len(positions) < self.config.max_positions:
                        entry_result = self._enter_position(
                            positions, signal_info, current_price, current_time, capital
                        )
                        if entry_result:
                            capital -= entry_result['position_value']
                
                # Record equity curve
                total_equity = capital + sum(pos.get('unrealized_pnl', 0) for pos in positions.values())
                equity_curve.append(total_equity)
                timestamps.append(current_time)
            
            # Close remaining positions
            for symbol_key in list(positions.keys()):
                exit_info = {
                    'symbol': symbol_key,
                    'price': data['close'].iloc[-1],
                    'reason': 'backtest_end'
                }
                trade = self._close_position(positions, exit_info, capital)
                if trade:
                    trades.append(trade)
                    capital += trade.pnl
            
            # Calculate final equity
            final_equity = capital
            equity_curve.append(final_equity)
            timestamps.append(data.index[-1])
            
            # Create equity curve series
            equity_series = pd.Series(equity_curve, index=timestamps)
            
            # Calculate comprehensive results
            results = self._calculate_results(
                trades, equity_series, self.config.initial_capital, data, symbol
            )
            
            # Store results
            self.results_history.append(results)
            
            # Save results
            self._save_results(results, symbol, mode)
            
            logger.info(f"Backtest completed for {symbol}: {len(trades)} trades, "
                       f"{results.total_return_pct:.2f}% return, {results.win_rate_pct:.1f}% win rate")
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
            raise
    
    def _filter_data_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data by configured date range"""
        if self.config.start_date:
            data = data[data.index >= self.config.start_date]
        if self.config.end_date:
            data = data[data.index <= self.config.end_date]
        return data
    
    def _generate_signals(self, 
                         data: pd.DataFrame, 
                         symbol: str, 
                         mode: BacktestMode,
                         strategy_func: Optional[Callable],
                         ml_model_type: str) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on backtesting mode
        """
        signals = []
        
        try:
            # Strategy signal
            strategy_signal = None
            if mode in [BacktestMode.STRATEGY_ONLY, BacktestMode.COMBINED, BacktestMode.COMPARISON]:
                if strategy_func:
                    strategy_signal = strategy_func(data, symbol)
                else:
                    strategy_signal = strategy_engine.generate_signal(data, symbol)
            
            # ML signal
            ml_signal = None
            if mode in [BacktestMode.ML_ONLY, BacktestMode.COMBINED, BacktestMode.COMPARISON]:
                ml_prediction = self.ml_manager.get_ml_prediction(symbol, data, ml_model_type)
                if not ml_prediction.get('error'):
                    ml_signal = ml_prediction
            
            # Combine signals based on mode
            if mode == BacktestMode.STRATEGY_ONLY and strategy_signal:
                if strategy_signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                    signals.append({
                        'direction': 'long',
                        'confidence': strategy_signal.confidence,
                        'strategy_signal': strategy_signal.signal.value,
                        'ml_signal': None,
                        'entry_price': strategy_signal.entry_price,
                        'stop_loss': strategy_signal.stop_loss,
                        'take_profit': strategy_signal.take_profit
                    })
                elif strategy_signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]:
                    signals.append({
                        'direction': 'short',
                        'confidence': strategy_signal.confidence,
                        'strategy_signal': strategy_signal.signal.value,
                        'ml_signal': None,
                        'entry_price': strategy_signal.entry_price,
                        'stop_loss': strategy_signal.stop_loss,
                        'take_profit': strategy_signal.take_profit
                    })
            
            elif mode == BacktestMode.ML_ONLY and ml_signal:
                if ml_signal['signal'] in [SignalType.BUY, SignalType.STRONG_BUY]:
                    signals.append({
                        'direction': 'long',
                        'confidence': ml_signal['confidence'],
                        'strategy_signal': None,
                        'ml_signal': ml_signal['signal'].value,
                        'entry_price': data['close'].iloc[-1],
                        'stop_loss': data['close'].iloc[-1] * (1 - self.config.stop_loss_pct),
                        'take_profit': data['close'].iloc[-1] * (1 + self.config.take_profit_pct)
                    })
                elif ml_signal['signal'] in [SignalType.SELL, SignalType.STRONG_SELL]:
                    signals.append({
                        'direction': 'short',
                        'confidence': ml_signal['confidence'],
                        'strategy_signal': None,
                        'ml_signal': ml_signal['signal'].value,
                        'entry_price': data['close'].iloc[-1],
                        'stop_loss': data['close'].iloc[-1] * (1 + self.config.stop_loss_pct),
                        'take_profit': data['close'].iloc[-1] * (1 - self.config.take_profit_pct)
                    })
            
            elif mode == BacktestMode.COMBINED and strategy_signal and ml_signal:
                # Require both signals to agree
                strategy_bullish = strategy_signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]
                ml_bullish = ml_signal['signal'] in [SignalType.BUY, SignalType.STRONG_BUY]
                strategy_bearish = strategy_signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]
                ml_bearish = ml_signal['signal'] in [SignalType.SELL, SignalType.STRONG_SELL]
                
                if strategy_bullish and ml_bullish:
                    combined_confidence = (strategy_signal.confidence + ml_signal['confidence']) / 2
                    signals.append({
                        'direction': 'long',
                        'confidence': combined_confidence,
                        'strategy_signal': strategy_signal.signal.value,
                        'ml_signal': ml_signal['signal'].value,
                        'entry_price': strategy_signal.entry_price,
                        'stop_loss': strategy_signal.stop_loss,
                        'take_profit': strategy_signal.take_profit
                    })
                elif strategy_bearish and ml_bearish:
                    combined_confidence = (strategy_signal.confidence + ml_signal['confidence']) / 2
                    signals.append({
                        'direction': 'short',
                        'confidence': combined_confidence,
                        'strategy_signal': strategy_signal.signal.value,
                        'ml_signal': ml_signal['signal'].value,
                        'entry_price': strategy_signal.entry_price,
                        'stop_loss': strategy_signal.stop_loss,
                        'take_profit': strategy_signal.take_profit
                    })
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _enter_position(self, 
                       positions: Dict[str, Any], 
                       signal_info: Dict[str, Any], 
                       current_price: float, 
                       current_time: datetime,
                       available_capital: float) -> Optional[Dict[str, Any]]:
        """
        Enter a new position based on signal
        """
        try:
            # Calculate position size
            position_value = available_capital * self.config.position_size_pct
            quantity = position_value / current_price
            
            # Apply slippage
            slippage = current_price * self.config.slippage_rate
            if signal_info['direction'] == 'long':
                entry_price = current_price + slippage
            else:
                entry_price = current_price - slippage
            
            # Calculate commission
            commission = position_value * self.config.commission_rate
            
            # Create position
            position_key = f"{signal_info['direction']}_{current_time.timestamp()}"
            positions[position_key] = {
                'direction': signal_info['direction'],
                'entry_price': entry_price,
                'quantity': quantity,
                'entry_time': current_time,
                'stop_loss': signal_info['stop_loss'],
                'take_profit': signal_info['take_profit'],
                'strategy_signal': signal_info['strategy_signal'],
                'ml_signal': signal_info['ml_signal'],
                'confidence': signal_info['confidence'],
                'commission': commission,
                'slippage': slippage,
                'unrealized_pnl': 0.0
            }
            
            return {
                'position_key': position_key,
                'position_value': position_value + commission
            }
            
        except Exception as e:
            logger.error(f"Error entering position: {e}")
            return None
    
    def _update_positions_pnl(self, positions: Dict[str, Any], current_price: float):
        """
        Update unrealized P&L for all positions
        """
        for pos in positions.values():
            if pos['direction'] == 'long':
                pos['unrealized_pnl'] = (current_price - pos['entry_price']) * pos['quantity']
            else:  # short
                pos['unrealized_pnl'] = (pos['entry_price'] - current_price) * pos['quantity']
    
    def _check_exit_conditions(self, 
                              positions: Dict[str, Any], 
                              current_price: float, 
                              current_time: datetime) -> List[Dict[str, Any]]:
        """
        Check exit conditions for all positions
        """
        exits = []
        
        for pos_key, pos in positions.items():
            exit_reason = None
            
            # Check stop loss
            if pos['direction'] == 'long' and current_price <= pos['stop_loss']:
                exit_reason = 'stop_loss'
            elif pos['direction'] == 'short' and current_price >= pos['stop_loss']:
                exit_reason = 'stop_loss'
            
            # Check take profit
            elif pos['direction'] == 'long' and current_price >= pos['take_profit']:
                exit_reason = 'take_profit'
            elif pos['direction'] == 'short' and current_price <= pos['take_profit']:
                exit_reason = 'take_profit'
            
            if exit_reason:
                exits.append({
                    'position_key': pos_key,
                    'price': current_price,
                    'time': current_time,
                    'reason': exit_reason
                })
        
        return exits
    
    def _close_position(self, 
                       positions: Dict[str, Any], 
                       exit_info: Dict[str, Any], 
                       current_capital: float) -> Optional[Trade]:
        """
        Close a position and create trade record
        """
        try:
            pos_key = exit_info['position_key']
            if pos_key not in positions:
                return None
            
            pos = positions[pos_key]
            exit_price = exit_info['price']
            exit_time = exit_info['time']
            
            # Apply slippage
            slippage = exit_price * self.config.slippage_rate
            if pos['direction'] == 'long':
                actual_exit_price = exit_price - slippage
            else:
                actual_exit_price = exit_price + slippage
            
            # Calculate P&L
            if pos['direction'] == 'long':
                pnl = (actual_exit_price - pos['entry_price']) * pos['quantity']
            else:  # short
                pnl = (pos['entry_price'] - actual_exit_price) * pos['quantity']
            
            # Subtract commissions
            exit_commission = abs(actual_exit_price * pos['quantity']) * self.config.commission_rate
            total_commission = pos['commission'] + exit_commission
            pnl -= total_commission
            
            # Calculate percentage return
            position_value = pos['entry_price'] * pos['quantity']
            pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0
            
            # Calculate duration
            duration = (exit_time - pos['entry_time']).total_seconds() / 3600  # hours
            
            # Create trade record
            trade = Trade(
                entry_time=pos['entry_time'],
                exit_time=exit_time,
                symbol=exit_info.get('symbol', 'UNKNOWN'),
                direction=pos['direction'],
                entry_price=pos['entry_price'],
                exit_price=actual_exit_price,
                quantity=pos['quantity'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=total_commission,
                slippage=pos['slippage'] + slippage,
                duration_hours=duration,
                exit_reason=exit_info['reason'],
                strategy_signal=pos['strategy_signal'],
                ml_signal=pos['ml_signal'],
                confidence=pos['confidence']
            )
            
            # Remove position
            del positions[pos_key]
            
            return trade
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def _calculate_results(self, 
                          trades: List[Trade], 
                          equity_curve: pd.Series, 
                          initial_capital: float,
                          data: pd.DataFrame,
                          symbol: str) -> BacktestResults:
        """
        Calculate comprehensive backtest results
        """
        try:
            if not trades:
                # Return empty results if no trades
                return BacktestResults(
                    initial_capital=initial_capital,
                    final_capital=initial_capital,
                    total_return_pct=0.0,
                    total_pnl=0.0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate_pct=0.0,
                    avg_win_pct=0.0,
                    avg_loss_pct=0.0,
                    largest_win_pct=0.0,
                    largest_loss_pct=0.0,
                    profit_factor=0.0,
                    max_drawdown_pct=0.0,
                    max_drawdown_duration_days=0,
                    sharpe_ratio=0.0,
                    sortino_ratio=0.0,
                    calmar_ratio=0.0,
                    var_95_pct=0.0,
                    expected_shortfall_pct=0.0,
                    recovery_factor=0.0,
                    total_duration_days=0,
                    avg_trade_duration_hours=0.0,
                    trades_per_month=0.0,
                    trades=[],
                    equity_curve=equity_curve,
                    drawdown_curve=pd.Series(),
                    monthly_returns=pd.Series()
                )
            
            # Basic metrics
            final_capital = equity_curve.iloc[-1]
            total_pnl = sum(trade.pnl for trade in trades)
            total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
            
            # Trade statistics
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            total_trades = len(trades)
            num_winning = len(winning_trades)
            num_losing = len(losing_trades)
            win_rate_pct = (num_winning / total_trades) * 100 if total_trades > 0 else 0
            
            # Performance metrics
            avg_win_pct = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
            avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
            largest_win_pct = max([t.pnl_pct for t in winning_trades]) if winning_trades else 0
            largest_loss_pct = min([t.pnl_pct for t in losing_trades]) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Drawdown analysis
            peak = equity_curve.expanding().max()
            drawdown = (peak - equity_curve) / peak
            max_drawdown_pct = drawdown.max() * 100
            
            # Find maximum drawdown duration
            in_drawdown = drawdown > 0
            drawdown_periods = []
            start = None
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start is None:
                    start = i
                elif not is_dd and start is not None:
                    drawdown_periods.append(i - start)
                    start = None
            if start is not None:
                drawdown_periods.append(len(in_drawdown) - start)
            
            max_drawdown_duration_days = max(drawdown_periods) if drawdown_periods else 0
            
            # Risk-adjusted returns
            returns = equity_curve.pct_change().dropna()
            
            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino_ratio = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Calmar ratio
            calmar_ratio = (total_return_pct / 100) / (max_drawdown_pct / 100) if max_drawdown_pct > 0 else 0
            
            # Value at Risk (95%)
            var_95_pct = np.percentile(returns, 5) * 100 if len(returns) > 0 else 0
            
            # Expected Shortfall (Conditional VaR)
            var_threshold = np.percentile(returns, 5) if len(returns) > 0 else 0
            tail_returns = returns[returns <= var_threshold]
            expected_shortfall_pct = tail_returns.mean() * 100 if len(tail_returns) > 0 else 0
            
            # Recovery factor
            recovery_factor = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
            
            # Time-based metrics
            if trades:
                start_time = min(t.entry_time for t in trades)
                end_time = max(t.exit_time for t in trades)
                total_duration_days = (end_time - start_time).days
                avg_trade_duration_hours = np.mean([t.duration_hours for t in trades])
                trades_per_month = total_trades / (total_duration_days / 30.44) if total_duration_days > 0 else 0
            else:
                total_duration_days = 0
                avg_trade_duration_hours = 0
                trades_per_month = 0
            
            # Monthly returns
            monthly_equity = equity_curve.resample('M').last()
            monthly_returns = monthly_equity.pct_change().dropna() * 100
            
            return BacktestResults(
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return_pct=total_return_pct,
                total_pnl=total_pnl,
                total_trades=total_trades,
                winning_trades=num_winning,
                losing_trades=num_losing,
                win_rate_pct=win_rate_pct,
                avg_win_pct=avg_win_pct,
                avg_loss_pct=avg_loss_pct,
                largest_win_pct=largest_win_pct,
                largest_loss_pct=largest_loss_pct,
                profit_factor=profit_factor,
                max_drawdown_pct=max_drawdown_pct,
                max_drawdown_duration_days=max_drawdown_duration_days,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                var_95_pct=var_95_pct,
                expected_shortfall_pct=expected_shortfall_pct,
                recovery_factor=recovery_factor,
                total_duration_days=total_duration_days,
                avg_trade_duration_hours=avg_trade_duration_hours,
                trades_per_month=trades_per_month,
                trades=trades,
                equity_curve=equity_curve,
                drawdown_curve=drawdown * 100,
                monthly_returns=monthly_returns
            )
            
        except Exception as e:
            logger.error(f"Error calculating backtest results: {e}")
            raise
    
    def _save_results(self, results: BacktestResults, symbol: str, mode: BacktestMode):
        """
        Save backtest results to files
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{symbol}_{mode.value}_{timestamp}"
            
            # Save summary as JSON
            summary = {
                'symbol': symbol,
                'mode': mode.value,
                'timestamp': timestamp,
                'initial_capital': results.initial_capital,
                'final_capital': results.final_capital,
                'total_return_pct': results.total_return_pct,
                'total_trades': results.total_trades,
                'win_rate_pct': results.win_rate_pct,
                'max_drawdown_pct': results.max_drawdown_pct,
                'sharpe_ratio': results.sharpe_ratio,
                'profit_factor': results.profit_factor
            }
            
            with open(self.results_dir / f"{filename_base}_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Save detailed trades as CSV
            if results.trades:
                trades_df = pd.DataFrame([asdict(trade) for trade in results.trades])
                trades_df.to_csv(self.results_dir / f"{filename_base}_trades.csv", index=False)
            
            # Save equity curve
            results.equity_curve.to_csv(self.results_dir / f"{filename_base}_equity.csv")
            
            logger.info(f"Backtest results saved: {filename_base}")
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    def compare_strategies(self, 
                          data: pd.DataFrame, 
                          symbol: str,
                          strategies: List[str] = None) -> Dict[str, BacktestResults]:
