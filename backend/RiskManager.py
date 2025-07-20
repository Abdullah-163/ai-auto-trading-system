"""
RiskManager Module for AI-powered Auto-Trading Application
Implements comprehensive risk management including position sizing, stop loss, take profit,
portfolio risk controls, and drawdown protection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from loguru import logger
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
from config import config

class RiskLevel(Enum):
    """Risk level enumeration"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

@dataclass
class PositionRisk:
    """
    Data class for position risk parameters
    """
    symbol: str
    position_size: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    max_loss_pct: float
    position_value: float

@dataclass
class PortfolioRisk:
    """
    Data class for portfolio-level risk metrics
    """
    total_capital: float
    available_capital: float
    used_capital: float
    total_risk: float
    max_drawdown: float
    current_drawdown: float
    daily_pnl: float
    open_positions: int
    risk_level: RiskLevel

class RiskManager:
    """
    Comprehensive risk management system for automated trading
    Handles position sizing, stop loss/take profit, portfolio risk, and drawdown protection
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize Risk Manager
        
        Args:
            initial_capital: Starting capital amount
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.available_capital = initial_capital
        
        # Risk parameters from config
        self.max_risk_per_trade = config.MAX_RISK_PER_TRADE
        self.max_position_size_pct = config.MAX_POSITION_SIZE_PCT
        self.default_stop_loss_pct = config.DEFAULT_STOP_LOSS_PCT
        self.default_take_profit_pct = config.DEFAULT_TAKE_PROFIT_PCT
        self.max_daily_trades = config.MAX_DAILY_TRADES
        
        # Additional risk parameters
        self.max_portfolio_risk = 0.15  # 15% max total portfolio risk
        self.max_drawdown_limit = 0.20  # 20% max drawdown before stopping
        self.max_correlation_exposure = 0.30  # 30% max exposure to correlated assets
        self.min_reward_risk_ratio = 1.5  # Minimum 1.5:1 reward to risk ratio
        
        # Tracking variables
        self.open_positions = {}  # symbol -> position info
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.trade_history = []
        self.risk_events = []
        
        # Volatility tracking for dynamic risk adjustment
        self.volatility_window = 20
        self.volatility_multiplier = 1.0
        
        logger.info(f"RiskManager initialized with capital: ${initial_capital:,.2f}")
    
    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss: float, 
                              risk_amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss: Stop loss price
            risk_amount: Custom risk amount (optional)
            
        Returns:
            Dictionary with position sizing information
        """
        try:
            # Use custom risk amount or default percentage
            if risk_amount is None:
                risk_amount = self.available_capital * self.max_risk_per_trade
            
            # Calculate risk per share/unit
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0:
                logger.error(f"Invalid stop loss for {symbol}: risk per unit is zero")
                return {'error': 'Invalid stop loss - no risk per unit'}
            
            # Calculate base position size
            base_position_size = risk_amount / risk_per_unit
            
            # Calculate position value
            position_value = base_position_size * entry_price
            
            # Apply maximum position size limit
            max_position_value = self.available_capital * self.max_position_size_pct
            if position_value > max_position_value:
                position_size = max_position_value / entry_price
                actual_risk = position_size * risk_per_unit
                logger.warning(f"Position size limited by max position size for {symbol}")
            else:
                position_size = base_position_size
                actual_risk = risk_amount
            
            # Check if we have enough capital
            if position_value > self.available_capital:
                logger.error(f"Insufficient capital for {symbol}: need ${position_value:,.2f}, have ${self.available_capital:,.2f}")
                return {'error': 'Insufficient capital'}
            
            # Calculate final metrics
            final_position_value = position_size * entry_price
            risk_percentage = (actual_risk / self.current_capital) * 100
            
            result = {
                'symbol': symbol,
                'position_size': position_size,
                'position_value': final_position_value,
                'risk_amount': actual_risk,
                'risk_percentage': risk_percentage,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'risk_per_unit': risk_per_unit,
                'max_loss_pct': (actual_risk / self.current_capital) * 100
            }
            
            logger.info(f"Position size calculated for {symbol}: {position_size:.4f} units, "
                       f"${final_position_value:,.2f} value, {risk_percentage:.2f}% risk")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return {'error': str(e)}
    
    def calculate_stop_loss(self, 
                           symbol: str, 
                           entry_price: float, 
                           direction: str, 
                           method: str = 'percentage',
                           atr_value: Optional[float] = None,
                           support_resistance: Optional[float] = None) -> float:
        """
        Calculate stop loss level using various methods
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            direction: 'long' or 'short'
            method: 'percentage', 'atr', 'support_resistance', 'volatility'
            atr_value: Average True Range value (for ATR method)
            support_resistance: Support/resistance level
            
        Returns:
            Stop loss price
        """
        try:
            if method == 'percentage':
                if direction.lower() == 'long':
                    stop_loss = entry_price * (1 - self.default_stop_loss_pct)
                else:  # short
                    stop_loss = entry_price * (1 + self.default_stop_loss_pct)
            
            elif method == 'atr' and atr_value is not None:
                # Use 2x ATR as stop loss distance
                atr_multiplier = 2.0
                if direction.lower() == 'long':
                    stop_loss = entry_price - (atr_value * atr_multiplier)
                else:  # short
                    stop_loss = entry_price + (atr_value * atr_multiplier)
            
            elif method == 'support_resistance' and support_resistance is not None:
                if direction.lower() == 'long':
                    # Place stop loss slightly below support
                    stop_loss = support_resistance * 0.995
                else:  # short
                    # Place stop loss slightly above resistance
                    stop_loss = support_resistance * 1.005
            
            elif method == 'volatility':
                # Adjust stop loss based on recent volatility
                volatility_adjustment = self.volatility_multiplier
                base_stop_pct = self.default_stop_loss_pct * volatility_adjustment
                
                if direction.lower() == 'long':
                    stop_loss = entry_price * (1 - base_stop_pct)
                else:  # short
                    stop_loss = entry_price * (1 + base_stop_pct)
            
            else:
                # Default to percentage method
                if direction.lower() == 'long':
                    stop_loss = entry_price * (1 - self.default_stop_loss_pct)
                else:  # short
                    stop_loss = entry_price * (1 + self.default_stop_loss_pct)
            
            logger.debug(f"Stop loss calculated for {symbol} ({direction}): {stop_loss:.4f} using {method} method")
            return stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating stop loss for {symbol}: {e}")
            # Return default stop loss
            if direction.lower() == 'long':
                return entry_price * (1 - self.default_stop_loss_pct)
            else:
                return entry_price * (1 + self.default_stop_loss_pct)
    
    def calculate_take_profit(self, 
                             symbol: str, 
                             entry_price: float, 
                             stop_loss: float, 
                             direction: str,
                             reward_risk_ratio: Optional[float] = None) -> float:
        """
        Calculate take profit level based on risk-reward ratio
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: 'long' or 'short'
            reward_risk_ratio: Custom reward to risk ratio
            
        Returns:
            Take profit price
        """
        try:
            # Use custom ratio or minimum required ratio
            if reward_risk_ratio is None:
                reward_risk_ratio = self.min_reward_risk_ratio
            
            # Calculate risk amount
            risk_amount = abs(entry_price - stop_loss)
            
            # Calculate reward amount
            reward_amount = risk_amount * reward_risk_ratio
            
            # Calculate take profit level
            if direction.lower() == 'long':
                take_profit = entry_price + reward_amount
            else:  # short
                take_profit = entry_price - reward_amount
            
            logger.debug(f"Take profit calculated for {symbol} ({direction}): {take_profit:.4f} "
                        f"with {reward_risk_ratio}:1 ratio")
            
            return take_profit
            
        except Exception as e:
            logger.error(f"Error calculating take profit for {symbol}: {e}")
            # Return default take profit
            if direction.lower() == 'long':
                return entry_price * (1 + self.default_take_profit_pct)
            else:
                return entry_price * (1 - self.default_take_profit_pct)
    
    def validate_trade(self, 
                      symbol: str, 
                      position_size: float, 
                      entry_price: float, 
                      stop_loss: float, 
                      take_profit: float,
                      direction: str) -> Dict[str, Any]:
        """
        Validate trade against all risk management rules
        
        Args:
            symbol: Trading symbol
            position_size: Proposed position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            direction: 'long' or 'short'
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                'approved': True,
                'warnings': [],
                'errors': [],
                'risk_metrics': {}
            }
            
            # Check daily trade limit
            if self._is_new_trading_day():
                self.daily_trades = 0
                self.daily_pnl = 0.0
            
            if self.daily_trades >= self.max_daily_trades:
                validation_results['approved'] = False
                validation_results['errors'].append(f"Daily trade limit reached: {self.daily_trades}/{self.max_daily_trades}")
            
            # Calculate position metrics
            position_value = position_size * entry_price
            risk_amount = position_size * abs(entry_price - stop_loss)
            reward_amount = position_size * abs(take_profit - entry_price)
            
            # Check available capital
            if position_value > self.available_capital:
                validation_results['approved'] = False
                validation_results['errors'].append(f"Insufficient capital: need ${position_value:,.2f}, have ${self.available_capital:,.2f}")
            
            # Check position size limits
            position_size_pct = (position_value / self.current_capital) * 100
            if position_size_pct > self.max_position_size_pct * 100:
                validation_results['approved'] = False
                validation_results['errors'].append(f"Position size too large: {position_size_pct:.1f}% > {self.max_position_size_pct*100:.1f}%")
            
            # Check risk per trade
            risk_pct = (risk_amount / self.current_capital) * 100
            if risk_pct > self.max_risk_per_trade * 100:
                validation_results['approved'] = False
                validation_results['errors'].append(f"Risk per trade too high: {risk_pct:.1f}% > {self.max_risk_per_trade*100:.1f}%")
            
            # Check reward to risk ratio
            reward_risk_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
            if reward_risk_ratio < self.min_reward_risk_ratio:
                validation_results['warnings'].append(f"Low reward:risk ratio: {reward_risk_ratio:.2f} < {self.min_reward_risk_ratio}")
            
            # Check portfolio risk
            total_portfolio_risk = self._calculate_total_portfolio_risk() + risk_amount
            portfolio_risk_pct = (total_portfolio_risk / self.current_capital) * 100
            if portfolio_risk_pct > self.max_portfolio_risk * 100:
                validation_results['approved'] = False
                validation_results['errors'].append(f"Portfolio risk too high: {portfolio_risk_pct:.1f}% > {self.max_portfolio_risk*100:.1f}%")
            
            # Check drawdown limits
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > self.max_drawdown_limit:
                validation_results['approved'] = False
                validation_results['errors'].append(f"Maximum drawdown exceeded: {current_drawdown*100:.1f}% > {self.max_drawdown_limit*100:.1f}%")
            
            # Check correlation limits (simplified check)
            correlation_exposure = self._calculate_correlation_exposure(symbol)
            if correlation_exposure > self.max_correlation_exposure:
                validation_results['warnings'].append(f"High correlation exposure: {correlation_exposure*100:.1f}%")
            
            # Store risk metrics
            validation_results['risk_metrics'] = {
                'position_value': position_value,
                'position_size_pct': position_size_pct,
                'risk_amount': risk_amount,
                'risk_pct': risk_pct,
                'reward_amount': reward_amount,
                'reward_risk_ratio': reward_risk_ratio,
                'portfolio_risk_pct': portfolio_risk_pct,
                'current_drawdown': current_drawdown,
                'correlation_exposure': correlation_exposure
            }
            
            # Log validation results
            if validation_results['approved']:
                logger.info(f"Trade validation passed for {symbol}: {position_size:.4f} units, "
                           f"{risk_pct:.2f}% risk, {reward_risk_ratio:.2f}:1 R:R")
            else:
                logger.warning(f"Trade validation failed for {symbol}: {validation_results['errors']}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating trade for {symbol}: {e}")
            return {
                'approved': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'risk_metrics': {}
            }
    
    def open_position(self, 
                     symbol: str, 
                     position_size: float, 
                     entry_price: float, 
                     stop_loss: float, 
                     take_profit: float,
                     direction: str,
                     strategy: str = "unknown") -> bool:
        """
        Register a new open position
        
        Args:
            symbol: Trading symbol
            position_size: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            direction: 'long' or 'short'
            strategy: Strategy name
            
        Returns:
            True if position opened successfully
        """
        try:
            # Validate trade first
            validation = self.validate_trade(symbol, position_size, entry_price, stop_loss, take_profit, direction)
            
            if not validation['approved']:
                logger.error(f"Cannot open position for {symbol}: {validation['errors']}")
                return False
            
            # Calculate position metrics
            position_value = position_size * entry_price
            risk_amount = position_size * abs(entry_price - stop_loss)
            
            # Create position record
            position = {
                'symbol': symbol,
                'position_size': position_size,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'direction': direction,
                'strategy': strategy,
                'position_value': position_value,
                'risk_amount': risk_amount,
                'entry_time': datetime.now(),
                'unrealized_pnl': 0.0
            }
            
            # Update tracking variables
            self.open_positions[symbol] = position
            self.available_capital -= position_value
            self.daily_trades += 1
            self.last_trade_date = datetime.now().date()
            
            # Log position opening
            logger.info(f"Position opened for {symbol}: {direction} {position_size:.4f} units at ${entry_price:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "manual") -> Dict[str, Any]:
        """
        Close an open position and calculate P&L
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            Dictionary with position closing results
        """
        try:
            if symbol not in self.open_positions:
                logger.error(f"No open position found for {symbol}")
                return {'error': f'No open position for {symbol}'}
            
            position = self.open_positions[symbol]
            
            # Calculate P&L
            if position['direction'].lower() == 'long':
                pnl = (exit_price - position['entry_price']) * position['position_size']
            else:  # short
                pnl = (position['entry_price'] - exit_price) * position['position_size']
            
            # Calculate returns
            return_pct = (pnl / position['position_value']) * 100
            
            # Update capital
            self.current_capital += pnl
            self.available_capital += position['position_value'] + pnl
            self.daily_pnl += pnl
            
            # Update peak capital for drawdown calculation
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            
            # Create trade record
            trade_record = {
                'symbol': symbol,
                'direction': position['direction'],
                'strategy': position['strategy'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'position_size': position['position_size'],
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(),
                'pnl': pnl,
                'return_pct': return_pct,
                'reason': reason,
                'risk_amount': position['risk_amount']
            }
            
            # Add to trade history
            self.trade_history.append(trade_record)
            
            # Remove from open positions
            del self.open_positions[symbol]
            
            # Log position closing
            logger.info(f"Position closed for {symbol}: {reason}, P&L: ${pnl:.2f} ({return_pct:.2f}%)")
            
            return {
                'success': True,
                'pnl': pnl,
                'return_pct': return_pct,
                'trade_record': trade_record
            }
            
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return {'error': str(e)}
    
    def update_position_pnl(self, symbol: str, current_price: float):
        """
        Update unrealized P&L for open position
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        try:
            if symbol in self.open_positions:
                position = self.open_positions[symbol]
                
                if position['direction'].lower() == 'long':
                    unrealized_pnl = (current_price - position['entry_price']) * position['position_size']
                else:  # short
                    unrealized_pnl = (position['entry_price'] - current_price) * position['position_size']
                
                position['unrealized_pnl'] = unrealized_pnl
                
        except Exception as e:
            logger.error(f"Error updating P&L for {symbol}: {e}")
    
    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if stop loss or take profit should be triggered
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            'stop_loss', 'take_profit', or None
        """
        try:
            if symbol not in self.open_positions:
                return None
            
            position = self.open_positions[symbol]
            
            if position['direction'].lower() == 'long':
                if current_price <= position['stop_loss']:
                    return 'stop_loss'
                elif current_price >= position['take_profit']:
                    return 'take_profit'
            else:  # short
                if current_price >= position['stop_loss']:
                    return 'stop_loss'
                elif current_price <= position['take_profit']:
                    return 'take_profit'
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking stop loss/take profit for {symbol}: {e}")
            return None
    
    def get_portfolio_risk_metrics(self) -> PortfolioRisk:
        """
        Calculate comprehensive portfolio risk metrics
        
        Returns:
            PortfolioRisk object with current metrics
        """
        try:
            # Calculate total risk from open positions
            total_risk = self._calculate_total_portfolio_risk()
            
            # Calculate current drawdown
            current_drawdown = self._calculate_current_drawdown()
            
            # Determine risk level
            risk_level = self._assess_risk_level(current_drawdown, total_risk)
            
            # Calculate used capital
            used_capital = sum(pos['position_value'] for pos in self.open_positions.values())
            
            portfolio_risk = PortfolioRisk(
                total_capital=self.current_capital,
                available_capital=self.available_capital,
                used_capital=used_capital,
                total_risk=total_risk,
                max_drawdown=self.max_drawdown_limit,
                current_drawdown=current_drawdown,
                daily_pnl=self.daily_pnl,
                open_positions=len(self.open_positions),
                risk_level=risk_level
            )
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            return PortfolioRisk(
                total_capital=self.current_capital,
                available_capital=self.available_capital,
                used_capital=0,
                total_risk=0,
                max_drawdown=self.max_drawdown_limit,
                current_drawdown=0,
                daily_pnl=0,
                open_positions=0,
                risk_level=RiskLevel.MEDIUM
            )
    
    def _calculate_total_portfolio_risk(self) -> float:
        """Calculate total risk from all open positions"""
        return sum(pos['risk_amount'] for pos in self.open_positions.values())
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_capital == 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital
    
    def _calculate_correlation_exposure(self, symbol: str) -> float:
        """Calculate exposure to correlated assets (simplified)"""
        # This is a simplified implementation
        # In practice, you would use correlation matrices
        similar_symbols = [s for s in self.open_positions.keys() if s[:3] == symbol[:3]]
        total_exposure = sum(self.open_positions[s]['position_value'] for s in similar_symbols)
        return total_exposure / self.current_capital if self.current_capital > 0 else 0
    
    def _assess_risk_level(self, drawdown: float, total_risk: float) -> RiskLevel:
        """Assess overall portfolio risk level"""
        risk_score = 0
        
        # Drawdown component
        if drawdown > 0.15:
            risk_score += 3
        elif drawdown > 0.10:
            risk_score += 2
        elif drawdown > 0.05:
            risk_score += 1
        
        # Total risk component
        risk_pct = total_risk / self.current_capital if self.current_capital > 0 else 0
        if risk_pct > 0.12:
            risk_score += 3
        elif risk_pct > 0.08:
            risk_score += 2
        elif risk_pct > 0.04:
            risk_score += 1
        
        # Map score to risk level
        if risk_score >= 5:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        elif risk_score >= 1:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
    
    def _is_new_trading_day(self) -> bool:
        """Check if it's a new trading day"""
        current_date = datetime.now().date()
        return self.last_trade_date != current_date
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk management summary
        
        Returns:
            Dictionary with risk summary
        """
        try:
            portfolio_risk = self.get_portfolio_risk_metrics()
            
            # Calculate performance metrics
            total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            
            # Calculate win rate from trade history
            if self.trade_history:
                winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
                win_rate = (winning_trades / len(self.trade_history)) * 100
                avg_win = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] > 0]) if winning_trades > 0 else 0
                avg_loss = np.mean([t['pnl'] for t in self.trade_history if t['pnl'] < 0]) if len(self.trade_history) - winning_trades > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            summary = {
                'capital': {
                    'initial': self.initial_capital,
                    'current': self.current_capital,
                    'available': self.available_capital,
                    'peak': self.peak_capital,
                    'total_return_pct': total_return
                },
                'risk_metrics': {
                    'current_drawdown_pct': portfolio_risk.current_drawdown * 100,
                    'max_drawdown_limit_pct': self.max_drawdown_limit * 100,
                    'total_risk': portfolio_risk.total_risk,
                    'risk_level': portfolio_risk.risk_level.value,
                    'daily_pnl': self.daily_pnl,
                    'daily_trades': self.daily_trades,
                    'max_daily_trades': self.max_daily_trades
                },
                'positions': {
                    'open_positions': len(self.open_positions),
                    'position_details': [
                        {
                            'symbol': pos['symbol'],
                            'direction': pos['direction'],
                            'size': pos['position_size'],
                            'value': pos['position_value'],
                            'unrealized_pnl': pos.get('unrealized_pnl', 0)
                        }
                        for pos in self.open_positions.values()
                    ]
                },
                'performance': {
                    'total_trades': len(self.trade_history),
                    'win_rate_pct': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor
                },
                'limits': {
                    'max_risk_per_trade_pct': self.max_risk_per_trade * 100,
                    'max_position_size_pct': self.max_position_size_pct * 100,
                    'max_portfolio_risk_pct': self.max_portfolio_risk * 100,
                    'min_reward_risk_ratio': self.min_reward_risk_ratio
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return {'error': str(e)}

# Global RiskManager instance
risk_manager = RiskManager()

# Example usage and testing
def test_risk_manager():
    """
    Test function to verify RiskManager functionality
    """
    logger.info("Testing RiskManager...")
    
    # Test position sizing
    position_info = risk_manager.calculate_position_size('BTCUSDT', 50000, 48000)
    if 'error' not in position_info:
        logger.info(f"
