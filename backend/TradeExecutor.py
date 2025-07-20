"""
TradeExecutor Module for AI-powered Auto-Trading Application
Handles trade execution across multiple exchanges: Binance, Bybit, and MetaTrader5
Implements order management, execution monitoring, and error handling
"""

import asyncio
import aiohttp
import requests
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from loguru import logger
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
import hmac
import hashlib
import base64
from urllib.parse import urlencode

# Exchange-specific imports
try:
    from binance.client import Client as BinanceClient
    from binance.exceptions import BinanceAPIException, BinanceOrderException
    BINANCE_AVAILABLE = True
except ImportError:
    logger.warning("Binance library not available")
    BINANCE_AVAILABLE = False

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.warning("MetaTrader5 library not available")
    MT5_AVAILABLE = False

from config import config
from RiskManager import risk_manager

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    """Order status enumeration"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"

class Exchange(Enum):
    """Supported exchanges"""
    BINANCE = "BINANCE"
    BYBIT = "BYBIT"
    METATRADER5 = "METATRADER5"

@dataclass
class OrderRequest:
    """
    Data class for order requests
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Canceled
    client_order_id: Optional[str] = None

@dataclass
class OrderResponse:
    """
    Data class for order responses
    """
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    filled_quantity: float
    average_price: Optional[float]
    commission: float
    timestamp: datetime
    exchange: Exchange
    error_message: Optional[str] = None

class TradeExecutor:
    """
    Comprehensive trade execution system supporting multiple exchanges
    Handles order placement, monitoring, and management with robust error handling
    """
    
    def __init__(self):
        """
        Initialize Trade Executor with exchange connections
        """
        self.binance_client = None
        self.bybit_session = None
        self.mt5_initialized = False
        
        # Order tracking
        self.active_orders = {}  # order_id -> OrderResponse
        self.order_history = []
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': 0.0,
            'total_commission': 0.0
        }
        
        # Rate limiting
        self.rate_limits = {
            Exchange.BINANCE: {'requests': 0, 'reset_time': time.time() + 60},
            Exchange.BYBIT: {'requests': 0, 'reset_time': time.time() + 60}
        }
        
        # Initialize exchange connections
        self._initialize_exchanges()
        
        logger.info("TradeExecutor initialized")
    
    def _initialize_exchanges(self):
        """
        Initialize connections to all supported exchanges
        """
        try:
            # Initialize Binance
            if BINANCE_AVAILABLE and config.BINANCE_API_KEY and config.BINANCE_SECRET_KEY:
                self.binance_client = BinanceClient(
                    api_key=config.BINANCE_API_KEY,
                    api_secret=config.BINANCE_SECRET_KEY,
                    testnet=False  # Set to True for testing
                )
                # Test connection
                account_info = self.binance_client.get_account()
                logger.info("Binance connection established successfully")
            else:
                logger.warning("Binance not configured or library not available")
            
            # Initialize MetaTrader5
            if MT5_AVAILABLE and all([config.METATRADER_LOGIN, config.METATRADER_PASSWORD, config.METATRADER_SERVER]):
                if mt5.initialize():
                    if mt5.login(
                        login=int(config.METATRADER_LOGIN),
                        password=config.METATRADER_PASSWORD,
                        server=config.METATRADER_SERVER
                    ):
                        self.mt5_initialized = True
                        logger.info("MetaTrader5 connection established successfully")
                    else:
                        logger.error("MetaTrader5 login failed")
                else:
                    logger.error("MetaTrader5 initialization failed")
            else:
                logger.warning("MetaTrader5 not configured or library not available")
                
        except Exception as e:
            logger.error(f"Error initializing exchanges: {e}")
    
    async def execute_trade(self, 
                           exchange: Exchange, 
                           order_request: OrderRequest,
                           validate_risk: bool = True) -> OrderResponse:
        """
        Execute trade on specified exchange
        
        Args:
            exchange: Target exchange
            order_request: Order details
            validate_risk: Whether to validate against risk management rules
            
        Returns:
            OrderResponse with execution results
        """
        try:
            logger.info(f"Executing {order_request.side.value} order for {order_request.symbol} "
                       f"on {exchange.value}: {order_request.quantity} units")
            
            # Risk validation
            if validate_risk:
                risk_check = self._validate_order_risk(order_request)
                if not risk_check['approved']:
                    return OrderResponse(
                        order_id="",
                        client_order_id=order_request.client_order_id,
                        symbol=order_request.symbol,
                        side=order_request.side,
                        order_type=order_request.order_type,
                        quantity=order_request.quantity,
                        price=order_request.price,
                        status=OrderStatus.REJECTED,
                        filled_quantity=0.0,
                        average_price=None,
                        commission=0.0,
                        timestamp=datetime.now(),
                        exchange=exchange,
                        error_message=f"Risk validation failed: {risk_check['errors']}"
                    )
            
            # Check rate limits
            if not self._check_rate_limit(exchange):
                return OrderResponse(
                    order_id="",
                    client_order_id=order_request.client_order_id,
                    symbol=order_request.symbol,
                    side=order_request.side,
                    order_type=order_request.order_type,
                    quantity=order_request.quantity,
                    price=order_request.price,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0.0,
                    average_price=None,
                    commission=0.0,
                    timestamp=datetime.now(),
                    exchange=exchange,
                    error_message="Rate limit exceeded"
                )
            
            # Execute on specific exchange
            if exchange == Exchange.BINANCE:
                return await self._execute_binance_order(order_request)
            elif exchange == Exchange.BYBIT:
                return await self._execute_bybit_order(order_request)
            elif exchange == Exchange.METATRADER5:
                return self._execute_mt5_order(order_request)
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
                
        except Exception as e:
            logger.error(f"Error executing trade on {exchange.value}: {e}")
            return OrderResponse(
                order_id="",
                client_order_id=order_request.client_order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                average_price=None,
                commission=0.0,
                timestamp=datetime.now(),
                exchange=exchange,
                error_message=str(e)
            )
    
    async def _execute_binance_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Execute order on Binance
        """
        try:
            if not self.binance_client:
                raise Exception("Binance client not initialized")
            
            # Prepare order parameters
            order_params = {
                'symbol': order_request.symbol,
                'side': order_request.side.value,
                'type': order_request.order_type.value,
                'quantity': order_request.quantity,
                'timeInForce': order_request.time_in_force
            }
            
            # Add price for limit orders
            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order_request.price is None:
                    raise ValueError("Price required for limit orders")
                order_params['price'] = order_request.price
            
            # Add stop price for stop orders
            if order_request.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
                if order_request.stop_price is None:
                    raise ValueError("Stop price required for stop orders")
                order_params['stopPrice'] = order_request.stop_price
            
            # Add client order ID if provided
            if order_request.client_order_id:
                order_params['newClientOrderId'] = order_request.client_order_id
            
            # Execute order
            if order_request.order_type == OrderType.MARKET:
                if order_request.side == OrderSide.BUY:
                    result = self.binance_client.order_market_buy(**order_params)
                else:
                    result = self.binance_client.order_market_sell(**order_params)
            else:
                result = self.binance_client.create_order(**order_params)
            
            # Parse response
            order_response = OrderResponse(
                order_id=str(result['orderId']),
                client_order_id=result.get('clientOrderId'),
                symbol=result['symbol'],
                side=OrderSide(result['side']),
                order_type=OrderType(result['type']),
                quantity=float(result['origQty']),
                price=float(result['price']) if result.get('price') else None,
                status=OrderStatus(result['status']),
                filled_quantity=float(result['executedQty']),
                average_price=float(result['price']) if result.get('price') and float(result['executedQty']) > 0 else None,
                commission=0.0,  # Will be calculated separately
                timestamp=datetime.fromtimestamp(result['transactTime'] / 1000),
                exchange=Exchange.BINANCE
            )
            
            # Update statistics
            self._update_execution_stats(order_response)
            
            # Store active order
            self.active_orders[order_response.order_id] = order_response
            
            logger.info(f"Binance order executed: {order_response.order_id}, Status: {order_response.status.value}")
            return order_response
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return self._create_error_response(order_request, Exchange.BINANCE, str(e))
        except Exception as e:
            logger.error(f"Error executing Binance order: {e}")
            return self._create_error_response(order_request, Exchange.BINANCE, str(e))
    
    async def _execute_bybit_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Execute order on Bybit
        """
        try:
            if not config.BYBIT_API_KEY or not config.BYBIT_SECRET_KEY:
                raise Exception("Bybit API credentials not configured")
            
            # Bybit API endpoint
            url = "https://api.bybit.com/v2/private/order/create"
            
            # Prepare order parameters
            params = {
                'api_key': config.BYBIT_API_KEY,
                'symbol': order_request.symbol,
                'side': 'Buy' if order_request.side == OrderSide.BUY else 'Sell',
                'order_type': order_request.order_type.value.title(),
                'qty': order_request.quantity,
                'time_in_force': order_request.time_in_force,
                'timestamp': int(time.time() * 1000)
            }
            
            # Add price for limit orders
            if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                if order_request.price is None:
                    raise ValueError("Price required for limit orders")
                params['price'] = order_request.price
            
            # Add stop price for stop orders
            if order_request.order_type in [OrderType.STOP_LOSS, OrderType.STOP_LIMIT]:
                if order_request.stop_price is None:
                    raise ValueError("Stop price required for stop orders")
                params['stop_px'] = order_request.stop_price
            
            # Generate signature
            signature = self._generate_bybit_signature(params)
            params['sign'] = signature
            
            # Execute request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=params) as response:
                    result = await response.json()
            
            if result['ret_code'] != 0:
                raise Exception(f"Bybit API error: {result['ret_msg']}")
            
            order_data = result['result']
            
            # Parse response
            order_response = OrderResponse(
                order_id=str(order_data['order_id']),
                client_order_id=order_data.get('order_link_id'),
                symbol=order_data['symbol'],
                side=OrderSide.BUY if order_data['side'] == 'Buy' else OrderSide.SELL,
                order_type=OrderType(order_data['order_type'].upper()),
                quantity=float(order_data['qty']),
                price=float(order_data['price']) if order_data.get('price') else None,
                status=self._map_bybit_status(order_data['order_status']),
                filled_quantity=0.0,  # Will be updated when filled
                average_price=None,
                commission=0.0,
                timestamp=datetime.now(),
                exchange=Exchange.BYBIT
            )
            
            # Update statistics
            self._update_execution_stats(order_response)
            
            # Store active order
            self.active_orders[order_response.order_id] = order_response
            
            logger.info(f"Bybit order executed: {order_response.order_id}, Status: {order_response.status.value}")
            return order_response
            
        except Exception as e:
            logger.error(f"Error executing Bybit order: {e}")
            return self._create_error_response(order_request, Exchange.BYBIT, str(e))
    
    def _execute_mt5_order(self, order_request: OrderRequest) -> OrderResponse:
        """
        Execute order on MetaTrader5
        """
        try:
            if not self.mt5_initialized:
                raise Exception("MetaTrader5 not initialized")
            
            # Map order type
            if order_request.order_type == OrderType.MARKET:
                if order_request.side == OrderSide.BUY:
                    order_type = mt5.ORDER_TYPE_BUY
                else:
                    order_type = mt5.ORDER_TYPE_SELL
            elif order_request.order_type == OrderType.LIMIT:
                if order_request.side == OrderSide.BUY:
                    order_type = mt5.ORDER_TYPE_BUY_LIMIT
                else:
                    order_type = mt5.ORDER_TYPE_SELL_LIMIT
            else:
                raise ValueError(f"Unsupported order type for MT5: {order_request.order_type}")
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order_request.symbol,
                "volume": order_request.quantity,
                "type": order_type,
                "deviation": 20,
                "magic": 234000,
                "comment": "AI Auto-Trader",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add price for limit orders
            if order_request.order_type == OrderType.LIMIT and order_request.price:
                request["price"] = order_request.price
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"MT5 order failed: {result.comment}")
            
            # Parse response
            order_response = OrderResponse(
                order_id=str(result.order),
                client_order_id=None,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=result.price,
                status=OrderStatus.FILLED if result.retcode == mt5.TRADE_RETCODE_DONE else OrderStatus.REJECTED,
                filled_quantity=result.volume,
                average_price=result.price,
                commission=0.0,  # MT5 doesn't provide commission in order result
                timestamp=datetime.now(),
                exchange=Exchange.METATRADER5
            )
            
            # Update statistics
            self._update_execution_stats(order_response)
            
            logger.info(f"MT5 order executed: {order_response.order_id}, Status: {order_response.status.value}")
            return order_response
            
        except Exception as e:
            logger.error(f"Error executing MT5 order: {e}")
            return self._create_error_response(order_request, Exchange.METATRADER5, str(e))
    
    def _generate_bybit_signature(self, params: Dict[str, Any]) -> str:
        """
        Generate signature for Bybit API requests
        """
        # Sort parameters
        sorted_params = sorted(params.items())
        query_string = '&'.join([f"{k}={v}" for k, v in sorted_params])
        
        # Generate signature
        signature = hmac.new(
            config.BYBIT_SECRET_KEY.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _map_bybit_status(self, bybit_status: str) -> OrderStatus:
        """
        Map Bybit order status to internal status
        """
        status_map = {
            'Created': OrderStatus.NEW,
            'New': OrderStatus.NEW,
            'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
            'Filled': OrderStatus.FILLED,
            'Cancelled': OrderStatus.CANCELED,
            'Rejected': OrderStatus.REJECTED
        }
        return status_map.get(bybit_status, OrderStatus.NEW)
    
    def _validate_order_risk(self, order_request: OrderRequest) -> Dict[str, Any]:
        """
        Validate order against risk management rules
        """
        try:
            # This is a simplified validation - in practice, you'd integrate with RiskManager
            validation = {
                'approved': True,
                'errors': [],
                'warnings': []
            }
            
            # Check minimum order size
            if order_request.quantity <= 0:
                validation['approved'] = False
                validation['errors'].append("Invalid quantity")
            
            # Check if price is reasonable for limit orders
            if order_request.order_type == OrderType.LIMIT and order_request.price and order_request.price <= 0:
                validation['approved'] = False
                validation['errors'].append("Invalid price")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating order risk: {e}")
            return {
                'approved': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }
    
    def _check_rate_limit(self, exchange: Exchange) -> bool:
        """
        Check if we're within rate limits for the exchange
        """
        try:
            current_time = time.time()
            rate_limit = self.rate_limits.get(exchange)
            
            if not rate_limit:
                return True
            
            # Reset counter if time window has passed
            if current_time > rate_limit['reset_time']:
                rate_limit['requests'] = 0
                rate_limit['reset_time'] = current_time + 60
            
            # Check if we're within limits (simplified - 100 requests per minute)
            if rate_limit['requests'] >= 100:
                logger.warning(f"Rate limit exceeded for {exchange.value}")
                return False
            
            # Increment counter
            rate_limit['requests'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return True  # Allow request if check fails
    
    def _create_error_response(self, order_request: OrderRequest, exchange: Exchange, error_message: str) -> OrderResponse:
        """
        Create error response for failed orders
        """
        return OrderResponse(
            order_id="",
            client_order_id=order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            status=OrderStatus.REJECTED,
            filled_quantity=0.0,
            average_price=None,
            commission=0.0,
            timestamp=datetime.now(),
            exchange=exchange,
            error_message=error_message
        )
    
    def _update_execution_stats(self, order_response: OrderResponse):
        """
        Update execution statistics
        """
        try:
            self.execution_stats['total_orders'] += 1
            
            if order_response.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                self.execution_stats['successful_orders'] += 1
                self.execution_stats['total_volume'] += order_response.filled_quantity * (order_response.average_price or 0)
            else:
                self.execution_stats['failed_orders'] += 1
            
            self.execution_stats['total_commission'] += order_response.commission
            
        except Exception as e:
            logger.error(f"Error updating execution stats: {e}")
    
    async def cancel_order(self, exchange: Exchange, order_id: str, symbol: str) -> bool:
        """
        Cancel an active order
        
        Args:
            exchange: Exchange where order was placed
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            True if cancellation successful
        """
        try:
            logger.info(f"Canceling order {order_id} on {exchange.value}")
            
            if exchange == Exchange.BINANCE and self.binance_client:
                result = self.binance_client.cancel_order(symbol=symbol, orderId=order_id)
                success = result['status'] == 'CANCELED'
                
            elif exchange == Exchange.BYBIT:
                # Bybit cancellation logic
                params = {
                    'api_key': config.BYBIT_API_KEY,
                    'symbol': symbol,
                    'order_id': order_id,
                    'timestamp': int(time.time() * 1000)
                }
                
                signature = self._generate_bybit_signature(params)
                params['sign'] = signature
                
                url = "https://api.bybit.com/v2/private/order/cancel"
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, data=params) as response:
                        result = await response.json()
                
                success = result['ret_code'] == 0
                
            elif exchange == Exchange.METATRADER5 and self.mt5_initialized:
                # MT5 cancellation logic
                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": int(order_id)
                }
                result = mt5.order_send(request)
                success = result.retcode == mt5.TRADE_RETCODE_DONE
                
            else:
                logger.error(f"Cannot cancel order - exchange {exchange.value} not available")
                return False
            
            if success:
                # Update order status
                if order_id in self.active_orders:
                    self.active_orders[order_id].status = OrderStatus.CANCELED
                logger.info(f"Order {order_id} canceled successfully")
            else:
                logger.error(f"Failed to cancel order {order_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    async def get_order_status(self, exchange: Exchange, order_id: str, symbol: str) -> Optional[OrderResponse]:
        """
        Get current status of an order
        
        Args:
            exchange: Exchange where order was placed
            order_id: Order ID to check
            symbol: Trading symbol
            
        Returns:
            OrderResponse with current status or None if not found
        """
        try:
            if exchange == Exchange.BINANCE and self.binance_client:
                result = self.binance_client.get_order(symbol=symbol, orderId=order_id)
                
                order_response = OrderResponse(
                    order_id=str(result['orderId']),
                    client_order_id=result.get('clientOrderId'),
                    symbol=result['symbol'],
                    side=OrderSide(result['side']),
                    order_type=OrderType(result['type']),
                    quantity=float(result['origQty']),
                    price=float(result['price']) if result.get('price') else None,
                    status=OrderStatus(result['status']),
                    filled_quantity=float(result['executedQty']),
                    average_price=float(result['price']) if result.get('price') and float(result['executedQty']) > 0 else None,
                    commission=0.0,
                    timestamp=datetime.fromtimestamp(result['time'] / 1000),
                    exchange=Exchange.BINANCE
                )
                
                # Update stored order
                self.active_orders[order_id] = order_response
                return order_response
                
            # Add similar logic for other exchanges...
            
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive execution summary
        
        Returns:
            Dictionary with execution statistics and metrics
        """
        try:
            # Calculate success rate
            success_rate = 0
            if self.execution_stats['total_orders'] > 0:
                success_rate = (self.execution_stats['successful_orders'] / self.execution_stats['total_orders']) * 100
            
            # Get active orders summary
            active_orders_summary = []
            for order_id, order in self.active_orders.items():
                active_orders_summary.append({
                    'order_id': order_id,
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'status': order.status.value,
                    'exchange': order.exchange.value
                })
            
            summary = {
                'statistics': {
                    'total_orders': self.execution_stats['total_orders'],
                    'successful_orders': self.execution_stats['successful_orders'],
                    'failed_orders': self.execution_stats['failed_orders'],
                    'success_rate_pct': success_rate,
                    'total_volume': self.execution_stats['total_volume'],
                    'total_commission': self.execution_stats['total_commission']
                },
                'active_orders': {
                    'count': len(self.active_orders),
                    'orders': active_orders_summary
                },
                'exchange_status': {
                    'binance': self.binance_client is not None,
                    'bybit': config.BYBIT_API_KEY is not None,
                    'metatrader5': self.mt5_initialized
                },
                'rate_limits': {
                    exchange.value.lower(): {
                        'requests': limit_info['requests'],
                        'reset_time': limit_info['reset_time']
                    }
                    for exchange, limit_info in self.rate_limits.items()
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating execution summary: {e}")
            return {'error': str(e)}
    
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

# Global TradeExecutor instance
trade_executor = TradeExecutor()

# Example usage and testing
async def test_trade_executor():
    """
    Test function to verify TradeExecutor functionality
    """
    logger.info("Testing TradeExecutor...")
    
    # Create test order request
    test_order = OrderRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=0.001,
        price=45000.0,
        client_order_id="test_order_001"
    )
    
    # Test order execution (this would place a real order in production)
    # result = await trade_executor.execute_trade(Exchange.BINANCE, test_order)
    # logger.info(f"Test order result: {result.status.value}")
    
    # Get execution summary
    summary = trade_executor.get_execution_summary()
    logger.info(f"Execution summary: {summary['statistics']}")

if __name__ == "__main__":
    # Run test when module is executed directly
    asyncio.run(test_trade_executor())
