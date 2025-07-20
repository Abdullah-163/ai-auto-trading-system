"""
UIController Module for AI-powered Auto-Trading Application
Provides FastAPI REST API endpoints for frontend interaction
Handles trading control, monitoring, backtesting, and system management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from loguru import logger
import asyncio
import json
from enum import Enum

# Import our modules
from config import config
from DataFetcher import data_fetcher
from StrategyEngine import strategy_engine, SignalType
from MLModel import MLModelManager
from RiskManager import risk_manager, RiskLevel
from TradeExecutor import trade_executor, Exchange, OrderType, OrderSide, OrderRequest
from BackTesting import BackTester, BacktestMode, BacktestConfig

# Pydantic models for API requests/responses
class TradingStatus(str, Enum):
    """Trading status enumeration"""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"

class SystemStatusResponse(BaseModel):
    """System status response model"""
    trading_enabled: bool
    trading_status: TradingStatus
    current_capital: float
    available_capital: float
    daily_pnl: float
    open_positions: int
    total_trades_today: int
    system_uptime: str
    last_update: datetime
    active_symbols: List[str]
    exchange_status: Dict[str, bool]
    risk_level: str

class ToggleRequest(BaseModel):
    """Toggle trading request model"""
    enable: bool
    reason: Optional[str] = None

class BacktestRequest(BaseModel):
    """Backtest request model"""
    symbol: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = Field(default=10000, gt=0)
    mode: str = Field(default="STRATEGY_ONLY")
    commission_rate: float = Field(default=0.001, ge=0, le=0.1)
    max_positions: int = Field(default=5, gt=0, le=20)

class TradeHistoryResponse(BaseModel):
    """Trade history response model"""
    trades: List[Dict[str, Any]]
    total_trades: int
    total_pnl: float
    win_rate: float
    profit_factor: float

class MarketDataRequest(BaseModel):
    """Market data request model"""
    symbols: List[str]
    timeframe: str = "1h"
    limit: int = Field(default=100, le=1000)

class SignalResponse(BaseModel):
    """Trading signal response model"""
    symbol: str
    signal: str
    confidence: float
    reasoning: List[str]
    timestamp: datetime
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

# Global variables for system state
system_start_time = datetime.now()
ml_manager = MLModelManager()
backtester = BackTester()
background_tasks_running = False

# FastAPI app initialization
app = FastAPI(
    title="AI Auto-Trading API",
    description="Comprehensive AI-powered auto-trading system API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting AI Auto-Trading API...")
    
    # Start background tasks
    global background_tasks_running
    if not background_tasks_running:
        asyncio.create_task(background_monitoring_task())
        background_tasks_running = True
    
    logger.info("AI Auto-Trading API started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down AI Auto-Trading API...")
    
    # Cleanup resources
    data_fetcher.cleanup()
    trade_executor.cleanup()
    
    logger.info("AI Auto-Trading API shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "uptime": str(datetime.now() - system_start_time)
    }

# System status endpoint
@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get risk metrics
        portfolio_risk = risk_manager.get_portfolio_risk_metrics()
        
        # Get execution stats
        execution_summary = trade_executor.get_execution_summary()
        
        # Calculate uptime
        uptime = str(datetime.now() - system_start_time)
        
        # Get active symbols
        active_symbols = config.get_active_symbols()
        
        # Determine trading status
        if config.is_trading_enabled():
            if portfolio_risk.risk_level in [RiskLevel.VERY_HIGH, RiskLevel.HIGH]:
                trading_status = TradingStatus.PAUSED
            else:
                trading_status = TradingStatus.ACTIVE
        else:
            trading_status = TradingStatus.INACTIVE
        
        return SystemStatusResponse(
            trading_enabled=config.is_trading_enabled(),
            trading_status=trading_status,
            current_capital=portfolio_risk.total_capital,
            available_capital=portfolio_risk.available_capital,
            daily_pnl=portfolio_risk.daily_pnl,
            open_positions=portfolio_risk.open_positions,
            total_trades_today=risk_manager.daily_trades,
            system_uptime=uptime,
            last_update=datetime.now(),
            active_symbols=active_symbols[:10],  # Limit to first 10
            exchange_status=execution_summary.get('exchange_status', {}),
            risk_level=portfolio_risk.risk_level.value
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Toggle trading endpoint
@app.post("/toggle")
async def toggle_trading(request: ToggleRequest):
    """Toggle auto-trading on/off"""
    try:
        previous_state = config.is_trading_enabled()
        new_state = config.toggle_trading() if request.enable != previous_state else previous_state
        
        # Log the change
        action = "enabled" if new_state else "disabled"
        reason = f" - Reason: {request.reason}" if request.reason else ""
        logger.info(f"Trading {action} via API{reason}")
        
        return {
            "success": True,
            "previous_state": previous_state,
            "new_state": new_state,
            "message": f"Trading {action}",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error toggling trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get trading signals
@app.get("/signals/{symbol}", response_model=SignalResponse)
async def get_trading_signal(symbol: str, timeframe: str = "1h", limit: int = 100):
    """Get current trading signal for a symbol"""
    try:
        # Fetch recent data
        data = await data_fetcher.fetch_binance_data(symbol, timeframe, limit)
        
        if data is None or data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Generate signal
        signal = strategy_engine.generate_signal(data, symbol)
        
        return SignalResponse(
            symbol=symbol,
            signal=signal.signal.value,
            confidence=signal.confidence,
            reasoning=signal.reasoning,
            timestamp=signal.timestamp,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get multiple signals
@app.post("/signals/batch")
async def get_batch_signals(request: MarketDataRequest):
    """Get trading signals for multiple symbols"""
    try:
        signals = []
        
        # Fetch data for all symbols concurrently
        data_results = await data_fetcher.fetch_multiple_symbols(
            request.symbols, 'binance'
        )
        
        # Generate signals for each symbol
        for symbol, data in data_results.items():
            try:
                if data is not None and not data.empty:
                    signal = strategy_engine.generate_signal(data, symbol)
                    signals.append({
                        "symbol": symbol,
                        "signal": signal.signal.value,
                        "confidence": signal.confidence,
                        "reasoning": signal.reasoning[:3],  # Limit reasoning
                        "timestamp": signal.timestamp,
                        "entry_price": signal.entry_price
                    })
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                signals.append({
                    "symbol": symbol,
                    "signal": "ERROR",
                    "confidence": 0.0,
                    "reasoning": [f"Error: {str(e)}"],
                    "timestamp": datetime.now(),
                    "entry_price": None
                })
        
        return {
            "signals": signals,
            "total_symbols": len(request.symbols),
            "successful_signals": len([s for s in signals if s["signal"] != "ERROR"]),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting batch signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get trade history
@app.get("/trades", response_model=TradeHistoryResponse)
async def get_trade_history(limit: int = 100, offset: int = 0):
    """Get trading history"""
    try:
        # Get trade history from risk manager
        all_trades = risk_manager.trade_history
        
        # Apply pagination
        trades = all_trades[offset:offset + limit]
        
        # Calculate metrics
        if trades:
            total_pnl = sum(trade['pnl'] for trade in trades)
            winning_trades = len([t for t in trades if t['pnl'] > 0])
            win_rate = (winning_trades / len(trades)) * 100
            
            # Calculate profit factor
            gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            total_pnl = 0
            win_rate = 0
            profit_factor = 0
        
        return TradeHistoryResponse(
            trades=trades,
            total_trades=len(all_trades),
            total_pnl=total_pnl,
            win_rate=win_rate,
            profit_factor=profit_factor
        )
        
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run backtest
@app.post("/backtest")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run backtesting for a strategy"""
    try:
        # Validate mode
        try:
            mode = BacktestMode(request.mode.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid backtest mode: {request.mode}")
        
        # Create backtest config
        backtest_config = BacktestConfig(
            initial_capital=request.initial_capital,
            start_date=request.start_date,
            end_date=request.end_date,
            commission_rate=request.commission_rate,
            max_positions=request.max_positions
        )
        
        # Initialize backtester
        backtester_instance = BackTester(backtest_config)
        
        # Fetch historical data
        logger.info(f"Fetching historical data for {request.symbol}")
        data = await data_fetcher.fetch_binance_data(
            request.symbol, 
            '1h', 
            min(2000, config.HISTORICAL_DATA_DAYS * 24)
        )
        
        if data is None or len(data) < 100:
            raise HTTPException(
                status_code=404, 
                detail=f"Insufficient historical data for {request.symbol}"
            )
        
        # Run backtest
        logger.info(f"Running backtest for {request.symbol} in {mode.value} mode")
        results = backtester_instance.run_backtest(data, request.symbol, mode)
        
        # Prepare response
        response = {
            "success": True,
            "symbol": request.symbol,
            "mode": mode.value,
            "config": {
                "initial_capital": request.initial_capital,
                "start_date": request.start_date,
                "end_date": request.end_date,
                "commission_rate": request.commission_rate
            },
            "results": {
                "initial_capital": results.initial_capital,
                "final_capital": results.final_capital,
                "total_return_pct": results.total_return_pct,
                "total_trades": results.total_trades,
                "winning_trades": results.winning_trades,
                "losing_trades": results.losing_trades,
                "win_rate_pct": results.win_rate_pct,
                "profit_factor": results.profit_factor,
                "max_drawdown_pct": results.max_drawdown_pct,
                "sharpe_ratio": results.sharpe_ratio,
                "sortino_ratio": results.sortino_ratio,
                "avg_trade_duration_hours": results.avg_trade_duration_hours,
                "trades_per_month": results.trades_per_month
            },
            "timestamp": datetime.now()
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get market data
@app.post("/market-data")
async def get_market_data(request: MarketDataRequest):
    """Get market data for symbols"""
    try:
        # Fetch data for all symbols
        data_results = await data_fetcher.fetch_multiple_symbols(
            request.symbols, 'binance'
        )
        
        # Format response
        market_data = {}
        for symbol, data in data_results.items():
            if data is not None and not data.empty:
                # Get latest values
                latest = data.iloc[-1]
                
                market_data[symbol] = {
                    "price": float(latest['close']),
                    "change_24h": float(((latest['close'] - data.iloc[-24]['close']) / data.iloc[-24]['close']) * 100) if len(data) >= 24 else 0,
                    "volume": float(latest['volume']),
                    "high_24h": float(data['high'].tail(24).max()) if len(data) >= 24 else float(latest['high']),
                    "low_24h": float(data['low'].tail(24).min()) if len(data) >= 24 else float(latest['low']),
                    "timestamp": latest.name.isoformat()
                }
            else:
                market_data[symbol] = {
                    "error": "No data available"
                }
        
        return {
            "market_data": market_data,
            "timestamp": datetime.now(),
            "symbols_requested": len(request.symbols),
            "symbols_successful": len([v for v in market_data.values() if "error" not in v])
        }
        
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get risk metrics
@app.get("/risk-metrics")
async def get_risk_metrics():
    """Get comprehensive risk management metrics"""
    try:
        risk_summary = risk_manager.get_risk_summary()
        return {
            "success": True,
            "risk_metrics": risk_summary,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get ML model status
@app.get("/ml-status")
async def get_ml_status():
    """Get machine learning model status"""
    try:
        # This would be expanded to show actual ML model status
        return {
            "models_available": ["random_forest", "gradient_boost", "neural_network"],
            "models_trained": 0,  # Would be calculated from ml_manager
            "last_training": None,
            "prediction_accuracy": 0.0,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting ML status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Train ML model
@app.post("/ml-train/{symbol}")
async def train_ml_model(symbol: str, model_type: str = "random_forest", background_tasks: BackgroundTasks):
    """Train ML model for a symbol"""
    try:
        # Fetch training data
        data = await data_fetcher.fetch_binance_data(symbol, '1h', 1000)
        
        if data is None or len(data) < 100:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient data for training {symbol}"
            )
        
        # Add training task to background
        background_tasks.add_task(
            train_model_background,
            symbol,
            data,
            model_type
        )
        
        return {
            "success": True,
            "message": f"ML model training started for {symbol}",
            "model_type": model_type,
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting ML training for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get system logs
@app.get("/logs")
async def get_system_logs(limit: int = 100, level: str = "INFO"):
    """Get recent system logs"""
    try:
        # This is a simplified implementation
        # In practice, you'd read from actual log files
        return {
            "logs": [
                {
                    "timestamp": datetime.now(),
                    "level": "INFO",
                    "message": "System running normally",
                    "module": "UIController"
                }
            ],
            "total_logs": 1,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def background_monitoring_task():
    """Background task for system monitoring"""
    while True:
        try:
            # Monitor system health
            await asyncio.sleep(60)  # Run every minute
            
            # Check if trading should be paused due to high risk
            portfolio_risk = risk_manager.get_portfolio_risk_metrics()
            if (config.is_trading_enabled() and 
                portfolio_risk.risk_level == RiskLevel.VERY_HIGH):
                logger.warning("High risk detected - consider pausing trading")
            
            # Update position P&L for open positions
            for symbol in risk_manager.open_positions:
                try:
                    latest_price = data_fetcher.get_latest_price(symbol, 'binance')
                    if latest_price:
                        risk_manager.update_position_pnl(symbol, latest_price)
                except Exception as e:
                    logger.error(f"Error updating P&L for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Error in background monitoring: {e}")
            await asyncio.sleep(60)

async def train_model_background(symbol: str, data, model_type: str):
    """Background task for ML model training"""
    try:
        logger.info(f"Starting background ML training for {symbol}")
        results = ml_manager.train_model_for_symbol(symbol, data, model_type)
        
        if results.get('success'):
            logger.info(f"ML model training completed for {symbol}: {results.get('metrics', {}).get('accuracy', 0):.3f} accuracy")
        else:
            logger.error(f"ML model training failed for {symbol}: {results.get('error')}")
            
    except Exception as e:
        logger.error(f"Error in background ML training for {symbol}: {e}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting AI Auto-Trading API server...")
    uvicorn.run(
        "UIController:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
