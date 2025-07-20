"""
Main Entry Point for AI-powered Auto-Trading Application
Initializes and runs the complete trading system with FastAPI server
Handles system startup, configuration, and graceful shutdown
"""

import asyncio
import signal
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
import uvicorn
from contextlib import asynccontextmanager

# Import our modules
from config import config, create_env_template
from DataFetcher import data_fetcher
from StrategyEngine import strategy_engine
from MLModel import MLModelManager
from RiskManager import risk_manager
from TradeExecutor import trade_executor
from BackTesting import BackTester
from UIController import app

# Global variables
trading_task = None
system_initialized = False
shutdown_event = asyncio.Event()

def setup_logging():
    """
    Configure logging for the application
    """
    try:
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure loguru
        logger.remove()  # Remove default handler
        
        # Add console handler
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=config.LOG_LEVEL,
            colorize=True
        )
        
        # Add file handler
        logger.add(
            config.LOG_FILE_PATH,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=config.LOG_LEVEL,
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        logger.info("Logging configured successfully")
        
    except Exception as e:
        print(f"Error setting up logging: {e}")
        sys.exit(1)

async def initialize_system():
    """
    Initialize all system components
    """
    global system_initialized
    
    try:
        logger.info("=" * 60)
        logger.info("AI-POWERED AUTO-TRADING SYSTEM")
        logger.info("=" * 60)
        logger.info("Initializing system components...")
        
        # Create necessary directories
        directories = ["models", "logs", "backtest_results", "data_cache"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        # Create environment template if .env doesn't exist
        if not Path(".env").exists():
            create_env_template()
            logger.warning("Created .env.template - Please configure your API keys in .env file")
        
        # Initialize data fetcher
        logger.info("Initializing data fetcher...")
        # Data fetcher initializes connections in its __init__
        
        # Initialize strategy engine
        logger.info("Initializing strategy engine...")
        # Strategy engine is ready to use
        
        # Initialize ML model manager
        logger.info("Initializing ML model manager...")
        ml_manager = MLModelManager()
        
        # Initialize risk manager
        logger.info("Initializing risk manager...")
        # Risk manager is ready to use
        
        # Initialize trade executor
        logger.info("Initializing trade executor...")
        # Trade executor initializes exchange connections in its __init__
        
        # Test system components
        await test_system_components()
        
        system_initialized = True
        logger.info("System initialization completed successfully")
        
        # Display system status
        await display_system_status()
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

async def test_system_components():
    """
    Test all system components to ensure they're working
    """
    try:
        logger.info("Testing system components...")
        
        # Test data fetcher
        try:
            test_data = await data_fetcher.fetch_binance_data('BTCUSDT', '1h', 10)
            if test_data is not None and not test_data.empty:
                logger.info("✓ Data fetcher working")
            else:
                logger.warning("⚠ Data fetcher test returned no data")
        except Exception as e:
            logger.warning(f"⚠ Data fetcher test failed: {e}")
        
        # Test strategy engine
        try:
            if test_data is not None and not test_data.empty:
                test_signal = strategy_engine.generate_signal(test_data, 'BTCUSDT')
                logger.info(f"✓ Strategy engine working - Generated {test_signal.signal.value} signal")
            else:
                logger.warning("⚠ Strategy engine test skipped - no test data")
        except Exception as e:
            logger.warning(f"⚠ Strategy engine test failed: {e}")
        
        # Test risk manager
        try:
            risk_summary = risk_manager.get_risk_summary()
            logger.info(f"✓ Risk manager working - Capital: ${risk_summary['capital']['current']:,.2f}")
        except Exception as e:
            logger.warning(f"⚠ Risk manager test failed: {e}")
        
        # Test trade executor
        try:
            execution_summary = trade_executor.get_execution_summary()
            logger.info(f"✓ Trade executor working - Exchange status: {execution_summary['exchange_status']}")
        except Exception as e:
            logger.warning(f"⚠ Trade executor test failed: {e}")
        
        logger.info("Component testing completed")
        
    except Exception as e:
        logger.error(f"Error testing system components: {e}")

async def display_system_status():
    """
    Display current system status
    """
    try:
        logger.info("=" * 60)
        logger.info("SYSTEM STATUS")
        logger.info("=" * 60)
        
        # Trading status
        trading_enabled = config.is_trading_enabled()
        logger.info(f"Trading Status: {'ENABLED' if trading_enabled else 'DISABLED'}")
        
        # Risk metrics
        risk_summary = risk_manager.get_risk_summary()
        logger.info(f"Current Capital: ${risk_summary['capital']['current']:,.2f}")
        logger.info(f"Available Capital: ${risk_summary['capital']['available']:,.2f}")
        logger.info(f"Risk Level: {risk_summary['risk_metrics']['risk_level']}")
        
        # Active symbols
        active_symbols = config.get_active_symbols()
        logger.info(f"Active Symbols: {len(active_symbols)} ({', '.join(active_symbols[:5])}{'...' if len(active_symbols) > 5 else ''})")
        
        # Exchange status
        execution_summary = trade_executor.get_execution_summary()
        exchange_status = execution_summary.get('exchange_status', {})
        logger.info(f"Exchange Connections:")
        for exchange, status in exchange_status.items():
            status_icon = "✓" if status else "✗"
            logger.info(f"  {status_icon} {exchange.title()}: {'Connected' if status else 'Disconnected'}")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error displaying system status: {e}")

async def trading_loop():
    """
    Main trading loop - monitors markets and executes trades
    """
    logger.info("Starting main trading loop...")
    
    loop_count = 0
    last_signal_check = {}
    
    while not shutdown_event.is_set():
        try:
            loop_count += 1
            current_time = datetime.now()
            
            # Only process if trading is enabled
            if not config.is_trading_enabled():
                await asyncio.sleep(30)  # Check every 30 seconds when disabled
                continue
            
            # Get active symbols to monitor
            symbols_to_monitor = config.get_active_symbols()[:10]  # Limit to 10 symbols
            
            logger.debug(f"Trading loop #{loop_count} - Monitoring {len(symbols_to_monitor)} symbols")
            
            # Process each symbol
            for symbol in symbols_to_monitor:
                try:
                    # Skip if we checked this symbol recently (avoid over-trading)
                    if symbol in last_signal_check:
                        time_since_last = (current_time - last_signal_check[symbol]).total_seconds()
                        if time_since_last < config.DATA_REFRESH_INTERVAL:
                            continue
                    
                    # Fetch recent data
                    data = await data_fetcher.fetch_binance_data(symbol, '1h', 100)
                    if data is None or data.empty:
                        logger.warning(f"No data available for {symbol}")
                        continue
                    
                    # Generate trading signal
                    signal = strategy_engine.generate_signal(data, symbol)
                    last_signal_check[symbol] = current_time
                    
                    # Check if we should act on the signal
                    if signal.confidence >= config.MIN_ACCURACY_THRESHOLD:
                        await process_trading_signal(symbol, signal, data)
                    
                    # Small delay between symbols to avoid rate limits
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} in trading loop: {e}")
                    continue
            
            # Update position P&L
            await update_positions()
            
            # Check for exit conditions
            await check_exit_conditions()
            
            # Log periodic status
            if loop_count % 10 == 0:  # Every 10 loops
                portfolio_risk = risk_manager.get_portfolio_risk_metrics()
                logger.info(f"Trading loop #{loop_count} - "
                           f"Capital: ${portfolio_risk.total_capital:,.2f}, "
                           f"Open positions: {portfolio_risk.open_positions}, "
                           f"Risk level: {portfolio_risk.risk_level.value}")
            
            # Wait before next loop
            await asyncio.sleep(config.DATA_REFRESH_INTERVAL)
            
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            await asyncio.sleep(60)  # Wait longer on error

async def process_trading_signal(symbol: str, signal, data):
    """
    Process a trading signal and potentially execute a trade
    """
    try:
        # Skip if signal is HOLD
        if signal.signal.value == 'HOLD':
            return
        
        # Check if we already have a position in this symbol
        if symbol in risk_manager.open_positions:
            logger.debug(f"Already have position in {symbol}, skipping")
            return
        
        # Determine trade direction
        direction = 'long' if signal.signal.value in ['BUY', 'STRONG_BUY'] else 'short'
        
        # Calculate position size
        position_info = risk_manager.calculate_position_size(
            symbol, signal.entry_price, signal.stop_loss
        )
        
        if 'error' in position_info:
            logger.warning(f"Cannot calculate position size for {symbol}: {position_info['error']}")
            return
        
        # Validate trade
        validation = risk_manager.validate_trade(
            symbol, position_info['position_size'], signal.entry_price,
            signal.stop_loss, signal.take_profit, direction
        )
        
        if not validation['approved']:
            logger.warning(f"Trade validation failed for {symbol}: {validation['errors']}")
            return
        
        # Execute trade (in demo mode for safety)
        logger.info(f"DEMO TRADE: {direction.upper()} {symbol} - "
                   f"Size: {position_info['position_size']:.6f}, "
                   f"Price: ${signal.entry_price:.4f}, "
                   f"Confidence: {signal.confidence:.2f}")
        
        # In production, you would uncomment this:
        # success = risk_manager.open_position(
        #     symbol, position_info['position_size'], signal.entry_price,
        #     signal.stop_loss, signal.take_profit, direction, "AI_Strategy"
        # )
        
        # For demo, we'll just log the trade
        success = True
        
        if success:
            logger.info(f"✓ Demo position opened for {symbol}")
        else:
            logger.error(f"✗ Failed to open position for {symbol}")
            
    except Exception as e:
        logger.error(f"Error processing trading signal for {symbol}: {e}")

async def update_positions():
    """
    Update P&L for all open positions
    """
    try:
        for symbol in list(risk_manager.open_positions.keys()):
            try:
                latest_price = data_fetcher.get_latest_price(symbol, 'binance')
                if latest_price:
                    risk_manager.update_position_pnl(symbol, latest_price)
            except Exception as e:
                logger.error(f"Error updating P&L for {symbol}: {e}")
                
    except Exception as e:
        logger.error(f"Error updating positions: {e}")

async def check_exit_conditions():
    """
    Check exit conditions for all open positions
    """
    try:
        for symbol in list(risk_manager.open_positions.keys()):
            try:
                latest_price = data_fetcher.get_latest_price(symbol, 'binance')
                if latest_price:
                    exit_reason = risk_manager.check_stop_loss_take_profit(symbol, latest_price)
                    if exit_reason:
                        logger.info(f"Exit condition triggered for {symbol}: {exit_reason}")
                        # In production, execute the exit trade
                        # For demo, just close the position in risk manager
                        result = risk_manager.close_position(symbol, latest_price, exit_reason)
                        if result.get('success'):
                            logger.info(f"✓ Position closed for {symbol}: P&L ${result['pnl']:.2f}")
                            
            except Exception as e:
                logger.error(f"Error checking exit conditions for {symbol}: {e}")
                
    except Exception as e:
        logger.error(f"Error checking exit conditions: {e}")

def signal_handler(signum, frame):
    """
    Handle shutdown signals gracefully
    """
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

async def cleanup_system():
    """
    Clean up system resources
    """
    try:
        logger.info("Cleaning up system resources...")
        
        # Cancel trading task
        global trading_task
        if trading_task and not trading_task.done():
            trading_task.cancel()
            try:
                await trading_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup components
        data_fetcher.cleanup()
        trade_executor.cleanup()
        
        # Close any remaining positions (in production)
        if risk_manager.open_positions:
            logger.warning(f"Closing {len(risk_manager.open_positions)} remaining positions...")
            # In production, you would close all positions here
        
        logger.info("System cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during system cleanup: {e}")

@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan context manager
    """
    # Startup
    await initialize_system()
    
    # Start trading loop
    global trading_task
    trading_task = asyncio.create_task(trading_loop())
    
    yield
    
    # Shutdown
    shutdown_event.set()
    await cleanup_system()

# Update FastAPI app with lifespan
app.router.lifespan_context = lifespan

async def main():
    """
    Main application entry point
    """
    try:
        # Setup logging
        setup_logging()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize system
        await initialize_system()
        
        # Start trading loop
        global trading_task
        trading_task = asyncio.create_task(trading_loop())
        
        # Start FastAPI server
        config_uvicorn = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config_uvicorn)
        
        logger.info("Starting FastAPI server on http://0.0.0.0:8001")
        logger.info("API Documentation available at http://0.0.0.0:8001/docs")
        
        # Run server
        await server.serve()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        raise
    finally:
        # Cleanup
        await cleanup_system()

if __name__ == "__main__":
    try:
        # Run the main application
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
