"""
Command-line interface for the trading bot.

This module provides a command-line interface for interacting with
the trading bot.
"""

import os
import logging
import asyncio
from typing import Optional, List, Dict
from pathlib import Path
import json
import time
from datetime import datetime

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler

from trading_bot.exchange.bitunix import BitunixClient
from trading_bot.strategy.lsob import LSOBDetector
from trading_bot.trading.position import PositionManager
from trading_bot.trading.metrics import KPITracker, TimeFrame
from trading_bot.data.storage import DatabaseManager
from trading_bot.backtest.engine import BacktestEngine
from trading_bot.ui.dashboard import run_dashboard

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("trading_bot")

# Create Typer app
app = typer.Typer(help="Bitcoin Futures Trading Bot")
console = Console()

# Default paths
DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_DB_PATH = "data/trades.db"
DEFAULT_BACKTEST_DATA_PATH = "data/backtest_data.csv"
DEFAULT_BACKTEST_RESULTS_PATH = "results/backtest"


@app.callback()
def callback():
    """
    Bitcoin Futures Trading Bot CLI.
    """
    pass


@app.command()
def init(
    config_path: str = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to the configuration file"
    ),
    api_key: str = typer.Option(None, "--api-key", "-k", help="Bitunix API key"),
    api_secret: str = typer.Option(
        None, "--api-secret", "-s", help="Bitunix API secret"
    ),
    testnet: bool = typer.Option(False, "--testnet", "-t", help="Use testnet API"),
    db_path: str = typer.Option(
        DEFAULT_DB_PATH, "--db", "-d", help="Path to the database file"
    ),
):
    """
    Initialize the trading bot.
    """
    # Create config directory if it doesn't exist
    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir)

    # Create data directory if it doesn't exist
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # Check if config file exists
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Use existing values if not provided
        api_key = api_key or config.get("api_key")
        api_secret = api_secret or config.get("api_secret")
        testnet = testnet or config.get("testnet", False)
        db_path = db_path or config.get("db_path", DEFAULT_DB_PATH)
    else:
        config = {}

    # Prompt for API key and secret if not provided
    if not api_key:
        api_key = typer.prompt("Enter your Bitunix API key")

    if not api_secret:
        api_secret = typer.prompt("Enter your Bitunix API secret", hide_input=True)

    # Create config
    config = {
        "api_key": api_key,
        "api_secret": api_secret,
        "testnet": testnet,
        "db_path": db_path,
        "strategy": {
            "lsob": {
                "imbalance_threshold": 0.3,
                "sweep_detection_window": 5,
                "min_sweep_percentage": 0.5,
                "confidence_threshold": 0.7,
            }
        },
        "trading": {
            "risk_per_trade": 0.01,
            "max_positions": 5,
            "max_positions_per_symbol": 1,
        },
        "symbols": ["BTCUSDT"],
    }

    # Save config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    console.print(f"[green]Configuration saved to {config_path}[/green]")

    # Initialize database
    db_manager = DatabaseManager(db_path)
    db_manager.connect()
    db_manager.init_db()
    db_manager.disconnect()

    console.print(f"[green]Database initialized at {db_path}[/green]")
    console.print("[green]Trading bot initialized successfully![/green]")


@app.command()
def run(
    config_path: str = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to the configuration file"
    ),
    symbol: str = typer.Option(
        None, "--symbol", "-s", help="Trading pair symbol (e.g., BTCUSDT)"
    ),
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run in daemon mode"),
):
    """
    Run the trading bot.
    """
    # Load config
    if not os.path.exists(config_path):
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        console.print("Run 'trading-bot init' to create a configuration file.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Get API credentials
    api_key = config["api_key"]
    api_secret = config["api_secret"]
    testnet = config.get("testnet", False)
    db_path = config.get("db_path", DEFAULT_DB_PATH)

    # Get strategy parameters
    lsob_config = config["strategy"]["lsob"]
    imbalance_threshold = lsob_config["imbalance_threshold"]
    sweep_detection_window = lsob_config["sweep_detection_window"]
    min_sweep_percentage = lsob_config["min_sweep_percentage"]
    confidence_threshold = lsob_config["confidence_threshold"]

    # Get trading parameters
    trading_config = config["trading"]
    risk_per_trade = trading_config["risk_per_trade"]
    max_positions = trading_config["max_positions"]
    max_positions_per_symbol = trading_config["max_positions_per_symbol"]

    # Get symbols
    symbols = [symbol] if symbol else config["symbols"]

    # Initialize components
    client = BitunixClient(api_key, api_secret, testnet=testnet)
    position_manager = PositionManager(
        client=client,
        risk_per_trade=risk_per_trade,
        max_positions=max_positions,
        max_positions_per_symbol=max_positions_per_symbol,
    )
    db_manager = DatabaseManager(db_path)
    kpi_tracker = KPITracker(db_path=db_path)

    # Initialize strategies
    strategies = {}
    for sym in symbols:
        strategies[sym] = LSOBDetector(
            symbol=sym,
            imbalance_threshold=imbalance_threshold,
            sweep_detection_window=sweep_detection_window,
            min_sweep_percentage=min_sweep_percentage,
            confidence_threshold=confidence_threshold,
        )

    # Run the trading loop
    try:
        asyncio.run(
            run_trading_loop(
                client=client,
                position_manager=position_manager,
                strategies=strategies,
                kpi_tracker=kpi_tracker,
                db_manager=db_manager,
                daemon=daemon,
            )
        )
    except KeyboardInterrupt:
        console.print("[yellow]Trading bot stopped by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("Error in trading loop")


async def run_trading_loop(
    client: BitunixClient,
    position_manager: PositionManager,
    strategies: Dict[str, LSOBDetector],
    kpi_tracker: KPITracker,
    db_manager: DatabaseManager,
    daemon: bool = False,
):
    """
    Run the trading loop.

    Args:
        client: Bitunix API client
        position_manager: Position manager
        strategies: Dictionary of strategies by symbol
        kpi_tracker: KPI tracker
        db_manager: Database manager
        daemon: Whether to run in daemon mode
    """
    console.print("[green]Starting trading bot...[/green]")

    # Connect to database
    db_manager.connect()

    try:
        # Main loop
        while True:
            # Update positions
            await position_manager.update_positions()

            # Process each symbol
            for symbol, strategy in strategies.items():
                try:
                    # Get order book
                    order_book = await client.get_order_book(symbol)

                    # Add order book to strategy
                    strategy.add_order_book(order_book)

                    # Check for signals
                    signal = strategy.detect_signal()

                    if signal:
                        console.print(
                            f"[bold blue]Signal detected for {symbol}:[/bold blue]"
                        )
                        console.print(f"  Type: {signal.type.value}")
                        console.print(f"  Price: {signal.price:.2f}")
                        console.print(f"  Confidence: {signal.confidence:.2f}")
                        console.print(f"  Target: {signal.target_price:.2f}")
                        console.print(f"  Stop Loss: {signal.stop_loss:.2f}")

                        # Open position
                        position_id = await position_manager.open_position_from_signal(
                            signal
                        )

                        if position_id:
                            console.print(
                                f"[green]Position opened: {position_id}[/green]"
                            )
                        else:
                            console.print("[yellow]Failed to open position[/yellow]")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")

            # Check for closed positions
            for position in position_manager.positions.values():
                if position.status == "CLOSED" and position.exit_time is not None:
                    # Add trade to KPI tracker
                    kpi_tracker.add_trade(position)

                    # Save trade to database
                    from trading_bot.data.storage import Trade

                    trade = Trade(
                        position_id=position.id,
                        symbol=position.symbol,
                        side=position.side,
                        entry_price=position.entry_price,
                        exit_price=position.exit_price,
                        quantity=position.quantity,
                        pnl=position.pnl,
                        pnl_percentage=position.pnl_percentage,
                        entry_time=position.entry_time,
                        exit_time=position.exit_time,
                        duration_ms=position.duration_ms,
                        status=position.status,
                        target_price=position.target_price,
                        stop_loss=position.stop_loss,
                    )

                    db_manager.insert_trade(trade)

            # Print status
            if not daemon:
                # Get active positions
                active_positions = await position_manager.get_active_positions()

                # Print active positions
                if active_positions:
                    table = Table(title="Active Positions")
                    table.add_column("ID")
                    table.add_column("Symbol")
                    table.add_column("Side")
                    table.add_column("Entry Price")
                    table.add_column("Current PnL")
                    table.add_column("PnL %")

                    for position in active_positions:
                        table.add_row(
                            position.id,
                            position.symbol,
                            position.side,
                            f"${position.entry_price:.2f}",
                            f"${position.pnl:.2f}"
                            if position.pnl is not None
                            else "N/A",
                            f"{position.pnl_percentage:.2f}%"
                            if position.pnl_percentage is not None
                            else "N/A",
                        )

                    console.print(table)

                # Print performance metrics
                metrics = kpi_tracker.get_metrics(TimeFrame.ALL_TIME)

                table = Table(title="Performance Metrics")
                table.add_column("Metric")
                table.add_column("Value")

                table.add_row("Total Trades", str(metrics.total_trades))
                table.add_row("Win Rate", f"{metrics.win_rate * 100:.2f}%")
                table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
                table.add_row("Total PnL", f"${metrics.total_pnl:.2f}")

                console.print(table)

            # Wait for next iteration
            await asyncio.sleep(5)

            # Clear screen if not in daemon mode
            if not daemon:
                console.clear()

    finally:
        # Disconnect from database
        db_manager.disconnect()

        # Close all connections
        await client.close_all_connections()


@app.command()
def backtest(
    config_path: str = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to the configuration file"
    ),
    data_path: str = typer.Option(
        DEFAULT_BACKTEST_DATA_PATH,
        "--data",
        "-d",
        help="Path to the backtest data file",
    ),
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s", help="Trading pair symbol"),
    output_dir: str = typer.Option(
        DEFAULT_BACKTEST_RESULTS_PATH,
        "--output",
        "-o",
        help="Output directory for backtest results",
    ),
    parameter_sweep: bool = typer.Option(
        False, "--sweep", help="Perform parameter sweep"
    ),
):
    """
    Run a backtest.
    """
    # Load config
    if not os.path.exists(config_path):
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        console.print("Run 'trading-bot init' to create a configuration file.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Check if data file exists
    if not os.path.exists(data_path):
        console.print(f"[red]Data file not found: {data_path}[/red]")
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    console.print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)

    # Initialize backtest engine
    engine = BacktestEngine(
        initial_balance=10000.0,
        risk_per_trade=config["trading"]["risk_per_trade"],
        commission_rate=0.0004,  # 0.04% per trade
        slippage=0.0001,  # 0.01% slippage
    )

    if parameter_sweep:
        # Perform parameter sweep
        console.print("Performing parameter sweep...")

        # Define parameter ranges
        imbalance_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        sweep_detection_windows = [3, 5, 7, 10]
        min_sweep_percentages = [0.3, 0.5, 0.7]
        confidence_thresholds = [0.5, 0.7, 0.9]

        # Run parameter sweep
        results = engine.parameter_sweep(
            data=data,
            symbol=symbol,
            imbalance_thresholds=imbalance_thresholds,
            sweep_detection_windows=sweep_detection_windows,
            min_sweep_percentages=min_sweep_percentages,
            confidence_thresholds=confidence_thresholds,
            output_dir=output_dir,
        )

        # Print best results
        console.print("\n[green]Best Parameters:[/green]")

        table = Table(title="Top 5 Parameter Combinations")
        table.add_column("Rank")
        table.add_column("Imbalance Threshold")
        table.add_column("Sweep Window")
        table.add_column("Min Sweep %")
        table.add_column("Confidence")
        table.add_column("Total PnL")
        table.add_column("Win Rate")
        table.add_column("Profit Factor")

        for i, result in enumerate(results[:5]):
            params = result.parameters
            metrics = result.metrics

            table.add_row(
                str(i + 1),
                str(params["imbalance_threshold"]),
                str(params["sweep_detection_window"]),
                str(params["min_sweep_percentage"]),
                str(params["confidence_threshold"]),
                f"${metrics.total_pnl:.2f}",
                f"{metrics.win_rate * 100:.2f}%",
                f"{metrics.profit_factor:.2f}",
            )

        console.print(table)
        console.print(f"\nResults saved to {output_dir}")

    else:
        # Run single backtest with parameters from config
        console.print("Running backtest...")

        lsob_config = config["strategy"]["lsob"]

        result = engine.backtest_lsob_strategy(
            data=data,
            symbol=symbol,
            imbalance_threshold=lsob_config["imbalance_threshold"],
            sweep_detection_window=lsob_config["sweep_detection_window"],
            min_sweep_percentage=lsob_config["min_sweep_percentage"],
            confidence_threshold=lsob_config["confidence_threshold"],
        )

        # Save results
        output_file = os.path.join(
            output_dir, f"backtest_{symbol}_{int(time.time())}.json"
        )
        result.save_to_json(output_file)

        # Print results
        console.print("\n[green]Backtest Results:[/green]")

        table = Table(title="Performance Metrics")
        table.add_column("Metric")
        table.add_column("Value")

        metrics = result.metrics

        table.add_row("Total Trades", str(metrics.total_trades))
        table.add_row("Winning Trades", str(metrics.winning_trades))
        table.add_row("Losing Trades", str(metrics.losing_trades))
        table.add_row("Win Rate", f"{metrics.win_rate * 100:.2f}%")
        table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
        table.add_row("Total PnL", f"${metrics.total_pnl:.2f}")
        table.add_row("Max Drawdown", f"${metrics.max_drawdown:.2f}")
        table.add_row("Average Win", f"${metrics.average_win:.2f}")
        table.add_row("Average Loss", f"${metrics.average_loss:.2f}")
        table.add_row("Largest Win", f"${metrics.largest_win:.2f}")
        table.add_row("Largest Loss", f"${metrics.largest_loss:.2f}")
        table.add_row(
            "Avg Trade Duration", f"{metrics.average_trade_duration_minutes:.2f} min"
        )

        console.print(table)
        console.print(f"\nResults saved to {output_file}")


@app.command()
def dashboard(
    config_path: str = typer.Option(
        DEFAULT_CONFIG_PATH, "--config", "-c", help="Path to the configuration file"
    ),
    port: int = typer.Option(8501, "--port", "-p", help="Port to run the dashboard on"),
):
    """
    Run the dashboard.
    """
    # Load config
    if not os.path.exists(config_path):
        console.print(f"[red]Configuration file not found: {config_path}[/red]")
        console.print("Run 'trading-bot init' to create a configuration file.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Get API credentials
    api_key = config["api_key"]
    api_secret = config["api_secret"]
    db_path = config.get("db_path", DEFAULT_DB_PATH)

    # Run dashboard
    console.print(f"[green]Starting dashboard on port {port}...[/green]")
    console.print("Press Ctrl+C to stop.")

    try:
        run_dashboard(
            db_path=db_path,
            api_key=api_key,
            api_secret=api_secret,
            refresh_interval=60,
        )
    except KeyboardInterrupt:
        console.print("[yellow]Dashboard stopped by user.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("Error in dashboard")


if __name__ == "__main__":
    app()
