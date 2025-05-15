# Bitcoin Futures Trading Bot - Project Context

## Project Overview
This project implements a Bitcoin Futures Trading Bot that uses the Limit State Order Book (LSOB) strategy to detect trading signals and execute trades on the Bitunix exchange. The bot includes functionality for backtesting, real-time trading, and performance monitoring through a dashboard.

## Project Structure
The project is organized into several modules:

- **exchange**: Handles communication with the Bitunix exchange API
- **strategy**: Implements the LSOB trading strategy
- **trading**: Manages positions and trading logic
- **data**: Handles data storage and retrieval
- **backtest**: Provides backtesting functionality
- **ui**: Implements the dashboard for monitoring performance
- **cli**: Provides a command-line interface for interacting with the bot

## Key Components

### Exchange Module
- `BitunixClient`: Client for interacting with the Bitunix exchange API
- `OrderBook`: Represents the order book data structure
- Authentication handling for API requests

### Strategy Module
- `LSOBDetector`: Implements the Limit State Order Book strategy
- Detects imbalances and sweeps in the order book
- Generates trading signals with confidence levels

### Trading Module
- `PositionManager`: Manages open positions and executes trades
- `TradePosition`: Represents a trading position
- `KPITracker`: Tracks key performance indicators
- `PerformanceMetrics`: Stores performance metrics

### Data Module
- `DatabaseManager`: Manages the SQLite database
- `Trade`: Represents a completed trade

### Backtest Module
- `BacktestEngine`: Runs backtests on historical data
- `BacktestResult`: Stores backtest results
- Parameter sweep functionality to optimize strategy parameters

### UI Module
- `Dashboard`: Streamlit dashboard for monitoring performance
- Various chart and table creation functions

### CLI Module
- Command-line interface for interacting with the bot
- Commands for initialization, running, backtesting, and launching the dashboard

## Configuration
The bot is configured through a JSON configuration file that includes:
- API credentials
- Strategy parameters
- Trading parameters
- Symbols to trade

## Implementation Details

### LSOB Strategy
The LSOB strategy detects trading signals based on:
- Order book imbalances
- Liquidity sweeps
- Confidence levels

Parameters:
- `imbalance_threshold`: Threshold for detecting significant imbalances
- `sweep_detection_window`: Number of order book updates to consider for sweep detection
- `min_sweep_percentage`: Minimum percentage of liquidity that must be swept
- `confidence_threshold`: Minimum confidence level for generating signals

### Risk Management
- Position sizing based on risk per trade
- Stop loss and take profit levels
- Maximum number of concurrent positions

### Performance Metrics
- Win rate
- Profit factor
- Total PnL
- Maximum drawdown
- Average win/loss
- Trade duration

### Backtesting
- Historical data analysis
- Parameter optimization
- Performance evaluation

### Dashboard
- Real-time monitoring of active positions
- Performance metrics visualization
- Equity curve and trade distribution charts

## Getting Started
1. Initialize the bot: `trading-bot init`
2. Run backtests: `trading-bot backtest`
3. Start trading: `trading-bot run`
4. Launch dashboard: `trading-bot dashboard`

## Dependencies
- Python 3.8+
- Pandas for data manipulation
- Plotly and Streamlit for visualization
- Typer for CLI
- SQLite for data storage
- Rich for terminal output formatting
