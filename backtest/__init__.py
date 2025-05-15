"""
Backtest module for simulating trading strategies.

This module provides functionality for backtesting trading strategies
using historical data.
"""

from .engine import BacktestEngine, BacktestResult

__all__ = ["BacktestEngine", "BacktestResult"]
