"""
Backtesting engine for simulating trading strategies.

This module provides functionality for backtesting trading strategies
using historical data.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import json
import csv
import os

import pandas as pd
import numpy as np

from trading_bot.strategy.lsob import LSOBDetector, LSOBSignal, SignalType
from trading_bot.trading.position import TradePosition, PositionStatus
from trading_bot.trading.metrics import TradeMetrics, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results of a backtest."""

    trades: List[TradePosition]
    metrics: PerformanceMetrics
    equity_curve: List[Tuple[int, float]]  # List of (timestamp, equity) tuples
    parameters: Dict[str, Any]
    start_time: int
    end_time: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert the backtest result to a dictionary."""
        return {
            "trades": [
                {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl": t.pnl,
                    "pnl_percentage": t.pnl_percentage,
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "duration_ms": t.duration_ms,
                    "status": t.status,
                    "target_price": t.target_price,
                    "stop_loss": t.stop_loss,
                }
                for t in self.trades
            ],
            "metrics": {
                "total_trades": self.metrics.total_trades,
                "winning_trades": self.metrics.winning_trades,
                "losing_trades": self.metrics.losing_trades,
                "total_pnl": self.metrics.total_pnl,
                "max_drawdown": self.metrics.max_drawdown,
                "win_rate": self.metrics.win_rate,
                "profit_factor": self.metrics.profit_factor,
                "average_win": self.metrics.average_win,
                "average_loss": self.metrics.average_loss,
                "largest_win": self.metrics.largest_win,
                "largest_loss": self.metrics.largest_loss,
                "average_trade_duration_minutes": self.metrics.average_trade_duration_minutes,
            },
            "equity_curve": self.equity_curve,
            "parameters": self.parameters,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    def save_to_json(self, file_path: str) -> None:
        """
        Save the backtest result to a JSON file.

        Args:
            file_path: Path to the output JSON file
        """
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_trades_to_csv(self, file_path: str) -> None:
        """
        Save the trades to a CSV file.

        Args:
            file_path: Path to the output CSV file
        """
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "ID",
                    "Symbol",
                    "Side",
                    "Entry Price",
                    "Exit Price",
                    "Quantity",
                    "PnL",
                    "PnL %",
                    "Entry Time",
                    "Exit Time",
                    "Duration (ms)",
                    "Status",
                    "Target Price",
                    "Stop Loss",
                ]
            )

            for trade in self.trades:
                writer.writerow(
                    [
                        trade.id,
                        trade.symbol,
                        trade.side,
                        trade.entry_price,
                        trade.exit_price,
                        trade.quantity,
                        trade.pnl,
                        trade.pnl_percentage,
                        trade.entry_time,
                        trade.exit_time,
                        trade.duration_ms,
                        trade.status,
                        trade.target_price,
                        trade.stop_loss,
                    ]
                )


class BacktestEngine:
    """
    Engine for backtesting trading strategies.

    This class provides functionality for backtesting trading strategies
    using historical data.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.01,
        commission_rate: float = 0.0004,  # 0.04% per trade
        slippage: float = 0.0001,  # 0.01% slippage
    ):
        """
        Initialize the backtest engine.

        Args:
            initial_balance: Initial account balance
            risk_per_trade: Maximum risk per trade as a fraction of account balance
            commission_rate: Commission rate per trade
            slippage: Slippage as a fraction of price
        """
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission_rate = commission_rate
        self.slippage = slippage

    def _calculate_position_size(
        self, balance: float, entry_price: float, stop_loss: float
    ) -> float:
        """
        Calculate the position size based on risk parameters.

        Args:
            balance: Account balance
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size in base currency
        """
        # Calculate risk amount
        risk_amount = balance * self.risk_per_trade

        # Calculate position size
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            logger.warning(f"Invalid price risk: {price_risk}")
            return 0.0

        position_size = risk_amount / price_risk

        # Round to appropriate precision
        precision = 5
        position_size = round(position_size, precision)

        return position_size

    def _calculate_trade_pnl(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
    ) -> Tuple[float, float]:
        """
        Calculate the profit/loss for a trade.

        Args:
            side: Order side (BUY or SELL)
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size

        Returns:
            Tuple of (PnL in quote currency, PnL percentage)
        """
        # Apply commission to entry and exit
        entry_commission = entry_price * quantity * self.commission_rate
        exit_commission = exit_price * quantity * self.commission_rate

        if side == "BUY":  # Long position
            pnl = (
                (exit_price - entry_price) * quantity
                - entry_commission
                - exit_commission
            )
            pnl_percentage = (exit_price / entry_price - 1) * 100
        else:  # Short position
            pnl = (
                (entry_price - exit_price) * quantity
                - entry_commission
                - exit_commission
            )
            pnl_percentage = (entry_price / exit_price - 1) * 100

        return pnl, pnl_percentage

    def _apply_slippage(self, price: float, side: str) -> float:
        """
        Apply slippage to a price.

        Args:
            price: Original price
            side: Order side (BUY or SELL)

        Returns:
            Price with slippage applied
        """
        if side == "BUY":
            # For buy orders, slippage increases the price
            return price * (1 + self.slippage)
        else:
            # For sell orders, slippage decreases the price
            return price * (1 - self.slippage)

    def _calculate_performance_metrics(
        self, trades: List[TradePosition]
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics from trades.

        Args:
            trades: List of trades

        Returns:
            Performance metrics
        """
        if not trades:
            return PerformanceMetrics()

        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        total_pnl = sum(t.pnl for t in trades)

        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Calculate profit factor
        total_profit = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Calculate average win/loss
        winning_trades_list = [t for t in trades if t.pnl > 0]
        losing_trades_list = [t for t in trades if t.pnl < 0]

        average_win = (
            sum(t.pnl for t in winning_trades_list) / len(winning_trades_list)
            if winning_trades_list
            else 0.0
        )
        average_loss = (
            sum(t.pnl for t in losing_trades_list) / len(losing_trades_list)
            if losing_trades_list
            else 0.0
        )

        # Calculate largest win/loss
        largest_win = max([t.pnl for t in trades if t.pnl > 0], default=0.0)
        largest_loss = min([t.pnl for t in trades if t.pnl < 0], default=0.0)

        # Calculate average trade duration
        durations = [
            t.duration_ms / (1000 * 60) for t in trades if t.duration_ms is not None
        ]
        average_trade_duration_minutes = (
            sum(durations) / len(durations) if durations else 0.0
        )

        # Calculate max drawdown
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        cumulative_pnl = 0.0
        peak = 0.0
        drawdown = 0.0
        max_drawdown = 0.0

        for trade in sorted_trades:
            cumulative_pnl += trade.pnl

            if cumulative_pnl > peak:
                peak = cumulative_pnl
                drawdown = 0.0
            else:
                drawdown = peak - cumulative_pnl

            max_drawdown = max(max_drawdown, drawdown)

        metrics = PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            average_trade_duration_minutes=average_trade_duration_minutes,
        )

        return metrics

    def backtest_lsob_strategy(
        self,
        data: pd.DataFrame,
        symbol: str,
        imbalance_threshold: float = 0.3,
        sweep_detection_window: int = 5,
        min_sweep_percentage: float = 0.5,
        confidence_threshold: float = 0.7,
    ) -> BacktestResult:
        """
        Backtest the LSOB strategy.

        Args:
            data: DataFrame with historical data (must have columns: timestamp, bid_price, ask_price, bid_size, ask_size)
            symbol: Trading pair symbol
            imbalance_threshold: Threshold for detecting significant imbalances
            sweep_detection_window: Number of order book updates to consider for sweep detection
            min_sweep_percentage: Minimum percentage of liquidity that must be swept
            confidence_threshold: Minimum confidence level for generating signals

        Returns:
            Backtest result
        """
        # Initialize strategy
        lsob_detector = LSOBDetector(
            symbol=symbol,
            imbalance_threshold=imbalance_threshold,
            sweep_detection_window=sweep_detection_window,
            min_sweep_percentage=min_sweep_percentage,
            confidence_threshold=confidence_threshold,
        )

        # Initialize variables
        balance = self.initial_balance
        trades = []
        equity_curve = [(data.iloc[0]["timestamp"], balance)]
        active_position = None
        next_position_id = 1

        # Iterate through data
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = row["timestamp"]
            bid_price = row["bid_price"]
            ask_price = row["ask_price"]
            bid_size = row["bid_size"]
            ask_size = row["ask_size"]

            # Create order book snapshot
            order_book = self._create_order_book_from_row(row)
            lsob_detector.add_order_book(order_book)

            # Check for signals
            signal = lsob_detector.detect_signal()

            # Process active position
            if active_position is not None:
                # Check if stop loss or target price is hit
                if active_position.side == "BUY":  # Long position
                    if bid_price <= active_position.stop_loss:
                        # Stop loss hit
                        exit_price = self._apply_slippage(
                            active_position.stop_loss, "SELL"
                        )
                        self._close_position(
                            active_position, exit_price, timestamp, "stop_loss"
                        )
                        active_position = None
                        balance += active_position.pnl
                        equity_curve.append((timestamp, balance))
                    elif bid_price >= active_position.target_price:
                        # Target price hit
                        exit_price = self._apply_slippage(
                            active_position.target_price, "SELL"
                        )
                        self._close_position(
                            active_position, exit_price, timestamp, "target"
                        )
                        active_position = None
                        balance += active_position.pnl
                        equity_curve.append((timestamp, balance))
                else:  # Short position
                    if ask_price >= active_position.stop_loss:
                        # Stop loss hit
                        exit_price = self._apply_slippage(
                            active_position.stop_loss, "BUY"
                        )
                        self._close_position(
                            active_position, exit_price, timestamp, "stop_loss"
                        )
                        active_position = None
                        balance += active_position.pnl
                        equity_curve.append((timestamp, balance))
                    elif ask_price <= active_position.target_price:
                        # Target price hit
                        exit_price = self._apply_slippage(
                            active_position.target_price, "BUY"
                        )
                        self._close_position(
                            active_position, exit_price, timestamp, "target"
                        )
                        active_position = None
                        balance += active_position.pnl
                        equity_curve.append((timestamp, balance))

            # Process signal
            if signal is not None and active_position is None:
                # Map signal type to order side
                side = "BUY" if signal.type == SignalType.LONG else "SELL"

                # Calculate position size
                quantity = self._calculate_position_size(
                    balance, signal.price, signal.stop_loss
                )

                if quantity <= 0:
                    logger.warning(f"Invalid position size: {quantity}")
                    continue

                # Apply slippage to entry price
                entry_price = self._apply_slippage(signal.price, side)

                # Create position
                position_id = f"P{next_position_id}"
                next_position_id += 1

                active_position = TradePosition(
                    id=position_id,
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    quantity=quantity,
                    entry_time=timestamp,
                    target_price=signal.target_price,
                    stop_loss=signal.stop_loss,
                    status=PositionStatus.OPEN,
                )

                logger.info(
                    f"Opened position {position_id}: {side} {quantity} {symbol} "
                    f"at {entry_price} (target: {signal.target_price}, stop: {signal.stop_loss})"
                )

        # Close any remaining position
        if active_position is not None:
            # Use the last price as exit price
            if active_position.side == "BUY":
                exit_price = self._apply_slippage(bid_price, "SELL")
            else:
                exit_price = self._apply_slippage(ask_price, "BUY")

            self._close_position(
                active_position, exit_price, data.iloc[-1]["timestamp"], "end_of_data"
            )
            trades.append(active_position)
            balance += active_position.pnl
            equity_curve.append((data.iloc[-1]["timestamp"], balance))

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(trades)

        # Create backtest result
        result = BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            parameters={
                "symbol": symbol,
                "imbalance_threshold": imbalance_threshold,
                "sweep_detection_window": sweep_detection_window,
                "min_sweep_percentage": min_sweep_percentage,
                "confidence_threshold": confidence_threshold,
                "initial_balance": self.initial_balance,
                "risk_per_trade": self.risk_per_trade,
                "commission_rate": self.commission_rate,
                "slippage": self.slippage,
            },
            start_time=data.iloc[0]["timestamp"],
            end_time=data.iloc[-1]["timestamp"],
        )

        return result

    def _create_order_book_from_row(self, row: pd.Series) -> Any:
        """
        Create an order book object from a DataFrame row.

        Args:
            row: DataFrame row with order book data

        Returns:
            Order book object
        """
        # TODO: Implement this based on the actual data format
        # This is a placeholder implementation
        from trading_bot.exchange.bitunix import OrderBook, OrderBookEntry

        # Create bid and ask entries
        bids = [OrderBookEntry(price=row["bid_price"], quantity=row["bid_size"])]
        asks = [OrderBookEntry(price=row["ask_price"], quantity=row["ask_size"])]

        # Create order book
        order_book = OrderBook(
            lastUpdateId=int(row["timestamp"]),
            bids=bids,
            asks=asks,
        )

        return order_book

    def _close_position(
        self,
        position: TradePosition,
        exit_price: float,
        exit_time: int,
        reason: str,
    ) -> None:
        """
        Close a position.

        Args:
            position: Position to close
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for closing the position
        """
        position.exit_price = exit_price
        position.exit_time = exit_time
        position.duration_ms = exit_time - position.entry_time
        position.status = PositionStatus.CLOSED

        # Calculate PnL
        position.pnl, position.pnl_percentage = self._calculate_trade_pnl(
            position.side,
            position.entry_price,
            exit_price,
            position.quantity,
        )

        logger.info(
            f"Closed position {position.id} ({reason}): {position.side} {position.quantity} "
            f"{position.symbol} at {exit_price} (PnL: {position.pnl:.2f}, {position.pnl_percentage:.2f}%)"
        )

    def parameter_sweep(
        self,
        data: pd.DataFrame,
        symbol: str,
        imbalance_thresholds: List[float],
        sweep_detection_windows: List[int],
        min_sweep_percentages: List[float],
        confidence_thresholds: List[float],
        output_dir: str,
    ) -> List[BacktestResult]:
        """
        Perform a parameter sweep to find the best parameters.

        Args:
            data: DataFrame with historical data
            symbol: Trading pair symbol
            imbalance_thresholds: List of imbalance thresholds to test
            sweep_detection_windows: List of sweep detection windows to test
            min_sweep_percentages: List of minimum sweep percentages to test
            confidence_thresholds: List of confidence thresholds to test
            output_dir: Directory to save results

        Returns:
            List of backtest results
        """
        results = []

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Iterate through all parameter combinations
        for imbalance_threshold in imbalance_thresholds:
            for sweep_detection_window in sweep_detection_windows:
                for min_sweep_percentage in min_sweep_percentages:
                    for confidence_threshold in confidence_thresholds:
                        # Run backtest with current parameters
                        result = self.backtest_lsob_strategy(
                            data=data,
                            symbol=symbol,
                            imbalance_threshold=imbalance_threshold,
                            sweep_detection_window=sweep_detection_window,
                            min_sweep_percentage=min_sweep_percentage,
                            confidence_threshold=confidence_threshold,
                        )

                        results.append(result)

                        # Save result to file
                        file_name = (
                            f"backtest_{symbol}_"
                            f"imb{imbalance_threshold:.2f}_"
                            f"win{sweep_detection_window}_"
                            f"swp{min_sweep_percentage:.2f}_"
                            f"conf{confidence_threshold:.2f}.json"
                        )
                        file_path = os.path.join(output_dir, file_name)
                        result.save_to_json(file_path)

                        logger.info(
                            f"Completed backtest with parameters: "
                            f"imbalance_threshold={imbalance_threshold}, "
                            f"sweep_detection_window={sweep_detection_window}, "
                            f"min_sweep_percentage={min_sweep_percentage}, "
                            f"confidence_threshold={confidence_threshold} "
                            f"(PnL: {result.metrics.total_pnl:.2f}, Win Rate: {result.metrics.win_rate:.2f})"
                        )

        # Sort results by total PnL
        results.sort(key=lambda r: r.metrics.total_pnl, reverse=True)

        # Save summary to file
        summary_path = os.path.join(output_dir, "summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Imbalance Threshold",
                    "Sweep Detection Window",
                    "Min Sweep Percentage",
                    "Confidence Threshold",
                    "Total PnL",
                    "Win Rate",
                    "Profit Factor",
                    "Max Drawdown",
                    "Total Trades",
                ]
            )

            for result in results:
                params = result.parameters
                metrics = result.metrics

                writer.writerow(
                    [
                        params["imbalance_threshold"],
                        params["sweep_detection_window"],
                        params["min_sweep_percentage"],
                        params["confidence_threshold"],
                        metrics.total_pnl,
                        metrics.win_rate,
                        metrics.profit_factor,
                        metrics.max_drawdown,
                        metrics.total_trades,
                    ]
                )

        logger.info(f"Parameter sweep completed. Results saved to {output_dir}")

        return results
