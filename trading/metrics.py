"""
KPI tracking and performance metrics.

This module provides functionality for tracking key performance indicators (KPIs)
and calculating performance metrics for trading strategies.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque
from enum import Enum
import json
import statistics
from datetime import datetime, timedelta

from trading_bot.trading.position import TradePosition, PositionStatus

logger = logging.getLogger(__name__)


@dataclass
class TradeMetrics:
    """Metrics for a single trade."""

    position_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percentage: float
    entry_time: int
    exit_time: int
    duration_ms: int

    @property
    def duration_minutes(self) -> float:
        """Get the trade duration in minutes."""
        return self.duration_ms / (1000 * 60)

    @property
    def is_win(self) -> bool:
        """Check if the trade was profitable."""
        return self.pnl > 0

    @classmethod
    def from_position(cls, position: TradePosition) -> Optional["TradeMetrics"]:
        """
        Create trade metrics from a position.

        Args:
            position: Closed trading position

        Returns:
            Trade metrics or None if position is not closed
        """
        if position.status != PositionStatus.CLOSED:
            return None

        if position.exit_price is None or position.exit_time is None:
            return None

        return cls(
            position_id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            quantity=position.quantity,
            pnl=position.pnl or 0.0,
            pnl_percentage=position.pnl_percentage or 0.0,
            entry_time=position.entry_time,
            exit_time=position.exit_time,
            duration_ms=position.exit_time - position.entry_time,
        )


@dataclass
class PerformanceMetrics:
    """Overall performance metrics."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_trade_duration_minutes: float = 0.0

    def update(self, trades: List[TradeMetrics]) -> None:
        """
        Update performance metrics based on a list of trades.

        Args:
            trades: List of trade metrics
        """
        if not trades:
            return

        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.is_win)
        self.losing_trades = self.total_trades - self.winning_trades
        self.total_pnl = sum(t.pnl for t in trades)

        # Calculate win rate
        self.win_rate = (
            self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        )

        # Calculate profit factor
        total_profit = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        self.profit_factor = (
            total_profit / total_loss if total_loss > 0 else float("inf")
        )

        # Calculate average win/loss
        winning_trades = [t for t in trades if t.is_win]
        losing_trades = [t for t in trades if not t.is_win]

        self.average_win = (
            statistics.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
        )
        self.average_loss = (
            statistics.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
        )

        # Calculate largest win/loss
        self.largest_win = max([t.pnl for t in trades if t.is_win], default=0.0)
        self.largest_loss = min([t.pnl for t in trades if not t.is_win], default=0.0)

        # Calculate average trade duration
        self.average_trade_duration_minutes = statistics.mean(
            [t.duration_minutes for t in trades]
        )

        # Calculate max drawdown
        self._calculate_max_drawdown(trades)

    def _calculate_max_drawdown(self, trades: List[TradeMetrics]) -> None:
        """
        Calculate the maximum drawdown from a list of trades.

        Args:
            trades: List of trade metrics
        """
        if not trades:
            self.max_drawdown = 0.0
            return

        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)

        # Calculate cumulative PnL
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

        self.max_drawdown = max_drawdown


class TimeFrame(str, Enum):
    """Time frames for KPI tracking."""

    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    ALL_TIME = "ALL_TIME"


class KPITracker:
    """
    Tracker for key performance indicators.

    This class tracks trading performance metrics over different time frames
    and provides functionality for auto-tuning strategy parameters.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_trades: int = 1000,
        auto_tune_enabled: bool = True,
        auto_tune_threshold: float = 0.4,
    ):
        """
        Initialize the KPI tracker.

        Args:
            db_path: Path to the SQLite database file
            max_trades: Maximum number of trades to keep in memory
            auto_tune_enabled: Whether auto-tuning is enabled
            auto_tune_threshold: Win rate threshold for auto-tuning
        """
        self.db_path = db_path
        self.max_trades = max_trades
        self.auto_tune_enabled = auto_tune_enabled
        self.auto_tune_threshold = auto_tune_threshold

        self.trades: List[TradeMetrics] = []
        self.performance_metrics: Dict[TimeFrame, PerformanceMetrics] = {
            tf: PerformanceMetrics() for tf in TimeFrame
        }

    def add_trade(self, position: TradePosition) -> None:
        """
        Add a closed trade to the tracker.

        Args:
            position: Closed trading position
        """
        if position.status != PositionStatus.CLOSED:
            return

        trade_metrics = TradeMetrics.from_position(position)
        if trade_metrics is None:
            return

        self.trades.append(trade_metrics)

        # Limit the number of trades in memory
        if len(self.trades) > self.max_trades:
            self.trades = self.trades[-self.max_trades :]

        # Update performance metrics
        self._update_performance_metrics()

        # Save trade to database
        self._save_trade_to_db(trade_metrics)

        # Check if auto-tuning is needed
        if self.auto_tune_enabled:
            self._check_auto_tune()

    def _update_performance_metrics(self) -> None:
        """Update performance metrics for all time frames."""
        now = datetime.now()

        # Update all-time metrics
        self.performance_metrics[TimeFrame.ALL_TIME].update(self.trades)

        # Update daily metrics
        today_start = datetime(now.year, now.month, now.day).timestamp() * 1000
        daily_trades = [t for t in self.trades if t.exit_time >= today_start]
        self.performance_metrics[TimeFrame.DAILY].update(daily_trades)

        # Update weekly metrics
        week_start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp() * 1000
        weekly_trades = [t for t in self.trades if t.exit_time >= week_start]
        self.performance_metrics[TimeFrame.WEEKLY].update(weekly_trades)

        # Update monthly metrics
        month_start = datetime(now.year, now.month, 1).timestamp() * 1000
        monthly_trades = [t for t in self.trades if t.exit_time >= month_start]
        self.performance_metrics[TimeFrame.MONTHLY].update(monthly_trades)

    def _save_trade_to_db(self, trade: TradeMetrics) -> None:
        """
        Save a trade to the database.

        Args:
            trade: Trade metrics to save
        """
        if self.db_path is None:
            return

        # TODO: Implement database saving
        pass

    def _check_auto_tune(self) -> None:
        """Check if auto-tuning is needed based on recent performance."""
        # Get daily and weekly metrics
        daily_metrics = self.performance_metrics[TimeFrame.DAILY]
        weekly_metrics = self.performance_metrics[TimeFrame.WEEKLY]

        # Check if we have enough trades
        if daily_metrics.total_trades < 5 or weekly_metrics.total_trades < 10:
            return

        # Check if win rate is below threshold
        if daily_metrics.win_rate < self.auto_tune_threshold:
            logger.warning(
                f"Daily win rate ({daily_metrics.win_rate:.2f}) is below threshold "
                f"({self.auto_tune_threshold:.2f}). Auto-tuning recommended."
            )
            # TODO: Implement auto-tuning logic

    def get_metrics(
        self, time_frame: TimeFrame = TimeFrame.ALL_TIME
    ) -> PerformanceMetrics:
        """
        Get performance metrics for a specific time frame.

        Args:
            time_frame: Time frame to get metrics for

        Returns:
            Performance metrics for the specified time frame
        """
        return self.performance_metrics[time_frame]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics for all time frames.

        Returns:
            Dictionary with performance metrics for all time frames
        """
        return {
            tf.value: {
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_pnl": metrics.total_pnl,
                "max_drawdown": metrics.max_drawdown,
            }
            for tf, metrics in self.performance_metrics.items()
        }

    def get_rolling_win_rate(self, window: int = 20) -> float:
        """
        Calculate the rolling win rate for the last N trades.

        Args:
            window: Number of trades to include in the calculation

        Returns:
            Rolling win rate (0.0 to 1.0)
        """
        if not self.trades or window <= 0:
            return 0.0

        recent_trades = self.trades[-min(window, len(self.trades)) :]
        wins = sum(1 for t in recent_trades if t.is_win)

        return wins / len(recent_trades)

    def get_symbol_performance(self, symbol: str) -> PerformanceMetrics:
        """
        Get performance metrics for a specific symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Performance metrics for the specified symbol
        """
        symbol_trades = [t for t in self.trades if t.symbol == symbol]
        metrics = PerformanceMetrics()
        metrics.update(symbol_trades)

        return metrics

    def export_trades_to_json(self, file_path: str) -> bool:
        """
        Export all trades to a JSON file.

        Args:
            file_path: Path to the output JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            trades_data = [
                {
                    "position_id": t.position_id,
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
                    "is_win": t.is_win,
                }
                for t in self.trades
            ]

            with open(file_path, "w") as f:
                json.dump(trades_data, f, indent=2)

            logger.info(f"Exported {len(self.trades)} trades to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export trades to {file_path}: {str(e)}")
            return False
