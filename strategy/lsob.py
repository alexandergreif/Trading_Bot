"""
Liquidity Sweep Order Block (LSOB) strategy implementation.

This module implements the LSOB strategy, which detects imbalances in the order book
and identifies potential sweep patterns that may indicate future price movements.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Deque
from collections import deque
import time

from pydantic import BaseModel, Field

from trading_bot.exchange.bitunix import OrderBook, OrderBookEntry

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Type of trading signal."""

    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class LSOBSignal:
    """Trading signal generated by the LSOB detector."""

    type: SignalType
    symbol: str
    price: float
    confidence: float  # 0.0 to 1.0
    timestamp: int
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate the risk-reward ratio of the signal."""
        if self.target_price is None or self.stop_loss is None:
            return None

        if self.type == SignalType.LONG:
            reward = self.target_price - self.price
            risk = self.price - self.stop_loss
        else:  # SHORT
            reward = self.price - self.target_price
            risk = self.stop_loss - self.price

        if risk <= 0:
            return None

        return reward / risk


class OrderBookSnapshot:
    """A snapshot of the order book at a specific time."""

    def __init__(self, order_book: OrderBook, timestamp: Optional[int] = None):
        """
        Initialize an order book snapshot.

        Args:
            order_book: The order book data
            timestamp: Timestamp of the snapshot (default: current time)
        """
        self.bids = order_book.bids
        self.asks = order_book.asks
        self.last_update_id = order_book.last_update_id
        self.timestamp = timestamp or int(time.time() * 1000)

    def get_bid_liquidity(self, levels: int = 10) -> float:
        """
        Calculate the total bid liquidity for the top N levels.

        Args:
            levels: Number of price levels to include

        Returns:
            Total bid liquidity
        """
        return sum(bid.quantity for bid in self.bids[:levels])

    def get_ask_liquidity(self, levels: int = 10) -> float:
        """
        Calculate the total ask liquidity for the top N levels.

        Args:
            levels: Number of price levels to include

        Returns:
            Total ask liquidity
        """
        return sum(ask.quantity for ask in self.asks[:levels])

    def get_liquidity_imbalance(self, levels: int = 10) -> float:
        """
        Calculate the liquidity imbalance ratio.

        A positive value indicates more bid liquidity (bullish),
        a negative value indicates more ask liquidity (bearish).

        Args:
            levels: Number of price levels to include

        Returns:
            Liquidity imbalance ratio (-1.0 to 1.0)
        """
        bid_liquidity = self.get_bid_liquidity(levels)
        ask_liquidity = self.get_ask_liquidity(levels)
        total_liquidity = bid_liquidity + ask_liquidity

        if total_liquidity == 0:
            return 0.0

        return (bid_liquidity - ask_liquidity) / total_liquidity

    def get_spread(self) -> float:
        """
        Calculate the bid-ask spread.

        Returns:
            Bid-ask spread as a percentage
        """
        if not self.bids or not self.asks:
            return 0.0

        best_bid = self.bids[0].price
        best_ask = self.asks[0].price

        if best_bid <= 0:
            return 0.0

        return (best_ask - best_bid) / best_bid


class LSOBDetector:
    """
    Detector for Liquidity Sweep Order Block patterns.

    This class analyzes order book data to detect imbalances and sweep patterns
    that may indicate future price movements.
    """

    def __init__(
        self,
        symbol: str,
        imbalance_threshold: float = 0.3,
        sweep_detection_window: int = 5,
        min_sweep_percentage: float = 0.5,
        confidence_threshold: float = 0.7,
        history_size: int = 100,
    ):
        """
        Initialize the LSOB detector.

        Args:
            symbol: Trading pair symbol
            imbalance_threshold: Threshold for detecting significant imbalances (0.0 to 1.0)
            sweep_detection_window: Number of order book updates to consider for sweep detection
            min_sweep_percentage: Minimum percentage of liquidity that must be swept
            confidence_threshold: Minimum confidence level for generating signals
            history_size: Number of order book snapshots to keep in history
        """
        self.symbol = symbol
        self.imbalance_threshold = imbalance_threshold
        self.sweep_detection_window = sweep_detection_window
        self.min_sweep_percentage = min_sweep_percentage
        self.confidence_threshold = confidence_threshold

        self.history: Deque[OrderBookSnapshot] = deque(maxlen=history_size)
        self.last_signal_time = 0
        self.min_signal_interval = 60000  # Minimum time between signals (ms)

    def add_order_book(self, order_book: OrderBook) -> None:
        """
        Add a new order book snapshot to the history.

        Args:
            order_book: The order book data
        """
        snapshot = OrderBookSnapshot(order_book)
        self.history.append(snapshot)

    def detect_signal(self) -> Optional[LSOBSignal]:
        """
        Detect trading signals based on the order book history.

        Returns:
            A trading signal if a pattern is detected, None otherwise
        """
        if len(self.history) < self.sweep_detection_window:
            return None

        # Check if enough time has passed since the last signal
        current_time = int(time.time() * 1000)
        if current_time - self.last_signal_time < self.min_signal_interval:
            return None

        # Get the current and previous snapshots
        current = self.history[-1]
        previous = self.history[-self.sweep_detection_window]

        # Check for liquidity imbalance
        imbalance = current.get_liquidity_imbalance()
        if abs(imbalance) < self.imbalance_threshold:
            return None

        # Detect sweep patterns
        signal_type, confidence, price = self._detect_sweep(previous, current)

        if signal_type == SignalType.NEUTRAL or confidence < self.confidence_threshold:
            return None

        # Calculate target price and stop loss
        target_price, stop_loss = self._calculate_targets(signal_type, price, current)

        # Create and return the signal
        signal = LSOBSignal(
            type=signal_type,
            symbol=self.symbol,
            price=price,
            confidence=confidence,
            timestamp=current_time,
            target_price=target_price,
            stop_loss=stop_loss,
        )

        self.last_signal_time = current_time
        logger.info(
            f"Generated {signal_type.value} signal for {self.symbol} at {price} (confidence: {confidence:.2f})"
        )

        return signal

    def _detect_sweep(
        self, previous: OrderBookSnapshot, current: OrderBookSnapshot
    ) -> Tuple[SignalType, float, float]:
        """
        Detect sweep patterns by comparing previous and current order book snapshots.

        Args:
            previous: Previous order book snapshot
            current: Current order book snapshot

        Returns:
            Tuple of (signal type, confidence, price)
        """
        # Calculate liquidity changes
        prev_bid_liquidity = previous.get_bid_liquidity()
        prev_ask_liquidity = previous.get_ask_liquidity()

        curr_bid_liquidity = current.get_bid_liquidity()
        curr_ask_liquidity = current.get_ask_liquidity()

        bid_change = (
            (curr_bid_liquidity - prev_bid_liquidity) / prev_bid_liquidity
            if prev_bid_liquidity > 0
            else 0
        )
        ask_change = (
            (curr_ask_liquidity - prev_ask_liquidity) / prev_ask_liquidity
            if prev_ask_liquidity > 0
            else 0
        )

        # Check for significant liquidity sweeps
        if bid_change < -self.min_sweep_percentage:
            # Significant bid liquidity was swept (bearish)
            confidence = min(1.0, abs(bid_change) * 1.5)
            return SignalType.SHORT, confidence, current.asks[0].price

        if ask_change < -self.min_sweep_percentage:
            # Significant ask liquidity was swept (bullish)
            confidence = min(1.0, abs(ask_change) * 1.5)
            return SignalType.LONG, confidence, current.bids[0].price

        # Check for imbalance-based signals
        imbalance = current.get_liquidity_imbalance()
        if imbalance > self.imbalance_threshold:
            # More bids than asks (bullish)
            confidence = min(1.0, imbalance * 1.2)
            return SignalType.LONG, confidence, current.bids[0].price

        if imbalance < -self.imbalance_threshold:
            # More asks than bids (bearish)
            confidence = min(1.0, abs(imbalance) * 1.2)
            return SignalType.SHORT, confidence, current.asks[0].price

        return SignalType.NEUTRAL, 0.0, 0.0

    def _calculate_targets(
        self, signal_type: SignalType, price: float, snapshot: OrderBookSnapshot
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate target price and stop loss levels based on the signal type and current market conditions.

        Args:
            signal_type: Type of trading signal
            price: Entry price
            snapshot: Current order book snapshot

        Returns:
            Tuple of (target price, stop loss)
        """
        if signal_type == SignalType.NEUTRAL:
            return None, None

        # Calculate average price gap in the order book
        if signal_type == SignalType.LONG:
            # For long positions, look at ask side
            price_gaps = [
                snapshot.asks[i + 1].price - snapshot.asks[i].price
                for i in range(min(5, len(snapshot.asks) - 1))
            ]
        else:
            # For short positions, look at bid side
            price_gaps = [
                snapshot.bids[i].price - snapshot.bids[i + 1].price
                for i in range(min(5, len(snapshot.bids) - 1))
            ]

        if not price_gaps:
            # Fallback to percentage-based targets
            if signal_type == SignalType.LONG:
                target_price = price * 1.02  # 2% profit target
                stop_loss = price * 0.99  # 1% stop loss
            else:
                target_price = price * 0.98  # 2% profit target
                stop_loss = price * 1.01  # 1% stop loss
            return target_price, stop_loss

        avg_gap = sum(price_gaps) / len(price_gaps)

        # Set targets based on average price gap
        if signal_type == SignalType.LONG:
            target_price = price + (avg_gap * 3)  # Target 3x the average gap
            stop_loss = price - (avg_gap * 1.5)  # Stop loss at 1.5x the average gap
        else:
            target_price = price - (avg_gap * 3)  # Target 3x the average gap
            stop_loss = price + (avg_gap * 1.5)  # Stop loss at 1.5x the average gap

        return target_price, stop_loss
