"""
Unit tests for the strategy module.

This module contains tests for the LSOB strategy implementation.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from trading_bot.strategy.lsob import (
    LSOBDetector,
    LSOBSignal,
    SignalType,
    OrderBookSnapshot,
)
from trading_bot.exchange.bitunix import OrderBook, OrderBookEntry


@pytest.fixture
def sample_order_book():
    """Create a sample order book for testing."""
    bids = [
        OrderBookEntry(price=50000.0, quantity=1.5),
        OrderBookEntry(price=49900.0, quantity=2.0),
        OrderBookEntry(price=49800.0, quantity=3.0),
        OrderBookEntry(price=49700.0, quantity=4.0),
        OrderBookEntry(price=49600.0, quantity=5.0),
    ]
    asks = [
        OrderBookEntry(price=50100.0, quantity=1.0),
        OrderBookEntry(price=50200.0, quantity=2.0),
        OrderBookEntry(price=50300.0, quantity=3.0),
        OrderBookEntry(price=50400.0, quantity=4.0),
        OrderBookEntry(price=50500.0, quantity=5.0),
    ]
    return OrderBook(lastUpdateId=1234567890, bids=bids, asks=asks)


@pytest.fixture
def imbalanced_order_book_bullish():
    """Create a bullish imbalanced order book for testing."""
    bids = [
        OrderBookEntry(price=50000.0, quantity=5.0),  # Much more bid liquidity
        OrderBookEntry(price=49900.0, quantity=6.0),
        OrderBookEntry(price=49800.0, quantity=7.0),
        OrderBookEntry(price=49700.0, quantity=8.0),
        OrderBookEntry(price=49600.0, quantity=9.0),
    ]
    asks = [
        OrderBookEntry(price=50100.0, quantity=1.0),
        OrderBookEntry(price=50200.0, quantity=1.2),
        OrderBookEntry(price=50300.0, quantity=1.3),
        OrderBookEntry(price=50400.0, quantity=1.4),
        OrderBookEntry(price=50500.0, quantity=1.5),
    ]
    return OrderBook(lastUpdateId=1234567890, bids=bids, asks=asks)


@pytest.fixture
def imbalanced_order_book_bearish():
    """Create a bearish imbalanced order book for testing."""
    bids = [
        OrderBookEntry(price=50000.0, quantity=1.0),
        OrderBookEntry(price=49900.0, quantity=1.2),
        OrderBookEntry(price=49800.0, quantity=1.3),
        OrderBookEntry(price=49700.0, quantity=1.4),
        OrderBookEntry(price=49600.0, quantity=1.5),
    ]
    asks = [
        OrderBookEntry(price=50100.0, quantity=5.0),  # Much more ask liquidity
        OrderBookEntry(price=50200.0, quantity=6.0),
        OrderBookEntry(price=50300.0, quantity=7.0),
        OrderBookEntry(price=50400.0, quantity=8.0),
        OrderBookEntry(price=50500.0, quantity=9.0),
    ]
    return OrderBook(lastUpdateId=1234567890, bids=bids, asks=asks)


@pytest.fixture
def sweep_order_book_bullish(sample_order_book):
    """Create a bullish sweep order book for testing."""
    # First, create a normal order book
    normal_order_book = sample_order_book

    # Then, create an order book after a sweep (much less ask liquidity)
    bids = [
        OrderBookEntry(price=50000.0, quantity=1.5),
        OrderBookEntry(price=49900.0, quantity=2.0),
        OrderBookEntry(price=49800.0, quantity=3.0),
        OrderBookEntry(price=49700.0, quantity=4.0),
        OrderBookEntry(price=49600.0, quantity=5.0),
    ]
    asks = [
        OrderBookEntry(price=50100.0, quantity=0.1),  # Ask liquidity was swept
        OrderBookEntry(price=50200.0, quantity=0.2),
        OrderBookEntry(price=50300.0, quantity=3.0),
        OrderBookEntry(price=50400.0, quantity=4.0),
        OrderBookEntry(price=50500.0, quantity=5.0),
    ]
    swept_order_book = OrderBook(lastUpdateId=1234567891, bids=bids, asks=asks)

    return normal_order_book, swept_order_book


@pytest.fixture
def sweep_order_book_bearish(sample_order_book):
    """Create a bearish sweep order book for testing."""
    # First, create a normal order book
    normal_order_book = sample_order_book

    # Then, create an order book after a sweep (much less bid liquidity)
    bids = [
        OrderBookEntry(price=50000.0, quantity=0.1),  # Bid liquidity was swept
        OrderBookEntry(price=49900.0, quantity=0.2),
        OrderBookEntry(price=49800.0, quantity=3.0),
        OrderBookEntry(price=49700.0, quantity=4.0),
        OrderBookEntry(price=49600.0, quantity=5.0),
    ]
    asks = [
        OrderBookEntry(price=50100.0, quantity=1.0),
        OrderBookEntry(price=50200.0, quantity=2.0),
        OrderBookEntry(price=50300.0, quantity=3.0),
        OrderBookEntry(price=50400.0, quantity=4.0),
        OrderBookEntry(price=50500.0, quantity=5.0),
    ]
    swept_order_book = OrderBook(lastUpdateId=1234567891, bids=bids, asks=asks)

    return normal_order_book, swept_order_book


class TestOrderBookSnapshot:
    """Tests for the OrderBookSnapshot class."""

    def test_get_bid_liquidity(self, sample_order_book):
        """Test calculating bid liquidity."""
        snapshot = OrderBookSnapshot(sample_order_book)

        # Calculate expected liquidity
        expected_liquidity = sum(bid.quantity for bid in sample_order_book.bids[:10])

        assert snapshot.get_bid_liquidity() == expected_liquidity
        assert snapshot.get_bid_liquidity(levels=3) == sum(
            bid.quantity for bid in sample_order_book.bids[:3]
        )

    def test_get_ask_liquidity(self, sample_order_book):
        """Test calculating ask liquidity."""
        snapshot = OrderBookSnapshot(sample_order_book)

        # Calculate expected liquidity
        expected_liquidity = sum(ask.quantity for ask in sample_order_book.asks[:10])

        assert snapshot.get_ask_liquidity() == expected_liquidity
        assert snapshot.get_ask_liquidity(levels=3) == sum(
            ask.quantity for ask in sample_order_book.asks[:3]
        )

    def test_get_liquidity_imbalance(
        self,
        sample_order_book,
        imbalanced_order_book_bullish,
        imbalanced_order_book_bearish,
    ):
        """Test calculating liquidity imbalance."""
        # Balanced order book
        balanced_snapshot = OrderBookSnapshot(sample_order_book)
        imbalance = balanced_snapshot.get_liquidity_imbalance()

        # Should be close to 0 for a balanced book
        assert abs(imbalance) < 0.3

        # Bullish imbalanced order book
        bullish_snapshot = OrderBookSnapshot(imbalanced_order_book_bullish)
        bullish_imbalance = bullish_snapshot.get_liquidity_imbalance()

        # Should be positive for a bullish book
        assert bullish_imbalance > 0.3

        # Bearish imbalanced order book
        bearish_snapshot = OrderBookSnapshot(imbalanced_order_book_bearish)
        bearish_imbalance = bearish_snapshot.get_liquidity_imbalance()

        # Should be negative for a bearish book
        assert bearish_imbalance < -0.3

    def test_get_spread(self, sample_order_book):
        """Test calculating bid-ask spread."""
        snapshot = OrderBookSnapshot(sample_order_book)

        # Calculate expected spread
        best_bid = sample_order_book.bids[0].price
        best_ask = sample_order_book.asks[0].price
        expected_spread = (best_ask - best_bid) / best_bid

        assert snapshot.get_spread() == expected_spread


class TestLSOBDetector:
    """Tests for the LSOBDetector class."""

    @pytest.fixture
    def detector(self):
        """Create an LSOBDetector instance for testing."""
        return LSOBDetector(
            symbol="BTCUSDT",
            imbalance_threshold=0.3,
            sweep_detection_window=2,
            min_sweep_percentage=0.5,
            confidence_threshold=0.7,
        )

    def test_add_order_book(self, detector, sample_order_book):
        """Test adding an order book to the detector."""
        # Initially, history should be empty
        assert len(detector.history) == 0

        # Add an order book
        detector.add_order_book(sample_order_book)

        # History should now have one entry
        assert len(detector.history) == 1
        assert isinstance(detector.history[0], OrderBookSnapshot)

    def test_detect_signal_imbalance_bullish(
        self, detector, imbalanced_order_book_bullish
    ):
        """Test detecting a bullish signal based on imbalance."""
        # Add an imbalanced order book
        detector.add_order_book(imbalanced_order_book_bullish)

        # Need at least sweep_detection_window entries in history
        detector.add_order_book(imbalanced_order_book_bullish)

        # Detect signal
        signal = detector.detect_signal()

        # Should detect a LONG signal
        assert signal is not None
        assert signal.type == SignalType.LONG
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence >= detector.confidence_threshold
        assert signal.price == imbalanced_order_book_bullish.bids[0].price
        assert signal.target_price is not None
        assert signal.stop_loss is not None
        assert signal.target_price > signal.price  # Target should be higher for LONG
        assert signal.stop_loss < signal.price  # Stop loss should be lower for LONG

    def test_detect_signal_imbalance_bearish(
        self, detector, imbalanced_order_book_bearish
    ):
        """Test detecting a bearish signal based on imbalance."""
        # Add an imbalanced order book
        detector.add_order_book(imbalanced_order_book_bearish)

        # Need at least sweep_detection_window entries in history
        detector.add_order_book(imbalanced_order_book_bearish)

        # Detect signal
        signal = detector.detect_signal()

        # Should detect a SHORT signal
        assert signal is not None
        assert signal.type == SignalType.SHORT
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence >= detector.confidence_threshold
        assert signal.price == imbalanced_order_book_bearish.asks[0].price
        assert signal.target_price is not None
        assert signal.stop_loss is not None
        assert signal.target_price < signal.price  # Target should be lower for SHORT
        assert signal.stop_loss > signal.price  # Stop loss should be higher for SHORT

    def test_detect_signal_sweep_bullish(self, detector, sweep_order_book_bullish):
        """Test detecting a bullish signal based on sweep."""
        normal_order_book, swept_order_book = sweep_order_book_bullish

        # Add order books to create a sweep pattern
        detector.add_order_book(normal_order_book)
        detector.add_order_book(swept_order_book)

        # Detect signal
        signal = detector.detect_signal()

        # Should detect a LONG signal
        assert signal is not None
        assert signal.type == SignalType.LONG
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence >= detector.confidence_threshold
        assert signal.price == swept_order_book.bids[0].price
        assert signal.target_price is not None
        assert signal.stop_loss is not None

    def test_detect_signal_sweep_bearish(self, detector, sweep_order_book_bearish):
        """Test detecting a bearish signal based on sweep."""
        normal_order_book, swept_order_book = sweep_order_book_bearish

        # Add order books to create a sweep pattern
        detector.add_order_book(normal_order_book)
        detector.add_order_book(swept_order_book)

        # Detect signal
        signal = detector.detect_signal()

        # Should detect a SHORT signal
        assert signal is not None
        assert signal.type == SignalType.SHORT
        assert signal.symbol == "BTCUSDT"
        assert signal.confidence >= detector.confidence_threshold
        assert signal.price == swept_order_book.asks[0].price
        assert signal.target_price is not None
        assert signal.stop_loss is not None

    def test_detect_signal_no_signal(self, detector, sample_order_book):
        """Test detecting no signal with a balanced order book."""
        # Add a balanced order book
        detector.add_order_book(sample_order_book)

        # Need at least sweep_detection_window entries in history
        detector.add_order_book(sample_order_book)

        # Detect signal
        signal = detector.detect_signal()

        # Should not detect a signal
        assert signal is None

    def test_signal_risk_reward_ratio(self):
        """Test calculating risk-reward ratio for a signal."""
        # Create a LONG signal
        long_signal = LSOBSignal(
            type=SignalType.LONG,
            symbol="BTCUSDT",
            price=50000.0,
            confidence=0.8,
            timestamp=int(time.time() * 1000),
            target_price=55000.0,  # 10% profit target
            stop_loss=47500.0,  # 5% stop loss
        )

        # Calculate expected risk-reward ratio
        expected_ratio = (55000.0 - 50000.0) / (50000.0 - 47500.0)

        assert long_signal.risk_reward_ratio == expected_ratio

        # Create a SHORT signal
        short_signal = LSOBSignal(
            type=SignalType.SHORT,
            symbol="BTCUSDT",
            price=50000.0,
            confidence=0.8,
            timestamp=int(time.time() * 1000),
            target_price=45000.0,  # 10% profit target
            stop_loss=52500.0,  # 5% stop loss
        )

        # Calculate expected risk-reward ratio
        expected_ratio = (50000.0 - 45000.0) / (52500.0 - 50000.0)

        assert short_signal.risk_reward_ratio == expected_ratio
