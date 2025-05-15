"""
Unit tests for the trading module.

This module contains tests for the position management and risk calculation.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

from trading_bot.trading.position import (
    PositionManager,
    TradePosition,
    PositionStatus,
)
from trading_bot.trading.metrics import (
    KPITracker,
    TradeMetrics,
    PerformanceMetrics,
    TimeFrame,
)
from trading_bot.strategy.lsob import LSOBSignal, SignalType
from trading_bot.exchange.bitunix import BitunixClient, OrderSide, OrderType


class TestTradePosition:
    """Tests for the TradePosition class."""

    def test_is_active(self):
        """Test checking if a position is active."""
        # Create a pending position
        pending_position = TradePosition(
            id="P1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
            status=PositionStatus.PENDING,
        )

        assert pending_position.is_active is True

        # Create an open position
        open_position = TradePosition(
            id="P2",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
            status=PositionStatus.OPEN,
        )

        assert open_position.is_active is True

        # Create a closed position
        closed_position = TradePosition(
            id="P3",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
            status=PositionStatus.CLOSED,
        )

        assert closed_position.is_active is False

    def test_is_long(self):
        """Test checking if a position is long."""
        # Create a long position
        long_position = TradePosition(
            id="P1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
        )

        assert long_position.is_long is True

        # Create a short position
        short_position = TradePosition(
            id="P2",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
        )

        assert short_position.is_long is False

    def test_calculate_pnl(self):
        """Test calculating profit/loss for a position."""
        # Create a long position
        long_position = TradePosition(
            id="P1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
        )

        # Calculate PnL at a higher price (profit)
        current_price = 55000.0
        pnl, pnl_percentage = long_position.calculate_pnl(current_price)

        expected_pnl = (
            current_price - long_position.entry_price
        ) * long_position.quantity
        expected_pnl_percentage = (current_price / long_position.entry_price - 1) * 100

        assert pnl == expected_pnl
        assert pnl_percentage == expected_pnl_percentage

        # Calculate PnL at a lower price (loss)
        current_price = 45000.0
        pnl, pnl_percentage = long_position.calculate_pnl(current_price)

        expected_pnl = (
            current_price - long_position.entry_price
        ) * long_position.quantity
        expected_pnl_percentage = (current_price / long_position.entry_price - 1) * 100

        assert pnl == expected_pnl
        assert pnl_percentage == expected_pnl_percentage

        # Create a short position
        short_position = TradePosition(
            id="P2",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
        )

        # Calculate PnL at a lower price (profit)
        current_price = 45000.0
        pnl, pnl_percentage = short_position.calculate_pnl(current_price)

        expected_pnl = (
            short_position.entry_price - current_price
        ) * short_position.quantity
        expected_pnl_percentage = (short_position.entry_price / current_price - 1) * 100

        assert pnl == expected_pnl
        assert pnl_percentage == expected_pnl_percentage

        # Calculate PnL at a higher price (loss)
        current_price = 55000.0
        pnl, pnl_percentage = short_position.calculate_pnl(current_price)

        expected_pnl = (
            short_position.entry_price - current_price
        ) * short_position.quantity
        expected_pnl_percentage = (short_position.entry_price / current_price - 1) * 100

        assert pnl == expected_pnl
        assert pnl_percentage == expected_pnl_percentage

    def test_update_status(self):
        """Test updating the status of a position."""
        # Create a position
        position = TradePosition(
            id="P1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
            status=PositionStatus.OPEN,
        )

        # Update status to closed with an exit price
        exit_price = 55000.0
        position.update_status(PositionStatus.CLOSED, exit_price)

        assert position.status == PositionStatus.CLOSED
        assert position.exit_price == exit_price
        assert position.exit_time is not None
        assert position.pnl is not None
        assert position.pnl_percentage is not None

        # Calculate expected PnL
        expected_pnl = (exit_price - position.entry_price) * position.quantity
        expected_pnl_percentage = (exit_price / position.entry_price - 1) * 100

        assert position.pnl == expected_pnl
        assert position.pnl_percentage == expected_pnl_percentage


class TestPositionManager:
    """Tests for the PositionManager class."""

    @pytest.fixture
    def client(self):
        """Create a BitunixClient instance for testing."""
        return AsyncMock(spec=BitunixClient)

    @pytest.fixture
    def position_manager(self, client):
        """Create a PositionManager instance for testing."""
        return PositionManager(
            client=client,
            risk_per_trade=0.01,
            max_positions=5,
            max_positions_per_symbol=1,
        )

    @pytest.mark.asyncio
    async def test_get_account_balance(self, position_manager, client):
        """Test getting the account balance."""
        # Mock the client.get_account_balance method
        mock_balance = MagicMock()
        mock_balance.asset = "USDT"
        mock_balance.wallet_balance = 10000.0
        client.get_account_balance.return_value = [mock_balance]

        # Get account balance
        balance = await position_manager.get_account_balance()

        assert balance == 10000.0
        client.get_account_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_active_positions(self, position_manager):
        """Test getting active positions."""
        # Add some positions
        position_manager.positions = {
            "P1": TradePosition(
                id="P1",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                entry_price=50000.0,
                quantity=0.1,
                entry_time=int(time.time() * 1000),
                status=PositionStatus.OPEN,
            ),
            "P2": TradePosition(
                id="P2",
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                entry_price=3000.0,
                quantity=1.0,
                entry_time=int(time.time() * 1000),
                status=PositionStatus.CLOSED,
            ),
            "P3": TradePosition(
                id="P3",
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                entry_price=51000.0,
                quantity=0.2,
                entry_time=int(time.time() * 1000),
                status=PositionStatus.PENDING,
            ),
        }

        # Get active positions
        active_positions = await position_manager.get_active_positions()

        assert len(active_positions) == 2
        assert active_positions[0].id == "P1"
        assert active_positions[1].id == "P3"

    @pytest.mark.asyncio
    async def test_get_active_positions_for_symbol(self, position_manager):
        """Test getting active positions for a specific symbol."""
        # Add some positions
        position_manager.positions = {
            "P1": TradePosition(
                id="P1",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                entry_price=50000.0,
                quantity=0.1,
                entry_time=int(time.time() * 1000),
                status=PositionStatus.OPEN,
            ),
            "P2": TradePosition(
                id="P2",
                symbol="ETHUSDT",
                side=OrderSide.SELL,
                entry_price=3000.0,
                quantity=1.0,
                entry_time=int(time.time() * 1000),
                status=PositionStatus.OPEN,
            ),
            "P3": TradePosition(
                id="P3",
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                entry_price=51000.0,
                quantity=0.2,
                entry_time=int(time.time() * 1000),
                status=PositionStatus.PENDING,
            ),
        }

        # Get active positions for BTCUSDT
        active_positions = await position_manager.get_active_positions_for_symbol(
            "BTCUSDT"
        )

        assert len(active_positions) == 2
        assert active_positions[0].id == "P1"
        assert active_positions[1].id == "P3"

        # Get active positions for ETHUSDT
        active_positions = await position_manager.get_active_positions_for_symbol(
            "ETHUSDT"
        )

        assert len(active_positions) == 1
        assert active_positions[0].id == "P2"

    @pytest.mark.asyncio
    async def test_can_open_position(self, position_manager):
        """Test checking if a new position can be opened."""
        # Initially, no positions
        position_manager.positions = {}

        # Should be able to open a position
        can_open = await position_manager.can_open_position("BTCUSDT")
        assert can_open is True

        # Add max positions for BTCUSDT
        position_manager.positions = {
            "P1": TradePosition(
                id="P1",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                entry_price=50000.0,
                quantity=0.1,
                entry_time=int(time.time() * 1000),
                status=PositionStatus.OPEN,
            ),
        }

        # Should not be able to open another BTCUSDT position
        can_open = await position_manager.can_open_position("BTCUSDT")
        assert can_open is False

        # Should be able to open an ETHUSDT position
        can_open = await position_manager.can_open_position("ETHUSDT")
        assert can_open is True

        # Add max positions overall
        position_manager.positions = {
            f"P{i}": TradePosition(
                id=f"P{i}",
                symbol=f"SYMBOL{i}",
                side=OrderSide.BUY,
                entry_price=50000.0,
                quantity=0.1,
                entry_time=int(time.time() * 1000),
                status=PositionStatus.OPEN,
            )
            for i in range(1, 6)  # 5 positions (max)
        }

        # Should not be able to open any more positions
        can_open = await position_manager.can_open_position("BTCUSDT")
        assert can_open is False

    @pytest.mark.asyncio
    async def test_calculate_position_size(self, position_manager):
        """Test calculating position size based on risk parameters."""
        # Mock the get_account_balance method
        with patch.object(
            position_manager, "get_account_balance", AsyncMock(return_value=10000.0)
        ):
            # Calculate position size
            symbol = "BTCUSDT"
            entry_price = 50000.0
            stop_loss = 49000.0

            position_size = await position_manager.calculate_position_size(
                symbol, entry_price, stop_loss
            )

            # Calculate expected position size
            risk_amount = 10000.0 * position_manager.risk_per_trade  # 100.0
            price_risk = abs(entry_price - stop_loss)  # 1000.0
            expected_position_size = risk_amount / price_risk  # 0.1

            assert position_size == expected_position_size

    @pytest.mark.asyncio
    async def test_open_position_from_signal(self, position_manager, client):
        """Test opening a position from a trading signal."""
        # Mock the necessary methods
        with (
            patch.object(
                position_manager, "can_open_position", AsyncMock(return_value=True)
            ),
            patch.object(
                position_manager, "calculate_position_size", AsyncMock(return_value=0.1)
            ),
        ):
            # Mock the client.create_order method
            mock_order = MagicMock()
            mock_order.order_id = "12345"
            mock_order.price = 50000.0
            client.create_order.return_value = mock_order

            # Create a signal
            signal = LSOBSignal(
                type=SignalType.LONG,
                symbol="BTCUSDT",
                price=50000.0,
                confidence=0.8,
                timestamp=int(time.time() * 1000),
                target_price=55000.0,
                stop_loss=47500.0,
            )

            # Open position from signal
            position_id = await position_manager.open_position_from_signal(signal)

            assert position_id is not None
            assert position_id in position_manager.positions
            assert position_manager.positions[position_id].symbol == "BTCUSDT"
            assert position_manager.positions[position_id].side == OrderSide.BUY
            assert position_manager.positions[position_id].entry_price == 50000.0
            assert position_manager.positions[position_id].quantity == 0.1
            assert position_manager.positions[position_id].status == PositionStatus.OPEN
            assert position_manager.positions[position_id].target_price == 55000.0
            assert position_manager.positions[position_id].stop_loss == 47500.0

            # Verify client.create_order was called
            client.create_order.assert_called_once_with(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=0.1,
            )

            # Verify _place_exit_orders was called
            client.create_order.reset_mock()

            # Mock the client.create_order method for exit orders
            client.create_order.side_effect = [
                MagicMock(order_id="SL12345"),
                MagicMock(order_id="TP12345"),
            ]

            # Call _place_exit_orders directly
            await position_manager._place_exit_orders(
                position_manager.positions[position_id]
            )

            # Verify client.create_order was called twice (for stop loss and take profit)
            assert client.create_order.call_count == 2

            # First call should be for stop loss
            args, kwargs = client.create_order.call_args_list[0]
            assert kwargs["symbol"] == "BTCUSDT"
            assert kwargs["side"] == OrderSide.SELL
            assert kwargs["order_type"] == OrderType.STOP_MARKET
            assert kwargs["quantity"] == 0.1
            assert kwargs["stop_price"] == 47500.0
            assert kwargs["reduce_only"] is True

            # Second call should be for take profit
            args, kwargs = client.create_order.call_args_list[1]
            assert kwargs["symbol"] == "BTCUSDT"
            assert kwargs["side"] == OrderSide.SELL
            assert kwargs["order_type"] == OrderType.TAKE_PROFIT_MARKET
            assert kwargs["quantity"] == 0.1
            assert kwargs["stop_price"] == 55000.0
            assert kwargs["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_close_position(self, position_manager, client):
        """Test closing a position manually."""
        # Create a position
        position = TradePosition(
            id="P1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000),
            status=PositionStatus.OPEN,
            tp_order_id="TP12345",
            sl_order_id="SL12345",
        )

        position_manager.positions = {"P1": position}

        # Mock the client.cancel_order and client.create_order methods
        client.cancel_order.return_value = MagicMock()

        mock_order = MagicMock()
        mock_order.price = 55000.0
        client.create_order.return_value = mock_order

        # Close the position
        result = await position_manager.close_position("P1", reason="test")

        assert result is True

        # Verify client.cancel_order was called twice (for stop loss and take profit)
        assert client.cancel_order.call_count == 2

        # Verify client.create_order was called once (for market order to close position)
        client.create_order.assert_called_once_with(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=0.1,
            reduce_only=True,
        )

        # Verify position status was updated
        assert position.status == PositionStatus.CLOSED
        assert position.exit_price == 55000.0
        assert position.exit_time is not None
        assert position.pnl is not None
        assert position.pnl_percentage is not None


class TestKPITracker:
    """Tests for the KPITracker class."""

    @pytest.fixture
    def kpi_tracker(self):
        """Create a KPITracker instance for testing."""
        return KPITracker(db_path=None, max_trades=100)

    @pytest.fixture
    def sample_position(self):
        """Create a sample closed position for testing."""
        position = TradePosition(
            id="P1",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            entry_price=50000.0,
            quantity=0.1,
            entry_time=int(time.time() * 1000) - 3600000,  # 1 hour ago
            status=PositionStatus.CLOSED,
            exit_price=55000.0,
            exit_time=int(time.time() * 1000),
            target_price=60000.0,
            stop_loss=45000.0,
        )

        # Calculate PnL
        position.pnl, position.pnl_percentage = position.calculate_pnl(
            position.exit_price
        )
        position.duration_ms = position.exit_time - position.entry_time

        return position

    def test_add_trade(self, kpi_tracker, sample_position):
        """Test adding a trade to the tracker."""
        # Initially, no trades
        assert len(kpi_tracker.trades) == 0

        # Add a trade
        kpi_tracker.add_trade(sample_position)

        # Should have one trade
        assert len(kpi_tracker.trades) == 1
        assert kpi_tracker.trades[0].position_id == sample_position.id
        assert kpi_tracker.trades[0].symbol == sample_position.symbol
        assert kpi_tracker.trades[0].side == sample_position.side
        assert kpi_tracker.trades[0].entry_price == sample_position.entry_price
        assert kpi_tracker.trades[0].exit_price == sample_position.exit_price
        assert kpi_tracker.trades[0].quantity == sample_position.quantity
        assert kpi_tracker.trades[0].pnl == sample_position.pnl
        assert kpi_tracker.trades[0].pnl_percentage == sample_position.pnl_percentage
        assert kpi_tracker.trades[0].entry_time == sample_position.entry_time
        assert kpi_tracker.trades[0].exit_time == sample_position.exit_time
        assert kpi_tracker.trades[0].duration_ms == sample_position.duration_ms

    def test_get_metrics(self, kpi_tracker, sample_position):
        """Test getting performance metrics."""
        # Add some trades
        for i in range(10):
            # Create a position with alternating profit/loss
            position = TradePosition(
                id=f"P{i + 1}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                entry_price=50000.0,
                quantity=0.1,
                entry_time=int(time.time() * 1000) - 3600000,  # 1 hour ago
                status=PositionStatus.CLOSED,
                exit_price=55000.0
                if i % 2 == 0
                else 45000.0,  # Alternating profit/loss
                exit_time=int(time.time() * 1000),
            )

            # Calculate PnL
            position.pnl, position.pnl_percentage = position.calculate_pnl(
                position.exit_price
            )
            position.duration_ms = position.exit_time - position.entry_time

            kpi_tracker.add_trade(position)

        # Get metrics
        metrics = kpi_tracker.get_metrics(TimeFrame.ALL_TIME)

        assert metrics.total_trades == 10
        assert metrics.winning_trades == 5
        assert metrics.losing_trades == 5
        assert metrics.win_rate == 0.5
        assert metrics.profit_factor > 0
        assert metrics.average_win > 0
        assert metrics.average_loss < 0
        assert metrics.largest_win > 0
        assert metrics.largest_loss < 0
        assert metrics.average_trade_duration_minutes > 0

    def test_get_rolling_win_rate(self, kpi_tracker):
        """Test calculating rolling win rate."""
        # Add some trades with alternating profit/loss
        for i in range(20):
            trade_metrics = TradeMetrics(
                position_id=f"P{i + 1}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                entry_price=50000.0,
                exit_price=55000.0
                if i % 2 == 0
                else 45000.0,  # Alternating profit/loss
                quantity=0.1,
                pnl=500.0 if i % 2 == 0 else -500.0,
                pnl_percentage=10.0 if i % 2 == 0 else -10.0,
                entry_time=int(time.time() * 1000) - 3600000,
                exit_time=int(time.time() * 1000),
                duration_ms=3600000,
            )

            kpi_tracker.trades.append(trade_metrics)

        # Get rolling win rate with window=10
        win_rate = kpi_tracker.get_rolling_win_rate(window=10)

        # Should be 0.5 (5 wins, 5 losses in the last 10 trades)
        assert win_rate == 0.5

        # Get rolling win rate with window=4
        win_rate = kpi_tracker.get_rolling_win_rate(window=4)

        # Should be 0.5 (2 wins, 2 losses in the last 4 trades)
        assert win_rate == 0.5

        # Add a few more winning trades
        for i in range(3):
            trade_metrics = TradeMetrics(
                position_id=f"P{i + 21}",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                entry_price=50000.0,
                exit_price=55000.0,
                quantity=0.1,
                pnl=500.0,
                pnl_percentage=10.0,
                entry_time=int(time.time() * 1000) - 3600000,
                exit_time=int(time.time() * 1000),
                duration_ms=3600000,
            )

            kpi_tracker.trades.append(trade_metrics)

        # Get rolling win rate with window=4
        win_rate = kpi_tracker.get_rolling_win_rate(window=4)

        # Should be 0.75 (3 wins, 1 loss in the last 4 trades)
        assert win_rate == 0.75
