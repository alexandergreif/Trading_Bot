"""
Unit tests for the UI module.

This module contains tests for the dashboard functionality.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock, AsyncMock
import time
from datetime import datetime

from trading_bot.ui.dashboard import (
    format_currency,
    format_percentage,
    create_equity_curve,
    create_win_loss_chart,
    create_symbol_performance_chart,
    create_win_rate_chart,
    create_trade_duration_chart,
    create_performance_metrics_table,
    create_active_positions_table,
    create_recent_trades_table,
    Dashboard,
)
from trading_bot.data.storage import Trade
from trading_bot.trading.metrics import TimeFrame


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    trades = [
        Trade(
            id=i + 1,
            position_id=f"P{i + 1}",
            symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT",
            side="BUY" if i % 3 == 0 else "SELL",
            entry_price=50000.0 if i % 2 == 0 else 3000.0,
            exit_price=55000.0 if i % 3 != 0 else 45000.0,  # 2/3 winning, 1/3 losing
            quantity=0.1 if i % 2 == 0 else 1.0,
            pnl=500.0 if i % 3 != 0 else -500.0,
            pnl_percentage=10.0 if i % 3 != 0 else -10.0,
            entry_time=1620000000000 + i * 3600000,
            exit_time=1620003600000 + i * 3600000,
            duration_ms=3600000,
            status="CLOSED",
        )
        for i in range(9)  # 9 trades: 6 winning, 3 losing
    ]
    return trades


@pytest.fixture
def sample_active_positions():
    """Create sample active positions for testing."""
    positions = [
        {
            "id": "P1",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "entry_price": 50000.0,
            "current_price": 55000.0,
            "quantity": 0.1,
            "pnl": 500.0,
            "pnl_percentage": 10.0,
        },
        {
            "id": "P2",
            "symbol": "ETHUSDT",
            "side": "SELL",
            "entry_price": 3000.0,
            "current_price": 2700.0,
            "quantity": 1.0,
            "pnl": 300.0,
            "pnl_percentage": 10.0,
        },
    ]
    return positions


class TestFormatters:
    """Tests for the formatting functions."""

    def test_format_currency(self):
        """Test formatting currency values."""
        assert format_currency(1234.56) == "$1234.56"
        assert format_currency(0.0) == "$0.00"
        assert format_currency(-1234.56) == "$-1234.56"

    def test_format_percentage(self):
        """Test formatting percentage values."""
        assert format_percentage(12.34) == "12.34%"
        assert format_percentage(0.0) == "0.00%"
        assert format_percentage(-12.34) == "-12.34%"


class TestChartCreation:
    """Tests for the chart creation functions."""

    def test_create_equity_curve_empty(self):
        """Test creating an equity curve with no trades."""
        fig = create_equity_curve([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Equity Curve"

    def test_create_equity_curve(self, sample_trades):
        """Test creating an equity curve with trades."""
        fig = create_equity_curve(sample_trades)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Equity Curve"
        assert len(fig.data) == 1
        assert fig.data[0].mode == "lines"

    def test_create_win_loss_chart_empty(self):
        """Test creating a win/loss chart with no trades."""
        fig = create_win_loss_chart([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Win/Loss Distribution"

    def test_create_win_loss_chart(self, sample_trades):
        """Test creating a win/loss chart with trades."""
        fig = create_win_loss_chart(sample_trades)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Win/Loss Distribution"
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"

    def test_create_symbol_performance_chart_empty(self):
        """Test creating a symbol performance chart with no trades."""
        fig = create_symbol_performance_chart([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Symbol Performance"

    def test_create_symbol_performance_chart(self, sample_trades):
        """Test creating a symbol performance chart with trades."""
        fig = create_symbol_performance_chart(sample_trades)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Symbol Performance"
        assert len(fig.data) == 1
        assert fig.data[0].type == "bar"

    def test_create_win_rate_chart_empty(self):
        """Test creating a win rate chart with no trades."""
        fig = create_win_rate_chart([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Rolling Win Rate (Window: 20)"

    def test_create_win_rate_chart_insufficient_trades(self, sample_trades):
        """Test creating a win rate chart with insufficient trades."""
        # Use a window larger than the number of trades
        fig = create_win_rate_chart(sample_trades[:3], window=5)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Rolling Win Rate (Window: 5)"

    def test_create_win_rate_chart(self, sample_trades):
        """Test creating a win rate chart with trades."""
        fig = create_win_rate_chart(sample_trades, window=3)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Rolling Win Rate (Window: 3)"
        assert len(fig.data) == 1
        assert fig.data[0].mode == "lines+markers"

    def test_create_trade_duration_chart_empty(self):
        """Test creating a trade duration chart with no trades."""
        fig = create_trade_duration_chart([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Trade Duration Distribution"

    def test_create_trade_duration_chart(self, sample_trades):
        """Test creating a trade duration chart with trades."""
        fig = create_trade_duration_chart(sample_trades)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Trade Duration Distribution"
        assert len(fig.data) == 1
        assert fig.data[0].type == "histogram"

    def test_create_performance_metrics_table(self):
        """Test creating a performance metrics table."""
        fig = create_performance_metrics_table(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_pnl=1000.0,
            max_drawdown=200.0,
            win_rate=0.6,
            profit_factor=2.0,
            average_win=200.0,
            average_loss=-100.0,
            largest_win=500.0,
            largest_loss=-200.0,
            average_trade_duration_minutes=60.0,
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Performance Metrics"
        assert len(fig.data) == 1
        assert fig.data[0].type == "table"
        assert len(fig.data[0].header.values) == 2
        assert len(fig.data[0].cells.values) == 2
        assert len(fig.data[0].cells.values[0]) == 12  # 12 metrics

    def test_create_active_positions_table_empty(self):
        """Test creating an active positions table with no positions."""
        fig = create_active_positions_table([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Active Positions"
        assert len(fig.data) == 1
        assert fig.data[0].type == "table"
        assert len(fig.data[0].header.values) == 8
        assert len(fig.data[0].cells.values) == 8
        assert len(fig.data[0].cells.values[0]) == 0  # No positions

    def test_create_active_positions_table(self, sample_active_positions):
        """Test creating an active positions table with positions."""
        fig = create_active_positions_table(sample_active_positions)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Active Positions"
        assert len(fig.data) == 1
        assert fig.data[0].type == "table"
        assert len(fig.data[0].header.values) == 8
        assert len(fig.data[0].cells.values) == 8
        assert len(fig.data[0].cells.values[0]) == 2  # 2 positions

    def test_create_recent_trades_table_empty(self):
        """Test creating a recent trades table with no trades."""
        fig = create_recent_trades_table([])
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Recent Trades"
        assert len(fig.data) == 1
        assert fig.data[0].type == "table"
        assert len(fig.data[0].header.values) == 9
        assert len(fig.data[0].cells.values) == 9
        assert len(fig.data[0].cells.values[0]) == 0  # No trades

    def test_create_recent_trades_table(self, sample_trades):
        """Test creating a recent trades table with trades."""
        fig = create_recent_trades_table(sample_trades, limit=5)
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Recent Trades"
        assert len(fig.data) == 1
        assert fig.data[0].type == "table"
        assert len(fig.data[0].header.values) == 9
        assert len(fig.data[0].cells.values) == 9
        assert len(fig.data[0].cells.values[0]) == 5  # 5 trades (limit)


class TestDashboard:
    """Tests for the Dashboard class."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock DatabaseManager."""
        mock_db = MagicMock()
        mock_db.get_trades_by_date_range.return_value = []
        return mock_db

    @pytest.fixture
    def mock_client(self):
        """Create a mock BitunixClient."""
        mock_client = AsyncMock()
        mock_client.get_positions.return_value = []
        return mock_client

    @pytest.fixture
    def dashboard(self, mock_db_manager, mock_client):
        """Create a Dashboard instance for testing."""
        with (
            patch(
                "trading_bot.ui.dashboard.DatabaseManager", return_value=mock_db_manager
            ),
            patch("trading_bot.ui.dashboard.BitunixClient", return_value=mock_client),
        ):
            dashboard = Dashboard(
                db_path="test.db",
                api_key="test_key",
                api_secret="test_secret",
                refresh_interval=60,
            )
            yield dashboard

    def test_init(self, dashboard, mock_db_manager, mock_client):
        """Test initializing the dashboard."""
        assert dashboard.db_path == "test.db"
        assert dashboard.api_key == "test_key"
        assert dashboard.api_secret == "test_secret"
        assert dashboard.refresh_interval == 60
        assert dashboard.db_manager == mock_db_manager
        assert dashboard.client == mock_client
        assert dashboard.active_positions == []
        assert dashboard.last_update_time == 0

    def test_load_trades(self, dashboard, mock_db_manager, sample_trades):
        """Test loading trades from the database."""
        # Mock the get_trades_by_date_range method
        mock_db_manager.get_trades_by_date_range.return_value = sample_trades

        # Test loading trades for different time frames
        for time_frame in TimeFrame:
            trades = dashboard._load_trades(time_frame)
            assert trades == sample_trades
            mock_db_manager.get_trades_by_date_range.assert_called()

    @pytest.mark.asyncio
    async def test_update_active_positions(self, dashboard, mock_client):
        """Test updating active positions from the exchange."""
        # Mock the get_positions method
        mock_client.get_positions.return_value = [
            MagicMock(
                symbol="BTCUSDT",
                position_amt=0.1,  # Long position
                entry_price=50000.0,
                mark_price=55000.0,
                unrealized_pnl=500.0,
                liquidation_price=45000.0,
                leverage=10,
                marginType="isolated",
            ),
            MagicMock(
                symbol="ETHUSDT",
                position_amt=-1.0,  # Short position
                entry_price=3000.0,
                mark_price=2700.0,
                unrealized_pnl=300.0,
                liquidation_price=3500.0,
                leverage=10,
                marginType="isolated",
            ),
            MagicMock(
                symbol="SOLUSDT",
                position_amt=0.0,  # No position
                entry_price=0.0,
                mark_price=0.0,
                unrealized_pnl=0.0,
                liquidation_price=0.0,
                leverage=10,
                marginType="isolated",
            ),
        ]

        # Update active positions
        await dashboard._update_active_positions()

        # Verify active positions
        assert len(dashboard.active_positions) == 2
        assert dashboard.active_positions[0]["symbol"] == "BTCUSDT"
        assert dashboard.active_positions[0]["side"] == "BUY"
        assert dashboard.active_positions[0]["entry_price"] == 50000.0
        assert dashboard.active_positions[0]["current_price"] == 55000.0
        assert dashboard.active_positions[0]["quantity"] == 0.1
        assert dashboard.active_positions[0]["pnl"] == 500.0
        assert dashboard.active_positions[0]["pnl_percentage"] == 10.0

        assert dashboard.active_positions[1]["symbol"] == "ETHUSDT"
        assert dashboard.active_positions[1]["side"] == "SELL"
        assert dashboard.active_positions[1]["entry_price"] == 3000.0
        assert dashboard.active_positions[1]["current_price"] == 2700.0
        assert dashboard.active_positions[1]["quantity"] == 1.0
        assert dashboard.active_positions[1]["pnl"] == 300.0
        assert dashboard.active_positions[1]["pnl_percentage"] == 10.0

        # Verify last update time
        assert dashboard.last_update_time > 0

    def test_calculate_performance_metrics(self, dashboard, sample_trades):
        """Test calculating performance metrics from trades."""
        # Calculate performance metrics
        metrics = dashboard._calculate_performance_metrics(sample_trades)

        # Verify the metrics
        assert metrics["total_trades"] == 9
        assert metrics["winning_trades"] == 6
        assert metrics["losing_trades"] == 3
        assert metrics["total_pnl"] == 1500.0  # 6 * 500 - 3 * 500
        assert metrics["win_rate"] == 6 / 9
        assert metrics["profit_factor"] == 6 / 3  # Total profit / Total loss
        assert metrics["average_win"] == 500.0
        assert metrics["average_loss"] == -500.0
        assert metrics["largest_win"] == 500.0
        assert metrics["largest_loss"] == -500.0
        assert metrics["average_trade_duration_minutes"] == 60.0

    def test_calculate_performance_metrics_empty(self, dashboard):
        """Test calculating performance metrics with no trades."""
        # Calculate performance metrics
        metrics = dashboard._calculate_performance_metrics([])

        # Verify the metrics
        assert metrics["total_trades"] == 0
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 0
        assert metrics["total_pnl"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0
        assert metrics["average_win"] == 0.0
        assert metrics["average_loss"] == 0.0
        assert metrics["largest_win"] == 0.0
        assert metrics["largest_loss"] == 0.0
        assert metrics["average_trade_duration_minutes"] == 0.0

    def test_run(self, dashboard, mock_db_manager, sample_trades):
        """Test running the dashboard."""
        # Mock the necessary methods and functions
        with (
            patch("trading_bot.ui.dashboard.st") as mock_st,
            patch.object(dashboard, "_load_trades", return_value=sample_trades),
            patch.object(
                dashboard,
                "_calculate_performance_metrics",
                return_value={
                    "total_trades": 9,
                    "winning_trades": 6,
                    "losing_trades": 3,
                    "total_pnl": 1500.0,
                    "max_drawdown": 200.0,
                    "win_rate": 6 / 9,
                    "profit_factor": 2.0,
                    "average_win": 500.0,
                    "average_loss": -500.0,
                    "largest_win": 500.0,
                    "largest_loss": -500.0,
                    "average_trade_duration_minutes": 60.0,
                },
            ),
            patch(
                "trading_bot.ui.dashboard.create_equity_curve", return_value=go.Figure()
            ),
            patch(
                "trading_bot.ui.dashboard.create_win_loss_chart",
                return_value=go.Figure(),
            ),
            patch(
                "trading_bot.ui.dashboard.create_symbol_performance_chart",
                return_value=go.Figure(),
            ),
            patch(
                "trading_bot.ui.dashboard.create_win_rate_chart",
                return_value=go.Figure(),
            ),
            patch(
                "trading_bot.ui.dashboard.create_performance_metrics_table",
                return_value=go.Figure(),
            ),
            patch(
                "trading_bot.ui.dashboard.create_active_positions_table",
                return_value=go.Figure(),
            ),
            patch(
                "trading_bot.ui.dashboard.create_recent_trades_table",
                return_value=go.Figure(),
            ),
            patch("trading_bot.ui.dashboard.time") as mock_time,
            patch("trading_bot.ui.dashboard.threading") as mock_threading,
        ):
            # Mock time.time() to return a value greater than last_update_time
            mock_time.time.return_value = (
                dashboard.last_update_time + dashboard.refresh_interval + 1
            )

            # Run the dashboard
            dashboard.run()

            # Verify that the necessary methods were called
            mock_st.set_page_config.assert_called_once()
            mock_st.title.assert_called_once()
            mock_st.sidebar.header.assert_called_once()
            mock_st.sidebar.selectbox.assert_called_once()
            assert mock_st.columns.call_count >= 2
            assert mock_st.plotly_chart.call_count >= 5
            mock_threading.Thread.assert_called_once()
            mock_threading.Thread.return_value.start.assert_called_once()
