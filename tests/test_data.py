"""
Unit tests for the data module.

This module contains tests for the database storage functionality.
"""

import pytest
import os
import sqlite3
import time
from unittest.mock import patch, MagicMock

from trading_bot.data.storage import DatabaseManager, Trade


class TestTrade:
    """Tests for the Trade class."""

    def test_from_row(self):
        """Test creating a Trade object from a database row."""
        # Create a sample database row
        row = (
            1,  # id
            "P1",  # position_id
            "BTCUSDT",  # symbol
            "BUY",  # side
            50000.0,  # entry_price
            55000.0,  # exit_price
            0.1,  # quantity
            500.0,  # pnl
            10.0,  # pnl_percentage
            1620000000000,  # entry_time
            1620003600000,  # exit_time
            3600000,  # duration_ms
            "CLOSED",  # status
            60000.0,  # target_price
            45000.0,  # stop_loss
        )

        # Create a Trade object from the row
        trade = Trade.from_row(row)

        # Verify the Trade object
        assert trade.id == 1
        assert trade.position_id == "P1"
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "BUY"
        assert trade.entry_price == 50000.0
        assert trade.exit_price == 55000.0
        assert trade.quantity == 0.1
        assert trade.pnl == 500.0
        assert trade.pnl_percentage == 10.0
        assert trade.entry_time == 1620000000000
        assert trade.exit_time == 1620003600000
        assert trade.duration_ms == 3600000
        assert trade.status == "CLOSED"
        assert trade.target_price == 60000.0
        assert trade.stop_loss == 45000.0

    def test_is_win(self):
        """Test checking if a trade was profitable."""
        # Create a winning trade
        winning_trade = Trade(
            position_id="P1",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            exit_price=55000.0,
            quantity=0.1,
            pnl=500.0,
            pnl_percentage=10.0,
            entry_time=1620000000000,
            exit_time=1620003600000,
            duration_ms=3600000,
            status="CLOSED",
        )

        assert winning_trade.is_win is True

        # Create a losing trade
        losing_trade = Trade(
            position_id="P2",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            exit_price=45000.0,
            quantity=0.1,
            pnl=-500.0,
            pnl_percentage=-10.0,
            entry_time=1620000000000,
            exit_time=1620003600000,
            duration_ms=3600000,
            status="CLOSED",
        )

        assert losing_trade.is_win is False


class TestDatabaseManager:
    """Tests for the DatabaseManager class."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a temporary database path for testing."""
        return os.path.join(tmp_path, "test.db")

    @pytest.fixture
    def db_manager(self, db_path):
        """Create a DatabaseManager instance for testing."""
        manager = DatabaseManager(db_path)
        manager.connect()
        manager.init_db()
        yield manager
        manager.disconnect()

    def test_connect_disconnect(self, db_path):
        """Test connecting to and disconnecting from the database."""
        # Create a database manager
        manager = DatabaseManager(db_path)

        # Initially, not connected
        assert manager.conn is None
        assert manager.cursor is None

        # Connect
        manager.connect()

        assert manager.conn is not None
        assert manager.cursor is not None

        # Disconnect
        manager.disconnect()

        assert manager.conn is None
        assert manager.cursor is None

    def test_context_manager(self, db_path):
        """Test using the database manager as a context manager."""
        with DatabaseManager(db_path) as manager:
            assert manager.conn is not None
            assert manager.cursor is not None

            # Execute a simple query
            manager.cursor.execute("SELECT sqlite_version()")
            version = manager.cursor.fetchone()
            assert version is not None

        # After exiting the context, should be disconnected
        assert manager.conn is None
        assert manager.cursor is None

    def test_init_db(self, db_manager):
        """Test initializing the database schema."""
        # Check if the trades table exists
        db_manager.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        result = db_manager.cursor.fetchone()
        assert result is not None
        assert result[0] == "trades"

        # Check if the kpi_metrics table exists
        db_manager.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kpi_metrics'"
        )
        result = db_manager.cursor.fetchone()
        assert result is not None
        assert result[0] == "kpi_metrics"

        # Check if the indexes exist
        db_manager.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_trades_position_id'"
        )
        result = db_manager.cursor.fetchone()
        assert result is not None

        db_manager.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_trades_symbol'"
        )
        result = db_manager.cursor.fetchone()
        assert result is not None

        db_manager.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_trades_entry_time'"
        )
        result = db_manager.cursor.fetchone()
        assert result is not None

        db_manager.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_kpi_metrics_time_frame_timestamp'"
        )
        result = db_manager.cursor.fetchone()
        assert result is not None

    def test_insert_trade(self, db_manager):
        """Test inserting a trade into the database."""
        # Create a trade
        trade = Trade(
            position_id="P1",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            exit_price=55000.0,
            quantity=0.1,
            pnl=500.0,
            pnl_percentage=10.0,
            entry_time=1620000000000,
            exit_time=1620003600000,
            duration_ms=3600000,
            status="CLOSED",
            target_price=60000.0,
            stop_loss=45000.0,
        )

        # Insert the trade
        trade_id = db_manager.insert_trade(trade)

        # Verify the trade was inserted
        assert trade_id is not None
        assert trade_id > 0

        # Retrieve the trade
        db_manager.cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = db_manager.cursor.fetchone()

        assert row is not None
        assert row[0] == trade_id
        assert row[1] == trade.position_id
        assert row[2] == trade.symbol
        assert row[3] == trade.side
        assert row[4] == trade.entry_price
        assert row[5] == trade.exit_price
        assert row[6] == trade.quantity
        assert row[7] == trade.pnl
        assert row[8] == trade.pnl_percentage
        assert row[9] == trade.entry_time
        assert row[10] == trade.exit_time
        assert row[11] == trade.duration_ms
        assert row[12] == trade.status
        assert row[13] == trade.target_price
        assert row[14] == trade.stop_loss

    def test_update_trade(self, db_manager):
        """Test updating a trade in the database."""
        # Create and insert a trade
        trade = Trade(
            position_id="P1",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            exit_price=55000.0,
            quantity=0.1,
            pnl=500.0,
            pnl_percentage=10.0,
            entry_time=1620000000000,
            exit_time=1620003600000,
            duration_ms=3600000,
            status="CLOSED",
            target_price=60000.0,
            stop_loss=45000.0,
        )

        trade_id = db_manager.insert_trade(trade)

        # Update the trade
        trade.id = trade_id
        trade.exit_price = 56000.0
        trade.pnl = 600.0
        trade.pnl_percentage = 12.0

        result = db_manager.update_trade(trade)

        # Verify the update was successful
        assert result is True

        # Retrieve the updated trade
        db_manager.cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = db_manager.cursor.fetchone()

        assert row is not None
        assert row[5] == 56000.0  # exit_price
        assert row[7] == 600.0  # pnl
        assert row[8] == 12.0  # pnl_percentage

    def test_get_trade_by_id(self, db_manager):
        """Test getting a trade by its ID."""
        # Create and insert a trade
        trade = Trade(
            position_id="P1",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            exit_price=55000.0,
            quantity=0.1,
            pnl=500.0,
            pnl_percentage=10.0,
            entry_time=1620000000000,
            exit_time=1620003600000,
            duration_ms=3600000,
            status="CLOSED",
            target_price=60000.0,
            stop_loss=45000.0,
        )

        trade_id = db_manager.insert_trade(trade)

        # Get the trade by ID
        retrieved_trade = db_manager.get_trade_by_id(trade_id)

        # Verify the retrieved trade
        assert retrieved_trade is not None
        assert retrieved_trade.id == trade_id
        assert retrieved_trade.position_id == trade.position_id
        assert retrieved_trade.symbol == trade.symbol
        assert retrieved_trade.side == trade.side
        assert retrieved_trade.entry_price == trade.entry_price
        assert retrieved_trade.exit_price == trade.exit_price
        assert retrieved_trade.quantity == trade.quantity
        assert retrieved_trade.pnl == trade.pnl
        assert retrieved_trade.pnl_percentage == trade.pnl_percentage
        assert retrieved_trade.entry_time == trade.entry_time
        assert retrieved_trade.exit_time == trade.exit_time
        assert retrieved_trade.duration_ms == trade.duration_ms
        assert retrieved_trade.status == trade.status
        assert retrieved_trade.target_price == trade.target_price
        assert retrieved_trade.stop_loss == trade.stop_loss

        # Try to get a non-existent trade
        non_existent_trade = db_manager.get_trade_by_id(9999)
        assert non_existent_trade is None

    def test_get_trade_by_position_id(self, db_manager):
        """Test getting a trade by its position ID."""
        # Create and insert a trade
        trade = Trade(
            position_id="P1",
            symbol="BTCUSDT",
            side="BUY",
            entry_price=50000.0,
            exit_price=55000.0,
            quantity=0.1,
            pnl=500.0,
            pnl_percentage=10.0,
            entry_time=1620000000000,
            exit_time=1620003600000,
            duration_ms=3600000,
            status="CLOSED",
            target_price=60000.0,
            stop_loss=45000.0,
        )

        db_manager.insert_trade(trade)

        # Get the trade by position ID
        retrieved_trade = db_manager.get_trade_by_position_id("P1")

        # Verify the retrieved trade
        assert retrieved_trade is not None
        assert retrieved_trade.position_id == "P1"
        assert retrieved_trade.symbol == trade.symbol
        assert retrieved_trade.side == trade.side
        assert retrieved_trade.entry_price == trade.entry_price
        assert retrieved_trade.exit_price == trade.exit_price

        # Try to get a non-existent trade
        non_existent_trade = db_manager.get_trade_by_position_id("P999")
        assert non_existent_trade is None

    def test_get_trades_by_symbol(self, db_manager):
        """Test getting trades by symbol."""
        # Create and insert some trades
        trades = [
            Trade(
                position_id=f"P{i}",
                symbol="BTCUSDT" if i % 2 == 0 else "ETHUSDT",
                side="BUY",
                entry_price=50000.0,
                exit_price=55000.0,
                quantity=0.1,
                pnl=500.0,
                pnl_percentage=10.0,
                entry_time=1620000000000 + i * 3600000,
                exit_time=1620003600000 + i * 3600000,
                duration_ms=3600000,
                status="CLOSED",
            )
            for i in range(10)
        ]

        for trade in trades:
            db_manager.insert_trade(trade)

        # Get trades by symbol
        btc_trades = db_manager.get_trades_by_symbol("BTCUSDT")
        eth_trades = db_manager.get_trades_by_symbol("ETHUSDT")

        # Verify the retrieved trades
        assert len(btc_trades) == 5
        assert len(eth_trades) == 5

        for trade in btc_trades:
            assert trade.symbol == "BTCUSDT"

        for trade in eth_trades:
            assert trade.symbol == "ETHUSDT"

    def test_get_trades_by_date_range(self, db_manager):
        """Test getting trades within a date range."""
        # Create and insert some trades
        base_time = 1620000000000  # May 3, 2021
        trades = [
            Trade(
                position_id=f"P{i}",
                symbol="BTCUSDT",
                side="BUY",
                entry_price=50000.0,
                exit_price=55000.0,
                quantity=0.1,
                pnl=500.0,
                pnl_percentage=10.0,
                entry_time=base_time + i * 86400000,  # Each trade 1 day apart
                exit_time=base_time + i * 86400000 + 3600000,
                duration_ms=3600000,
                status="CLOSED",
            )
            for i in range(10)
        ]

        for trade in trades:
            db_manager.insert_trade(trade)

        # Get trades within a date range (days 3-6)
        start_time = base_time + 3 * 86400000
        end_time = base_time + 6 * 86400000

        date_range_trades = db_manager.get_trades_by_date_range(start_time, end_time)

        # Verify the retrieved trades
        assert len(date_range_trades) == 4  # Days 3, 4, 5, 6

        for i, trade in enumerate(date_range_trades):
            assert trade.entry_time >= start_time
            assert trade.entry_time <= end_time

    def test_get_all_trades(self, db_manager):
        """Test getting all trades."""
        # Create and insert some trades
        trades = [
            Trade(
                position_id=f"P{i}",
                symbol="BTCUSDT",
                side="BUY",
                entry_price=50000.0,
                exit_price=55000.0,
                quantity=0.1,
                pnl=500.0,
                pnl_percentage=10.0,
                entry_time=1620000000000 + i * 3600000,
                exit_time=1620003600000 + i * 3600000,
                duration_ms=3600000,
                status="CLOSED",
            )
            for i in range(20)
        ]

        for trade in trades:
            db_manager.insert_trade(trade)

        # Get all trades with default limit
        all_trades = db_manager.get_all_trades()

        # Verify the retrieved trades
        assert len(all_trades) == 20

        # Get trades with a limit
        limited_trades = db_manager.get_all_trades(limit=5)

        # Verify the retrieved trades
        assert len(limited_trades) == 5

        # Trades should be ordered by entry_time DESC
        for i in range(len(limited_trades) - 1):
            assert limited_trades[i].entry_time > limited_trades[i + 1].entry_time

    def test_save_kpi_metrics(self, db_manager):
        """Test saving KPI metrics to the database."""
        # Save KPI metrics
        kpi_id = db_manager.save_kpi_metrics(
            time_frame="DAILY",
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

        # Verify the KPI metrics were saved
        assert kpi_id is not None
        assert kpi_id > 0

        # Retrieve the KPI metrics
        db_manager.cursor.execute("SELECT * FROM kpi_metrics WHERE id = ?", (kpi_id,))
        row = db_manager.cursor.fetchone()

        assert row is not None
        assert row[0] == kpi_id
        assert row[2] == "DAILY"
        assert row[3] == 10  # total_trades
        assert row[4] == 6  # winning_trades
        assert row[5] == 4  # losing_trades
        assert row[6] == 1000.0  # total_pnl
        assert row[7] == 200.0  # max_drawdown
        assert row[8] == 0.6  # win_rate
        assert row[9] == 2.0  # profit_factor
        assert row[10] == 200.0  # average_win
        assert row[11] == -100.0  # average_loss
        assert row[12] == 500.0  # largest_win
        assert row[13] == -200.0  # largest_loss
        assert row[14] == 60.0  # average_trade_duration_minutes

    def test_get_latest_kpi_metrics(self, db_manager):
        """Test getting the latest KPI metrics for a specific time frame."""
        # Save some KPI metrics
        time_frames = ["DAILY", "WEEKLY", "MONTHLY", "ALL_TIME"]

        for time_frame in time_frames:
            db_manager.save_kpi_metrics(
                time_frame=time_frame,
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

        # Save another set of metrics for DAILY (should be the latest)
        db_manager.save_kpi_metrics(
            time_frame="DAILY",
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            total_pnl=2000.0,
            max_drawdown=400.0,
            win_rate=0.6,
            profit_factor=2.0,
            average_win=200.0,
            average_loss=-100.0,
            largest_win=500.0,
            largest_loss=-200.0,
            average_trade_duration_minutes=60.0,
        )

        # Get the latest KPI metrics for DAILY
        daily_metrics = db_manager.get_latest_kpi_metrics("DAILY")

        # Verify the retrieved metrics
        assert daily_metrics is not None
        assert daily_metrics["time_frame"] == "DAILY"
        assert daily_metrics["total_trades"] == 20
        assert daily_metrics["winning_trades"] == 12
        assert daily_metrics["losing_trades"] == 8
        assert daily_metrics["total_pnl"] == 2000.0

        # Get the latest KPI metrics for WEEKLY
        weekly_metrics = db_manager.get_latest_kpi_metrics("WEEKLY")

        # Verify the retrieved metrics
        assert weekly_metrics is not None
        assert weekly_metrics["time_frame"] == "WEEKLY"
        assert weekly_metrics["total_trades"] == 10

        # Try to get metrics for a non-existent time frame
        non_existent_metrics = db_manager.get_latest_kpi_metrics("NON_EXISTENT")
        assert non_existent_metrics is None
