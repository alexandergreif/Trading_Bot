"""
Unit tests for the CLI module.

This module contains tests for the command-line interface.
"""

import pytest
import os
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typer.testing import CliRunner

from trading_bot.cli.main import app
from trading_bot.exchange.bitunix import BitunixClient
from trading_bot.strategy.lsob import LSOBDetector
from trading_bot.trading.position import PositionManager
from trading_bot.data.storage import DatabaseManager


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock configuration file."""
    config = {
        "api_key": "test_key",
        "api_secret": "test_secret",
        "testnet": True,
        "db_path": str(tmp_path / "test.db"),
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

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    return str(config_path)


class TestCLI:
    """Tests for the CLI module."""

    def test_init_command(self, runner, tmp_path):
        """Test the init command."""
        config_path = str(tmp_path / "config.json")
        db_path = str(tmp_path / "test.db")

        # Run the init command
        result = runner.invoke(
            app,
            [
                "init",
                "--config",
                config_path,
                "--api-key",
                "test_key",
                "--api-secret",
                "test_secret",
                "--testnet",
                "--db",
                db_path,
            ],
        )

        # Verify the command executed successfully
        assert result.exit_code == 0
        assert "Configuration saved to" in result.stdout
        assert "Database initialized at" in result.stdout
        assert "Trading bot initialized successfully" in result.stdout

        # Verify the config file was created
        assert os.path.exists(config_path)

        # Verify the database file was created
        assert os.path.exists(db_path)

        # Verify the config file contains the expected values
        with open(config_path, "r") as f:
            config = json.load(f)

        assert config["api_key"] == "test_key"
        assert config["api_secret"] == "test_secret"
        assert config["testnet"] is True
        assert config["db_path"] == db_path
        assert "strategy" in config
        assert "lsob" in config["strategy"]
        assert "trading" in config
        assert "symbols" in config

    def test_run_command(self, runner, mock_config):
        """Test the run command."""
        # Mock the necessary components
        with (
            patch("trading_bot.cli.main.BitunixClient") as mock_client,
            patch("trading_bot.cli.main.PositionManager") as mock_position_manager,
            patch("trading_bot.cli.main.DatabaseManager") as mock_db_manager,
            patch("trading_bot.cli.main.KPITracker") as mock_kpi_tracker,
            patch("trading_bot.cli.main.LSOBDetector") as mock_lsob_detector,
            patch("trading_bot.cli.main.asyncio.run") as mock_asyncio_run,
        ):
            # Run the run command
            result = runner.invoke(
                app,
                [
                    "run",
                    "--config",
                    mock_config,
                    "--symbol",
                    "BTCUSDT",
                ],
            )

            # Verify the command executed successfully
            assert result.exit_code == 0

            # Verify the components were initialized
            mock_client.assert_called_once()
            mock_position_manager.assert_called_once()
            mock_db_manager.assert_called_once()
            mock_kpi_tracker.assert_called_once()
            mock_lsob_detector.assert_called_once()

            # Verify asyncio.run was called
            mock_asyncio_run.assert_called_once()

    def test_backtest_command(self, runner, mock_config, tmp_path):
        """Test the backtest command."""
        # Create a mock backtest data file
        data_path = tmp_path / "backtest_data.csv"
        with open(data_path, "w") as f:
            f.write("timestamp,bid_price,ask_price,bid_size,ask_size\n")
            f.write("1620000000000,50000,50100,1.5,1.0\n")
            f.write("1620000001000,50010,50110,1.6,1.1\n")

        output_dir = tmp_path / "results"

        # Mock the necessary components
        with (
            patch("trading_bot.cli.main.BacktestEngine") as mock_backtest_engine,
            patch("trading_bot.cli.main.pd.read_csv") as mock_read_csv,
        ):
            # Mock the backtest_lsob_strategy method
            mock_backtest_engine.return_value.backtest_lsob_strategy.return_value = (
                MagicMock(
                    metrics=MagicMock(
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
                    ),
                    save_to_json=MagicMock(),
                )
            )

            # Run the backtest command
            result = runner.invoke(
                app,
                [
                    "backtest",
                    "--config",
                    mock_config,
                    "--data",
                    str(data_path),
                    "--symbol",
                    "BTCUSDT",
                    "--output",
                    str(output_dir),
                ],
            )

            # Verify the command executed successfully
            assert result.exit_code == 0

            # Verify the components were initialized
            mock_backtest_engine.assert_called_once()
            mock_read_csv.assert_called_once_with(str(data_path))

            # Verify backtest_lsob_strategy was called
            mock_backtest_engine.return_value.backtest_lsob_strategy.assert_called_once()

            # Verify save_to_json was called
            mock_backtest_engine.return_value.backtest_lsob_strategy.return_value.save_to_json.assert_called_once()

    def test_parameter_sweep(self, runner, mock_config, tmp_path):
        """Test the backtest command with parameter sweep."""
        # Create a mock backtest data file
        data_path = tmp_path / "backtest_data.csv"
        with open(data_path, "w") as f:
            f.write("timestamp,bid_price,ask_price,bid_size,ask_size\n")
            f.write("1620000000000,50000,50100,1.5,1.0\n")
            f.write("1620000001000,50010,50110,1.6,1.1\n")

        output_dir = tmp_path / "results"

        # Mock the necessary components
        with (
            patch("trading_bot.cli.main.BacktestEngine") as mock_backtest_engine,
            patch("trading_bot.cli.main.pd.read_csv") as mock_read_csv,
        ):
            # Mock the parameter_sweep method
            mock_backtest_engine.return_value.parameter_sweep.return_value = [
                MagicMock(
                    parameters={
                        "imbalance_threshold": 0.3,
                        "sweep_detection_window": 5,
                        "min_sweep_percentage": 0.5,
                        "confidence_threshold": 0.7,
                    },
                    metrics=MagicMock(
                        total_trades=10,
                        winning_trades=6,
                        losing_trades=4,
                        total_pnl=1000.0,
                        max_drawdown=200.0,
                        win_rate=0.6,
                        profit_factor=2.0,
                    ),
                ),
                MagicMock(
                    parameters={
                        "imbalance_threshold": 0.4,
                        "sweep_detection_window": 7,
                        "min_sweep_percentage": 0.6,
                        "confidence_threshold": 0.8,
                    },
                    metrics=MagicMock(
                        total_trades=8,
                        winning_trades=5,
                        losing_trades=3,
                        total_pnl=800.0,
                        max_drawdown=150.0,
                        win_rate=0.625,
                        profit_factor=2.2,
                    ),
                ),
            ]

            # Run the backtest command with parameter sweep
            result = runner.invoke(
                app,
                [
                    "backtest",
                    "--config",
                    mock_config,
                    "--data",
                    str(data_path),
                    "--symbol",
                    "BTCUSDT",
                    "--output",
                    str(output_dir),
                    "--sweep",
                ],
            )

            # Verify the command executed successfully
            assert result.exit_code == 0

            # Verify the components were initialized
            mock_backtest_engine.assert_called_once()
            mock_read_csv.assert_called_once_with(str(data_path))

            # Verify parameter_sweep was called
            mock_backtest_engine.return_value.parameter_sweep.assert_called_once()

    def test_dashboard_command(self, runner, mock_config):
        """Test the dashboard command."""
        # Mock the necessary components
        with patch("trading_bot.cli.main.run_dashboard") as mock_run_dashboard:
            # Run the dashboard command
            result = runner.invoke(
                app,
                [
                    "dashboard",
                    "--config",
                    mock_config,
                    "--port",
                    "8501",
                ],
            )

            # Verify the command executed successfully
            assert result.exit_code == 0

            # Verify run_dashboard was called
            mock_run_dashboard.assert_called_once()

    def test_run_trading_loop(self):
        """Test the run_trading_loop function."""
        # Mock the necessary components
        mock_client = AsyncMock(spec=BitunixClient)
        mock_position_manager = AsyncMock(spec=PositionManager)
        mock_lsob_detector = MagicMock(spec=LSOBDetector)
        mock_kpi_tracker = MagicMock()
        mock_db_manager = MagicMock(spec=DatabaseManager)

        # Mock the get_order_book method
        mock_client.get_order_book.return_value = MagicMock()

        # Mock the detect_signal method to return None (no signal)
        mock_lsob_detector.detect_signal.return_value = None

        # Mock the update_positions method
        mock_position_manager.update_positions.return_value = None

        # Mock the get_active_positions method to return an empty list
        mock_position_manager.get_active_positions.return_value = []

        # Create a mock for asyncio.sleep that raises an exception after the first call
        # to break out of the infinite loop
        mock_sleep = AsyncMock(side_effect=[None, KeyboardInterrupt])

        # Test the run_trading_loop function
        with patch("trading_bot.cli.main.asyncio.sleep", mock_sleep):
            from trading_bot.cli.main import run_trading_loop

            # Run the trading loop
            with pytest.raises(KeyboardInterrupt):
                asyncio.run(
                    run_trading_loop(
                        client=mock_client,
                        position_manager=mock_position_manager,
                        strategies={"BTCUSDT": mock_lsob_detector},
                        kpi_tracker=mock_kpi_tracker,
                        db_manager=mock_db_manager,
                        daemon=False,
                    )
                )

            # Verify the methods were called
            mock_db_manager.connect.assert_called_once()
            mock_position_manager.update_positions.assert_called_once()
            mock_client.get_order_book.assert_called_once_with("BTCUSDT")
            mock_lsob_detector.add_order_book.assert_called_once()
            mock_lsob_detector.detect_signal.assert_called_once()
            mock_position_manager.get_active_positions.assert_called_once()
            mock_sleep.assert_called_once_with(5)
