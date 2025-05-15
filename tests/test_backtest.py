"""
Unit tests for the backtest module.

This module contains tests for the backtesting engine.
"""

import pytest
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from trading_bot.backtest.engine import BacktestEngine, BacktestResult
from trading_bot.strategy.lsob import SignalType
from trading_bot.trading.position import PositionStatus


@pytest.fixture
def sample_data():
    """Create sample data for backtesting."""
    # Create a DataFrame with sample data
    data = pd.DataFrame(
        {
            "timestamp": [
                1620000000000 + i * 60000 for i in range(100)
            ],  # 1 minute intervals
            "bid_price": [
                50000.0 + i * 10 + np.sin(i / 10) * 100 for i in range(100)
            ],  # Increasing with some oscillation
            "ask_price": [
                50100.0 + i * 10 + np.sin(i / 10) * 100 for i in range(100)
            ],  # Bid + 100
            "bid_size": [1.5 + np.random.random() for _ in range(100)],  # Random sizes
            "ask_size": [1.0 + np.random.random() for _ in range(100)],  # Random sizes
        }
    )
    return data


@pytest.fixture
def backtest_engine():
    """Create a BacktestEngine instance for testing."""
    return BacktestEngine(
        initial_balance=10000.0,
        risk_per_trade=0.01,
        commission_rate=0.0004,
        slippage=0.0001,
    )


class TestBacktestResult:
    """Tests for the BacktestResult class."""

    def test_to_dict(self):
        """Test converting a backtest result to a dictionary."""
        # Create a sample backtest result
        trades = [
            MagicMock(
                id="P1",
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
                status=PositionStatus.CLOSED,
                target_price=60000.0,
                stop_loss=45000.0,
            ),
            MagicMock(
                id="P2",
                symbol="BTCUSDT",
                side="SELL",
                entry_price=55000.0,
                exit_price=50000.0,
                quantity=0.1,
                pnl=500.0,
                pnl_percentage=10.0,
                entry_time=1620003600000,
                exit_time=1620007200000,
                duration_ms=3600000,
                status=PositionStatus.CLOSED,
                target_price=45000.0,
                stop_loss=60000.0,
            ),
        ]

        metrics = MagicMock(
            total_trades=2,
            winning_trades=2,
            losing_trades=0,
            total_pnl=1000.0,
            max_drawdown=0.0,
            win_rate=1.0,
            profit_factor=float("inf"),
            average_win=500.0,
            average_loss=0.0,
            largest_win=500.0,
            largest_loss=0.0,
            average_trade_duration_minutes=60.0,
        )

        equity_curve = [
            (1620000000000, 10000.0),
            (1620003600000, 10500.0),
            (1620007200000, 11000.0),
        ]

        parameters = {
            "imbalance_threshold": 0.3,
            "sweep_detection_window": 5,
            "min_sweep_percentage": 0.5,
            "confidence_threshold": 0.7,
        }

        result = BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            parameters=parameters,
            start_time=1620000000000,
            end_time=1620007200000,
        )

        # Convert to dictionary
        result_dict = result.to_dict()

        # Verify the dictionary
        assert "trades" in result_dict
        assert "metrics" in result_dict
        assert "equity_curve" in result_dict
        assert "parameters" in result_dict
        assert "start_time" in result_dict
        assert "end_time" in result_dict

        assert len(result_dict["trades"]) == 2
        assert result_dict["trades"][0]["id"] == "P1"
        assert result_dict["trades"][1]["id"] == "P2"

        assert result_dict["metrics"]["total_trades"] == 2
        assert result_dict["metrics"]["winning_trades"] == 2
        assert result_dict["metrics"]["total_pnl"] == 1000.0

        assert len(result_dict["equity_curve"]) == 3
        assert result_dict["equity_curve"][0] == (1620000000000, 10000.0)
        assert result_dict["equity_curve"][2] == (1620007200000, 11000.0)

        assert result_dict["parameters"]["imbalance_threshold"] == 0.3
        assert result_dict["parameters"]["sweep_detection_window"] == 5

        assert result_dict["start_time"] == 1620000000000
        assert result_dict["end_time"] == 1620007200000

    def test_save_to_json(self, tmp_path):
        """Test saving a backtest result to a JSON file."""
        # Create a sample backtest result
        trades = [
            MagicMock(
                id="P1",
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
                status=PositionStatus.CLOSED,
                target_price=60000.0,
                stop_loss=45000.0,
            ),
        ]

        metrics = MagicMock(
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            total_pnl=500.0,
            max_drawdown=0.0,
            win_rate=1.0,
            profit_factor=float("inf"),
            average_win=500.0,
            average_loss=0.0,
            largest_win=500.0,
            largest_loss=0.0,
            average_trade_duration_minutes=60.0,
        )

        equity_curve = [
            (1620000000000, 10000.0),
            (1620003600000, 10500.0),
        ]

        parameters = {
            "imbalance_threshold": 0.3,
            "sweep_detection_window": 5,
            "min_sweep_percentage": 0.5,
            "confidence_threshold": 0.7,
        }

        result = BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            parameters=parameters,
            start_time=1620000000000,
            end_time=1620003600000,
        )

        # Save to JSON
        file_path = tmp_path / "backtest_result.json"
        result.save_to_json(str(file_path))

        # Verify the file was created
        assert os.path.exists(file_path)

        # Verify the file contains the expected data
        with open(file_path, "r") as f:
            data = json.load(f)

        assert "trades" in data
        assert "metrics" in data
        assert "equity_curve" in data
        assert "parameters" in data
        assert "start_time" in data
        assert "end_time" in data

        assert len(data["trades"]) == 1
        assert data["trades"][0]["id"] == "P1"

        assert data["metrics"]["total_trades"] == 1
        assert data["metrics"]["winning_trades"] == 1
        assert data["metrics"]["total_pnl"] == 500.0

        assert len(data["equity_curve"]) == 2
        assert data["equity_curve"][0] == [1620000000000, 10000.0]
        assert data["equity_curve"][1] == [1620003600000, 10500.0]

        assert data["parameters"]["imbalance_threshold"] == 0.3
        assert data["parameters"]["sweep_detection_window"] == 5

        assert data["start_time"] == 1620000000000
        assert data["end_time"] == 1620003600000

    def test_save_trades_to_csv(self, tmp_path):
        """Test saving trades to a CSV file."""
        # Create a sample backtest result
        trades = [
            MagicMock(
                id="P1",
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
                status=PositionStatus.CLOSED,
                target_price=60000.0,
                stop_loss=45000.0,
            ),
            MagicMock(
                id="P2",
                symbol="BTCUSDT",
                side="SELL",
                entry_price=55000.0,
                exit_price=50000.0,
                quantity=0.1,
                pnl=500.0,
                pnl_percentage=10.0,
                entry_time=1620003600000,
                exit_time=1620007200000,
                duration_ms=3600000,
                status=PositionStatus.CLOSED,
                target_price=45000.0,
                stop_loss=60000.0,
            ),
        ]

        metrics = MagicMock(
            total_trades=2,
            winning_trades=2,
            losing_trades=0,
            total_pnl=1000.0,
            max_drawdown=0.0,
            win_rate=1.0,
            profit_factor=float("inf"),
            average_win=500.0,
            average_loss=0.0,
            largest_win=500.0,
            largest_loss=0.0,
            average_trade_duration_minutes=60.0,
        )

        equity_curve = [
            (1620000000000, 10000.0),
            (1620003600000, 10500.0),
            (1620007200000, 11000.0),
        ]

        parameters = {
            "imbalance_threshold": 0.3,
            "sweep_detection_window": 5,
            "min_sweep_percentage": 0.5,
            "confidence_threshold": 0.7,
        }

        result = BacktestResult(
            trades=trades,
            metrics=metrics,
            equity_curve=equity_curve,
            parameters=parameters,
            start_time=1620000000000,
            end_time=1620007200000,
        )

        # Save trades to CSV
        file_path = tmp_path / "trades.csv"
        result.save_trades_to_csv(str(file_path))

        # Verify the file was created
        assert os.path.exists(file_path)

        # Verify the file contains the expected data
        df = pd.read_csv(file_path)

        assert len(df) == 2
        assert df.iloc[0]["ID"] == "P1"
        assert df.iloc[0]["Symbol"] == "BTCUSDT"
        assert df.iloc[0]["Side"] == "BUY"
        assert df.iloc[0]["Entry Price"] == 50000.0
        assert df.iloc[0]["Exit Price"] == 55000.0
        assert df.iloc[0]["PnL"] == 500.0

        assert df.iloc[1]["ID"] == "P2"
        assert df.iloc[1]["Symbol"] == "BTCUSDT"
        assert df.iloc[1]["Side"] == "SELL"
        assert df.iloc[1]["Entry Price"] == 55000.0
        assert df.iloc[1]["Exit Price"] == 50000.0
        assert df.iloc[1]["PnL"] == 500.0


class TestBacktestEngine:
    """Tests for the BacktestEngine class."""

    def test_calculate_position_size(self, backtest_engine):
        """Test calculating position size based on risk parameters."""
        # Calculate position size
        balance = 10000.0
        entry_price = 50000.0
        stop_loss = 49000.0

        position_size = backtest_engine._calculate_position_size(
            balance, entry_price, stop_loss
        )

        # Calculate expected position size
        risk_amount = balance * backtest_engine.risk_per_trade  # 100.0
        price_risk = abs(entry_price - stop_loss)  # 1000.0
        expected_position_size = risk_amount / price_risk  # 0.1

        assert position_size == expected_position_size

    def test_calculate_trade_pnl(self, backtest_engine):
        """Test calculating profit/loss for a trade."""
        # Calculate PnL for a long trade
        side = "BUY"
        entry_price = 50000.0
        exit_price = 55000.0
        quantity = 0.1

        pnl, pnl_percentage = backtest_engine._calculate_trade_pnl(
            side, entry_price, exit_price, quantity
        )

        # Calculate expected PnL
        entry_commission = entry_price * quantity * backtest_engine.commission_rate
        exit_commission = exit_price * quantity * backtest_engine.commission_rate
        expected_pnl = (
            (exit_price - entry_price) * quantity - entry_commission - exit_commission
        )
        expected_pnl_percentage = (exit_price / entry_price - 1) * 100

        assert pnl == pytest.approx(expected_pnl)
        assert pnl_percentage == pytest.approx(expected_pnl_percentage)

        # Calculate PnL for a short trade
        side = "SELL"
        entry_price = 55000.0
        exit_price = 50000.0
        quantity = 0.1

        pnl, pnl_percentage = backtest_engine._calculate_trade_pnl(
            side, entry_price, exit_price, quantity
        )

        # Calculate expected PnL
        entry_commission = entry_price * quantity * backtest_engine.commission_rate
        exit_commission = exit_price * quantity * backtest_engine.commission_rate
        expected_pnl = (
            (entry_price - exit_price) * quantity - entry_commission - exit_commission
        )
        expected_pnl_percentage = (entry_price / exit_price - 1) * 100

        assert pnl == pytest.approx(expected_pnl)
        assert pnl_percentage == pytest.approx(expected_pnl_percentage)

    def test_apply_slippage(self, backtest_engine):
        """Test applying slippage to a price."""
        # Apply slippage to a buy order
        price = 50000.0
        side = "BUY"

        price_with_slippage = backtest_engine._apply_slippage(price, side)

        # Calculate expected price with slippage
        expected_price = price * (1 + backtest_engine.slippage)

        assert price_with_slippage == expected_price

        # Apply slippage to a sell order
        price = 50000.0
        side = "SELL"

        price_with_slippage = backtest_engine._apply_slippage(price, side)

        # Calculate expected price with slippage
        expected_price = price * (1 - backtest_engine.slippage)

        assert price_with_slippage == expected_price

    def test_calculate_performance_metrics(self, backtest_engine):
        """Test calculating performance metrics from trades."""
        # Create some sample trades
        trades = [
            MagicMock(
                id=f"P{i + 1}",
                symbol="BTCUSDT",
                side="BUY" if i % 2 == 0 else "SELL",
                entry_price=50000.0,
                exit_price=55000.0
                if i % 3 != 0
                else 45000.0,  # 2/3 winning, 1/3 losing
                quantity=0.1,
                pnl=500.0 if i % 3 != 0 else -500.0,
                pnl_percentage=10.0 if i % 3 != 0 else -10.0,
                entry_time=1620000000000 + i * 3600000,
                exit_time=1620003600000 + i * 3600000,
                duration_ms=3600000,
                status=PositionStatus.CLOSED,
            )
            for i in range(9)  # 9 trades: 6 winning, 3 losing
        ]

        # Calculate performance metrics
        metrics = backtest_engine._calculate_performance_metrics(trades)

        # Verify the metrics
        assert metrics.total_trades == 9
        assert metrics.winning_trades == 6
        assert metrics.losing_trades == 3
        assert metrics.total_pnl == 1500.0  # 6 * 500 - 3 * 500
        assert metrics.win_rate == 6 / 9
        assert metrics.profit_factor == 6 / 3  # Total profit / Total loss
        assert metrics.average_win == 500.0
        assert metrics.average_loss == -500.0
        assert metrics.largest_win == 500.0
        assert metrics.largest_loss == -500.0
        assert metrics.average_trade_duration_minutes == 60.0

    def test_backtest_lsob_strategy(self, backtest_engine, sample_data):
        """Test backtesting the LSOB strategy."""
        # Mock the necessary methods
        with (
            patch.object(
                backtest_engine, "_create_order_book_from_row"
            ) as mock_create_order_book,
            patch.object(
                backtest_engine, "_calculate_position_size", return_value=0.1
            ) as mock_calculate_position_size,
            patch.object(
                backtest_engine, "_calculate_trade_pnl", return_value=(500.0, 10.0)
            ) as mock_calculate_trade_pnl,
            patch.object(
                backtest_engine,
                "_apply_slippage",
                side_effect=lambda price, side: price,
            ) as mock_apply_slippage,
            patch.object(
                backtest_engine, "_calculate_performance_metrics"
            ) as mock_calculate_performance_metrics,
        ):
            # Mock the LSOBDetector
            mock_lsob_detector = MagicMock()
            mock_lsob_detector.add_order_book = MagicMock()

            # Mock the detect_signal method to return a signal every 10 rows
            def mock_detect_signal():
                if mock_detect_signal.call_count % 10 == 0:
                    return MagicMock(
                        type=SignalType.LONG
                        if mock_detect_signal.call_count % 20 == 0
                        else SignalType.SHORT,
                        price=50000.0,
                        target_price=55000.0,
                        stop_loss=45000.0,
                    )
                return None

            mock_detect_signal.call_count = 0
            mock_lsob_detector.detect_signal = MagicMock(side_effect=mock_detect_signal)

            # Mock the LSOBDetector constructor
            with patch(
                "trading_bot.strategy.lsob.LSOBDetector",
                return_value=mock_lsob_detector,
            ) as mock_lsob_detector_class:
                # Run the backtest
                result = backtest_engine.backtest_lsob_strategy(
                    data=sample_data,
                    symbol="BTCUSDT",
                    imbalance_threshold=0.3,
                    sweep_detection_window=5,
                    min_sweep_percentage=0.5,
                    confidence_threshold=0.7,
                )

                # Verify the result
                assert isinstance(result, BacktestResult)
                assert result.start_time == sample_data.iloc[0]["timestamp"]
                assert result.end_time == sample_data.iloc[-1]["timestamp"]
                assert "imbalance_threshold" in result.parameters
                assert result.parameters["imbalance_threshold"] == 0.3

                # Verify the methods were called
                mock_lsob_detector_class.assert_called_once_with(
                    symbol="BTCUSDT",
                    imbalance_threshold=0.3,
                    sweep_detection_window=5,
                    min_sweep_percentage=0.5,
                    confidence_threshold=0.7,
                )

                assert mock_create_order_book.call_count == len(sample_data)
                assert mock_lsob_detector.add_order_book.call_count == len(sample_data)
                assert mock_lsob_detector.detect_signal.call_count == len(sample_data)

                # Verify that calculate_performance_metrics was called
                mock_calculate_performance_metrics.assert_called_once()

    def test_parameter_sweep(self, backtest_engine, sample_data, tmp_path):
        """Test parameter sweep for optimization."""
        # Mock the backtest_lsob_strategy method
        with patch.object(
            backtest_engine, "backtest_lsob_strategy"
        ) as mock_backtest_lsob_strategy:
            # Mock the return value of backtest_lsob_strategy
            mock_backtest_lsob_strategy.return_value = MagicMock(
                metrics=MagicMock(
                    total_trades=10,
                    winning_trades=6,
                    losing_trades=4,
                    total_pnl=1000.0,
                    max_drawdown=200.0,
                    win_rate=0.6,
                    profit_factor=2.0,
                ),
                parameters={},
                save_to_json=MagicMock(),
            )

            # Define parameter ranges
            imbalance_thresholds = [0.1, 0.3, 0.5]
            sweep_detection_windows = [3, 5, 7]
            min_sweep_percentages = [0.3, 0.5, 0.7]
            confidence_thresholds = [0.5, 0.7, 0.9]

            # Create output directory
            output_dir = tmp_path / "results"
            os.makedirs(output_dir, exist_ok=True)

            # Run parameter sweep
            results = backtest_engine.parameter_sweep(
                data=sample_data,
                symbol="BTCUSDT",
                imbalance_thresholds=imbalance_thresholds,
                sweep_detection_windows=sweep_detection_windows,
                min_sweep_percentages=min_sweep_percentages,
                confidence_thresholds=confidence_thresholds,
                output_dir=str(output_dir),
            )

            # Verify the results
            assert len(results) == len(imbalance_thresholds) * len(
                sweep_detection_windows
            ) * len(min_sweep_percentages) * len(confidence_thresholds)

            # Verify that backtest_lsob_strategy was called for each parameter combination
            assert mock_backtest_lsob_strategy.call_count == len(results)

            # Verify that save_to_json was called for each result
            for result in results:
                result.save_to_json.assert_called_once()

            # Verify that the summary file was created
            assert os.path.exists(output_dir / "summary.csv")
