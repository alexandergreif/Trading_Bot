"""
Pytest configuration file.

This module contains fixtures that can be used across all test files.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import MagicMock, AsyncMock

from trading_bot.exchange.bitunix import BitunixClient
from trading_bot.trading.position import PositionManager
from trading_bot.data.storage import DatabaseManager
from trading_bot.trading.metrics import KPITracker
from trading_bot.strategy.lsob import LSOBDetector


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def config_file(temp_dir):
    """Create a temporary configuration file for testing."""
    config = {
        "api_key": "test_key",
        "api_secret": "test_secret",
        "testnet": True,
        "db_path": os.path.join(temp_dir, "test.db"),
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

    config_path = os.path.join(temp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    return config_path


@pytest.fixture
def mock_client():
    """Create a mock BitunixClient."""
    client = AsyncMock(spec=BitunixClient)
    return client


@pytest.fixture
def mock_position_manager():
    """Create a mock PositionManager."""
    manager = AsyncMock(spec=PositionManager)
    return manager


@pytest.fixture
def mock_db_manager():
    """Create a mock DatabaseManager."""
    manager = MagicMock(spec=DatabaseManager)
    return manager


@pytest.fixture
def mock_kpi_tracker():
    """Create a mock KPITracker."""
    tracker = MagicMock(spec=KPITracker)
    return tracker


@pytest.fixture
def mock_lsob_detector():
    """Create a mock LSOBDetector."""
    detector = MagicMock(spec=LSOBDetector)
    return detector
