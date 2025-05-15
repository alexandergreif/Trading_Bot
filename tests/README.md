# Trading Bot Tests

This directory contains unit tests for the trading bot project.

## Running Tests

To run all tests:

```bash
cd trading_bot
pytest
```

To run tests with coverage report:

```bash
pytest --cov=trading_bot --cov-report=html
```

This will generate an HTML coverage report in the `htmlcov` directory.

## Test Categories

Tests are organized by module:

- `test_exchange.py`: Tests for the exchange module (API client, authentication)
- `test_strategy.py`: Tests for the strategy module (LSOB detector)
- `test_trading.py`: Tests for the trading module (position management, metrics)
- `test_data.py`: Tests for the data module (database storage)
- `test_backtest.py`: Tests for the backtest module (backtesting engine)
- `test_ui.py`: Tests for the UI module (dashboard)
- `test_cli.py`: Tests for the CLI module (command-line interface)

## Test Markers

You can use the following markers to run specific types of tests:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run slow tests (normally skipped)
pytest -m slow

# Run API tests (normally skipped)
pytest -m api
```

## Fixtures

Common fixtures are defined in `conftest.py` and can be used across all test files:

- `temp_dir`: Creates a temporary directory for test files
- `config_file`: Creates a temporary configuration file
- `mock_client`: Creates a mock BitunixClient
- `mock_position_manager`: Creates a mock PositionManager
- `mock_db_manager`: Creates a mock DatabaseManager
- `mock_kpi_tracker`: Creates a mock KPITracker
- `mock_lsob_detector`: Creates a mock LSOBDetector

## Adding New Tests

When adding new tests:

1. Create a new test file in the `tests` directory with the name `test_*.py`
2. Import the module you want to test
3. Create test classes with the name `Test*`
4. Create test functions with the name `test_*`
5. Use fixtures from `conftest.py` as needed
6. Add appropriate markers to categorize your tests
