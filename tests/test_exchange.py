"""
Unit tests for the exchange module.

This module contains tests for the Bitunix API client and authentication.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import patch, MagicMock, AsyncMock

from trading_bot.exchange.auth import generate_signature, add_auth_headers
from trading_bot.exchange.bitunix import (
    BitunixClient,
    OrderBook,
    OrderBookEntry,
    OrderSide,
    OrderType,
)


class TestAuth:
    """Tests for authentication functions."""

    def test_generate_signature(self):
        """Test signature generation."""
        api_secret = "test_secret"
        params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "price": "50000"}
        timestamp = 1620000000000

        signature = generate_signature(api_secret, params, timestamp)

        # Verify the signature is a valid hex string
        assert all(c in "0123456789abcdef" for c in signature)
        assert len(signature) == 64  # SHA256 produces a 64-character hex string

    def test_add_auth_headers(self):
        """Test adding authentication headers."""
        api_key = "test_key"
        api_secret = "test_secret"
        params = {"symbol": "BTCUSDT", "side": "BUY", "type": "LIMIT", "price": "50000"}

        headers = add_auth_headers(api_key, api_secret, params)

        assert "X-API-KEY" in headers
        assert headers["X-API-KEY"] == api_key
        assert "X-TIMESTAMP" in headers
        assert "X-SIGNATURE" in headers


class TestBitunixClient:
    """Tests for the Bitunix API client."""

    @pytest.fixture
    def client(self):
        """Create a BitunixClient instance for testing."""
        return BitunixClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
            request_timeout=1.0,
            max_retries=1,
            retry_delay=0.1,
        )

    @pytest.mark.asyncio
    async def test_get_order_book(self, client):
        """Test getting the order book."""
        # Mock the _make_request method
        mock_response = {
            "lastUpdateId": 1234567890,
            "bids": [["50000", "1.5"], ["49900", "2.0"]],
            "asks": [["50100", "1.0"], ["50200", "3.0"]],
        }

        with patch.object(
            client, "_make_request", AsyncMock(return_value=mock_response)
        ):
            order_book = await client.get_order_book("BTCUSDT")

            assert isinstance(order_book, OrderBook)
            assert order_book.last_update_id == 1234567890
            assert len(order_book.bids) == 2
            assert len(order_book.asks) == 2
            assert order_book.bids[0].price == 50000
            assert order_book.bids[0].quantity == 1.5
            assert order_book.asks[0].price == 50100
            assert order_book.asks[0].quantity == 1.0

    @pytest.mark.asyncio
    async def test_create_order(self, client):
        """Test creating an order."""
        # Mock the _make_request method
        mock_response = {
            "orderId": "12345",
            "symbol": "BTCUSDT",
            "status": "NEW",
            "side": "BUY",
            "price": 50000,
            "quantity": 0.1,
            "executedQty": 0,
            "timeInForce": "GTC",
            "type": "LIMIT",
            "reduceOnly": False,
            "time": 1620000000000,
        }

        with patch.object(
            client, "_make_request", AsyncMock(return_value=mock_response)
        ):
            order = await client.create_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.1,
                price=50000,
                time_in_force="GTC",
            )

            assert order.order_id == "12345"
            assert order.symbol == "BTCUSDT"
            assert order.status == "NEW"
            assert order.side == "BUY"
            assert order.price == 50000
            assert order.quantity == 0.1
            assert order.executed_qty == 0
            assert order.time_in_force == "GTC"
            assert order.type == "LIMIT"
            assert order.reduce_only is False
            assert order.created_time == 1620000000000

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test canceling an order."""
        # Mock the _make_request method
        mock_response = {
            "orderId": "12345",
            "symbol": "BTCUSDT",
            "status": "CANCELED",
            "side": "BUY",
            "price": 50000,
            "quantity": 0.1,
            "executedQty": 0,
            "timeInForce": "GTC",
            "type": "LIMIT",
            "reduceOnly": False,
            "time": 1620000000000,
        }

        with patch.object(
            client, "_make_request", AsyncMock(return_value=mock_response)
        ):
            order = await client.cancel_order(symbol="BTCUSDT", order_id="12345")

            assert order.order_id == "12345"
            assert order.symbol == "BTCUSDT"
            assert order.status == "CANCELED"

    @pytest.mark.asyncio
    async def test_get_account_balance(self, client):
        """Test getting account balance."""
        # Mock the _make_request method
        mock_response = [
            {
                "asset": "USDT",
                "walletBalance": 10000,
                "unrealizedPnl": 0,
                "marginBalance": 10000,
                "availableBalance": 10000,
            },
            {
                "asset": "BTC",
                "walletBalance": 1.5,
                "unrealizedPnl": 0,
                "marginBalance": 1.5,
                "availableBalance": 1.5,
            },
        ]

        with patch.object(
            client, "_make_request", AsyncMock(return_value=mock_response)
        ):
            balances = await client.get_account_balance()

            assert len(balances) == 2
            assert balances[0].asset == "USDT"
            assert balances[0].wallet_balance == 10000
            assert balances[0].available_balance == 10000
            assert balances[1].asset == "BTC"
            assert balances[1].wallet_balance == 1.5
            assert balances[1].available_balance == 1.5

    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """Test getting positions."""
        # Mock the _make_request method
        mock_response = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": 0.1,
                "entryPrice": 50000,
                "markPrice": 51000,
                "unrealizedPnl": 100,
                "liquidationPrice": 45000,
                "leverage": 10,
                "marginType": "isolated",
            },
            {
                "symbol": "ETHUSDT",
                "positionAmt": -2.0,
                "entryPrice": 3000,
                "markPrice": 2900,
                "unrealizedPnl": 200,
                "liquidationPrice": 3500,
                "leverage": 10,
                "marginType": "isolated",
            },
        ]

        with patch.object(
            client, "_make_request", AsyncMock(return_value=mock_response)
        ):
            positions = await client.get_positions()

            assert len(positions) == 2
            assert positions[0].symbol == "BTCUSDT"
            assert positions[0].position_amt == 0.1
            assert positions[0].entry_price == 50000
            assert positions[0].mark_price == 51000
            assert positions[0].unrealized_pnl == 100
            assert positions[1].symbol == "ETHUSDT"
            assert positions[1].position_amt == -2.0
            assert positions[1].entry_price == 3000
            assert positions[1].mark_price == 2900
            assert positions[1].unrealized_pnl == 200

    @pytest.mark.asyncio
    async def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        # Mock the websockets.connect function
        mock_ws = AsyncMock()
        mock_ws.open = True
        mock_ws.recv = AsyncMock(
            side_effect=[
                json.dumps({"data": "test1"}),
                json.dumps({"data": "test2"}),
                Exception("Connection closed"),
            ]
        )

        with patch("websockets.connect", AsyncMock(return_value=mock_ws)):
            # Create a mock callback
            callback = AsyncMock()

            # Connect to a stream
            await client._ws_connect("btcusdt@depth", callback)

            # Wait for the message handler to process messages
            await asyncio.sleep(0.1)

            # Verify the callback was called with the expected data
            callback.assert_called_with({"data": "test1"})

            # Close all connections
            await client.close_all_connections()
