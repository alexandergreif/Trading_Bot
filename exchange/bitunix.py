"""
Bitunix Futures API client implementation.

This module provides a client for interacting with the Bitunix Futures API,
including REST API calls and WebSocket connections for real-time data.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable
from urllib.parse import urlencode

import httpx
import websockets
from pydantic import BaseModel, Field

from .auth import add_auth_headers, generate_signature

logger = logging.getLogger(__name__)


class OrderSide(str):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(str):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderResponse(BaseModel):
    order_id: str = Field(..., alias="orderId")
    symbol: str
    status: str
    side: str
    price: Optional[float] = None
    quantity: float
    executed_qty: float = Field(..., alias="executedQty")
    time_in_force: str = Field(..., alias="timeInForce")
    type: str
    reduce_only: bool = Field(..., alias="reduceOnly")
    created_time: int = Field(..., alias="time")


class AccountBalance(BaseModel):
    asset: str
    wallet_balance: float = Field(..., alias="walletBalance")
    unrealized_pnl: float = Field(..., alias="unrealizedPnl")
    margin_balance: float = Field(..., alias="marginBalance")
    available_balance: float = Field(..., alias="availableBalance")


class Position(BaseModel):
    symbol: str
    position_amt: float = Field(..., alias="positionAmt")
    entry_price: float = Field(..., alias="entryPrice")
    mark_price: float = Field(..., alias="markPrice")
    unrealized_pnl: float = Field(..., alias="unrealizedPnl")
    liquidation_price: float = Field(..., alias="liquidationPrice")
    leverage: int
    margin_type: str = Field(..., alias="marginType")


class OrderBookEntry(BaseModel):
    price: float
    quantity: float


class OrderBook(BaseModel):
    last_update_id: int = Field(..., alias="lastUpdateId")
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]


class BitunixClient:
    """
    Client for interacting with the Bitunix Futures API.
    """

    BASE_URL = "https://api.bitunix.com"
    WS_URL = "wss://stream.bitunix.com/ws"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        request_timeout: float = 10.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Bitunix client.

        Args:
            api_key: API key for authentication
            api_secret: API secret for authentication
            testnet: Whether to use the testnet API
            request_timeout: Timeout for HTTP requests in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret

        if testnet:
            self.BASE_URL = "https://testnet-api.bitunix.com"
            self.WS_URL = "wss://testnet-stream.bitunix.com/ws"

        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._ws_connections = {}
        self._ws_callbacks = {}
        self._ws_reconnect_tasks = {}

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        auth_required: bool = True,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the Bitunix API.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            auth_required: Whether authentication is required

        Returns:
            API response as a dictionary
        """
        url = f"{self.BASE_URL}{endpoint}"
        headers = {}

        if params is None:
            params = {}

        if auth_required:
            headers.update(add_auth_headers(self.api_key, self.api_secret, params))

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                    if method == "GET":
                        response = await client.get(url, params=params, headers=headers)
                    elif method == "POST":
                        response = await client.post(
                            url, json=data, params=params, headers=headers
                        )
                    elif method == "DELETE":
                        response = await client.delete(
                            url, params=params, headers=headers
                        )
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error: {e.response.status_code} - {e.response.text}"
                )
                if attempt == self.max_retries - 1:
                    raise
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                logger.error(f"Request error: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise

            await asyncio.sleep(self.retry_delay * (2**attempt))  # Exponential backoff

    async def get_order_book(self, symbol: str, limit: int = 100) -> OrderBook:
        """
        Get the order book for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            limit: Number of price levels to return (default: 100, max: 1000)

        Returns:
            Order book data
        """
        params = {"symbol": symbol, "limit": limit}
        data = await self._make_request(
            "GET", "/fapi/v1/depth", params=params, auth_required=False
        )

        # Convert the raw data to OrderBookEntry objects
        bids = [
            OrderBookEntry(price=float(item[0]), quantity=float(item[1]))
            for item in data["bids"]
        ]
        asks = [
            OrderBookEntry(price=float(item[0]), quantity=float(item[1]))
            for item in data["asks"]
        ]

        return OrderBook(lastUpdateId=data["lastUpdateId"], bids=bids, asks=asks)

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: Optional[str] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
    ) -> OrderResponse:
        """
        Create a new order.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            side: Order side (BUY or SELL)
            order_type: Order type (MARKET, LIMIT, etc.)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            stop_price: Stop price (required for STOP and TAKE_PROFIT orders)
            reduce_only: Whether the order should only reduce position

        Returns:
            Order response data
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
            "reduceOnly": reduce_only,
        }

        if price is not None:
            params["price"] = price

        if time_in_force is not None:
            params["timeInForce"] = time_in_force

        if stop_price is not None:
            params["stopPrice"] = stop_price

        data = await self._make_request("POST", "/fapi/v1/order", params=params)
        return OrderResponse(**data)

    async def cancel_order(self, symbol: str, order_id: str) -> OrderResponse:
        """
        Cancel an existing order.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            order_id: Order ID to cancel

        Returns:
            Order response data
        """
        params = {
            "symbol": symbol,
            "orderId": order_id,
        }

        data = await self._make_request("DELETE", "/fapi/v1/order", params=params)
        return OrderResponse(**data)

    async def get_account_balance(self) -> List[AccountBalance]:
        """
        Get account balance information.

        Returns:
            List of account balances
        """
        data = await self._make_request("GET", "/fapi/v1/balance", params={})
        return [AccountBalance(**item) for item in data]

    async def get_positions(self) -> List[Position]:
        """
        Get current positions.

        Returns:
            List of positions
        """
        data = await self._make_request("GET", "/fapi/v1/positionRisk", params={})
        return [Position(**item) for item in data]

    async def _ws_connect(
        self, stream_name: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """
        Connect to a WebSocket stream.

        Args:
            stream_name: Name of the stream to connect to
            callback: Callback function to handle incoming messages
        """
        self._ws_callbacks[stream_name] = callback

        if (
            stream_name in self._ws_connections
            and self._ws_connections[stream_name].open
        ):
            return

        url = f"{self.WS_URL}/{stream_name}"

        try:
            ws = await websockets.connect(url)
            self._ws_connections[stream_name] = ws

            # Start the message handling task
            asyncio.create_task(self._ws_message_handler(stream_name, ws))

            logger.info(f"Connected to WebSocket stream: {stream_name}")
        except Exception as e:
            logger.error(
                f"Failed to connect to WebSocket stream {stream_name}: {str(e)}"
            )
            # Schedule reconnection
            self._schedule_reconnect(stream_name)

    async def _ws_message_handler(
        self, stream_name: str, ws: websockets.WebSocketClientProtocol
    ):
        """
        Handle incoming WebSocket messages.

        Args:
            stream_name: Name of the stream
            ws: WebSocket connection
        """
        try:
            while True:
                message = await ws.recv()
                data = json.loads(message)

                callback = self._ws_callbacks.get(stream_name)
                if callback:
                    await callback(data)
        except websockets.ConnectionClosed:
            logger.warning(f"WebSocket connection closed for stream: {stream_name}")
            self._schedule_reconnect(stream_name)
        except Exception as e:
            logger.error(
                f"Error in WebSocket message handler for stream {stream_name}: {str(e)}"
            )
            self._schedule_reconnect(stream_name)

    def _schedule_reconnect(self, stream_name: str):
        """
        Schedule a reconnection for a WebSocket stream.

        Args:
            stream_name: Name of the stream to reconnect
        """
        if (
            stream_name in self._ws_reconnect_tasks
            and not self._ws_reconnect_tasks[stream_name].done()
        ):
            return

        self._ws_reconnect_tasks[stream_name] = asyncio.create_task(
            self._ws_reconnect(stream_name)
        )

    async def _ws_reconnect(self, stream_name: str):
        """
        Reconnect to a WebSocket stream with exponential backoff.

        Args:
            stream_name: Name of the stream to reconnect
        """
        max_retries = 10
        base_delay = 1.0

        for attempt in range(max_retries):
            delay = base_delay * (2**attempt)
            logger.info(
                f"Reconnecting to {stream_name} in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})"
            )

            await asyncio.sleep(delay)

            try:
                callback = self._ws_callbacks.get(stream_name)
                if callback:
                    await self._ws_connect(stream_name, callback)
                    return
            except Exception as e:
                logger.error(f"Failed to reconnect to {stream_name}: {str(e)}")

        logger.error(
            f"Failed to reconnect to {stream_name} after {max_retries} attempts"
        )

    async def subscribe_depth_stream(
        self, symbol: str, callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ):
        """
        Subscribe to the depth stream for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            callback: Callback function to handle incoming messages
        """
        stream_name = f"{symbol.lower()}@depth"
        await self._ws_connect(stream_name, callback)

    async def subscribe_kline_stream(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ):
        """
        Subscribe to the kline/candlestick stream for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            callback: Callback function to handle incoming messages
        """
        stream_name = f"{symbol.lower()}@kline_{interval}"
        await self._ws_connect(stream_name, callback)

    async def close_all_connections(self):
        """
        Close all WebSocket connections.
        """
        for stream_name, ws in self._ws_connections.items():
            if ws.open:
                await ws.close()
                logger.info(f"Closed WebSocket connection for stream: {stream_name}")

        self._ws_connections = {}

        # Cancel all reconnect tasks
        for task in self._ws_reconnect_tasks.values():
            if not task.done():
                task.cancel()

        self._ws_reconnect_tasks = {}
