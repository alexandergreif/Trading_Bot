"""
Position management and risk calculation.

This module provides functionality for managing trading positions,
calculating position sizes based on risk parameters, and executing
orders with proper risk management.
"""

import logging
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Awaitable
import time

from trading_bot.exchange.bitunix import (
    BitunixClient,
    OrderType,
    OrderSide,
    OrderResponse,
    Position,
)
from trading_bot.strategy.lsob import LSOBSignal, SignalType

logger = logging.getLogger(__name__)


class PositionStatus(str, Enum):
    """Status of a trading position."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


@dataclass
class TradePosition:
    """Represents a trading position."""

    id: str
    symbol: str
    side: str
    entry_price: float
    quantity: float
    entry_time: int
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[int] = None
    status: PositionStatus = PositionStatus.PENDING
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    entry_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Check if the position is active."""
        return self.status in (PositionStatus.PENDING, PositionStatus.OPEN)

    @property
    def is_long(self) -> bool:
        """Check if the position is a long position."""
        return self.side == OrderSide.BUY

    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """
        Calculate the profit/loss for the position.

        Args:
            current_price: Current market price

        Returns:
            Tuple of (PnL in quote currency, PnL percentage)
        """
        if self.is_long:
            pnl = (current_price - self.entry_price) * self.quantity
            pnl_percentage = (current_price / self.entry_price - 1) * 100
        else:
            pnl = (self.entry_price - current_price) * self.quantity
            pnl_percentage = (self.entry_price / current_price - 1) * 100

        return pnl, pnl_percentage

    def update_status(
        self, status: PositionStatus, exit_price: Optional[float] = None
    ) -> None:
        """
        Update the status of the position.

        Args:
            status: New status
            exit_price: Exit price (if position is being closed)
        """
        self.status = status

        if status == PositionStatus.CLOSED and exit_price is not None:
            self.exit_price = exit_price
            self.exit_time = int(time.time() * 1000)
            self.pnl, self.pnl_percentage = self.calculate_pnl(exit_price)

            logger.info(
                f"Position {self.id} closed: {self.side} {self.quantity} {self.symbol} "
                f"at {self.exit_price} (PnL: {self.pnl:.2f}, {self.pnl_percentage:.2f}%)"
            )


class PositionManager:
    """
    Manager for trading positions.

    This class handles position creation, risk calculation, and order execution
    with proper risk management.
    """

    def __init__(
        self,
        client: BitunixClient,
        risk_per_trade: float = 0.01,
        max_positions: int = 5,
        max_positions_per_symbol: int = 1,
    ):
        """
        Initialize the position manager.

        Args:
            client: Bitunix API client
            risk_per_trade: Maximum risk per trade as a fraction of account balance (0.01 = 1%)
            max_positions: Maximum number of open positions
            max_positions_per_symbol: Maximum number of open positions per symbol
        """
        self.client = client
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.max_positions_per_symbol = max_positions_per_symbol

        self.positions: Dict[str, TradePosition] = {}
        self.next_position_id = 1

    async def get_account_balance(self) -> float:
        """
        Get the total account balance.

        Returns:
            Total account balance in quote currency
        """
        balances = await self.client.get_account_balance()

        # Find USDT balance (or other stablecoin)
        for balance in balances:
            if balance.asset == "USDT":
                return balance.wallet_balance

        # If no USDT balance found, return the first balance
        if balances:
            return balances[0].wallet_balance

        return 0.0

    async def get_active_positions(self) -> List[TradePosition]:
        """
        Get all active positions.

        Returns:
            List of active positions
        """
        return [p for p in self.positions.values() if p.is_active]

    async def get_active_positions_for_symbol(self, symbol: str) -> List[TradePosition]:
        """
        Get active positions for a specific symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            List of active positions for the symbol
        """
        return [
            p for p in self.positions.values() if p.is_active and p.symbol == symbol
        ]

    async def can_open_position(self, symbol: str) -> bool:
        """
        Check if a new position can be opened.

        Args:
            symbol: Trading pair symbol

        Returns:
            True if a new position can be opened, False otherwise
        """
        active_positions = await self.get_active_positions()
        active_positions_for_symbol = await self.get_active_positions_for_symbol(symbol)

        return (
            len(active_positions) < self.max_positions
            and len(active_positions_for_symbol) < self.max_positions_per_symbol
        )

    async def calculate_position_size(
        self, symbol: str, entry_price: float, stop_loss: float
    ) -> float:
        """
        Calculate the position size based on risk parameters.

        Args:
            symbol: Trading pair symbol
            entry_price: Entry price
            stop_loss: Stop loss price

        Returns:
            Position size in base currency
        """
        # Get account balance
        account_balance = await self.get_account_balance()

        # Calculate risk amount
        risk_amount = account_balance * self.risk_per_trade

        # Calculate position size
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            logger.warning(f"Invalid price risk for {symbol}: {price_risk}")
            return 0.0

        position_size = risk_amount / price_risk

        # Round to appropriate precision
        # TODO: Get symbol precision from exchange info
        precision = 5
        position_size = round(position_size, precision)

        logger.info(
            f"Calculated position size for {symbol}: {position_size} "
            f"(risk: {risk_amount:.2f}, price risk: {price_risk:.2f})"
        )

        return position_size

    async def open_position_from_signal(self, signal: LSOBSignal) -> Optional[str]:
        """
        Open a new position based on a trading signal.

        Args:
            signal: Trading signal

        Returns:
            Position ID if successful, None otherwise
        """
        # Check if we can open a new position
        if not await self.can_open_position(signal.symbol):
            logger.warning(
                f"Cannot open position for {signal.symbol}: maximum positions reached"
            )
            return None

        # Check if signal has stop loss
        if signal.stop_loss is None:
            logger.warning(
                f"Cannot open position for {signal.symbol}: no stop loss provided"
            )
            return None

        # Map signal type to order side
        side = OrderSide.BUY if signal.type == SignalType.LONG else OrderSide.SELL

        # Calculate position size
        quantity = await self.calculate_position_size(
            signal.symbol, signal.price, signal.stop_loss
        )

        if quantity <= 0:
            logger.warning(
                f"Cannot open position for {signal.symbol}: invalid position size"
            )
            return None

        # Create position object
        position_id = f"P{self.next_position_id}"
        self.next_position_id += 1

        position = TradePosition(
            id=position_id,
            symbol=signal.symbol,
            side=side,
            entry_price=signal.price,
            quantity=quantity,
            entry_time=int(time.time() * 1000),
            target_price=signal.target_price,
            stop_loss=signal.stop_loss,
            status=PositionStatus.PENDING,
        )

        # Store position
        self.positions[position_id] = position

        # Execute entry order
        try:
            order = await self.client.create_order(
                symbol=signal.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )

            position.entry_order_id = order.order_id
            position.entry_price = (
                order.price or signal.price
            )  # Use actual fill price if available
            position.status = PositionStatus.OPEN

            logger.info(
                f"Opened position {position_id}: {side} {quantity} {signal.symbol} "
                f"at {position.entry_price}"
            )

            # Place take profit and stop loss orders
            await self._place_exit_orders(position)

            return position_id
        except Exception as e:
            logger.error(f"Failed to open position for {signal.symbol}: {str(e)}")
            position.status = PositionStatus.ERROR
            return None

    async def _place_exit_orders(self, position: TradePosition) -> None:
        """
        Place take profit and stop loss orders for a position.

        Args:
            position: Trading position
        """
        if position.status != PositionStatus.OPEN:
            return

        # Determine exit order sides (opposite of entry side)
        exit_side = OrderSide.SELL if position.is_long else OrderSide.BUY

        try:
            # Place stop loss order
            if position.stop_loss is not None:
                sl_order = await self.client.create_order(
                    symbol=position.symbol,
                    side=exit_side,
                    order_type=OrderType.STOP_MARKET,
                    quantity=position.quantity,
                    stop_price=position.stop_loss,
                    reduce_only=True,
                )

                position.sl_order_id = sl_order.order_id
                logger.info(
                    f"Placed stop loss order for position {position.id} at {position.stop_loss}"
                )

            # Place take profit order
            if position.target_price is not None:
                tp_order = await self.client.create_order(
                    symbol=position.symbol,
                    side=exit_side,
                    order_type=OrderType.TAKE_PROFIT_MARKET,
                    quantity=position.quantity,
                    stop_price=position.target_price,
                    reduce_only=True,
                )

                position.tp_order_id = tp_order.order_id
                logger.info(
                    f"Placed take profit order for position {position.id} at {position.target_price}"
                )
        except Exception as e:
            logger.error(
                f"Failed to place exit orders for position {position.id}: {str(e)}"
            )

    async def close_position(self, position_id: str, reason: str = "manual") -> bool:
        """
        Close a position manually.

        Args:
            position_id: Position ID
            reason: Reason for closing the position

        Returns:
            True if successful, False otherwise
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return False

        position = self.positions[position_id]

        if not position.is_active:
            logger.warning(f"Position {position_id} is not active")
            return False

        # Cancel existing exit orders
        if position.tp_order_id:
            try:
                await self.client.cancel_order(position.symbol, position.tp_order_id)
                logger.info(f"Cancelled take profit order for position {position_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to cancel take profit order for position {position_id}: {str(e)}"
                )

        if position.sl_order_id:
            try:
                await self.client.cancel_order(position.symbol, position.sl_order_id)
                logger.info(f"Cancelled stop loss order for position {position_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to cancel stop loss order for position {position_id}: {str(e)}"
                )

        # Execute market order to close position
        exit_side = OrderSide.SELL if position.is_long else OrderSide.BUY

        try:
            order = await self.client.create_order(
                symbol=position.symbol,
                side=exit_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                reduce_only=True,
            )

            # Update position status
            position.update_status(PositionStatus.CLOSED, order.price)

            logger.info(
                f"Closed position {position_id} ({reason}): {exit_side} {position.quantity} "
                f"{position.symbol} at {position.exit_price}"
            )

            return True
        except Exception as e:
            logger.error(f"Failed to close position {position_id}: {str(e)}")
            return False

    async def update_positions(self) -> None:
        """
        Update the status of all active positions.

        This method checks the exchange for position updates and updates
        the local position objects accordingly.
        """
        try:
            # Get current positions from exchange
            exchange_positions = await self.client.get_positions()

            # Get active positions
            active_positions = await self.get_active_positions()

            # Update position status based on exchange data
            for position in active_positions:
                # Find matching exchange position
                exchange_position = next(
                    (p for p in exchange_positions if p.symbol == position.symbol), None
                )

                if exchange_position is None:
                    # Position not found on exchange, might be closed
                    if position.status == PositionStatus.OPEN:
                        # Check if we have exit price information
                        if position.exit_price is not None:
                            position.update_status(
                                PositionStatus.CLOSED, position.exit_price
                            )
                        else:
                            # Get current market price
                            # TODO: Implement this
                            pass
                else:
                    # Position found on exchange
                    if abs(exchange_position.position_amt) < 0.0001:
                        # Position is closed on exchange
                        if position.status == PositionStatus.OPEN:
                            # Update position status
                            position.update_status(
                                PositionStatus.CLOSED, exchange_position.mark_price
                            )
                    else:
                        # Position is still open
                        # Update PnL
                        position.pnl, position.pnl_percentage = position.calculate_pnl(
                            exchange_position.mark_price
                        )
        except Exception as e:
            logger.error(f"Failed to update positions: {str(e)}")
