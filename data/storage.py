"""
Database storage for trading data.

This module provides functionality for storing and retrieving
trading data from a SQLite database.
"""

import logging
import sqlite3
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a trade record in the database."""

    id: Optional[int] = None
    position_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    entry_time: int = 0
    exit_time: int = 0
    duration_ms: int = 0
    status: str = ""
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

    @property
    def is_win(self) -> bool:
        """Check if the trade was profitable."""
        return self.pnl > 0

    @classmethod
    def from_row(cls, row: Tuple) -> "Trade":
        """
        Create a Trade object from a database row.

        Args:
            row: Database row tuple

        Returns:
            Trade object
        """
        return cls(
            id=row[0],
            position_id=row[1],
            symbol=row[2],
            side=row[3],
            entry_price=row[4],
            exit_price=row[5],
            quantity=row[6],
            pnl=row[7],
            pnl_percentage=row[8],
            entry_time=row[9],
            exit_time=row[10],
            duration_ms=row[11],
            status=row[12],
            target_price=row[13],
            stop_loss=row[14],
        )


class DatabaseManager:
    """
    Manager for database operations.

    This class provides functionality for storing and retrieving
    trading data from a SQLite database.
    """

    def __init__(self, db_path: str):
        """
        Initialize the database manager.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        """Connect to the database."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Disconnect from the database."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.info(f"Disconnected from database: {self.db_path}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def init_db(self) -> None:
        """Initialize the database schema."""
        if not self.conn:
            self.connect()

        try:
            # Create trades table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl REAL,
                    pnl_percentage REAL,
                    entry_time INTEGER NOT NULL,
                    exit_time INTEGER,
                    duration_ms INTEGER,
                    status TEXT NOT NULL,
                    target_price REAL,
                    stop_loss REAL
                )
            """)

            # Create index on position_id
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_position_id
                ON trades (position_id)
            """)

            # Create index on symbol
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades (symbol)
            """)

            # Create index on entry_time
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time
                ON trades (entry_time)
            """)

            # Create KPI table for storing aggregated metrics
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS kpi_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    time_frame TEXT NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    total_pnl REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    average_win REAL NOT NULL,
                    average_loss REAL NOT NULL,
                    largest_win REAL NOT NULL,
                    largest_loss REAL NOT NULL,
                    average_trade_duration_minutes REAL NOT NULL
                )
            """)

            # Create index on time_frame and timestamp
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_kpi_metrics_time_frame_timestamp
                ON kpi_metrics (time_frame, timestamp)
            """)

            self.conn.commit()
            logger.info("Database schema initialized")
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to initialize database schema: {str(e)}")
            raise

    def insert_trade(self, trade: Trade) -> int:
        """
        Insert a new trade into the database.

        Args:
            trade: Trade to insert

        Returns:
            ID of the inserted trade
        """
        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(
                """
                INSERT INTO trades (
                    position_id, symbol, side, entry_price, exit_price,
                    quantity, pnl, pnl_percentage, entry_time, exit_time,
                    duration_ms, status, target_price, stop_loss
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.position_id,
                    trade.symbol,
                    trade.side,
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.pnl,
                    trade.pnl_percentage,
                    trade.entry_time,
                    trade.exit_time,
                    trade.duration_ms,
                    trade.status,
                    trade.target_price,
                    trade.stop_loss,
                ),
            )

            self.conn.commit()
            trade_id = self.cursor.lastrowid
            logger.info(f"Inserted trade {trade.position_id} with ID {trade_id}")
            return trade_id
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to insert trade: {str(e)}")
            raise

    def update_trade(self, trade: Trade) -> bool:
        """
        Update an existing trade in the database.

        Args:
            trade: Trade to update

        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            self.connect()

        if trade.id is None:
            logger.warning("Cannot update trade without ID")
            return False

        try:
            self.cursor.execute(
                """
                UPDATE trades SET
                    position_id = ?,
                    symbol = ?,
                    side = ?,
                    entry_price = ?,
                    exit_price = ?,
                    quantity = ?,
                    pnl = ?,
                    pnl_percentage = ?,
                    entry_time = ?,
                    exit_time = ?,
                    duration_ms = ?,
                    status = ?,
                    target_price = ?,
                    stop_loss = ?
                WHERE id = ?
            """,
                (
                    trade.position_id,
                    trade.symbol,
                    trade.side,
                    trade.entry_price,
                    trade.exit_price,
                    trade.quantity,
                    trade.pnl,
                    trade.pnl_percentage,
                    trade.entry_time,
                    trade.exit_time,
                    trade.duration_ms,
                    trade.status,
                    trade.target_price,
                    trade.stop_loss,
                    trade.id,
                ),
            )

            self.conn.commit()
            logger.info(f"Updated trade {trade.position_id} with ID {trade.id}")
            return True
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to update trade: {str(e)}")
            raise

    def get_trade_by_id(self, trade_id: int) -> Optional[Trade]:
        """
        Get a trade by its ID.

        Args:
            trade_id: Trade ID

        Returns:
            Trade object or None if not found
        """
        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(
                """
                SELECT * FROM trades WHERE id = ?
            """,
                (trade_id,),
            )

            row = self.cursor.fetchone()
            if row:
                return Trade.from_row(row)
            else:
                logger.warning(f"Trade with ID {trade_id} not found")
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to get trade by ID: {str(e)}")
            raise

    def get_trade_by_position_id(self, position_id: str) -> Optional[Trade]:
        """
        Get a trade by its position ID.

        Args:
            position_id: Position ID

        Returns:
            Trade object or None if not found
        """
        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(
                """
                SELECT * FROM trades WHERE position_id = ?
            """,
                (position_id,),
            )

            row = self.cursor.fetchone()
            if row:
                return Trade.from_row(row)
            else:
                logger.warning(f"Trade with position ID {position_id} not found")
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to get trade by position ID: {str(e)}")
            raise

    def get_trades_by_symbol(self, symbol: str) -> List[Trade]:
        """
        Get all trades for a specific symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            List of trades
        """
        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(
                """
                SELECT * FROM trades WHERE symbol = ? ORDER BY entry_time DESC
            """,
                (symbol,),
            )

            rows = self.cursor.fetchall()
            return [Trade.from_row(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get trades by symbol: {str(e)}")
            raise

    def get_trades_by_date_range(self, start_time: int, end_time: int) -> List[Trade]:
        """
        Get all trades within a date range.

        Args:
            start_time: Start timestamp (milliseconds since epoch)
            end_time: End timestamp (milliseconds since epoch)

        Returns:
            List of trades
        """
        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(
                """
                SELECT * FROM trades 
                WHERE entry_time >= ? AND entry_time <= ?
                ORDER BY entry_time DESC
            """,
                (start_time, end_time),
            )

            rows = self.cursor.fetchall()
            return [Trade.from_row(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get trades by date range: {str(e)}")
            raise

    def get_all_trades(self, limit: int = 1000) -> List[Trade]:
        """
        Get all trades.

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trades
        """
        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(
                """
                SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?
            """,
                (limit,),
            )

            rows = self.cursor.fetchall()
            return [Trade.from_row(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Failed to get all trades: {str(e)}")
            raise

    def save_kpi_metrics(
        self,
        time_frame: str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        max_drawdown: float,
        win_rate: float,
        profit_factor: float,
        average_win: float,
        average_loss: float,
        largest_win: float,
        largest_loss: float,
        average_trade_duration_minutes: float,
    ) -> int:
        """
        Save KPI metrics to the database.

        Args:
            time_frame: Time frame (DAILY, WEEKLY, MONTHLY, ALL_TIME)
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            total_pnl: Total profit/loss
            max_drawdown: Maximum drawdown
            win_rate: Win rate
            profit_factor: Profit factor
            average_win: Average win
            average_loss: Average loss
            largest_win: Largest win
            largest_loss: Largest loss
            average_trade_duration_minutes: Average trade duration in minutes

        Returns:
            ID of the inserted KPI metrics
        """
        if not self.conn:
            self.connect()

        try:
            timestamp = int(datetime.now().timestamp() * 1000)

            self.cursor.execute(
                """
                INSERT INTO kpi_metrics (
                    timestamp, time_frame, total_trades, winning_trades,
                    losing_trades, total_pnl, max_drawdown, win_rate,
                    profit_factor, average_win, average_loss, largest_win,
                    largest_loss, average_trade_duration_minutes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    time_frame,
                    total_trades,
                    winning_trades,
                    losing_trades,
                    total_pnl,
                    max_drawdown,
                    win_rate,
                    profit_factor,
                    average_win,
                    average_loss,
                    largest_win,
                    largest_loss,
                    average_trade_duration_minutes,
                ),
            )

            self.conn.commit()
            kpi_id = self.cursor.lastrowid
            logger.info(f"Saved KPI metrics for {time_frame} with ID {kpi_id}")
            return kpi_id
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Failed to save KPI metrics: {str(e)}")
            raise

    def get_latest_kpi_metrics(self, time_frame: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest KPI metrics for a specific time frame.

        Args:
            time_frame: Time frame (DAILY, WEEKLY, MONTHLY, ALL_TIME)

        Returns:
            Dictionary with KPI metrics or None if not found
        """
        if not self.conn:
            self.connect()

        try:
            self.cursor.execute(
                """
                SELECT * FROM kpi_metrics
                WHERE time_frame = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (time_frame,),
            )

            row = self.cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "timestamp": row[1],
                    "time_frame": row[2],
                    "total_trades": row[3],
                    "winning_trades": row[4],
                    "losing_trades": row[5],
                    "total_pnl": row[6],
                    "max_drawdown": row[7],
                    "win_rate": row[8],
                    "profit_factor": row[9],
                    "average_win": row[10],
                    "average_loss": row[11],
                    "largest_win": row[12],
                    "largest_loss": row[13],
                    "average_trade_duration_minutes": row[14],
                }
            else:
                logger.warning(f"No KPI metrics found for time frame {time_frame}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Failed to get latest KPI metrics: {str(e)}")
            raise
