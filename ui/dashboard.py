"""
Streamlit dashboard for monitoring trading performance.

This module provides a web-based dashboard for monitoring
trading performance and visualizing trading data.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import threading
import asyncio
from datetime import datetime, timedelta

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from trading_bot.data.storage import DatabaseManager, Trade
from trading_bot.trading.metrics import TimeFrame
from trading_bot.exchange.bitunix import BitunixClient

logger = logging.getLogger(__name__)


def format_currency(value: float) -> str:
    """Format a currency value."""
    return f"${value:.2f}"


def format_percentage(value: float) -> str:
    """Format a percentage value."""
    return f"{value:.2f}%"


def create_equity_curve(trades: List[Trade]) -> go.Figure:
    """
    Create an equity curve chart.

    Args:
        trades: List of trades

    Returns:
        Plotly figure
    """
    if not trades:
        # Create empty chart
        fig = go.Figure()
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity",
            height=400,
        )
        return fig

    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)

    # Calculate cumulative PnL
    dates = []
    equity = []
    cumulative_pnl = 0.0

    for trade in sorted_trades:
        if trade.exit_time is None:
            continue

        cumulative_pnl += trade.pnl
        dates.append(datetime.fromtimestamp(trade.exit_time / 1000))
        equity.append(cumulative_pnl)

    # Create DataFrame
    df = pd.DataFrame({"Date": dates, "Equity": equity})

    # Create figure
    fig = px.line(
        df,
        x="Date",
        y="Equity",
        title="Equity Curve",
        height=400,
    )

    # Add annotations for significant points
    if len(equity) > 0:
        max_equity = max(equity)
        max_index = equity.index(max_equity)
        min_equity = min(equity)
        min_index = equity.index(min_equity)

        fig.add_annotation(
            x=dates[max_index],
            y=max_equity,
            text=f"Max: {format_currency(max_equity)}",
            showarrow=True,
            arrowhead=1,
        )

        fig.add_annotation(
            x=dates[min_index],
            y=min_equity,
            text=f"Min: {format_currency(min_equity)}",
            showarrow=True,
            arrowhead=1,
        )

    return fig


def create_win_loss_chart(trades: List[Trade]) -> go.Figure:
    """
    Create a win/loss chart.

    Args:
        trades: List of trades

    Returns:
        Plotly figure
    """
    if not trades:
        # Create empty chart
        fig = go.Figure()
        fig.update_layout(
            title="Win/Loss Distribution",
            xaxis_title="Trade",
            yaxis_title="PnL",
            height=400,
        )
        return fig

    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)

    # Extract PnL values
    trade_ids = [f"T{i + 1}" for i in range(len(sorted_trades))]
    pnl_values = [trade.pnl for trade in sorted_trades]
    colors = ["green" if pnl >= 0 else "red" for pnl in pnl_values]

    # Create figure
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=trade_ids,
            y=pnl_values,
            marker_color=colors,
            name="PnL",
        )
    )

    fig.update_layout(
        title="Win/Loss Distribution",
        xaxis_title="Trade",
        yaxis_title="PnL",
        height=400,
    )

    return fig


def create_symbol_performance_chart(trades: List[Trade]) -> go.Figure:
    """
    Create a symbol performance chart.

    Args:
        trades: List of trades

    Returns:
        Plotly figure
    """
    if not trades:
        # Create empty chart
        fig = go.Figure()
        fig.update_layout(
            title="Symbol Performance",
            xaxis_title="Symbol",
            yaxis_title="PnL",
            height=400,
        )
        return fig

    # Group trades by symbol
    symbol_pnl = {}
    for trade in trades:
        if trade.symbol not in symbol_pnl:
            symbol_pnl[trade.symbol] = 0.0
        symbol_pnl[trade.symbol] += trade.pnl

    # Sort symbols by PnL
    sorted_symbols = sorted(
        symbol_pnl.keys(), key=lambda s: symbol_pnl[s], reverse=True
    )
    sorted_pnl = [symbol_pnl[s] for s in sorted_symbols]
    colors = ["green" if pnl >= 0 else "red" for pnl in sorted_pnl]

    # Create figure
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sorted_symbols,
            y=sorted_pnl,
            marker_color=colors,
            name="PnL",
        )
    )

    fig.update_layout(
        title="Symbol Performance",
        xaxis_title="Symbol",
        yaxis_title="PnL",
        height=400,
    )

    return fig


def create_win_rate_chart(trades: List[Trade], window: int = 20) -> go.Figure:
    """
    Create a rolling win rate chart.

    Args:
        trades: List of trades
        window: Window size for rolling win rate

    Returns:
        Plotly figure
    """
    if not trades or len(trades) < window:
        # Create empty chart
        fig = go.Figure()
        fig.update_layout(
            title=f"Rolling Win Rate (Window: {window})",
            xaxis_title="Trade",
            yaxis_title="Win Rate",
            height=400,
        )
        return fig

    # Sort trades by exit time
    sorted_trades = sorted(trades, key=lambda t: t.exit_time)

    # Calculate rolling win rate
    win_rates = []
    trade_ids = []

    for i in range(window, len(sorted_trades) + 1):
        window_trades = sorted_trades[i - window : i]
        wins = sum(1 for t in window_trades if t.pnl > 0)
        win_rate = wins / window
        win_rates.append(win_rate)
        trade_ids.append(f"T{i}")

    # Create figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=trade_ids,
            y=win_rates,
            mode="lines+markers",
            name="Win Rate",
            line=dict(color="blue"),
        )
    )

    # Add threshold line
    fig.add_shape(
        type="line",
        x0=trade_ids[0],
        y0=0.5,
        x1=trade_ids[-1],
        y1=0.5,
        line=dict(color="red", dash="dash"),
    )

    fig.update_layout(
        title=f"Rolling Win Rate (Window: {window})",
        xaxis_title="Trade",
        yaxis_title="Win Rate",
        height=400,
    )

    return fig


def create_trade_duration_chart(trades: List[Trade]) -> go.Figure:
    """
    Create a trade duration chart.

    Args:
        trades: List of trades

    Returns:
        Plotly figure
    """
    if not trades:
        # Create empty chart
        fig = go.Figure()
        fig.update_layout(
            title="Trade Duration Distribution",
            xaxis_title="Duration (minutes)",
            yaxis_title="Count",
            height=400,
        )
        return fig

    # Calculate trade durations in minutes
    durations = [
        trade.duration_ms / (1000 * 60)
        for trade in trades
        if trade.duration_ms is not None
    ]

    if not durations:
        # Create empty chart
        fig = go.Figure()
        fig.update_layout(
            title="Trade Duration Distribution",
            xaxis_title="Duration (minutes)",
            yaxis_title="Count",
            height=400,
        )
        return fig

    # Create figure
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=durations,
            nbinsx=20,
            marker_color="blue",
            name="Duration",
        )
    )

    fig.update_layout(
        title="Trade Duration Distribution",
        xaxis_title="Duration (minutes)",
        yaxis_title="Count",
        height=400,
    )

    return fig


def create_performance_metrics_table(
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
) -> go.Figure:
    """
    Create a performance metrics table.

    Args:
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
        Plotly figure
    """
    # Create table data
    metrics = [
        "Total Trades",
        "Winning Trades",
        "Losing Trades",
        "Total PnL",
        "Max Drawdown",
        "Win Rate",
        "Profit Factor",
        "Average Win",
        "Average Loss",
        "Largest Win",
        "Largest Loss",
        "Avg Trade Duration",
    ]

    values = [
        str(total_trades),
        str(winning_trades),
        str(losing_trades),
        format_currency(total_pnl),
        format_currency(max_drawdown),
        format_percentage(win_rate * 100),
        f"{profit_factor:.2f}",
        format_currency(average_win),
        format_currency(average_loss),
        format_currency(largest_win),
        format_currency(largest_loss),
        f"{average_trade_duration_minutes:.2f} min",
    ]

    # Create figure
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="royalblue",
                    align="left",
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=[metrics, values],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(
        title="Performance Metrics",
        height=400,
    )

    return fig


def create_active_positions_table(positions: List[Dict[str, Any]]) -> go.Figure:
    """
    Create an active positions table.

    Args:
        positions: List of active positions

    Returns:
        Plotly figure
    """
    if not positions:
        # Create empty table
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=[
                            "ID",
                            "Symbol",
                            "Side",
                            "Entry Price",
                            "Current Price",
                            "Quantity",
                            "PnL",
                            "PnL %",
                        ],
                        fill_color="royalblue",
                        align="left",
                        font=dict(color="white", size=12),
                    ),
                    cells=dict(
                        values=[[], [], [], [], [], [], [], []],
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Active Positions",
            height=400,
        )

        return fig

    # Extract position data
    ids = [p["id"] for p in positions]
    symbols = [p["symbol"] for p in positions]
    sides = [p["side"] for p in positions]
    entry_prices = [p["entry_price"] for p in positions]
    current_prices = [p["current_price"] for p in positions]
    quantities = [p["quantity"] for p in positions]
    pnls = [p["pnl"] for p in positions]
    pnl_percentages = [p["pnl_percentage"] for p in positions]

    # Format values
    entry_prices_fmt = [f"${price:.2f}" for price in entry_prices]
    current_prices_fmt = [f"${price:.2f}" for price in current_prices]
    quantities_fmt = [f"{qty:.6f}" for qty in quantities]
    pnls_fmt = [f"${pnl:.2f}" for pnl in pnls]
    pnl_percentages_fmt = [f"{pct:.2f}%" for pct in pnl_percentages]

    # Create figure
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "ID",
                        "Symbol",
                        "Side",
                        "Entry Price",
                        "Current Price",
                        "Quantity",
                        "PnL",
                        "PnL %",
                    ],
                    fill_color="royalblue",
                    align="left",
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=[
                        ids,
                        symbols,
                        sides,
                        entry_prices_fmt,
                        current_prices_fmt,
                        quantities_fmt,
                        pnls_fmt,
                        pnl_percentages_fmt,
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(
        title="Active Positions",
        height=400,
    )

    return fig


def create_recent_trades_table(trades: List[Trade], limit: int = 10) -> go.Figure:
    """
    Create a recent trades table.

    Args:
        trades: List of trades
        limit: Maximum number of trades to show

    Returns:
        Plotly figure
    """
    if not trades:
        # Create empty table
        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=[
                            "ID",
                            "Symbol",
                            "Side",
                            "Entry Price",
                            "Exit Price",
                            "Quantity",
                            "PnL",
                            "PnL %",
                            "Duration",
                        ],
                        fill_color="royalblue",
                        align="left",
                        font=dict(color="white", size=12),
                    ),
                    cells=dict(
                        values=[[], [], [], [], [], [], [], [], []],
                        fill_color="lavender",
                        align="left",
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Recent Trades",
            height=400,
        )

        return fig

    # Sort trades by exit time (descending)
    sorted_trades = sorted(trades, key=lambda t: t.exit_time or 0, reverse=True)

    # Limit the number of trades
    recent_trades = sorted_trades[:limit]

    # Extract trade data
    ids = [t.position_id for t in recent_trades]
    symbols = [t.symbol for t in recent_trades]
    sides = [t.side for t in recent_trades]
    entry_prices = [t.entry_price for t in recent_trades]
    exit_prices = [t.exit_price or 0.0 for t in recent_trades]
    quantities = [t.quantity for t in recent_trades]
    pnls = [t.pnl or 0.0 for t in recent_trades]
    pnl_percentages = [t.pnl_percentage or 0.0 for t in recent_trades]
    durations = [
        t.duration_ms / (1000 * 60) if t.duration_ms else 0.0 for t in recent_trades
    ]

    # Format values
    entry_prices_fmt = [f"${price:.2f}" for price in entry_prices]
    exit_prices_fmt = [f"${price:.2f}" for price in exit_prices]
    quantities_fmt = [f"{qty:.6f}" for qty in quantities]
    pnls_fmt = [f"${pnl:.2f}" for pnl in pnls]
    pnl_percentages_fmt = [f"{pct:.2f}%" for pct in pnl_percentages]
    durations_fmt = [f"{dur:.2f} min" for dur in durations]

    # Create figure
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "ID",
                        "Symbol",
                        "Side",
                        "Entry Price",
                        "Exit Price",
                        "Quantity",
                        "PnL",
                        "PnL %",
                        "Duration",
                    ],
                    fill_color="royalblue",
                    align="left",
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=[
                        ids,
                        symbols,
                        sides,
                        entry_prices_fmt,
                        exit_prices_fmt,
                        quantities_fmt,
                        pnls_fmt,
                        pnl_percentages_fmt,
                        durations_fmt,
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(
        title="Recent Trades",
        height=400,
    )

    return fig


class Dashboard:
    """
    Streamlit dashboard for monitoring trading performance.
    """

    def __init__(
        self,
        db_path: str,
        api_key: str,
        api_secret: str,
        refresh_interval: int = 60,
    ):
        """
        Initialize the dashboard.

        Args:
            db_path: Path to the SQLite database file
            api_key: Bitunix API key
            api_secret: Bitunix API secret
            refresh_interval: Dashboard refresh interval in seconds
        """
        self.db_path = db_path
        self.api_key = api_key
        self.api_secret = api_secret
        self.refresh_interval = refresh_interval

        self.db_manager = DatabaseManager(db_path)
        self.client = BitunixClient(api_key, api_secret)

        self.active_positions = []
        self.last_update_time = 0

    def _load_trades(self, time_frame: TimeFrame) -> List[Trade]:
        """
        Load trades from the database.

        Args:
            time_frame: Time frame to load trades for

        Returns:
            List of trades
        """
        now = datetime.now()

        if time_frame == TimeFrame.DAILY:
            start_time = datetime(now.year, now.month, now.day).timestamp() * 1000
            end_time = now.timestamp() * 1000
        elif time_frame == TimeFrame.WEEKLY:
            start_time = (now - timedelta(days=now.weekday())).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp() * 1000
            end_time = now.timestamp() * 1000
        elif time_frame == TimeFrame.MONTHLY:
            start_time = datetime(now.year, now.month, 1).timestamp() * 1000
            end_time = now.timestamp() * 1000
        else:  # ALL_TIME
            start_time = 0
            end_time = now.timestamp() * 1000

        return self.db_manager.get_trades_by_date_range(start_time, end_time)

    async def _update_active_positions(self) -> None:
        """Update active positions from the exchange."""
        try:
            # Get positions from exchange
            positions = await self.client.get_positions()

            # Filter active positions
            active_positions = []
            for position in positions:
                if abs(position.position_amt) > 0.0001:
                    # Calculate PnL
                    entry_price = position.entry_price
                    current_price = position.mark_price
                    quantity = abs(position.position_amt)

                    if position.position_amt > 0:  # Long position
                        side = "BUY"
                        pnl = (current_price - entry_price) * quantity
                        pnl_percentage = (current_price / entry_price - 1) * 100
                    else:  # Short position
                        side = "SELL"
                        pnl = (entry_price - current_price) * quantity
                        pnl_percentage = (entry_price / current_price - 1) * 100

                    active_positions.append(
                        {
                            "id": f"P{len(active_positions) + 1}",
                            "symbol": position.symbol,
                            "side": side,
                            "entry_price": entry_price,
                            "current_price": current_price,
                            "quantity": quantity,
                            "pnl": pnl,
                            "pnl_percentage": pnl_percentage,
                        }
                    )

            self.active_positions = active_positions
            self.last_update_time = int(time.time())
        except Exception as e:
            logger.error(f"Failed to update active positions: {str(e)}")

    def _calculate_performance_metrics(self, trades: List[Trade]) -> Dict[str, Any]:
        """
        Calculate performance metrics from trades.

        Args:
            trades: List of trades

        Returns:
            Dictionary with performance metrics
        """
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_pnl": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "average_trade_duration_minutes": 0.0,
            }

        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades
        total_pnl = sum(t.pnl for t in trades)

        # Calculate win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # Calculate profit factor
        total_profit = sum(t.pnl for t in trades if t.pnl > 0)
        total_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Calculate average win/loss
        winning_trades_list = [t for t in trades if t.pnl > 0]
        losing_trades_list = [t for t in trades if t.pnl < 0]

        average_win = (
            sum(t.pnl for t in winning_trades_list) / len(winning_trades_list)
            if winning_trades_list
            else 0.0
        )
        average_loss = (
            sum(t.pnl for t in losing_trades_list) / len(losing_trades_list)
            if losing_trades_list
            else 0.0
        )

        # Calculate largest win/loss
        largest_win = max([t.pnl for t in trades if t.pnl > 0], default=0.0)
        largest_loss = min([t.pnl for t in trades if t.pnl < 0], default=0.0)

        # Calculate average trade duration
        durations = [
            t.duration_ms / (1000 * 60) for t in trades if t.duration_ms is not None
        ]
        average_trade_duration_minutes = (
            sum(durations) / len(durations) if durations else 0.0
        )

        # Calculate max drawdown
        sorted_trades = sorted(trades, key=lambda t: t.exit_time)
        cumulative_pnl = 0.0
        peak = 0.0
        drawdown = 0.0
        max_drawdown = 0.0

        for trade in sorted_trades:
            cumulative_pnl += trade.pnl

            if cumulative_pnl > peak:
                peak = cumulative_pnl
                drawdown = 0.0
            else:
                drawdown = peak - cumulative_pnl

            max_drawdown = max(max_drawdown, drawdown)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "average_trade_duration_minutes": average_trade_duration_minutes,
        }

    def run(self) -> None:
        """Run the dashboard."""
        st.set_page_config(
            page_title="Trading Bot Dashboard",
            page_icon="📈",
            layout="wide",
        )

        st.title("Bitcoin Futures Trading Bot Dashboard")

        # Sidebar
        st.sidebar.header("Settings")

        time_frame = st.sidebar.selectbox(
            "Time Frame",
            [
                TimeFrame.DAILY.value,
                TimeFrame.WEEKLY.value,
                TimeFrame.MONTHLY.value,
                TimeFrame.ALL_TIME.value,
            ],
            index=3,  # Default to ALL_TIME
        )

        # Convert string to TimeFrame enum
        time_frame_enum = TimeFrame(time_frame)

        # Load trades
        trades = self._load_trades(time_frame_enum)

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(trades)

        # Update active positions
        if time.time() - self.last_update_time > self.refresh_interval:
            # Run in a separate thread to avoid blocking the UI
            def update_positions():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._update_active_positions())
                loop.close()

            threading.Thread(target=update_positions).start()

        # Display KPI metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total PnL",
                value=format_currency(metrics["total_pnl"]),
                delta=None,
            )

        with col2:
            st.metric(
                label="Win Rate",
                value=format_percentage(metrics["win_rate"] * 100),
                delta=None,
            )

        with col3:
            st.metric(
                label="Profit Factor",
                value=f"{metrics['profit_factor']:.2f}",
                delta=None,
            )

        with col4:
            st.metric(
                label="Max Drawdown",
                value=format_currency(metrics["max_drawdown"]),
                delta=None,
            )

        # Display charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_equity_curve(trades),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                create_win_loss_chart(trades),
                use_container_width=True,
            )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_symbol_performance_chart(trades),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                create_win_rate_chart(trades),
                use_container_width=True,
            )

        # Display tables
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_performance_metrics_table(
                    metrics["total_trades"],
                    metrics["winning_trades"],
                    metrics["losing_trades"],
                    metrics["total_pnl"],
                    metrics["max_drawdown"],
                    metrics["win_rate"],
                    metrics["profit_factor"],
                    metrics["average_win"],
                    metrics["average_loss"],
                    metrics["largest_win"],
                    metrics["largest_loss"],
                    metrics["average_trade_duration_minutes"],
                ),
                use_container_width=True,
            )

        with col2:
            st.plotly_chart(
                create_active_positions_table(self.active_positions),
                use_container_width=True,
            )

        # Display recent trades
        st.plotly_chart(
            create_recent_trades_table(trades),
            use_container_width=True,
        )

        # Add refresh button
        if st.button("Refresh Data"):
            st.experimental_rerun()

        # Add auto-refresh
        st.sidebar.write(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

        if auto_refresh:
            time.sleep(self.refresh_interval)
            st.experimental_rerun()


def run_dashboard(
    db_path: str,
    api_key: str,
    api_secret: str,
    refresh_interval: int = 60,
) -> None:
    """
    Run the dashboard.

    Args:
        db_path: Path to the SQLite database file
        api_key: Bitunix API key
        api_secret: Bitunix API secret
        refresh_interval: Dashboard refresh interval in seconds
    """
    dashboard = Dashboard(db_path, api_key, api_secret, refresh_interval)
    dashboard.run()
