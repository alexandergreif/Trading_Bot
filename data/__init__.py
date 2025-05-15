"""
Data module for database operations.

This module provides functionality for storing and retrieving
trading data from a SQLite database.
"""

from .storage import DatabaseManager, Trade

__all__ = ["DatabaseManager", "Trade"]
