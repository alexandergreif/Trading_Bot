"""
Authentication utilities for Bitunix API.

This module provides functions for generating authentication signatures
required for Bitunix API requests.
"""

import hmac
import hashlib
import time
from typing import Dict, Any, Optional


def generate_signature(
    api_secret: str, params: Dict[str, Any], timestamp: Optional[int] = None
) -> str:
    """
    Generate HMAC SHA256 signature for Bitunix API authentication.

    Args:
        api_secret: The API secret key
        params: Request parameters to sign
        timestamp: Optional timestamp to use (defaults to current time)

    Returns:
        Hexadecimal signature string
    """
    if timestamp is None:
        timestamp = int(time.time() * 1000)

    # Add timestamp to parameters
    params["timestamp"] = timestamp

    # Create query string
    query_string = "&".join([f"{key}={params[key]}" for key in sorted(params.keys())])

    # Create signature
    signature = hmac.new(
        api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return signature


def add_auth_headers(
    api_key: str, api_secret: str, params: Dict[str, Any]
) -> Dict[str, str]:
    """
    Add authentication headers to a request.

    Args:
        api_key: The API key
        api_secret: The API secret key
        params: Request parameters

    Returns:
        Dictionary of headers to add to the request
    """
    timestamp = int(time.time() * 1000)
    signature = generate_signature(api_secret, params, timestamp)

    return {
        "X-API-KEY": api_key,
        "X-TIMESTAMP": str(timestamp),
        "X-SIGNATURE": signature,
    }
