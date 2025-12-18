"""
Input validators for FX-Kline
"""

from typing import List


# Supported currency pairs
SUPPORTED_CURRENCY_PAIRS = {
    "USDJPY": "USD/JPY",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "AUDUSD": "AUD/USD",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "AUDJPY": "AUD/JPY",
    "XAUUSD": "XAU/USD (Gold)",
}

# Supported timeframes
SUPPORTED_TIMEFRAMES = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day",
    "1wk": "1 week",
    "1mo": "1 month",
}

# For yfinance API
YFINANCE_PAIR_SUFFIX = "=X"

# Preset timeframes for UI
PRESET_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# Preset currency pairs for UI
PRESET_CURRENCY_PAIRS = [
    "USDJPY",
    "EURUSD",
    "GBPUSD",
    "AUDUSD",
    "EURJPY",
    "GBPJPY",
    "AUDJPY",
    "XAUUSD",
]

# Default business-day lookback per timeframe
DEFAULT_TIMEFRAME_BUSINESS_DAYS = {
    "1m": 1,
    "5m": 1,
    "15m": 1,
    "30m": 2,
    "1h": 5,
    "4h": 10,
    "1d": 20,
    "1wk": 26,
    "1mo": 52,
}

FALLBACK_TIMEFRAME_BUSINESS_DAYS = 5


class ValidationError(Exception):
    """Custom validation error"""

    pass


def validate_currency_pair(pair: str) -> str:
    """
    Validate currency pair

    Args:
        pair: Currency pair code (e.g., 'USDJPY')

    Returns:
        Formatted pair string with suffix for yfinance

    Raises:
        ValidationError: If pair is not supported
    """
    pair_upper = pair.upper()

    if pair_upper not in SUPPORTED_CURRENCY_PAIRS:
        raise ValidationError(
            f"Unsupported currency pair: {pair}\n"
            f"Supported pairs: {', '.join(SUPPORTED_CURRENCY_PAIRS.keys())}"
        )

    # Special case for gold (XAUUSD) - use GC=F (Gold Futures)
    # GC=F is the only reliable gold price source in yfinance
    # Note: GC=F has different trading hours than FX (commodity futures market)
    if pair_upper == "XAUUSD":
        return "GC=F"

    # Add yfinance suffix for FX currency pairs
    if not pair_upper.endswith(YFINANCE_PAIR_SUFFIX):
        return f"{pair_upper}{YFINANCE_PAIR_SUFFIX}"
    return pair_upper


def validate_timeframe(interval: str) -> str:
    """
    Validate timeframe

    Args:
        interval: Timeframe (e.g., '1h', '1d')

    Returns:
        Validated timeframe string

    Raises:
        ValidationError: If timeframe is not supported
    """
    interval_lower = interval.lower()

    if interval_lower not in SUPPORTED_TIMEFRAMES:
        raise ValidationError(
            f"Unsupported timeframe: {interval}\n"
            f"Supported timeframes: {', '.join(SUPPORTED_TIMEFRAMES.keys())}"
        )

    return interval_lower


def validate_period(period: str) -> str:
    """
    Validate period string

    Args:
        period: Period (e.g., '30d', '1mo')

    Returns:
        Validated period string
    """
    # yfinance accepts periods like: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    period_lower = period.lower()

    # Extract the unit
    import re

    match = re.match(r"^(\d+)([a-z]+)$", period_lower)

    if not match and period_lower != "ytd" and period_lower != "max":
        raise ValidationError(
            f"Invalid period format: {period}\n"
            f"Valid formats: 1d, 5d, 1mo, 3mo, 1y, 2y, ytd, max"
        )

    return period_lower


def validate_business_days(days: int) -> int:
    """
    Validate number of business days

    Args:
        days: Number of business days

    Returns:
        Validated days count

    Raises:
        ValidationError: If days is invalid
    """
    if not isinstance(days, int) or days <= 0:
        raise ValidationError(f"Business days must be a positive integer, got: {days}")

    if days > 500:
        raise ValidationError(f"Business days must be <= 500, got: {days}")

    return days


def get_supported_pairs() -> dict:
    """Get all supported currency pairs"""
    return SUPPORTED_CURRENCY_PAIRS.copy()


def get_supported_timeframes() -> dict:
    """Get all supported timeframes"""
    return SUPPORTED_TIMEFRAMES.copy()


def get_preset_pairs() -> List[str]:
    """Get preset currency pairs for UI"""
    return PRESET_CURRENCY_PAIRS.copy()


def get_preset_timeframes() -> List[str]:
    """Get preset timeframes for UI"""
    return PRESET_TIMEFRAMES.copy()


def get_default_business_days_for_timeframe(interval: str) -> int:
    """Return preferred business-day lookback for a timeframe"""
    interval_lower = interval.lower()
    return DEFAULT_TIMEFRAME_BUSINESS_DAYS.get(
        interval_lower, FALLBACK_TIMEFRAME_BUSINESS_DAYS
    )
