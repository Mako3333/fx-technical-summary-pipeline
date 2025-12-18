"""
Timezone utilities for FX-Kline
Handles UTC to JST conversion with DST (daylight saving time) awareness
"""

from datetime import datetime
import pytz
import pandas as pd
from typing import Optional


# Timezone objects
UTC_TZ = pytz.UTC
JST_TZ = pytz.timezone("Asia/Tokyo")
US_EASTERN_TZ = pytz.timezone("US/Eastern")


def utc_to_jst(dt: datetime) -> datetime:
    """
    Convert UTC datetime to JST (Japan Standard Time)

    Args:
        dt: Datetime object in UTC (timezone-naive or timezone-aware)

    Returns:
        Datetime object in JST
    """
    # If naive, assume UTC
    if dt.tzinfo is None:
        dt = UTC_TZ.localize(dt)
    elif dt.tzinfo != UTC_TZ:
        # Convert to UTC first
        dt = dt.astimezone(UTC_TZ)

    # Convert to JST
    return dt.astimezone(JST_TZ)


def convert_dataframe_to_jst(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DataFrame index from UTC to JST

    Args:
        df: DataFrame with DatetimeIndex in UTC

    Returns:
        DataFrame with DatetimeIndex in JST
    """
    # Create a copy to avoid modifying original
    df_copy = df.copy()

    # If index is not timezone-aware, assume UTC
    if df_copy.index.tzinfo is None:
        df_copy.index = df_copy.index.tz_localize(UTC_TZ)

    # Convert to JST
    df_copy.index = df_copy.index.tz_convert(JST_TZ)

    return df_copy


def get_us_market_hours_in_jst() -> dict:
    """
    Get US market hours in JST

    Returns:
        Dictionary with market hours in JST (accounting for DST)
    """
    # Create a reference date to check if DST is active
    now = datetime.now(US_EASTERN_TZ)

    if now.dst():
        # EDT (UTC-4) - Daylight Saving Time
        # US market: 9:30 AM to 4:00 PM EDT
        # In JST: 10:30 PM (previous day) to 5:00 AM (same day)
        return {
            "open": "22:30 JST (previous day)",
            "close": "05:00 JST (same day)",
            "open_utc": "13:30 UTC",
            "close_utc": "20:00 UTC",
            "timezone": "EDT (UTC-4)",
        }
    else:
        # EST (UTC-5) - Standard Time
        # US market: 9:30 AM to 4:00 PM EST
        # In JST: 11:30 PM (previous day) to 6:00 AM (same day)
        return {
            "open": "23:30 JST (previous day)",
            "close": "06:00 JST (same day)",
            "open_utc": "14:30 UTC",
            "close_utc": "21:00 UTC",
            "timezone": "EST (UTC-5)",
        }


def is_us_dst_active(dt: Optional[datetime] = None) -> bool:
    """
    Check if US Daylight Saving Time is active

    Args:
        dt: Datetime to check (if None, uses current time)

    Returns:
        True if DST is active, False otherwise
    """
    if dt is None:
        dt = datetime.now(US_EASTERN_TZ)
    else:
        if dt.tzinfo is None:
            dt = UTC_TZ.localize(dt)
        dt = dt.astimezone(US_EASTERN_TZ)

    return bool(dt.dst())


def format_timestamp_jst(dt: datetime) -> str:
    """
    Format datetime as JST string

    Args:
        dt: Datetime object

    Returns:
        Formatted string in JST
    """
    jst_dt = utc_to_jst(dt) if dt.tzinfo is not None else JST_TZ.localize(dt)
    return jst_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def get_jst_now() -> datetime:
    """Get current time in JST"""
    return datetime.now(JST_TZ)


def get_business_day_offset_jst(days: int) -> datetime:
    """
    Get date that is N business days back from today (JST)

    Args:
        days: Number of business days

    Returns:
        Datetime object in JST
    """
    today_jst = get_jst_now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Use pandas business day offset
    offset = pd.tseries.offsets.BDay(days)
    target_date = today_jst - offset

    return target_date
