"""
Business day utilities for FX-Kline
Handles filtering weekends and calculating past N business days
"""

import pandas as pd
from datetime import date, datetime, time, timedelta
from typing import Tuple, Optional
import pytz

JST_TZ = pytz.timezone("Asia/Tokyo")
US_EASTERN_TZ = pytz.timezone("US/Eastern")
EUROPE_TZ = pytz.timezone("Europe/London")


def _ensure_jst(dt: datetime) -> datetime:
    """Return a JST-aware datetime for inputs that may be naive or in other zones."""
    if dt.tzinfo is None:
        return JST_TZ.localize(dt)
    return dt.astimezone(JST_TZ)


def is_combined_dst_active(dt: Optional[datetime] = None) -> bool:
    """
    Check if both US and Europe are in Daylight Saving Time
    Uses the later of the two for standard (both must be in DST)

    - Europe DST: Last Sunday of March to Last Sunday of October
    - US DST: Second Sunday of March to First Sunday of November
    - Combined DST: Last Sunday of March to First Sunday of November

    Args:
        dt: Datetime to check (if None, uses current time in JST)

    Returns:
        True if both US and Europe are in DST, False otherwise
    """
    if dt is None:
        dt = datetime.now(JST_TZ)
    else:
        # Ensure dt is timezone-aware
        if dt.tzinfo is None:
            dt = JST_TZ.localize(dt)

    # Check both US and Europe DST status
    us_dt = dt.astimezone(US_EASTERN_TZ)
    eu_dt = dt.astimezone(EUROPE_TZ)

    us_dst = bool(us_dt.dst())
    eu_dst = bool(eu_dt.dst())

    # Both must be in DST (align to later standard)
    return us_dst and eu_dst


def get_fx_market_close_hour_jst(dt: Optional[datetime] = None) -> int:
    """
    Get FX market close hour in JST (Saturday morning)

    - During combined DST: Saturday 6:00 JST (NY closes at 17:00 EDT)
    - During standard time: Saturday 7:00 JST (NY closes at 17:00 EST)

    Args:
        dt: Datetime to check (if None, uses current time)

    Returns:
        Hour in JST (6 or 7)
    """
    if is_combined_dst_active(dt):
        return 6  # EDT: UTC-4, so 21:00 UTC Friday = 06:00 JST Saturday
    else:
        return 7  # EST: UTC-5, so 22:00 UTC Friday = 07:00 JST Saturday


def filter_business_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove weekend data from DataFrame (simple weekday filter)

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with only weekday data (Monday=0, Friday=4)
    """
    if df.empty:
        return df

    # Check if index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    # Filter to business days only (Monday=0, Sunday=6)
    return df[df.index.dayofweek < 5]


def is_gold_futures(symbol: str) -> bool:
    """
    Check if symbol is Gold Futures (GC=F)

    Args:
        symbol: yfinance symbol string

    Returns:
        True if symbol is gold futures
    """
    return symbol == "GC=F"


def filter_business_days_fx(
    df: pd.DataFrame, interval: str, symbol: str = ""
) -> pd.DataFrame:
    """
    Remove weekend data considering market-specific trading hours

    For Gold Futures (GC=F):
    - Trades Sun 18:00 ET - Fri 17:00 ET (commodity futures market)
    - Uses simple weekday filter (no Saturday data exists)

    For FX pairs:
    - For daily/weekly/monthly intervals: Uses simple weekday filter
    - For intraday intervals (minutes/hours): Considers FX market hours with DST
    - FX Market hours (JST):
      - Close: Saturday 6:00 (DST) or 7:00 (Standard) - NY market closes at 17:00
      - Open: Monday 6:00 (DST) or 7:00 (Standard) - Sydney market opens
    - This keeps Friday night session data that extends into Saturday morning

    Args:
        df: DataFrame with DatetimeIndex in JST
        interval: Timeframe string (e.g., '1h', '15m', '1d')
        symbol: yfinance symbol (e.g., 'GC=F', 'USDJPY=X')

    Returns:
        DataFrame with weekend data filtered appropriately
    """
    if df.empty:
        return df

    # Check if index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    # For Gold Futures: Use simple weekday filter (commodity market hours, no Saturday data)
    if is_gold_futures(symbol):
        return df[df.index.dayofweek < 5]

    # For FX pairs: For daily/weekly/monthly data, use simple weekday filter
    if interval.endswith("d") or interval.endswith("wk") or interval.endswith("mo"):
        return df[df.index.dayofweek < 5]

    # For FX intraday data, consider FX market trading hours
    def is_fx_trading_time(dt: datetime) -> bool:
        """Check if datetime is within FX market trading hours"""
        weekday = dt.dayofweek  # Monday=0, Sunday=6
        hour = dt.hour

        # Monday to Friday: All hours are trading time
        if weekday < 5:
            return True

        # Get market close hour based on DST
        market_close_hour = get_fx_market_close_hour_jst(dt)

        # Saturday: Before market close (6:00 or 7:00 JST)
        # This is the continuation of Friday night session
        if weekday == 5 and hour < market_close_hour:
            return True

        # Sunday: After market open (6:00 or 7:00 JST)
        # This is the beginning of Monday session
        if weekday == 6 and hour >= market_close_hour:
            return True

        return False

    return df[df.index.map(is_fx_trading_time)]


def get_fx_trading_date(dt_jst: datetime) -> date:
    """
    Map a JST datetime to the FX trading date using NY close (US/Eastern 17:00) as the boundary.

    Naive inputs are assumed to be JST. DST is handled by converting to US/Eastern and checking
    the 17:00 local-time cutoff; times on/after the cutoff belong to the next trading date.
    """
    jst_dt = _ensure_jst(dt_jst)
    eastern_dt = jst_dt.astimezone(US_EASTERN_TZ)
    cutoff = time(17, 0)

    trading_date = eastern_dt.date()
    if eastern_dt.timetz().replace(tzinfo=None) >= cutoff:
        trading_date = trading_date + timedelta(days=1)

    return trading_date


def get_fx_trading_dates(index: pd.DatetimeIndex) -> pd.Series:
    """
    Vectorized helper to derive FX trading dates for a DatetimeIndex in JST.

    Args:
        index: DatetimeIndex (assumed JST-aware; naive will be treated as JST)

    Returns:
        Series indexed like the input with trading-date labels
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a DatetimeIndex")

    return pd.Series(
        index.map(get_fx_trading_date), index=index, name="fx_trading_date"
    )


def is_business_day(dt: datetime) -> bool:
    """
    Check if a datetime is a business day (weekday)

    Args:
        dt: Datetime object

    Returns:
        True if weekday (Monday-Friday), False if weekend
    """
    # weekday(): Monday=0, Sunday=6
    return dt.weekday() < 5


def get_business_days_back(
    days: int, as_of_date: Optional[datetime] = None
) -> datetime:
    """
    Calculate the date that is N business days back

    Args:
        days: Number of business days back
        as_of_date: Reference date (if None, uses today in JST)

    Returns:
        Datetime object for N business days back
    """
    if as_of_date is None:
        as_of_date = datetime.now(JST_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif as_of_date.tzinfo is None:
        as_of_date = JST_TZ.localize(
            as_of_date.replace(hour=0, minute=0, second=0, microsecond=0)
        )

    # Use pandas BDay for accurate business day calculation
    bday_offset = pd.tseries.offsets.BDay(days)
    target_date = as_of_date - bday_offset

    return target_date


def get_business_day_range(
    days: int, as_of_date: Optional[datetime] = None
) -> Tuple[datetime, datetime]:
    """
    Get date range for N business days back to today

    Args:
        days: Number of business days
        as_of_date: End date reference (if None, uses today in JST)

    Returns:
        Tuple of (start_date, end_date) in JST
    """
    if as_of_date is None:
        end_date = datetime.now(JST_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    elif as_of_date.tzinfo is None:
        end_date = JST_TZ.localize(
            as_of_date.replace(hour=0, minute=0, second=0, microsecond=0)
        )
    else:
        end_date = as_of_date.astimezone(JST_TZ).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    start_date = get_business_days_back(days, end_date)

    return start_date, end_date


def count_business_days(start_date: datetime, end_date: datetime) -> int:
    """
    Count business days between two dates (inclusive)

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        Number of business days
    """
    # Ensure dates are timezone-naive for comparison
    if start_date.tzinfo is not None:
        start_date = start_date.replace(tzinfo=None)
    if end_date.tzinfo is not None:
        end_date = end_date.replace(tzinfo=None)

    date_range = pd.bdate_range(start=start_date, end=end_date)
    return len(date_range)


def validate_data_coverage(
    df: pd.DataFrame, expected_days: int, tolerance_percent: float = 10.0
) -> Tuple[bool, str]:
    """
    Validate that fetched data has expected coverage

    Args:
        df: DataFrame with OHLC data
        expected_days: Expected number of business days
        tolerance_percent: Acceptable tolerance (e.g., 10% = 0.1)

    Returns:
        Tuple of (is_valid, message)
    """
    if df.empty:
        return False, f"No data available. Expected {expected_days} business days."

    actual_days = len(df)
    min_expected = int(expected_days * (1 - tolerance_percent / 100))

    if actual_days < min_expected:
        return False, (
            f"Insufficient data: Got {actual_days} days, expected ~{expected_days} days "
            f"(minimum {min_expected} acceptable). "
            f"This may indicate missing data or insufficient market data available."
        )

    return (
        True,
        f"Data coverage OK: {actual_days} days fetched (expected ~{expected_days})",
    )


def get_latest_business_day(df: pd.DataFrame) -> Optional[datetime]:
    """
    Get the latest business day in the DataFrame

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Latest datetime or None if empty
    """
    if df.empty:
        return None

    return df.index[-1]


def get_data_date_range(
    df: pd.DataFrame,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Get the date range of data in DataFrame

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        Tuple of (start_date, end_date) or (None, None) if empty
    """
    if df.empty:
        return None, None

    return df.index[0], df.index[-1]
