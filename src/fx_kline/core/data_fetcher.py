"""
Data fetcher for FX OHLC data using yfinance
Supports parallel fetching with trading-day aware filtering and JST timezone conversion
"""

import asyncio
import io
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

from .business_days import (
    filter_business_days_fx,
    get_business_days_back,
    get_fx_trading_date,
    get_fx_trading_dates,
)
from .models import BatchOHLCResponse, FetchError, OHLCData, OHLCRequest
from .validators import validate_currency_pair, validate_period, validate_timeframe
from .timezone_utils import convert_dataframe_to_jst, get_jst_now, JST_TZ

logger = logging.getLogger(__name__)

# Load .env for local development; tolerate absence of python-dotenv (e.g., CI)
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    try:
        load_dotenv()
    except Exception:  # pragma: no cover - best-effort load
        logger.debug("Failed to load .env file via python-dotenv.")

# Thread pool for async yfinance calls
# NOTE: max_workers=1 to avoid yfinance parallel execution bug
# yfinance is not thread-safe and returns incorrect data when called in parallel
# See: https://github.com/ranaroussi/yfinance/issues (known issue)
_executor = ThreadPoolExecutor(max_workers=1)

_PERIOD_PATTERN = re.compile(r"^(\d+)([a-z]+)$")
_INTERVAL_PATTERN = re.compile(r"^(\d+)([a-z]+)$")

# COMEX holidays and special dates for warning purposes
THANKSGIVING_START = date(2025, 11, 25)
THANKSGIVING_END = date(2025, 11, 28)


def _is_comex_holiday(dt: date) -> bool | str:
    """
    Check if date is a known COMEX holiday or shortened trading day.

    Returns:
        True: Known full holiday (no trading)
        "shortened": Known shortened trading hours
        False: Regular trading day (or unknown)

    Note: This function only tracks explicitly known holidays.
    If data is missing for an unknown date, it may still be a holiday.
    """
    # Known full holidays (examples - not exhaustive)
    # Thanksgiving (4th Thursday of November)
    if dt.month == 11 and 22 <= dt.day <= 28 and dt.weekday() == 3:
        return True
    # Christmas Day
    if dt.month == 12 and dt.day == 25:
        return True

    # Known shortened trading days (examples)
    # Day after Thanksgiving (Friday)
    if dt.month == 11 and 23 <= dt.day <= 29 and dt.weekday() == 4:
        # Check if previous day was Thanksgiving
        from datetime import timedelta

        thanksgiving = dt - timedelta(days=1)
        if thanksgiving.weekday() == 3:
            return "shortened"

    return False


def check_missing_data_reason(symbol: str, missing_date: date) -> str:
    """
    Provide a likely reason when daily data is missing.

    Returns user-friendly message about possible market closure.
    """
    holiday_status = _is_comex_holiday(missing_date)

    if holiday_status is True:
        return f"⚠️ {missing_date}: Known COMEX holiday (market closed)"
    elif holiday_status == "shortened":
        return f"⚠️ {missing_date}: Known shortened trading day (early close)"
    else:
        return f"⚠️ {missing_date}: Data missing (possible market holiday or data delay)"


def _check_data_quality_warnings(
    df: pd.DataFrame,
    pair: str,
    expected_trading_days: Optional[int],
    actual_trading_days: int,
    fallback_used: bool,
) -> List[str]:
    """
    Check for data quality issues and generate warnings.

    Args:
        df: DataFrame with JST DatetimeIndex
        pair: Currency pair/symbol
        expected_trading_days: Expected number of trading days
        actual_trading_days: Actual number of trading days returned
        fallback_used: Whether fallback fetch was necessary

    Returns:
        List of warning messages
    """
    warnings = []

    if df.empty:
        return warnings

    # Check for Thanksgiving period data gap (COMEX holiday)
    if hasattr(df.index, "date"):
        df_dates_array = df.index.date
        df_dates = pd.DatetimeIndex(df_dates_array)
    else:
        df_dates = pd.to_datetime(df.index)

    # For COMEX symbols (Gold), check for known holidays and missing data
    if pair == "GC=F":
        try:
            # Convert dates to comparable format
            if isinstance(df_dates, pd.DatetimeIndex):
                df_date_values = set(df_dates.date)
            else:
                df_date_values = set(
                    [d.date() if hasattr(d, "date") else d for d in df_dates]
                )

            # Check expected date range for missing dates
            if not df.empty:
                date_range_start = min(df_date_values)
                date_range_end = max(df_date_values)

                # Check for missing weekdays in the range
                current_date = date_range_start
                missing_dates = []
                while current_date <= date_range_end:
                    # Only check weekdays
                    if (
                        current_date.weekday() < 5
                        and current_date not in df_date_values
                    ):
                        missing_dates.append(current_date)
                    from datetime import timedelta

                    current_date = current_date + timedelta(days=1)

                # Provide specific reasons for missing dates
                if missing_dates:
                    for missing_date in missing_dates[
                        :3
                    ]:  # Limit to first 3 missing dates
                        reason = check_missing_data_reason(pair, missing_date)
                        warnings.append(reason)

                    if len(missing_dates) > 3:
                        warnings.append(
                            f"[INFO] {len(missing_dates) - 3} additional date(s) are also missing from the dataset."
                        )
        except Exception:
            # Silently skip if date comparison fails
            pass

    # Check for zero volume data
    if "Volume" in df.columns:
        zero_volume = df[df["Volume"] == 0]
        if not zero_volume.empty:
            dates_str = ", ".join([d.strftime("%m/%d") for d in zero_volume.index])
            warnings.append(f"[WARNING] Zero volume data detected on: {dates_str}")

    # Check if fallback was needed (indicates insufficient recent data)
    if fallback_used and expected_trading_days is not None:
        if actual_trading_days < expected_trading_days:
            warnings.append(
                f"[WARNING] Requested {expected_trading_days} trading days but only {actual_trading_days} available. "
                f"Data coverage may be incomplete."
            )
        else:
            warnings.append(
                "[INFO] Requested data was incomplete, so fallback fetch was used to extend the date range."
            )

    return warnings


def _build_daily_from_intraday(
    symbol: str, start_date: date, end_date: date
) -> pd.DataFrame:
    """
    Build daily OHLC from intraday data when yfinance daily data is incomplete.

    This is particularly useful for shortened trading days where daily candles
    are not properly formed by yfinance.

    Args:
        symbol: yfinance symbol (e.g., 'GC=F')
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with daily OHLC aggregated from intraday data
    """
    # Fetch 30-minute intraday data (good balance of granularity and API limits)
    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval="30m",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return df

    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Group by date and aggregate to daily OHLC
    df_daily = df.groupby(df.index.date).agg(
        {
            "Open": "first",  # First price of the day
            "High": "max",  # Highest price of the day
            "Low": "min",  # Lowest price of the day
            "Close": "last",  # Last price of the day
            "Volume": "sum",  # Total volume
        }
    )

    # Convert index back to DatetimeIndex
    df_daily.index = pd.to_datetime(df_daily.index)
    df_daily.index.name = "Datetime"

    return df_daily


def _download_with_period(pair: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch data from yfinance using period-based window."""
    return yf.download(
        pair, interval=interval, period=period, auto_adjust=False, progress=False
    )


def _normalize_twelvedata_interval(interval: str) -> str:
    """Map internal interval to Twelve Data expected format."""
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1day",
    }
    return mapping.get(interval, interval)


def _fetch_twelvedata_ohlc(
    pair: str,
    interval: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetch OHLC data for XAUUSD from Twelve Data.

    Args:
        pair: Symbol to fetch (expected 'XAUUSD' for logging)
        interval: Timeframe (e.g., '1h', '15m')
        start_date: Optional start date (YYYY-MM-DD) for server-side filtering
        end_date: Optional end date (YYYY-MM-DD) for server-side filtering

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns matching yfinance style.
    """
    api_key = os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        logger.error(
            "TWELVEDATA_API_KEY is not set; cannot fetch %s from Twelve Data", pair
        )
        return pd.DataFrame()

    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "format": "CSV",
        "outputsize": 5000,
        "apikey": api_key,
        # Force UTC so downstream tz_localize/convert is correct (avoid US/Eastern shift)
        "timezone": "UTC",
    }
    if start_date:
        params["start_date"] = start_date.isoformat()
    if end_date:
        params["end_date"] = end_date.isoformat()

    # Respect Twelve Data free tier rate limits (8 req/min)
    time.sleep(2)

    try:
        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params=params,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch %s from Twelve Data: %s", pair, exc)
        return pd.DataFrame()

    if resp.status_code != 200:
        logger.error(
            "Twelve Data returned %s for %s: %s",
            resp.status_code,
            pair,
            resp.text[:200],
        )
        return pd.DataFrame()

    try:
        # Twelve Data API returns semicolon-separated CSV
        df = pd.read_csv(io.StringIO(resp.text), sep=";")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse Twelve Data CSV for %s: %s", pair, exc)
        return pd.DataFrame()

    if df.empty or "datetime" not in df.columns:
        logger.error("Twelve Data response missing expected data for %s", pair)
        return pd.DataFrame()

    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    try:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse datetime for %s from Twelve Data: %s", pair, exc)
        return pd.DataFrame()

    df = df.set_index("datetime").sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def _download_with_fallback_window(
    pair: str, interval: str, trading_days: int
) -> pd.DataFrame:
    """Fetch data using an explicit date window when period-based fetch under-delivers."""
    end_jst = get_jst_now()
    # Pad the lookback so we have slack for holidays and API quirks
    buffer_days = max(2, trading_days // 2)
    lookback_days = trading_days + buffer_days
    start_jst = get_business_days_back(lookback_days, end_jst)

    start_utc = start_jst.astimezone(timezone.utc)
    end_utc = end_jst.astimezone(timezone.utc)

    return yf.download(
        pair,
        interval=interval,
        start=start_utc,
        end=end_utc,
        auto_adjust=False,
        progress=False,
    )


def _prepare_dataframe(
    df: pd.DataFrame, interval: str, symbol: str, exclude_weekends: bool
) -> pd.DataFrame:
    """Apply weekend filtering, flatten columns, and convert to JST."""
    if df.empty:
        return df

    processed = df.copy()

    # Flatten multi-index columns first (before timezone conversion)
    if isinstance(processed.columns, pd.MultiIndex):
        processed.columns = processed.columns.get_level_values(0)

    if processed.empty:
        return processed

    # Convert to JST timezone
    processed = convert_dataframe_to_jst(processed)

    if processed.empty:
        return processed

    # Guardrail: drop any rows that would land in the "future" relative to current JST
    # (can happen with provider clock skew or timezone mislabeling).
    now_jst = get_jst_now()
    future_mask = processed.index > now_jst
    if future_mask.any():
        dropped = future_mask.sum()
        max_future = processed.index[future_mask].max()
        logger.warning(
            "Dropping %s future-dated rows (max=%s JST) for %s %s",
            dropped,
            max_future,
            symbol,
            interval,
        )
        processed = processed[~future_mask]
        if processed.empty:
            return processed

    # Apply market-specific weekend filtering (after timezone conversion to JST)
    if exclude_weekends:
        processed = filter_business_days_fx(processed, interval, symbol)

    return processed


def _extract_trading_days(period: str) -> Optional[int]:
    """Extract desired trading-day lookback from a period string like '5d'."""
    match = _PERIOD_PATTERN.match(period)
    if not match:
        return None

    value, unit = match.groups()
    if unit != "d":
        return None

    try:
        days = int(value)
    except ValueError:
        return None

    return days if days > 0 else None


def _parse_interval(interval: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract numeric value and unit suffix from an interval string."""
    match = _INTERVAL_PATTERN.match(interval)
    if not match:
        return None, None

    value_str, unit = match.groups()
    try:
        value = int(value_str)
    except ValueError:
        return None, unit

    return value, unit


def _fx_trading_date_series(df: pd.DataFrame) -> pd.Series:
    """Label each row with its FX trading date based on the JST index."""
    if df.empty:
        return pd.Series([], index=df.index, name="fx_trading_date", dtype="object")
    return get_fx_trading_dates(df.index)


def _count_relevant_trading_days(
    df: pd.DataFrame,
    interval_unit: Optional[str],
    current_trading_date: date,
) -> int:
    """
    Count unique FX trading days, optionally excluding the in-progress day for daily intervals.
    """
    trading_dates = _fx_trading_date_series(df)
    if interval_unit == "d":
        trading_dates = trading_dates[trading_dates != current_trading_date]

    try:
        return trading_dates.nunique()
    except Exception:
        return len(set(trading_dates))


def _apply_trading_day_filters(
    df: pd.DataFrame,
    interval_unit: Optional[str],
    expected_trading_days: Optional[int],
    current_trading_date: date,
) -> pd.DataFrame:
    """
    Apply trading-day based trimming:
    - Drop the in-progress trading day for daily intervals
    - Keep only the most recent N trading days when requested
    """
    if df.empty:
        return df

    trading_dates = _fx_trading_date_series(df)

    if interval_unit == "d":
        closed_mask = trading_dates != current_trading_date
        df = df[closed_mask]
        trading_dates = trading_dates[closed_mask]

    if expected_trading_days is not None and not df.empty:
        ordered_unique = trading_dates.drop_duplicates()
        if len(ordered_unique) > expected_trading_days:
            keep_dates = set(ordered_unique.iloc[-expected_trading_days:])
            df = df[trading_dates.isin(keep_dates)]

    return df


def fetch_ohlc_range_dataframe(
    pair: str,
    interval: str,
    start: datetime,
    end: datetime,
    exclude_weekends: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLC data for a specific start/end window, returned as a JST-indexed DataFrame.

    Args:
        pair: Currency pair code (e.g., 'USDJPY')
        interval: Timeframe (e.g., '1h', '15m', '1d')
        start: Start datetime (timezone-aware preferred, naive datetimes assumed to be JST)
        end: End datetime (timezone-aware preferred, naive datetimes assumed to be JST)
        exclude_weekends: Filter out weekend data

    Returns:
        DataFrame with JST-indexed DatetimeIndex
    """
    pair_upper = pair.upper()
    use_twelve_data = pair_upper == "XAUUSD"

    # For XAUUSD we still validate to GC=F for downstream filtering logic
    if use_twelve_data:
        filter_symbol = validate_currency_pair(pair)  # GC=F, also validates support
        pair_formatted = pair_upper
    else:
        pair_formatted = validate_currency_pair(pair)
        filter_symbol = pair_formatted
    interval_validated = validate_timeframe(interval)

    # Handle naive datetimes by assuming they are JST (project standard)
    if start.tzinfo is None:
        start = JST_TZ.localize(start)
    if end.tzinfo is None:
        end = JST_TZ.localize(end)

    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)
    start_jst_date = start.astimezone(JST_TZ).date()
    end_jst_date = end.astimezone(JST_TZ).date()

    if use_twelve_data:
        td_interval = _normalize_twelvedata_interval(interval_validated)
        df = _fetch_twelvedata_ohlc(
            pair_formatted,
            td_interval,
            start_date=start_jst_date,
            end_date=end_jst_date,
        )
        if not df.empty:
            mask = (df.index >= start_utc) & (df.index < end_utc)
            df = df.loc[mask]
    else:
        df = yf.download(
            pair_formatted,
            interval=interval_validated,
            start=start_utc,
            end=end_utc,
            auto_adjust=False,
            progress=False,
        )

    if df.empty:
        return df

    return _prepare_dataframe(df, interval_validated, filter_symbol, exclude_weekends)


def fetch_single_ohlc(
    pair: str, interval: str, period: str, exclude_weekends: bool = True
) -> Tuple[Optional[OHLCData], Optional[FetchError]]:
    """
    Fetch OHLC data for a single currency pair (synchronous)

    Args:
        pair: Currency pair code (e.g., 'USDJPY')
        interval: Timeframe (e.g., '1h', '1d')
        period: Period (e.g., '30d')
        exclude_weekends: Filter out weekend data

    Returns:
        Tuple of (OHLCData or None, FetchError or None)
        One will be populated, the other None
    """
    try:
        # Validate inputs
        pair_upper = pair.upper()
        use_twelve_data = pair_upper == "XAUUSD"

        # Validate pair; for XAUUSD keep GC=F for filtering, but fetch using Twelve Data
        if use_twelve_data:
            filter_symbol = validate_currency_pair(
                pair
            )  # returns GC=F, ensures support
            pair_formatted = pair_upper
        else:
            pair_formatted = validate_currency_pair(pair)
            filter_symbol = pair_formatted

        interval_validated = validate_timeframe(interval)
        period_validated = validate_period(period)

        expected_trading_days = _extract_trading_days(period_validated)
        _, interval_unit = _parse_interval(interval_validated)
        is_minute_interval = interval_unit == "m"
        current_trading_date = get_fx_trading_date(get_jst_now())

        # Fetch data (Twelve Data for XAUUSD, yfinance otherwise)
        if use_twelve_data:
            td_interval = _normalize_twelvedata_interval(interval_validated)
            df = _fetch_twelvedata_ohlc(pair_formatted, td_interval)
            raw_data_present = not df.empty
            df_jst = _prepare_dataframe(
                df, interval_validated, filter_symbol, exclude_weekends
            )
            fallback_used = False
            relevant_trading_day_count = _count_relevant_trading_days(
                df_jst, interval_unit, current_trading_date
            )
        else:
            df = _download_with_period(
                pair_formatted, interval_validated, period_validated
            )
            raw_data_present = not df.empty

            df_jst = _prepare_dataframe(
                df, interval_validated, filter_symbol, exclude_weekends
            )
            relevant_trading_day_count = _count_relevant_trading_days(
                df_jst, interval_unit, current_trading_date
            )

            # Track whether fallback was used
            fallback_used = False

            # Attempt fallback if coverage is clearly insufficient
            if (
                not is_minute_interval
                and expected_trading_days is not None
                and relevant_trading_day_count < expected_trading_days
            ):
                fallback_used = True
                df_fallback = _download_with_fallback_window(
                    pair_formatted, interval_validated, expected_trading_days
                )
                raw_data_present = raw_data_present or not df_fallback.empty
                df_fallback_jst = _prepare_dataframe(
                    df_fallback, interval_validated, filter_symbol, exclude_weekends
                )
                fallback_trading_day_count = _count_relevant_trading_days(
                    df_fallback_jst, interval_unit, current_trading_date
                )

                if fallback_trading_day_count >= relevant_trading_day_count:
                    df_jst = df_fallback_jst.copy()
                    relevant_trading_day_count = fallback_trading_day_count
                elif df_jst.empty and not df_fallback_jst.empty:
                    df_jst = df_fallback_jst.copy()
                    relevant_trading_day_count = fallback_trading_day_count

        # Handle empty data
        if df_jst.empty:
            error_type = (
                "AllWeekendData"
                if raw_data_present and exclude_weekends
                else "NoDataAvailable"
            )
            error = FetchError(
                pair=pair,
                interval=interval,
                period=period,
                error_type=error_type,
                error_message=f"No data available for {pair} with interval {interval} and period {period}",
            )
            return None, error

        # Apply FX trading-day rules: drop in-progress daily bar, then keep the requested window
        df_jst = _apply_trading_day_filters(
            df_jst,
            interval_unit,
            expected_trading_days,
            current_trading_date,
        )

        if df_jst.empty:
            error = FetchError(
                pair=pair,
                interval=interval,
                period=period,
                error_type="NoDataAvailable",
                error_message=f"No data available for {pair} with interval {interval} and period {period}",
            )
            return None, error

        # Prepare OHLC data
        ohlc_columns = []
        for col in ["Open", "High", "Low", "Close"]:
            if col in df_jst.columns:
                ohlc_columns.append(col)

        if "Volume" in df_jst.columns:
            ohlc_columns.append("Volume")

        if not ohlc_columns:
            error = FetchError(
                pair=pair,
                interval=interval,
                period=period,
                error_type="NoOHLCColumns",
                error_message="No OHLC columns found in fetched data.",
            )
            return None, error

        df_ohlc = df_jst[ohlc_columns].copy()

        # Convert DataFrame to list of dicts
        rows = []
        for idx, row in df_ohlc.iterrows():
            try:
                row_dict = {
                    "Datetime": (
                        idx.strftime("%Y-%m-%d %H:%M:%S %Z")
                        if hasattr(idx, "strftime")
                        else str(idx)
                    ),
                    "Open": float(row["Open"]) if "Open" in ohlc_columns else 0.0,
                    "High": float(row["High"]) if "High" in ohlc_columns else 0.0,
                    "Low": float(row["Low"]) if "Low" in ohlc_columns else 0.0,
                    "Close": float(row["Close"]) if "Close" in ohlc_columns else 0.0,
                }
                if "Volume" in ohlc_columns:
                    row_dict["Volume"] = (
                        int(row["Volume"]) if pd.notna(row["Volume"]) else 0
                    )
                rows.append(row_dict)
            except Exception as e:
                # Log and skip rows with errors
                logger.debug(f"Skipping row {idx} due to conversion error: {e}")
                continue

        if not rows:
            error = FetchError(
                pair=pair,
                interval=interval,
                period=period,
                error_type="NoDataAvailable",
                error_message=f"No valid OHLC rows generated for {pair} ({interval}/{period})",
            )
            return None, error

        # Generate data quality warnings
        warnings = _check_data_quality_warnings(
            df_jst,
            filter_symbol,
            expected_trading_days,
            relevant_trading_day_count,
            fallback_used,
        )

        ohlc_data = OHLCData(
            pair=pair,
            interval=interval,
            period=period,
            data_count=len(rows),
            columns=ohlc_columns,
            rows=rows,
            timestamp_jst=get_jst_now(),
            warnings=warnings,
        )

        return ohlc_data, None

    except Exception as e:
        error = FetchError(
            pair=pair,
            interval=interval,
            period=period,
            error_type=type(e).__name__,
            error_message=str(e),
        )
        return None, error


async def fetch_single_ohlc_async(
    pair: str, interval: str, period: str, exclude_weekends: bool = True
) -> Tuple[Optional[OHLCData], Optional[FetchError]]:
    """
    Async wrapper for single OHLC fetch

    Args:
        pair: Currency pair code
        interval: Timeframe
        period: Period
        exclude_weekends: Filter out weekend data

    Returns:
        Tuple of (OHLCData or None, FetchError or None)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, fetch_single_ohlc, pair, interval, period, exclude_weekends
    )


async def fetch_batch_ohlc(
    requests: List[OHLCRequest], exclude_weekends: bool = True
) -> BatchOHLCResponse:
    """
    Fetch OHLC data for multiple currency pairs in parallel

    Args:
        requests: List of OHLCRequest objects
        exclude_weekends: Filter out weekend data for all requests

    Returns:
        BatchOHLCResponse with successful and failed requests
    """
    # Create async tasks
    tasks = [
        fetch_single_ohlc_async(req.pair, req.interval, req.period, exclude_weekends)
        for req in requests
    ]

    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Separate successes and failures
    successful = []
    failed = []

    for result in results:
        ohlc_data, error = result
        if ohlc_data:
            successful.append(ohlc_data)
        elif error:
            failed.append(error)

    # Create response
    response = BatchOHLCResponse(
        successful=successful,
        failed=failed,
        total_requested=len(requests),
        total_succeeded=len(successful),
        total_failed=len(failed),
    )

    return response


def fetch_batch_ohlc_sync(
    requests: List[OHLCRequest], exclude_weekends: bool = True
) -> BatchOHLCResponse:
    """
    Synchronous wrapper for batch OHLC fetch

    Args:
        requests: List of OHLCRequest objects
        exclude_weekends: Filter out weekend data for all requests

    Returns:
        BatchOHLCResponse with successful and failed requests
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(fetch_batch_ohlc(requests, exclude_weekends))
    finally:
        loop.close()


def export_to_csv(ohlc_data: OHLCData) -> str:
    """
    Export OHLC data to CSV format

    Args:
        ohlc_data: OHLCData object

    Returns:
        CSV string
    """
    # Create DataFrame from rows
    df = pd.DataFrame(ohlc_data.rows)

    # Convert to CSV
    csv_string = df.to_csv(index=False)
    return csv_string


def export_to_json(ohlc_data: OHLCData) -> str:
    """
    Export OHLC data to JSON format

    Args:
        ohlc_data: OHLCData object

    Returns:
        JSON string
    """
    import json

    # Convert to dict and then to JSON
    data_dict = {
        "pair": ohlc_data.pair,
        "interval": ohlc_data.interval,
        "period": ohlc_data.period,
        "data_count": ohlc_data.data_count,
        "rows": ohlc_data.rows,
    }

    return json.dumps(data_dict, indent=2, ensure_ascii=False)


def export_to_csv_string(ohlc_data: OHLCData, include_header: bool = True) -> str:
    """
    Export OHLC data to comma-separated string (for clipboard)

    Args:
        ohlc_data: OHLCData object
        include_header: Include header row

    Returns:
        Comma-separated string
    """
    lines = []

    if include_header and ohlc_data.rows:
        # Get column names from first row
        header = ",".join(ohlc_data.rows[0].keys())
        lines.append(header)

    # Add data rows
    for row in ohlc_data.rows:
        values = [str(v) for v in row.values()]
        lines.append(",".join(values))

    return "\n".join(lines)


def get_batch_csv_export(response: BatchOHLCResponse) -> Dict[str, str]:
    """
    Export all successful batch results to CSV

    Args:
        response: BatchOHLCResponse object

    Returns:
        Dictionary with pair as key and CSV string as value
    """
    exports = {}
    for ohlc_data in response.successful:
        key = f"{ohlc_data.pair}_{ohlc_data.interval}_{ohlc_data.period}"
        exports[key] = export_to_csv(ohlc_data)

    return exports


def get_batch_json_export(response: BatchOHLCResponse) -> str:
    """
    Export all successful batch results to JSON

    Args:
        response: BatchOHLCResponse object

    Returns:
        JSON string with all results
    """
    import json

    data = {
        "summary": response.summary,
        "successful": [
            {
                "pair": ohlc.pair,
                "interval": ohlc.interval,
                "period": ohlc.period,
                "data_count": ohlc.data_count,
                "rows": ohlc.rows,
            }
            for ohlc in response.successful
        ],
        "failed": [
            {
                "pair": err.pair,
                "interval": err.interval,
                "period": err.period,
                "error_type": err.error_type,
                "error_message": err.error_message,
            }
            for err in response.failed
        ],
    }

    return json.dumps(data, indent=2, ensure_ascii=False)
