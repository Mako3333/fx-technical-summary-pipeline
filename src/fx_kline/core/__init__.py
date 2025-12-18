"""
Core module for FX-Kline
Provides data fetching, validation, and timezone handling
"""

from .models import (
    OHLCRequest,
    BatchOHLCRequest,
    FetchError,
    OHLCData,
    BatchOHLCResponse,
)

from .validators import (
    validate_currency_pair,
    validate_timeframe,
    validate_period,
    validate_business_days,
    get_supported_pairs,
    get_supported_timeframes,
    get_preset_pairs,
    get_preset_timeframes,
    get_default_business_days_for_timeframe,
    SUPPORTED_CURRENCY_PAIRS,
    SUPPORTED_TIMEFRAMES,
    DEFAULT_TIMEFRAME_BUSINESS_DAYS,
    FALLBACK_TIMEFRAME_BUSINESS_DAYS,
    ValidationError,
)

from .data_fetcher import (
    fetch_single_ohlc,
    fetch_single_ohlc_async,
    fetch_batch_ohlc,
    fetch_batch_ohlc_sync,
    export_to_csv,
    export_to_json,
    export_to_csv_string,
    get_batch_csv_export,
    get_batch_json_export,
)

from .business_days import (
    filter_business_days,
    is_business_day,
    get_business_days_back,
    get_business_day_range,
    count_business_days,
    validate_data_coverage,
    get_latest_business_day,
    get_data_date_range,
    get_fx_trading_date,
    get_fx_trading_dates,
    get_fx_market_close_hour_jst,
)

from .timezone_utils import (
    utc_to_jst,
    convert_dataframe_to_jst,
    get_us_market_hours_in_jst,
    is_us_dst_active,
    format_timestamp_jst,
    get_jst_now,
    get_business_day_offset_jst,
)

from .summary_consolidator import (
    consolidate_reports_batch,
)

__all__ = [
    # Models
    "OHLCRequest",
    "BatchOHLCRequest",
    "FetchError",
    "OHLCData",
    "BatchOHLCResponse",
    # Validators
    "validate_currency_pair",
    "validate_timeframe",
    "validate_period",
    "validate_business_days",
    "get_supported_pairs",
    "get_supported_timeframes",
    "get_preset_pairs",
    "get_preset_timeframes",
    "get_default_business_days_for_timeframe",
    "SUPPORTED_CURRENCY_PAIRS",
    "SUPPORTED_TIMEFRAMES",
    "DEFAULT_TIMEFRAME_BUSINESS_DAYS",
    "FALLBACK_TIMEFRAME_BUSINESS_DAYS",
    "ValidationError",
    # Data fetcher
    "fetch_single_ohlc",
    "fetch_single_ohlc_async",
    "fetch_batch_ohlc",
    "fetch_batch_ohlc_sync",
    "export_to_csv",
    "export_to_json",
    "export_to_csv_string",
    "get_batch_csv_export",
    "get_batch_json_export",
    # Business days
    "filter_business_days",
    "is_business_day",
    "get_business_days_back",
    "get_business_day_range",
    "count_business_days",
    "validate_data_coverage",
    "get_latest_business_day",
    "get_data_date_range",
    "get_fx_trading_date",
    "get_fx_trading_dates",
    "get_fx_market_close_hour_jst",
    # Timezone utils
    "utc_to_jst",
    "convert_dataframe_to_jst",
    "get_us_market_hours_in_jst",
    "is_us_dst_active",
    "format_timestamp_jst",
    "get_jst_now",
    "get_business_day_offset_jst",
    # Summary consolidator
    "consolidate_reports_batch",
]
