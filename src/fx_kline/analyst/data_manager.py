"""
Helpers for managing HITL trading data directories (date-partitioned storage).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path


def _project_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[3]


def get_data_root() -> Path:
    """Path to the top-level data directory (repo_root/data)."""
    return _project_root() / "data"


def get_daily_data_dir(day: date) -> Path:
    """
    Path for a specific day under data/YYYY/MM/DD.

    The caller is responsible for creating the directory when needed.
    """
    return get_data_root() / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}"


def get_daily_summaries_dir(day: date) -> Path:
    """
    Path for L2 summaries under data/YYYY/MM/DD/summaries.

    Directory is created if it does not exist.
    """
    target = get_daily_data_dir(day) / "summaries"
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_daily_ohlc_dir(day: date) -> Path:
    """
    Path for OHLC archives under data/YYYY/MM/DD/ohlc.

    Directory is created if it does not exist.
    """
    target = get_daily_data_dir(day) / "ohlc"
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_daily_ohlc_filepath(day: date, pair: str, timeframe: str = "15m") -> Path:
    """
    File path for an OHLC CSV: data/YYYY/MM/DD/ohlc/{PAIR}_{timeframe}.csv
    """
    # Sanitize pair and timeframe to prevent path traversal and invalid characters
    safe_pair = pair.replace("/", "_").replace("\\", "_")
    safe_timeframe = timeframe.replace("/", "_").replace("\\", "_")

    # Ensure no path traversal
    if ".." in safe_pair or ".." in safe_timeframe:
        raise ValueError(
            f"Invalid characters in pair or timeframe: {pair}, {timeframe}"
        )

    return get_daily_ohlc_dir(day) / f"{safe_pair}_{safe_timeframe}.csv"


__all__ = [
    "get_data_root",
    "get_daily_data_dir",
    "get_daily_summaries_dir",
    "get_daily_ohlc_dir",
    "get_daily_ohlc_filepath",
]
