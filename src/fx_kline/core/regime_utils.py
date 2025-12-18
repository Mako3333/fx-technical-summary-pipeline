from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

SUMMARY_TIMEFRAMES = ("1d", "4h", "1h")


def load_regime_from_summary(
    date_str: str, pair: str
) -> Tuple[dict[str, Any] | None, str | None, List[str]]:
    """
    Load regime information from consolidated summary JSON.

    Args:
        date_str: Target date in YYYY-MM-DD format.
        pair: Currency pair (e.g., USDJPY).

    Returns:
        regime_detail: Structured detail per timeframe, or None.
        regime_str: Display-friendly summary string, or None.
        warnings: List of warning messages encountered while loading.
    """
    warnings: List[str] = []

    try:
        # Validate YYYY-MM-DD format with zero-padding
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        year, month, day = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    except ValueError:
        warnings.append(f"Invalid date format for summary lookup: {date_str}")
        return None, None, warnings

    summary_path = Path(f"data/{year}/{month}/{day}/summaries/{pair.upper()}_summary.json")
    if not summary_path.exists():
        warnings.append(f"Summary not found: {summary_path}")
        return None, None, warnings

    try:
        with summary_path.open("r", encoding="utf-8") as fp:
            summary = json.load(fp)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to load summary {summary_path.name}: {exc}")
        return None, None, warnings

    timeframes = summary.get("timeframes", {})
    regime_timeframes: Dict[str, Dict[str, Any]] = {}

    for tf in SUMMARY_TIMEFRAMES:
        tf_data = timeframes.get(tf)
        if not tf_data:
            warnings.append(f"{tf} timeframe missing in {summary_path.name}")
            continue

        trend = tf_data.get("trend")
        if not trend:
            warnings.append(f"trend missing for {tf} in {summary_path.name}")
            continue

        regime_timeframes[tf] = {
            "trend": trend,
            "atr": tf_data.get("atr"),
            "timestamp": tf_data.get("data_timestamp") or tf_data.get("generated_at"),
        }

    if not regime_timeframes:
        warnings.append(f"No regime data found for {pair.upper()} in {summary_path.name}")
        return None, None, warnings

    regime_detail = {
        "source": "summary",
        "pair": pair.upper(),
        "date": date_str,
        "timeframes": regime_timeframes,
    }
    regime_str = " / ".join(f"{tf}:{info['trend']}" for tf, info in regime_timeframes.items())

    return regime_detail, regime_str, warnings

