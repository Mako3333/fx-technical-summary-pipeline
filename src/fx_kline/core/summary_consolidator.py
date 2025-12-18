"""
Consolidate multi-timeframe analysis reports into unified summary files per currency pair.

Accepts individual analysis JSON files (schema_version >= 1) from reports/ directory
and produces consolidated summary JSON files (schema_version=2.1) in summary_reports/ directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .timezone_utils import get_jst_now

logger = logging.getLogger(__name__)

_ANALYSIS_FILE_PATTERN = re.compile(r"^([A-Z]+)_(.+)_analysis\.json$")
_EXPECTED_TIMEFRAMES = {"1h", "4h", "1d"}
_TIMEFRAME_ORDER = ["1d", "4h", "1h"]  # Macro to micro view
_CONSOLIDATION_VERSION = "1.2.0"


@dataclass
class TimeframeAnalysis:
    """Single timeframe data extracted from individual analysis JSON."""

    interval: str
    period: str
    trend: str
    support_levels: List[float]
    resistance_levels: List[float]
    rsi: Optional[float]
    atr: Optional[float]
    average_volatility: Optional[float]
    data_timestamp: str
    sma: Optional[dict] = None
    ema: Optional[dict] = None
    timeframe: Optional[str] = None
    time_of_day: Optional[dict] = None


@dataclass
class ConsolidatedSummary:
    """Multi-timeframe summary for a single currency pair."""

    pair: str
    schema_version: float
    generated_at: str
    timeframes: Dict[str, dict]
    metadata: dict

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict with proper nested dataclass handling."""
        result = {
            "pair": self.pair,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "timeframes": {},
            "metadata": copy.deepcopy(self.metadata),
        }

        # Convert TimeframeAnalysis objects in timeframes dict
        for interval, tf_data in self.timeframes.items():
            if hasattr(tf_data, "__dataclass_fields__"):
                result["timeframes"][interval] = asdict(tf_data)
            else:
                result["timeframes"][interval] = tf_data

        return result


def discover_analysis_files(reports_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover *_analysis.json files and group by currency pair.

    Args:
        reports_dir: Directory containing analysis JSON files

    Returns:
        Dict mapping pair name to list of analysis file paths
        Example: {"USDJPY": [Path("USDJPY_1h_10d_analysis.json"), ...], ...}
    """
    if not reports_dir.exists():
        logger.warning(f"Reports directory does not exist: {reports_dir}")
        return {}

    grouped_files: Dict[str, List[Path]] = defaultdict(list)

    for file_path in reports_dir.glob("*_analysis.json"):
        match = _ANALYSIS_FILE_PATTERN.match(file_path.name)
        if not match:
            logger.debug(f"Skipping file with unexpected name format: {file_path.name}")
            continue

        pair = match.group(1).upper()
        grouped_files[pair].append(file_path)

    # Sort files within each group for consistency
    for pair in grouped_files:
        grouped_files[pair] = sorted(grouped_files[pair], key=lambda p: p.name)

    return dict(grouped_files)


def load_analysis_file(file_path: Path) -> Optional[TimeframeAnalysis]:
    """
    Load and validate a single analysis JSON file.

    Args:
        file_path: Path to analysis JSON file

    Returns:
        TimeframeAnalysis object or None if loading fails
    """
    try:
        with file_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as exc:
        logger.error(f"Corrupt JSON in {file_path.name}: {exc}")
        return None
    except OSError as exc:
        logger.error(f"Failed to read {file_path.name}: {exc}")
        return None

    # Validate schema version
    schema_version = data.get("schema_version")
    if schema_version not in {1, 2, 2.1}:
        logger.error(
            f"Invalid schema_version in {file_path.name}: expected 1, 2, or 2.1, got {schema_version}"
        )
        return None

    # Extract required fields
    try:
        timeframe_analysis = TimeframeAnalysis(
            interval=data["interval"],
            period=data["period"],
            trend=data["trend"],
            support_levels=data.get("support_levels", []),
            resistance_levels=data.get("resistance_levels", []),
            rsi=data.get("rsi"),
            atr=data.get("atr"),
            average_volatility=data.get("average_volatility"),
            data_timestamp=data["generated_at"],  # Rename to data_timestamp
            sma=data.get("sma"),
            ema=data.get("ema"),
            time_of_day=data.get("time_of_day"),
            timeframe=data.get("timeframe") or data.get("interval"),
        )
    except KeyError as exc:
        logger.error(f"Missing required field in {file_path.name}: {exc}")
        return None

    return timeframe_analysis


def consolidate_pair_analyses(
    pair: str, analysis_files: List[Path]
) -> ConsolidatedSummary:
    """
    Merge multiple timeframe analyses for a single currency pair.

    Args:
        pair: Currency pair name (e.g., "USDJPY")
        analysis_files: List of analysis JSON file paths for this pair

    Returns:
        ConsolidatedSummary with all timeframe data
    """
    timeframes_dict: Dict[str, TimeframeAnalysis] = {}
    source_files: List[str] = []

    # Load each analysis file
    for file_path in analysis_files:
        tf_analysis = load_analysis_file(file_path)
        if tf_analysis is None:
            logger.warning(f"Skipping invalid analysis file: {file_path.name}")
            continue

        interval = tf_analysis.interval

        # Handle duplicate timeframes - use most recent
        if interval in timeframes_dict:
            existing_ts = timeframes_dict[interval].data_timestamp
            new_ts = tf_analysis.data_timestamp
            if new_ts > existing_ts:
                logger.warning(
                    f"Duplicate {interval} timeframe for {pair}, using newer data"
                )
                timeframes_dict[interval] = tf_analysis
                # Update source file reference
                old_file = next(
                    (f for f in source_files if f.startswith(f"{pair}_{interval}_")),
                    None,
                )
                if old_file:
                    source_files.remove(old_file)
                source_files.append(file_path.name)
            else:
                logger.warning(
                    f"Duplicate {interval} timeframe for {pair}, keeping existing data"
                )
        else:
            timeframes_dict[interval] = tf_analysis
            source_files.append(file_path.name)

    # Order timeframes: 1d → 4h → 1h (macro to micro)
    ordered_timeframes: Dict[str, TimeframeAnalysis] = {}
    for interval in _TIMEFRAME_ORDER:
        if interval in timeframes_dict:
            ordered_timeframes[interval] = timeframes_dict[interval]

    # Add any other timeframes not in the expected order
    for interval, tf_data in timeframes_dict.items():
        if interval not in ordered_timeframes:
            ordered_timeframes[interval] = tf_data

    # Detect missing expected timeframes
    found_intervals = set(timeframes_dict.keys())
    missing_timeframes = sorted(_EXPECTED_TIMEFRAMES - found_intervals)

    # Build metadata
    metadata = {
        "source_files": sorted(source_files),
        "consolidation_version": _CONSOLIDATION_VERSION,
        "total_timeframes": len(timeframes_dict),
        "missing_timeframes": missing_timeframes,
    }

    return ConsolidatedSummary(
        pair=pair,
        schema_version=2.1,
        generated_at=get_jst_now().isoformat(),
        timeframes=ordered_timeframes,
        metadata=metadata,
    )


def write_summary(summary: ConsolidatedSummary, output_path: Path) -> None:
    """
    Write consolidated summary to JSON with stable formatting.

    Args:
        summary: ConsolidatedSummary to write
        output_path: Destination path for JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(summary.to_dict(), fp, ensure_ascii=True, indent=2)

    logger.debug(f"Wrote summary to {output_path}")


def consolidate_reports_batch(
    reports_dir: Path, output_dir: Path, pairs_filter: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Batch consolidation of all currency pairs in reports directory.

    Args:
        reports_dir: Input directory with *_analysis.json files
        output_dir: Output directory for summary files
        pairs_filter: Optional list of pairs to process (e.g., ["USDJPY", "EURUSD"])

    Returns:
        Dict mapping pair name to output file path
    """
    grouped_files = discover_analysis_files(reports_dir)

    if not grouped_files:
        logger.warning(f"No analysis files found in {reports_dir}")
        return {}

    # Apply pairs filter if provided
    if pairs_filter:
        pairs_filter_upper = [p.upper() for p in pairs_filter]
        grouped_files = {
            pair: files
            for pair, files in grouped_files.items()
            if pair in pairs_filter_upper
        }
        if not grouped_files:
            logger.warning(f"No matching pairs found for filter: {pairs_filter}")
            return {}

    results: Dict[str, Path] = {}

    for pair, analysis_files in grouped_files.items():
        logger.info(f"Consolidating {len(analysis_files)} file(s) for {pair}")

        try:
            summary = consolidate_pair_analyses(pair, analysis_files)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Failed to consolidate {pair}: {exc}")
            continue

        output_path = output_dir / f"{pair}_summary.json"

        try:
            write_summary(summary, output_path)
            results[pair] = output_path
            logger.info(f"Generated summary for {pair} -> {output_path.name}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error(f"Failed to write summary for {pair}: {exc}")
            continue

    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    CLI entry point for consolidating analysis reports.

    Returns:
        0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(
        description="Consolidate multi-timeframe analysis reports into unified summaries per currency pair."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("./reports"),
        help="Input directory containing analysis JSON files (default: ./reports).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./summary_reports"),
        help="Output directory for consolidated summaries (default: ./summary_reports).",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help="Optional filter for specific currency pairs (e.g., USDJPY EURUSD).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    results = consolidate_reports_batch(
        reports_dir=args.reports_dir,
        output_dir=args.output_dir,
        pairs_filter=args.pairs,
    )

    if not results:
        logger.warning("No summaries were generated.")
        return 1

    logger.info(
        f"Successfully generated {len(results)} summary file(s) in {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
