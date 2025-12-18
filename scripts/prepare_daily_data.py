#!/usr/bin/env python3
"""
Copy summary_reports JSON files into date-partitioned data/YYYY/MM/DD/summaries.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Sequence

# Ensure src/ is importable when running as a standalone script
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fx_kline.analyst import data_manager  # noqa: E402
from fx_kline.core.timezone_utils import get_jst_now  # noqa: E402


def _parse_date(date_str: Optional[str]) -> date:
    if date_str:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    # Default: today's date in JST
    return get_jst_now().date()


def _collect_summary_files(summary_dir: Path) -> list[Path]:
    return sorted(summary_dir.glob("*_summary.json"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy summary_reports/*.json into data/YYYY/MM/DD/summaries/"
    )
    parser.add_argument(
        "--date",
        help="Target date in YYYY-MM-DD (default: today in JST).",
    )
    args = parser.parse_args(argv)

    target_date = _parse_date(args.date)
    dest_dir = data_manager.get_daily_summaries_dir(target_date)
    project_root = data_manager.get_data_root().parent
    source_dir = project_root / "summary_reports"

    if not source_dir.exists():
        print(f"[WARN] summary_reports directory not found: {source_dir}")
        return 1

    summary_files = _collect_summary_files(source_dir)
    if not summary_files:
        print(f"[WARN] No *_summary.json files found in {source_dir}")
        return 1

    for src in summary_files:
        shutil.copy2(src, dest_dir / src.name)

    print(f"Copied {len(summary_files)} file(s) to {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
