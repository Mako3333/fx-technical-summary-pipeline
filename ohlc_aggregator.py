#!/usr/bin/env python3
"""
CLI entrypoint for OHLC aggregation.

Usage example:
    python ohlc_aggregator.py --input-dir ./csv_data --output-dir ./reports
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"


def main() -> int:
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

    from fx_kline.core.ohlc_aggregator import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
