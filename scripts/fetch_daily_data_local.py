#!/usr/bin/env python3
"""
Fetch the same data that GitHub Actions uses for daily analysis
Useful for local verification and debugging
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"


def _load_fx_kline_core():
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))

    from fx_kline.core import OHLCRequest, export_to_csv, fetch_batch_ohlc_sync

    return OHLCRequest, fetch_batch_ohlc_sync, export_to_csv


OHLCRequest, fetch_batch_ohlc_sync, export_to_csv = _load_fx_kline_core()

# Same configuration as GitHub Actions
PAIRS = ["USDJPY", "EURUSD", "AUDJPY", "AUDUSD", "EURJPY", "XAUUSD"]
INTERVALS = ["1h", "4h", "1d"]
PERIOD_MAP = {
    "1h": "10d",
    "4h": "35d",
    "1d": "200d",
}


def main():
    print("=" * 80)
    print("FETCH DAILY ANALYSIS DATA (Same as GitHub Actions)")
    print("=" * 80)
    print(f"\nPairs: {', '.join(PAIRS)}")
    print(f"Intervals: {', '.join(INTERVALS)}")
    print(f"Period map: {PERIOD_MAP}")

    # Create output directory
    csv_dir = REPO_ROOT / "csv_data"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Build requests
    requests = [
        OHLCRequest(pair=p, interval=i, period=PERIOD_MAP[i])
        for p in PAIRS
        for i in INTERVALS
    ]

    print(f"\nFetching {len(requests)} datasets...")
    print("-" * 80)

    # Fetch data
    response = fetch_batch_ohlc_sync(requests)

    # Save to CSV
    success = 0
    for ohlc in response.successful:
        out_path = csv_dir / f"{ohlc.pair}_{ohlc.interval}_{ohlc.period}.csv"
        out_path.write_text(export_to_csv(ohlc), encoding="utf-8")
        success += 1
        print(f"[OK] {out_path.name:30s} - {ohlc.data_count:4d} rows")

    # Report failures
    if response.failed:
        print("\n" + "-" * 80)
        print("FAILURES:")
        for err in response.failed:
            print(
                f"[ERR] {err.pair}_{err.interval}_{err.period}: {err.error_type} -> {err.error_message}"
            )

    # Summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {success}/{response.total_requested} successful")
    print(f"Output directory: {csv_dir}")
    print("=" * 80)

    # Show daily data files specifically
    print("\n📊 DAILY DATA FILES (used for summary generation):")
    daily_files = sorted(csv_dir.glob("*_1d_*.csv"))

    if daily_files:
        for f in daily_files:
            print(f"  - {f.name}")
        print("\nTo view a daily data file:")
        print(f"  cat csv_data/{daily_files[0].name}")
        print("  # or in Python:")
        print("  import pandas as pd")
        print(f"  df = pd.read_csv('csv_data/{daily_files[0].name}')")
        print("  df.tail(30)  # Last 30 days")

    return 0 if not response.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
