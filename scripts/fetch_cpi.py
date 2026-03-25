#!/usr/bin/env python3
"""Download monthly CPI data from FRED and write data/CPI_monthly.csv.

Source: FRED CPIAUCSL — Consumer Price Index for All Urban Consumers
        https://fred.stlouisfed.org/series/CPIAUCSL
        Monthly, seasonally adjusted, index 1982-84=100.
        Series starts January 1947; no API key required.

Output format: newest-first CSV with a Month-over-month column (decimal),
matching the pre-computed format used by GOOG_monthly.csv and accepted by
sim/returns.py load_monthly_returns().

Usage:
    python scripts/fetch_cpi.py
    python scripts/fetch_cpi.py --start 1970-01-01   # trim to post-1970
    python scripts/fetch_cpi.py --out data/CPI_monthly.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "CPI_monthly.csv"


def fetch_cpi(start: str | None = None, out: Path = DEFAULT_OUT) -> None:
    print(f"Fetching CPIAUCSL from FRED...", file=sys.stderr)
    try:
        with urllib.request.urlopen(FRED_URL, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:
        print(f"Error: could not download CPI data: {e}", file=sys.stderr)
        sys.exit(1)

    lines = raw.strip().splitlines()
    # FRED CSV: header "DATE,CPIAUCSL", then rows chronologically oldest-first
    reader = csv.DictReader(lines)
    rows = [(r["DATE"], float(r["CPIAUCSL"])) for r in reader if r["CPIAUCSL"].strip()]

    if not rows:
        print("Error: no data rows found in FRED response", file=sys.stderr)
        sys.exit(1)

    # Optional start filter
    if start:
        rows = [(d, v) for d, v in rows if d >= start]
        if not rows:
            print(f"Error: no rows remain after filtering to start={start}", file=sys.stderr)
            sys.exit(1)

    # Compute month-over-month decimal returns: (P_t / P_{t-1}) - 1
    # rows is chronological; output will be reversed (newest-first)
    dates = [d for d, _ in rows]
    prices = [v for _, v in rows]
    returns: list[str] = [""]  # oldest row has no prior month
    for i in range(1, len(prices)):
        ret = prices[i] / prices[i - 1] - 1.0
        returns.append(f"{ret:.10f}")

    # Reverse to newest-first (matches GOOG_monthly.csv convention)
    dates_rev = list(reversed(dates))
    returns_rev = list(reversed(returns))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Month-over-month"])
        for date, ret in zip(dates_rev, returns_rev):
            writer.writerow([date, ret])

    n = len(dates_rev)
    print(f"Wrote {n} rows ({dates_rev[-1]} – {dates_rev[0]}) to {out}", file=sys.stderr)
    print(f"\nTo use in your config (bootstrap mode):", file=sys.stderr)
    print(f"  return_sources:", file=sys.stderr)
    print(f'    cpi: "{out}"', file=sys.stderr)
    print(f"  inflation_source: cpi", file=sys.stderr)
    print(f"  inflation_blend: 0.7", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", metavar="YYYY-MM-DD", default=None,
                        help="Earliest date to include (default: full series from 1947)")
    parser.add_argument("--out", metavar="PATH", default=str(DEFAULT_OUT),
                        help=f"Output path (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    fetch_cpi(start=args.start, out=Path(args.out))


if __name__ == "__main__":
    main()
