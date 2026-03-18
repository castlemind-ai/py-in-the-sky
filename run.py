#!/usr/bin/env python3
"""CLI entry point for the Monte Carlo wealth simulation."""

import argparse
import sys

from sim.config import load_config
from sim.engine import run
from sim.output import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo wealth & retirement simulation")
    parser.add_argument("config", nargs="?", default="example_config.yaml",
                        help="Path to YAML config file (default: example_config.yaml)")
    parser.add_argument("--output-dir", default="./output",
                        help="Directory for chart PNGs (default: ./output)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation, print stats only")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    result = run(config)

    if args.no_plot:
        from sim.output import print_summary
        print_summary(result)
    else:
        generate_report(result, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
