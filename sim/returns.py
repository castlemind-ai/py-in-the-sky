"""Historical return loading and block-bootstrap sampling."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def load_monthly_returns(path: str | Path) -> np.ndarray:
    """Load monthly returns from a CSV file.

    Auto-detects format:
    - If a 'Month-over-month' column exists (e.g. GOOG): use it directly.
    - Otherwise: compute returns from 'Adj Close', filtering dividend rows.

    Returns a 1-D numpy array of monthly decimal returns in **chronological** order.
    """
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        # Strip non-breaking spaces and whitespace from headers
        header = [h.replace("\u00a0", " ").strip() for h in raw_header]

        rows = list(reader)

    # Detect format
    if "Month-over-month" in header:
        return _load_precomputed(header, rows)
    elif "Adj Close" in header:
        return _load_from_adj_close(header, rows)
    else:
        raise ValueError(
            f"Cannot detect return format in {path}. "
            f"Expected 'Month-over-month' or 'Adj Close' column. Got: {header}"
        )


def _load_precomputed(header: list[str], rows: list[list[str]]) -> np.ndarray:
    """Load returns from a pre-computed 'Month-over-month' column (e.g. GOOG)."""
    col_idx = header.index("Month-over-month")
    returns = []
    for row in rows:
        val = row[col_idx].strip()
        if val:  # skip empty (first month has no return)
            returns.append(float(val))
    # File is newest-first; reverse to chronological
    returns.reverse()
    return np.array(returns)


def _load_from_adj_close(header: list[str], rows: list[list[str]]) -> np.ndarray:
    """Compute returns from 'Adj Close' prices, filtering dividend rows (e.g. VTI)."""
    adj_close_idx = header.index("Adj Close")
    open_idx = header.index("Open")

    prices = []
    for row in rows:
        # Skip dividend rows: Open field contains "Dividend"
        open_val = row[open_idx].strip()
        if "Dividend" in open_val:
            continue
        adj = row[adj_close_idx].strip()
        if adj:
            prices.append(float(adj))

    # File is newest-first; reverse to chronological
    prices.reverse()
    prices_arr = np.array(prices)

    # Compute month-over-month returns: r_t = (P_t / P_{t-1}) - 1
    returns = prices_arr[1:] / prices_arr[:-1] - 1.0
    return returns


@dataclass
class ReturnSampler:
    """Manages historical return data and provides block-bootstrap sampling.

    Block bootstrap: for each simulation-year, sample a random start index,
    take 12 consecutive monthly returns, and compound them into an annual return.
    This preserves within-year serial correlation and volatility clustering.

    Correlated sampling: all sources share the same random start indices,
    preserving cross-asset correlation (e.g. GOOG and VTI both crash in 2008).
    """

    sources: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_config(cls, return_sources: dict[str, str], base_path: Path | None = None) -> ReturnSampler:
        """Load return sources from a config mapping of name -> CSV path."""
        sources = {}
        for name, csv_path in return_sources.items():
            p = Path(csv_path)
            if base_path and not p.is_absolute():
                p = base_path / p
            sources[name] = load_monthly_returns(p)
        return cls(sources=sources)

    def bootstrap_correlated(
        self,
        rng: np.random.Generator,
        asset_sources: list[str | None],
        n_sims: int,
        n_periods: int,
        months_per_period: int = 12,
        blend_weights: list[float] | None = None,
        blend_means: np.ndarray | None = None,
        blend_stds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate correlated block-bootstrap returns for multiple assets.

        Uses the same random start indices for all historical sources,
        preserving cross-asset temporal correlation.

        Args:
            rng: numpy random generator
            asset_sources: list of source names per asset (None = parametric)
            n_sims: number of simulations
            n_periods: number of time periods (years for drawdown, or months for accumulation)
            months_per_period: months per block (12 for annual, 1 for monthly)
            blend_weights: per-asset bootstrap weight (1.0=pure bootstrap, 0.5=50/50 blend).
                          If None, all assets use pure bootstrap.
            blend_means: per-asset parametric mean returns (annual for drawdown, monthly for accum).
                        Required if any blend_weight < 1.0.
            blend_stds: per-asset parametric std devs. Required if any blend_weight < 1.0.

        Returns:
            np.ndarray of shape (n_sims, n_periods, num_assets) with compounded returns.
            Assets with source=None get np.nan (caller must fill with parametric).
        """
        num_assets = len(asset_sources)
        result = np.full((n_sims, n_periods, num_assets), np.nan)

        # Find unique sources and their min length for correlated indexing
        unique_sources = set(s for s in asset_sources if s is not None)
        if not unique_sources:
            return result

        # Use the shortest source to determine valid start range for correlation
        min_len = min(len(self.sources[s]) for s in unique_sources)
        max_start = min_len - months_per_period
        if max_start < 1:
            raise ValueError(
                f"Historical data too short for {months_per_period}-month blocks. "
                f"Shortest source has {min_len} months."
            )

        # Draw shared random start indices: shape (n_sims, n_periods)
        start_indices = rng.integers(0, max_start, size=(n_sims, n_periods))

        # For each source, extract blocks and compound
        for source_name in unique_sources:
            monthly = self.sources[source_name]
            src_len = len(monthly)

            # Map shared indices into this source's range
            src_max_start = src_len - months_per_period
            # Use modular mapping if this source is longer than the shortest
            mapped_indices = start_indices % (src_max_start + 1)

            # Vectorized block extraction and compounding
            # Build offset array: [0, 1, ..., months_per_period-1]
            offsets = np.arange(months_per_period)
            # Expand indices: (n_sims, n_periods, 1) + (months_per_period,) -> (n_sims, n_periods, months_per_period)
            block_indices = mapped_indices[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
            # Gather monthly returns
            block_returns = monthly[block_indices]  # (n_sims, n_periods, months_per_period)
            # Compound: product of (1 + r) - 1
            compounded = np.prod(1 + block_returns, axis=2) - 1.0  # (n_sims, n_periods)

            # Assign to all assets using this source
            for i, src in enumerate(asset_sources):
                if src == source_name:
                    result[:, :, i] = compounded

        # Apply blending: mix bootstrap returns with parametric draws
        if blend_weights is not None and blend_means is not None and blend_stds is not None:
            for i, src in enumerate(asset_sources):
                if src is not None and blend_weights[i] < 1.0:
                    w = blend_weights[i]
                    parametric = rng.normal(blend_means[i], blend_stds[i], size=(n_sims, n_periods))
                    result[:, :, i] = w * result[:, :, i] + (1 - w) * parametric

        return result

    def bootstrap_monthly(
        self,
        rng: np.random.Generator,
        asset_sources: list[str | None],
        n_sims: int,
        n_months: int,
    ) -> np.ndarray:
        """Generate correlated bootstrap monthly returns for accumulation phase.

        Draws individual months (block size = 1) with shared indices across sources.

        Returns:
            np.ndarray of shape (n_sims, n_months, num_assets).
            Assets with source=None get np.nan.
        """
        return self.bootstrap_correlated(
            rng, asset_sources, n_sims, n_months, months_per_period=1,
        )
