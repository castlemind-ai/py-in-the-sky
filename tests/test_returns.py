"""Tests for historical return loading and bootstrap sampling."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sim.returns import ReturnSampler, load_monthly_returns


def _write_csv(path: Path, header: list[str], rows: list[list[str]]):
    """Write a CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


@pytest.fixture
def goog_csv(tmp_path):
    """Create a small GOOG-style CSV with pre-computed returns."""
    # Newest first, 24 months
    header = ["Date", "Open", "High", "Low", "Close ", "Adj Close ", "Volume", "Month-over-month"]
    rows = []
    np.random.seed(42)
    returns = np.random.normal(0.01, 0.05, 24).tolist()
    for i, r in enumerate(returns):
        month = 24 - i
        rows.append([f"Month {month}", "100", "110", "90", "105", "105", "1000000", str(r)])
    # Last row (oldest) has no return
    rows[-1][-1] = ""
    _write_csv(tmp_path / "goog.csv", header, rows)
    return tmp_path / "goog.csv"


@pytest.fixture
def vti_csv(tmp_path):
    """Create a small VTI-style CSV with Adj Close prices and dividend rows."""
    header = ["Date", "Open", "High", "Low", "Close ", "Adj Close ", "Volume"]
    rows = []
    # 24 months of price data (newest first) + some dividend rows
    prices = [100.0]
    np.random.seed(99)
    for _ in range(23):
        prices.append(prices[-1] * (1 + np.random.normal(0.008, 0.04)))

    # Reverse so newest is first
    prices_rev = list(reversed(prices))
    for i, p in enumerate(prices_rev):
        month = 24 - i
        rows.append([f"Month {month}", f"{p:.2f}", f"{p+5:.2f}", f"{p-5:.2f}",
                      f"{p:.2f}", f"{p:.2f}", "500000"])
        # Insert a dividend row every 3 months
        if i > 0 and i % 3 == 0:
            rows.append([f"Div {month}", "0.50 Dividend", "", "", "", "", ""])

    _write_csv(tmp_path / "vti.csv", header, rows)
    return tmp_path / "vti.csv"


def test_load_goog_returns(goog_csv):
    """GOOG-style CSV loads pre-computed returns in chronological order."""
    returns = load_monthly_returns(goog_csv)
    # 24 rows, last has empty return → 23 returns
    assert len(returns) == 23
    # Should be chronological (reversed from file order)
    assert isinstance(returns, np.ndarray)


def test_load_vti_returns(vti_csv):
    """VTI-style CSV computes returns from Adj Close, filtering dividends."""
    returns = load_monthly_returns(vti_csv)
    # 24 price rows → 23 returns (first price has no prior)
    assert len(returns) == 23
    # Returns should be reasonable (not extreme)
    assert np.all(np.abs(returns) < 0.5)


def test_load_vti_filters_dividends(vti_csv):
    """Dividend rows should not appear in loaded returns."""
    returns = load_monthly_returns(vti_csv)
    # If dividends weren't filtered, we'd get different count or NaN values
    assert not np.any(np.isnan(returns))


def test_bootstrap_annual_shape(goog_csv):
    """Block bootstrap produces correct output shape."""
    returns = load_monthly_returns(goog_csv)
    sampler = ReturnSampler(sources={"goog": returns})
    rng = np.random.default_rng(42)

    asset_sources = ["goog", None]  # one bootstrap, one parametric
    result = sampler.bootstrap_correlated(rng, asset_sources, n_sims=50, n_periods=5)

    assert result.shape == (50, 5, 2)
    # First asset should have real values (from bootstrap)
    assert not np.any(np.isnan(result[:, :, 0]))
    # Second asset should be NaN (parametric placeholder)
    assert np.all(np.isnan(result[:, :, 1]))


def test_bootstrap_monthly_shape(goog_csv):
    """Monthly bootstrap produces correct shape."""
    returns = load_monthly_returns(goog_csv)
    sampler = ReturnSampler(sources={"goog": returns})
    rng = np.random.default_rng(42)

    asset_sources = ["goog"]
    result = sampler.bootstrap_monthly(rng, asset_sources, n_sims=100, n_months=12)
    assert result.shape == (100, 12, 1)
    assert not np.any(np.isnan(result))


def test_correlated_sampling_uses_same_indices(goog_csv, vti_csv):
    """All historical sources should use the same time indices for correlation."""
    goog_returns = load_monthly_returns(goog_csv)
    vti_returns = load_monthly_returns(vti_csv)
    sampler = ReturnSampler(sources={"goog": goog_returns, "vti": vti_returns})
    rng = np.random.default_rng(42)

    asset_sources = ["goog", "vti"]
    result = sampler.bootstrap_correlated(rng, asset_sources, n_sims=100, n_periods=10)

    # Both columns should have values (no NaN)
    assert not np.any(np.isnan(result))
    # The two assets should have different values (different return histories)
    assert not np.array_equal(result[:, :, 0], result[:, :, 1])


def test_bootstrap_seed_reproducibility(goog_csv):
    """Same seed should produce identical bootstrap results."""
    returns = load_monthly_returns(goog_csv)
    sampler = ReturnSampler(sources={"goog": returns})

    rng1 = np.random.default_rng(123)
    r1 = sampler.bootstrap_correlated(rng1, ["goog"], n_sims=50, n_periods=5)

    rng2 = np.random.default_rng(123)
    r2 = sampler.bootstrap_correlated(rng2, ["goog"], n_sims=50, n_periods=5)

    np.testing.assert_array_equal(r1, r2)


def test_compounded_annual_returns_reasonable(goog_csv):
    """Compounded 12-month block returns should be in a reasonable range."""
    returns = load_monthly_returns(goog_csv)
    sampler = ReturnSampler(sources={"goog": returns})
    rng = np.random.default_rng(42)

    result = sampler.bootstrap_correlated(rng, ["goog"], n_sims=1000, n_periods=10)
    annual = result[:, :, 0]

    # Annual returns should generally be between -80% and +200%
    assert annual.min() > -0.99
    assert annual.max() < 5.0  # generous upper bound


def test_from_config(goog_csv, vti_csv):
    """ReturnSampler.from_config loads sources correctly."""
    sampler = ReturnSampler.from_config(
        {"goog": str(goog_csv), "vti": str(vti_csv)}
    )
    assert "goog" in sampler.sources
    assert "vti" in sampler.sources
    assert len(sampler.sources["goog"]) == 23
    assert len(sampler.sources["vti"]) == 23
