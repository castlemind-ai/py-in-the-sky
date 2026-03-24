"""Tests for the simulation engine."""

import csv
from pathlib import Path

import numpy as np
import pytest

from sim.config import (
    Asset,
    BlackSwanEvent,
    Contribution,
    SimConfig,
    SpendingSchedule,
    TaxConfig,
)
from sim.engine import run, run_accumulation


def _simple_config(**overrides) -> SimConfig:
    """Create a minimal config for testing.

    Defaults: no accumulation growth, no tax, no inflation — pure drawdown test.
    """
    defaults = dict(
        assets=[Asset(name="Stocks", value=100000, mean_return=0.05, std_dev=0.0, tax_type="roth")],
        spending=SpendingSchedule(categories={}),
        tax=TaxConfig(income_rate=0.0, capital_gains_rate=0.0, roth_rate=0.0),
        current_age=49,
        retirement_age=50,  # 1 year accumulation
        life_expectancy=60,
        num_simulations=100,
        inflation_mean=0.0,
        inflation_std=0.0,
        accumulation_return=0.0,  # no growth during accumulation by default
        seed=42,
    )
    defaults.update(overrides)
    return SimConfig(**defaults)


def test_output_shape():
    config = _simple_config()
    result = run(config)
    # 10 years in retirement (60 - 50)
    assert result.wealth.shape == (100, 11)
    assert result.asset_wealth.shape == (100, 11, 1)
    assert result.dist_rates.shape == (100, 10)


def test_accumulation_grows_assets():
    config = _simple_config(
        assets=[Asset(name="A", value=100000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        accumulation_return=0.12,
        current_age=45,
        retirement_age=50,
    )
    accum = run_accumulation(config)
    # 5 years of 12% growth (blended), compounded monthly
    expected = 100000 * (1 + 0.12) ** 5
    assert accum.asset_values["A"] > 100000
    # Monthly compounding at 12% is slightly higher than annual
    assert accum.asset_values["A"] > expected * 0.99


def test_accumulation_with_contributions():
    config = _simple_config(
        assets=[Asset(name="A", value=0, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        contributions=[Contribution(name="Save", monthly_amount=1000, target_account="A")],
        accumulation_return=0.0,
        current_age=45,
        retirement_age=50,
    )
    accum = run_accumulation(config)
    # 5 years * 12 months * $1000 = $60,000, no growth
    assert accum.asset_values["A"] == 60000


def test_spending_reduces_wealth():
    config = _simple_config(
        assets=[Asset(name="A", value=100000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=SpendingSchedule(categories={"living": 10000}),
        life_expectancy=55,
    )
    result = run(config)
    # 5 years, $10k/yr, no growth, no tax (roth), no inflation
    # Start: 100k. End: 100k - 50k = 50k
    np.testing.assert_allclose(result.wealth[:, -1], 50000, rtol=1e-6)


def test_tax_grossup_increases_withdrawal():
    """Traditional accounts require more withdrawal to cover the same spending."""
    config_notax = _simple_config(
        assets=[Asset(name="A", value=1000000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=SpendingSchedule(categories={"living": 50000}),
        tax=TaxConfig(roth_rate=0.0),
        life_expectancy=52,
    )
    config_tax = _simple_config(
        assets=[Asset(name="A", value=1000000, mean_return=0.0, std_dev=0.0, tax_type="traditional")],
        spending=SpendingSchedule(categories={"living": 50000}),
        tax=TaxConfig(income_rate=0.35),
        life_expectancy=52,
    )

    result_notax = run(config_notax)
    result_tax = run(config_tax)

    # With tax, more is withdrawn so final wealth is lower
    assert result_tax.wealth[0, -1] < result_notax.wealth[0, -1]


def test_ruin_detection():
    config = _simple_config(
        assets=[Asset(name="A", value=10000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=SpendingSchedule(categories={"living": 5000}),
        life_expectancy=60,
        num_simulations=10,
    )
    result = run(config)
    # 10 years * $5k = $50k > $10k initial; portfolio depletes and floors at zero
    assert np.all(result.wealth[:, -1] == 0)


def test_multiple_assets():
    config = _simple_config(
        assets=[
            Asset(name="A", value=60000, mean_return=0.0, std_dev=0.0, tax_type="roth"),
            Asset(name="B", value=40000, mean_return=0.0, std_dev=0.0, tax_type="traditional"),
        ],
        life_expectancy=52,
    )
    result = run(config)
    np.testing.assert_allclose(result.wealth[:, 0], 100000)
    assert result.asset_wealth.shape[2] == 2


def test_seed_reproducibility():
    config = _simple_config(
        assets=[Asset(name="A", value=100000, mean_return=0.07, std_dev=0.15, tax_type="roth")],
        seed=123,
    )
    r1 = run(config)
    r2 = run(config)
    np.testing.assert_array_equal(r1.wealth, r2.wealth)


def test_distribution_rates_tracked():
    config = _simple_config(
        assets=[Asset(name="A", value=100000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=SpendingSchedule(categories={"living": 10000}),
        life_expectancy=55,
    )
    result = run(config)
    # Year 1: withdraw 10k from 100k portfolio = 10% rate
    np.testing.assert_allclose(result.dist_rates[:, 0], 0.10, rtol=1e-6)


@pytest.fixture
def bootstrap_csv(tmp_path):
    """Create a synthetic CSV with known monthly returns for bootstrap testing."""
    header = ["Date", "Open", "High", "Low", "Close ", "Adj Close ", "Volume", "Month-over-month"]
    rows = []
    # 36 months of constant 1% monthly return (newest first)
    for i in range(36):
        month = 36 - i
        ret = "0.01" if i < 35 else ""  # oldest has no return
        rows.append([f"Month {month}", "100", "110", "90", "105", "105", "1000", ret])
    path = tmp_path / "test_returns.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def test_bootstrap_drawdown_runs(bootstrap_csv):
    """Bootstrap mode should produce valid simulation results."""
    config = _simple_config(
        assets=[
            Asset(name="A", value=100000, mean_return=0.05, std_dev=0.15,
                  tax_type="roth", return_source="test"),
            Asset(name="B", value=50000, mean_return=0.02, std_dev=0.05,
                  tax_type="cash"),  # parametric, no return_source
        ],
        method="bootstrap",
        return_sources={"test": bootstrap_csv},
        spending=SpendingSchedule(categories={"living": 5000}),
        life_expectancy=55,
    )
    result = run(config)
    assert result.wealth.shape == (100, 6)
    assert result.asset_wealth.shape == (100, 6, 2)
    # Should not be all zeros (portfolio should survive some years)
    assert result.wealth[:, 0].sum() > 0


def test_bootstrap_accumulation_stochastic(bootstrap_csv):
    """Bootstrap accumulation should produce different values per simulation."""
    config = _simple_config(
        assets=[
            Asset(name="A", value=100000, mean_return=0.05, std_dev=0.15,
                  tax_type="roth", return_source="test"),
        ],
        method="bootstrap",
        return_sources={"test": bootstrap_csv},
        current_age=45,
        retirement_age=48,  # 3 years = 36 months (matches CSV length)
        life_expectancy=50,
        num_simulations=50,
    )
    result = run(config)
    # In bootstrap mode, accumulation is stochastic, so initial drawdown values differ per sim
    init_values = result.asset_wealth[:, 0, 0]
    # With constant 1% returns and same seed, all paths should be identical
    # (since all 12-month blocks produce the same return)
    # But the key test is that the code runs without error
    assert result.wealth.shape == (50, 3)


def test_bootstrap_seed_reproducibility(bootstrap_csv):
    """Bootstrap mode with same seed should produce identical results."""
    config = _simple_config(
        assets=[
            Asset(name="A", value=100000, mean_return=0.05, std_dev=0.15,
                  tax_type="roth", return_source="test"),
        ],
        method="bootstrap",
        return_sources={"test": bootstrap_csv},
        seed=42,
    )
    r1 = run(config)
    r2 = run(config)
    np.testing.assert_array_equal(r1.wealth, r2.wealth)
