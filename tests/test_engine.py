"""Tests for the simulation engine."""

import numpy as np

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
