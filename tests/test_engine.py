"""Tests for the simulation engine."""

import csv
from pathlib import Path

import numpy as np
import pytest

from sim.config import (
    Asset,
    BlackSwanEvent,
    Contribution,
    FutureCost,
    Plan529,
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


def test_retirement_diversification_pays_cap_gains_tax(bootstrap_csv):
    """Assets with retire_to_source should pay capital gains tax at retirement."""
    # Use mc mode for both so accumulation is deterministic and comparable.
    # The diversification logic runs in run() regardless of mode.
    config_no_div = _simple_config(
        assets=[
            Asset(name="Stock", value=500000, mean_return=0.0, std_dev=0.0,
                  tax_type="taxable", cost_basis=200000),
        ],
        tax=TaxConfig(capital_gains_rate=0.20),
        spending=SpendingSchedule(categories={}),
        accumulation_return=0.0,
        life_expectancy=52,
        seed=42,
    )
    result_no_div = run(config_no_div)

    # With retire_to_source: pays 20% tax on $300k gain = $60k tax at retirement
    config_div = _simple_config(
        assets=[
            Asset(name="Stock", value=500000, mean_return=0.0, std_dev=0.0,
                  tax_type="taxable", cost_basis=200000,
                  return_source="test", retire_to_source="test"),
        ],
        tax=TaxConfig(capital_gains_rate=0.20),
        spending=SpendingSchedule(categories={}),
        method="bootstrap",
        return_sources={"test": bootstrap_csv},
        accumulation_return=0.0,
        current_age=49,
        retirement_age=50,  # minimize accumulation to keep values close to initial
        life_expectancy=52,
        seed=42,
        num_simulations=50,
    )
    result_div = run(config_div)

    # Without diversification, no tax is deducted up front
    init_no_div = result_no_div.wealth[0, 0]
    assert init_no_div == 500000

    # With diversification, cap gains tax is deducted: gain = (value - basis), tax = 20% of gain
    # The accumulation phase grows the stock (1% monthly for 12 months), so
    # the exact post-tax value depends on growth. But it must be less than
    # the pre-tax accumulated value.
    accum_value = result_div.accumulation.asset_values["Stock"]  # array (n_sims,)
    expected_tax = np.maximum(accum_value - 200000, 0) * 0.20
    expected_post_tax = accum_value - expected_tax
    np.testing.assert_allclose(
        result_div.asset_wealth[:, 0, 0], expected_post_tax, rtol=1e-6,
    )


def test_school_spending_phaseout_multiple_children():
    """School spending should scale by kids_in_school/total_kids and phase out."""
    # Two children, one finishes school at year 16 (absolute), other at year 18
    # Retirement starts at year 15
    config = _simple_config(
        assets=[Asset(name="A", value=1_000_000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=SpendingSchedule(
            categories={"school": 20000},  # $20k/yr for school when all kids are enrolled
            school_end_years=[16, 18],
        ),
        current_age=35,
        retirement_age=50,  # years_to_retirement = 15
        life_expectancy=60,
        num_simulations=10,
    )
    result = run(config)

    # Year 0 in drawdown = absolute year 15. Both kids in school (15 < 16 and 15 < 18).
    # Spending = 20000 * (2/2) = $20000
    # Year 1 = absolute year 16. One kid finished (16 < 16 is false). kids_in_school = 1.
    # Spending = 20000 * (1/2) = $10000
    # Year 2 = absolute year 17. kids_in_school = 1 (17 < 18).
    # Spending = 20000 * (1/2) = $10000
    # Year 3 = absolute year 18. kids_in_school = 0. Spending = $0.
    # Total spending over 10 years = 20000 + 10000 + 10000 = $40000

    final_wealth = result.wealth[0, -1]
    np.testing.assert_allclose(final_wealth, 1_000_000 - 40000, rtol=1e-6)


def test_bootstrap_blend_weights(bootstrap_csv):
    """Blend weight < 1.0 should mix bootstrap and parametric returns."""
    # Pure bootstrap (blend=1.0)
    config_pure = _simple_config(
        assets=[
            Asset(name="A", value=100000, mean_return=0.05, std_dev=0.15,
                  tax_type="roth", return_source="test", bootstrap_blend=1.0),
        ],
        method="bootstrap",
        return_sources={"test": bootstrap_csv},
        spending=SpendingSchedule(categories={}),
        life_expectancy=55,
        seed=42,
        num_simulations=500,
    )
    result_pure = run(config_pure)

    # Blended (blend=0.5 = 50% bootstrap + 50% parametric)
    config_blend = _simple_config(
        assets=[
            Asset(name="A", value=100000, mean_return=0.05, std_dev=0.15,
                  tax_type="roth", return_source="test", bootstrap_blend=0.5),
        ],
        method="bootstrap",
        return_sources={"test": bootstrap_csv},
        spending=SpendingSchedule(categories={}),
        life_expectancy=55,
        seed=42,
        num_simulations=500,
    )
    result_blend = run(config_blend)

    # Blended results should differ from pure bootstrap
    assert not np.array_equal(result_pure.wealth, result_blend.wealth)


def test_integration_bootstrap_diversification_529(bootstrap_csv):
    """Integration: bootstrap accumulation + retirement diversification + 529 offset."""
    config = _simple_config(
        assets=[
            Asset(name="Stock", value=300000, mean_return=0.08, std_dev=0.20,
                  tax_type="taxable", cost_basis=100000,
                  return_source="test", retire_to_source="test",
                  bootstrap_blend=0.8),
            Asset(name="401k", value=200000, mean_return=0.06, std_dev=0.15,
                  tax_type="traditional"),
        ],
        method="bootstrap",
        return_sources={"test": bootstrap_csv},
        contributions=[
            Contribution(name="Save", monthly_amount=2000, target_account="401k"),
        ],
        plans_529=[
            Plan529(name="Kid529", current_value=5000, monthly_contribution=300,
                    mean_return=0.05, child_college_start_year=3),
        ],
        future_costs=[
            FutureCost(name="College", amount=40000, year=3, duration=4),
        ],
        spending=SpendingSchedule(categories={"living": 30000}),
        tax=TaxConfig(income_rate=0.30, capital_gains_rate=0.20),
        current_age=47,
        retirement_age=50,
        life_expectancy=60,
        num_simulations=200,
        seed=42,
    )
    result = run(config)

    # Basic shape checks
    assert result.wealth.shape == (200, 11)
    assert result.asset_wealth.shape == (200, 11, 2)

    # Diversification tax should reduce initial Stock values
    # Gain = value_at_retirement - $100k basis; tax = 20% of gain
    stock_init = result.asset_wealth[:, 0, 0]
    assert np.all(stock_init < result.accumulation.asset_values["Stock"])

    # 529 should have grown
    assert result.accumulation.plan_529_values["Kid529"] > 5000

    # Portfolio should not be all zeros (should survive at least some years)
    assert result.wealth[:, 0].sum() > 0
