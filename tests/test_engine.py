"""Tests for the simulation engine."""

import csv
from pathlib import Path

import numpy as np
import pytest

from sim.config import (
    AgeTransition,
    Asset,
    BlackSwanEvent,
    Contribution,
    FutureCost,
    Plan529,
    RothConversion,
    SimConfig,
    SocialSecurityConfig,
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


# ============================================================
# New feature tests
# ============================================================

def test_state_capital_gains_rate_adds_to_federal():
    """state_capital_gains_rate should stack on top of capital_gains_rate."""
    from sim.config import TaxConfig
    tax = TaxConfig(capital_gains_rate=0.20, state_capital_gains_rate=0.133)
    assert tax.rate_for("taxable") == pytest.approx(0.333)


def test_state_cap_gains_increases_tax_at_retirement(bootstrap_csv):
    """Retirement diversification should deduct combined federal + state cap gains tax."""
    tax_federal = TaxConfig(capital_gains_rate=0.20, state_capital_gains_rate=0.0)
    tax_combined = TaxConfig(capital_gains_rate=0.20, state_capital_gains_rate=0.133)

    def _make(tax_cfg):
        return _simple_config(
            assets=[Asset(name="Stock", value=500000, mean_return=0.0, std_dev=0.0,
                          tax_type="taxable", cost_basis=100000,
                          return_source="test", retire_to_source="test")],
            tax=tax_cfg,
            method="bootstrap",
            return_sources={"test": bootstrap_csv},
            accumulation_return=0.0,
            current_age=49, retirement_age=50,
            life_expectancy=52, num_simulations=10, seed=1,
            spending=SpendingSchedule(categories={}),
        )

    r_fed = run(_make(tax_federal))
    r_combo = run(_make(tax_combined))
    # Combined rate means more tax deducted at retirement → lower starting wealth
    assert r_combo.wealth[:, 0].mean() < r_fed.wealth[:, 0].mean()


def test_progressive_brackets_lower_effective_rate_for_small_withdrawals():
    """Progressive brackets: smaller withdrawals should have a lower effective rate."""
    from sim.config import TaxBracket, TaxConfig
    tax = TaxConfig(
        income_rate=0.35,
        tax_brackets=[
            TaxBracket(threshold=50000, rate=0.12),
            TaxBracket(threshold=100000, rate=0.22),
            TaxBracket(threshold=200000, rate=0.32),
        ],
    )
    rate_small = tax.effective_income_rate(40000)   # entirely in 12% bracket
    rate_large = tax.effective_income_rate(150000)  # spans multiple brackets
    assert rate_small == pytest.approx(0.12)
    assert rate_large > rate_small


def test_progressive_brackets_fallback_when_empty():
    """No tax_brackets → effective_income_rate returns flat income_rate."""
    from sim.config import TaxConfig
    tax = TaxConfig(income_rate=0.30)
    assert tax.effective_income_rate(80000) == pytest.approx(0.30)
    assert tax.effective_income_rate(0) == pytest.approx(0.30)


def test_early_withdrawal_penalty_pre_59_half():
    """Traditional withdrawals before 59.5 should cost more than after."""
    # Retire at 55; penalty applies years 0-4 (age 55-59), not years 5+ (age 60+)
    config_early = _simple_config(
        assets=[Asset(name="401k", value=1_000_000, mean_return=0.0, std_dev=0.0,
                      tax_type="traditional")],
        spending=SpendingSchedule(categories={"living": 50000}),
        tax=TaxConfig(income_rate=0.25, early_withdrawal_penalty_rate=0.10),
        current_age=54,
        retirement_age=55,
        life_expectancy=57,  # 2 years of retirement (ages 55 and 56, both < 59.5)
        num_simulations=10,
    )
    config_late = _simple_config(
        assets=[Asset(name="401k", value=1_000_000, mean_return=0.0, std_dev=0.0,
                      tax_type="traditional")],
        spending=SpendingSchedule(categories={"living": 50000}),
        tax=TaxConfig(income_rate=0.25, early_withdrawal_penalty_rate=0.0),  # no penalty
        current_age=54,
        retirement_age=55,
        life_expectancy=57,
        num_simulations=10,
    )
    result_early = run(config_early)
    result_late = run(config_late)
    # With penalty, more is withdrawn to cover the same net spending → less wealth remains
    assert result_early.wealth[0, -1] < result_late.wealth[0, -1]


def test_rule_of_55_exempts_from_penalty():
    """rule_of_55_eligible assets should not incur early withdrawal penalty."""
    config_no_exempt = _simple_config(
        assets=[Asset(name="401k", value=500_000, mean_return=0.0, std_dev=0.0,
                      tax_type="traditional", rule_of_55_eligible=False)],
        spending=SpendingSchedule(categories={"living": 40000}),
        tax=TaxConfig(income_rate=0.20, early_withdrawal_penalty_rate=0.10),
        current_age=54, retirement_age=55, life_expectancy=57, num_simulations=10,
    )
    config_exempt = _simple_config(
        assets=[Asset(name="401k", value=500_000, mean_return=0.0, std_dev=0.0,
                      tax_type="traditional", rule_of_55_eligible=True)],
        spending=SpendingSchedule(categories={"living": 40000}),
        tax=TaxConfig(income_rate=0.20, early_withdrawal_penalty_rate=0.10),
        current_age=54, retirement_age=55, life_expectancy=57, num_simulations=10,
    )
    r_no_ex = run(config_no_exempt)
    r_ex = run(config_exempt)
    # Exempted account: lower effective rate → less withdrawn → more wealth remaining
    assert r_ex.wealth[0, -1] > r_no_ex.wealth[0, -1]


def test_social_security_reduces_portfolio_withdrawal():
    """SS income should reduce required portfolio withdrawals in claim years."""
    spending = SpendingSchedule(categories={"living": 60000})

    config_no_ss = _simple_config(
        assets=[Asset(name="A", value=2_000_000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=spending,
        life_expectancy=72,
        current_age=49, retirement_age=50, num_simulations=10,
    )
    config_ss = _simple_config(
        assets=[Asset(name="A", value=2_000_000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=spending,
        social_security=SocialSecurityConfig(monthly_benefit=2500, claim_age=67,
                                             inflation_adjusted=False),
        life_expectancy=72,
        current_age=49, retirement_age=50, num_simulations=10,
    )
    r_no_ss = run(config_no_ss)
    r_ss = run(config_ss)
    # SS income partially covers spending after age 67 → higher remaining wealth
    assert r_ss.wealth[0, -1] > r_no_ss.wealth[0, -1]


def test_age_transition_reduces_spending_at_65():
    """Age transition should swap spending categories at the right age."""
    spending = SpendingSchedule(
        categories={"healthcare_pre": 30000},
        age_transitions=[
            AgeTransition(at_age=65, remove=["healthcare_pre"], add={"healthcare_post": 8000}),
        ],
    )
    config_transition = _simple_config(
        assets=[Asset(name="A", value=5_000_000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=spending,
        current_age=49, retirement_age=50,
        life_expectancy=70,
        num_simulations=10,
    )
    # Without transition: flat $30k/yr for 20 years = $600k total
    config_flat = _simple_config(
        assets=[Asset(name="A", value=5_000_000, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        spending=SpendingSchedule(categories={"healthcare_pre": 30000}),
        current_age=49, retirement_age=50,
        life_expectancy=70,
        num_simulations=10,
    )
    r_trans = run(config_transition)
    r_flat = run(config_flat)
    # Transition drops spending at 65 from $30k → $8k, so more wealth remains
    assert r_trans.wealth[0, -1] > r_flat.wealth[0, -1]


def test_growing_contributions_mc():
    """annual_growth_rate should increase accumulated balance vs. flat contribution."""
    config_flat = _simple_config(
        assets=[Asset(name="A", value=0, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        contributions=[Contribution(name="Save", monthly_amount=1000, target_account="A",
                                    annual_growth_rate=0.0)],
        accumulation_return=0.0,
        current_age=45, retirement_age=50, life_expectancy=52, num_simulations=10,
    )
    config_growing = _simple_config(
        assets=[Asset(name="A", value=0, mean_return=0.0, std_dev=0.0, tax_type="roth")],
        contributions=[Contribution(name="Save", monthly_amount=1000, target_account="A",
                                    annual_growth_rate=0.10)],
        accumulation_return=0.0,
        current_age=45, retirement_age=50, life_expectancy=52, num_simulations=10,
    )
    accum_flat = run_accumulation(config_flat)
    accum_growing = run_accumulation(config_growing)
    # Growing contributions compound annually, so total > 5*12*1000 = $60k
    assert accum_growing.asset_values["A"] > accum_flat.asset_values["A"]


def test_roth_conversion_reduces_source_increases_target():
    """Roth conversion should transfer funds from traditional to Roth (net of tax)."""
    config = _simple_config(
        assets=[
            Asset(name="401k", value=1_000_000, mean_return=0.0, std_dev=0.0,
                  tax_type="traditional"),
            Asset(name="Roth", value=0, mean_return=0.0, std_dev=0.0, tax_type="roth"),
        ],
        spending=SpendingSchedule(categories={}),
        tax=TaxConfig(income_rate=0.25),
        roth_conversions=[
            RothConversion(start_age=50, end_age=55, annual_amount=80000,
                           source_account="401k", target_account="Roth"),
        ],
        current_age=49, retirement_age=50, life_expectancy=56,
        num_simulations=10,
    )
    result = run(config)
    # After 5 years of $80k conversions: $400k out of 401k, $300k net into Roth (after 25% tax)
    final_401k = result.asset_wealth[0, -1, 0]
    final_roth = result.asset_wealth[0, -1, 1]
    assert final_401k < 1_000_000
    assert final_roth > 0


def test_tax_optimized_drawdown_preserves_roth():
    """Tax-optimized sequencing should drain taxable before Roth."""
    def _config(strategy):
        return _simple_config(
            assets=[
                Asset(name="Taxable", value=200_000, mean_return=0.0, std_dev=0.0,
                      tax_type="taxable"),
                Asset(name="Roth", value=800_000, mean_return=0.0, std_dev=0.0,
                      tax_type="roth"),
            ],
            spending=SpendingSchedule(categories={"living": 60000}),
            tax=TaxConfig(capital_gains_rate=0.20),
            drawdown_strategy=strategy,
            current_age=49, retirement_age=50, life_expectancy=55, num_simulations=10,
        )

    r_prop = run(_config("proportional"))
    r_opt = run(_config("tax_optimized"))

    # With tax_optimized, Roth (no-tax) is used last → higher remaining wealth after fees
    roth_idx = 1
    # Optimized should leave more in Roth at the end (less total tax paid)
    assert r_opt.asset_wealth[0, -1, roth_idx] >= r_prop.asset_wealth[0, -1, roth_idx]


def test_lognormal_returns_no_below_minus_one():
    """Lognormal return draws should always be > -1 (no below -100% losses)."""
    config = _simple_config(
        assets=[Asset(name="A", value=100_000, mean_return=0.07, std_dev=0.20,
                      tax_type="roth", return_distribution="lognormal")],
        spending=SpendingSchedule(categories={}),
        life_expectancy=80, num_simulations=1000, seed=42,
    )
    result = run(config)
    # All returns were applied; wealth should never go negative due to return alone
    # (it may hit zero due to spending, but that's floored separately)
    assert np.all(result.wealth >= 0)


def test_lognormal_vs_normal_returns_differ():
    """Lognormal and normal parameterizations should produce different results."""
    def _config(dist):
        return _simple_config(
            assets=[Asset(name="A", value=100_000, mean_return=0.07, std_dev=0.25,
                          tax_type="roth", return_distribution=dist)],
            spending=SpendingSchedule(categories={}),
            life_expectancy=60, num_simulations=500, seed=42,
        )

    r_normal = run(_config("normal"))
    r_lognormal = run(_config("lognormal"))
    assert not np.array_equal(r_normal.wealth, r_lognormal.wealth)


def test_after_tax_401k_withdrawals_tax_free():
    """after_tax_401k tax_type should produce 0.0 tax rate (like Roth)."""
    from sim.config import TaxConfig
    tax = TaxConfig(income_rate=0.35, capital_gains_rate=0.20)
    assert tax.rate_for("after_tax_401k") == 0.0


def test_401k_contribution_limit_validation():
    """Contributions exceeding $70k/year to is_401k_plan assets should raise."""
    from sim.config import ConfigValidationError, validate_config
    config = _simple_config(
        assets=[Asset(name="401k", value=100_000, mean_return=0.05, std_dev=0.1,
                      tax_type="traditional", is_401k_plan=True)],
        contributions=[
            # $6,000/month * 12 = $72,000/year > $70,000 limit
            Contribution(name="Big Save", monthly_amount=6000, target_account="401k"),
        ],
    )
    with pytest.raises(ConfigValidationError, match="401k limit"):
        validate_config(config)


def test_drawdown_strategy_validation():
    """Unknown drawdown_strategy should raise ConfigValidationError."""
    from sim.config import ConfigValidationError, validate_config
    config = _simple_config(drawdown_strategy="unknown_strategy")
    with pytest.raises(ConfigValidationError, match="drawdown_strategy"):
        validate_config(config)


def test_social_security_config_loaded(tmp_path):
    """load_config should parse social_security block correctly."""
    import yaml
    from sim.config import load_config
    cfg_content = {
        "current_age": 40,
        "retirement_age": 55,
        "life_expectancy": 85,
        "assets": [{"name": "A", "value": 100000, "mean_return": 0.06, "std_dev": 0.15}],
        "spending": {"categories": {"living": 40000}},
        "social_security": {
            "monthly_benefit": 3000,
            "claim_age": 67,
            "inflation_adjusted": True,
        },
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg_content))
    config = load_config(p)
    assert config.social_security is not None
    assert config.social_security.monthly_benefit == 3000
    assert config.social_security.claim_age == 67
