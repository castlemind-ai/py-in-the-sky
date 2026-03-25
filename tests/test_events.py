"""Tests for event resolution helpers."""

import numpy as np
import pytest

from sim.config import AgeTransition, FutureCost, MortgageConfig, Plan529, SpendingSchedule
from sim.events import (
    amortize_mortgage,
    annual_mortgage_step,
    compute_rmd,
    gross_up_for_tax,
    grow_529,
    resolve_active_spending_categories,
    resolve_future_costs,
)


# --- Mortgage tests ---

def test_amortize_mortgage_reduces_principal():
    mort = MortgageConfig(
        purchase_price=1000000, down_payment_pct=0.20,
        interest_rate=0.05, monthly_payment=4500,
        purchase_date="2020-01-01", non_mortgage_monthly=1000,
    )
    principal_start = 800000
    principal_after = amortize_mortgage(mort, 12)
    assert principal_after < principal_start
    assert principal_after > 0


def test_amortize_mortgage_zero_months():
    mort = MortgageConfig(
        purchase_price=500000, down_payment_pct=0.20,
        interest_rate=0.03, monthly_payment=2000,
        purchase_date="2020-01-01", non_mortgage_monthly=500,
    )
    assert amortize_mortgage(mort, 0) == 400000


def test_annual_mortgage_step():
    mort = MortgageConfig(
        purchase_price=1000000, down_payment_pct=0.20,
        interest_rate=0.04, monthly_payment=4000,
        purchase_date="2020-01-01", non_mortgage_monthly=1000,
    )
    new_p, cost = annual_mortgage_step(800000, mort)
    assert new_p < 800000
    assert cost == 4000 * 12


def test_annual_mortgage_step_paid_off():
    mort = MortgageConfig(
        purchase_price=100000, down_payment_pct=0.20,
        interest_rate=0.04, monthly_payment=1000,
        purchase_date="2020-01-01", non_mortgage_monthly=500,
    )
    new_p, cost = annual_mortgage_step(0, mort)
    assert new_p == 0
    assert cost == 0


# --- Tax tests ---

def test_gross_up_for_tax():
    # $100k spending at 35% tax = $153,846 withdrawal
    gross = gross_up_for_tax(100000, 0.35)
    assert gross == pytest.approx(153846.15, rel=0.01)


def test_gross_up_no_tax():
    assert gross_up_for_tax(100000, 0.0) == 100000


# --- Future costs ---

def test_resolve_future_costs_single():
    costs = [FutureCost(name="Wedding", amount=30000, year=5)]
    result = resolve_future_costs(costs, 10)
    assert result[5] == 30000
    assert result.sum() == 30000


def test_resolve_future_costs_multi_year():
    costs = [FutureCost(name="College", amount=75000, year=10, duration=4)]
    result = resolve_future_costs(costs, 20)
    expected_annual = 75000 / 4
    for y in range(10, 14):
        assert result[y] == expected_annual
    assert result.sum() == pytest.approx(75000)


def test_resolve_future_costs_beyond_horizon():
    costs = [FutureCost(name="Late", amount=100000, year=8, duration=4)]
    result = resolve_future_costs(costs, 10)
    assert result[8] == 100000 / 4
    assert result[9] == 100000 / 4
    assert result.sum() == pytest.approx(50000)


# --- 529 tests ---

def test_grow_529_basic():
    plan = Plan529(
        name="Test", current_value=10000,
        monthly_contribution=1000, mean_return=0.0,
        child_college_start_year=10,
    )
    # 0% return, 12 months, $1k/mo = 10000 + 12000 = 22000
    result = grow_529(plan, 12, 0.0)
    assert result == pytest.approx(22000)


def test_grow_529_with_delayed_start():
    plan = Plan529(
        name="Test", current_value=0,
        monthly_contribution=1000, mean_return=0.0,
        child_college_start_year=10, start_month=6,
    )
    # First 6 months: no contributions. Next 6: $1k each = $6000
    result = grow_529(plan, 12, 0.0)
    assert result == pytest.approx(6000)


# --- RMD tests ---

def test_compute_rmd_before_73():
    assert compute_rmd(1_000_000, 70) == 0.0
    assert compute_rmd(1_000_000, 72) == 0.0


def test_compute_rmd_at_73():
    # IRS divisor at 73 is 26.5
    assert compute_rmd(1_000_000, 73) == pytest.approx(1_000_000 / 26.5)


def test_compute_rmd_at_80():
    # IRS divisor at 80 is 20.2
    assert compute_rmd(500_000, 80) == pytest.approx(500_000 / 20.2)


def test_compute_rmd_zero_balance():
    assert compute_rmd(0, 75) == 0.0


# --- Age transition tests ---

def test_resolve_active_spending_no_transitions():
    spending = SpendingSchedule(categories={"food": 10000, "travel": 5000})
    result = resolve_active_spending_categories(spending, age=60)
    assert result == {"food": 10000, "travel": 5000}


def test_resolve_active_spending_transition_fires_at_age():
    spending = SpendingSchedule(
        categories={"healthcare_pre": 30000},
        age_transitions=[
            AgeTransition(at_age=65, remove=["healthcare_pre"], add={"healthcare_post": 8000}),
        ],
    )
    # Before age 65: original category present
    before = resolve_active_spending_categories(spending, age=64)
    assert "healthcare_pre" in before
    assert "healthcare_post" not in before

    # At age 65: transition fires
    at = resolve_active_spending_categories(spending, age=65)
    assert "healthcare_pre" not in at
    assert at["healthcare_post"] == 8000


def test_resolve_active_spending_multiple_transitions():
    spending = SpendingSchedule(
        categories={"a": 1000, "b": 2000},
        age_transitions=[
            AgeTransition(at_age=60, remove=["a"], add={"c": 3000}),
            AgeTransition(at_age=65, remove=["b"], add={"d": 4000}),
        ],
    )
    result = resolve_active_spending_categories(spending, age=67)
    assert result == {"c": 3000, "d": 4000}
