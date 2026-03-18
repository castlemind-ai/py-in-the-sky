"""Tests for event resolution helpers."""

import numpy as np
import pytest

from sim.config import FutureCost, MortgageConfig, Plan529, SpendingSchedule
from sim.events import (
    amortize_mortgage,
    annual_mortgage_step,
    gross_up_for_tax,
    grow_529,
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
