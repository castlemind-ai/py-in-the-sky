"""Helpers for mortgage, spending, tax, and 529 plan resolution."""

from __future__ import annotations

from datetime import date

import numpy as np

from sim.config import (
    FutureCost,
    MortgageConfig,
    Plan529,
    SpendingSchedule,
    TaxConfig,
)


def months_since_purchase(mortgage: MortgageConfig, ref_date: date | None = None) -> int:
    """Compute months elapsed since mortgage purchase date."""
    purchase = date.fromisoformat(mortgage.purchase_date)
    ref = ref_date or date.today()
    return (ref.year - purchase.year) * 12 + (ref.month - purchase.month)


def amortize_mortgage(mortgage: MortgageConfig, total_months: int) -> float:
    """Compute remaining principal balance after total_months of payments."""
    principal = mortgage.purchase_price * (1 - mortgage.down_payment_pct)
    monthly_rate = mortgage.interest_rate / 12

    for _ in range(total_months):
        if principal <= 0:
            break
        month_interest = principal * monthly_rate
        principal_paid = mortgage.monthly_payment - month_interest
        principal = max(principal - principal_paid, 0)

    return principal


def annual_mortgage_step(principal: float, mortgage: MortgageConfig) -> tuple[float, float]:
    """Simulate one year of mortgage payments.

    Returns (new_principal, annual_mortgage_cost).
    If principal is paid off, mortgage cost drops to zero.
    """
    if principal <= 0:
        return 0.0, 0.0

    annual_interest = principal * mortgage.interest_rate
    annual_payment = mortgage.monthly_payment * 12
    principal_paid = annual_payment - annual_interest
    new_principal = max(principal - principal_paid, 0)

    return new_principal, annual_payment


def resolve_spending_year(
    spending: SpendingSchedule,
    year: int,
    mortgage: MortgageConfig | None,
    mortgage_principal: float,
    inflation_factor: float,
) -> tuple[float, float]:
    """Compute total annual spending for a given year.

    Returns (total_costs_nominal, new_mortgage_principal).

    - Non-housing categories are inflation-adjusted.
    - School costs drop to zero after all kids finish (based on school_end_years).
    - Mortgage payment is fixed; non-mortgage housing inflates.
    """
    total = 0.0

    for category, amount in spending.categories.items():
        if category == "school":
            # Count how many kids are still in school this year
            kids_in_school = sum(1 for end_yr in spending.school_end_years if year < end_yr)
            if kids_in_school == 0:
                continue
            # Scale: original amount assumed all kids in school
            total_kids = len(spending.school_end_years)
            if total_kids > 0:
                total += amount * (kids_in_school / total_kids) * inflation_factor
        else:
            total += amount * inflation_factor

    # Housing: mortgage (fixed) + non-mortgage (inflated)
    new_principal = mortgage_principal
    if mortgage is not None:
        new_principal, mort_cost = annual_mortgage_step(mortgage_principal, mortgage)
        total += mort_cost
        # Non-mortgage housing costs (property tax, insurance, etc.) inflate
        total += mortgage.non_mortgage_monthly * 12 * inflation_factor

    return total, new_principal


def resolve_future_costs(costs: list[FutureCost], years: int) -> np.ndarray:
    """Return an array of total future costs per year in today's dollars.

    Multi-year costs are spread evenly across their duration.
    """
    result = np.zeros(years)
    for cost in costs:
        annual_amount = cost.amount / cost.duration if cost.duration > 1 else cost.amount
        for y in range(cost.year, min(cost.year + cost.duration, years)):
            result[y] += annual_amount
    return result


def gross_up_for_tax(net_amount: float, tax_rate: float) -> float:
    """Gross up a net spending amount to account for taxes on withdrawal."""
    if tax_rate >= 1.0:
        return net_amount  # safety: avoid division by zero
    return net_amount / (1 - tax_rate)


def grow_529(plan: Plan529, months: int, monthly_rate: float) -> float:
    """Grow a 529 plan over a number of months with contributions.

    Returns the final value.
    """
    value = plan.current_value
    for m in range(months):
        if m < plan.start_month:
            # Contributions haven't started yet (baby not born)
            value *= (1 + monthly_rate)
        else:
            value *= (1 + monthly_rate)
            value += plan.monthly_contribution
    return value
