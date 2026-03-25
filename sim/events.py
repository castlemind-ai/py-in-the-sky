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
    """Simulate one year of mortgage payments using monthly amortization.

    Returns (new_principal, annual_mortgage_cost).
    Uses the same monthly logic as the accumulation phase for consistency.
    If principal is paid off, mortgage cost drops to zero.
    """
    if principal <= 0:
        return 0.0, 0.0

    monthly_rate = mortgage.interest_rate / 12
    total_paid = 0.0
    for _ in range(12):
        if principal <= 0:
            break
        month_interest = principal * monthly_rate
        principal_paid = mortgage.monthly_payment - month_interest
        principal = max(principal - principal_paid, 0)
        total_paid += mortgage.monthly_payment

    return principal, total_paid


def resolve_spending_year(
    spending: SpendingSchedule,
    year: int,
    mortgage: MortgageConfig | None,
    mortgage_principal: float,
    inflation_factor: float,
    current_age: int = 0,
) -> tuple[float, float]:
    """Compute total annual spending for a given year.

    Returns (total_costs_nominal, new_mortgage_principal).

    - Applies age_transitions via resolve_active_spending_categories so that
      category changes fire at the right ages (consistent with the drawdown loop).
    - School costs drop to zero after all kids finish (based on school_end_years).
    - Mortgage payment is fixed; non-mortgage housing inflates.
    """
    total = 0.0
    active_categories = resolve_active_spending_categories(spending, current_age)

    for category, amount in active_categories.items():
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


def resolve_active_spending_categories(
    spending: "SpendingSchedule",
    current_age: int = 0,
    age: int | None = None,
) -> dict[str, float]:
    """Return the effective spending categories after applying age transitions.

    Starts from spending.categories, then fires each transition whose at_age
    is <= current_age in ascending order, removing and adding categories.
    """
    effective_age = age if age is not None else current_age
    categories = dict(spending.categories)
    for transition in sorted(spending.age_transitions, key=lambda t: t.at_age):
        if effective_age >= transition.at_age:
            for key in transition.remove:
                categories.pop(key, None)
            categories.update(transition.add)
    return categories


# IRS Uniform Lifetime Table (age -> distribution period divisor), ages 73–120.
# Source: IRS Publication 590-B, Appendix B, Table III.
_IRS_ULT: dict[int, float] = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94: 9.5,  95: 8.9,  96: 8.4,  97: 7.8,  98: 7.3,  99: 6.8,  100: 6.4,
    101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1,
    108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0,
}


def compute_rmd(balance: float, age: int) -> float:
    """Compute the IRS Required Minimum Distribution for a given balance and age.

    Uses the IRS Uniform Lifetime Table (Publication 590-B, Table III).
    Returns 0.0 if age < 73 (RMDs not yet required).
    """
    if age < 73 or balance <= 0:
        return 0.0
    divisor = _IRS_ULT.get(age, 2.0)  # age 120+ uses 2.0
    return balance / divisor


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
