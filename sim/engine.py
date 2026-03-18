"""Two-phase Monte Carlo simulation engine: accumulation + drawdown."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.config import SimConfig
from sim.events import (
    amortize_mortgage,
    annual_mortgage_step,
    gross_up_for_tax,
    grow_529,
    months_since_purchase,
    resolve_future_costs,
    resolve_spending_year,
)


@dataclass
class AccumulationResult:
    """Results of the accumulation phase."""

    asset_values: dict[str, float]  # asset name -> value at retirement
    mortgage_principal: float
    plan_529_values: dict[str, float]  # plan name -> value at retirement
    months_elapsed_mortgage: int  # total months of mortgage payments by retirement


@dataclass
class SimResult:
    """Results of the full Monte Carlo simulation."""

    wealth: np.ndarray  # shape (num_sims, years_in_retirement + 1)
    asset_wealth: np.ndarray  # shape (num_sims, years_in_retirement + 1, num_assets)
    dist_rates: np.ndarray  # shape (num_sims, years_in_retirement) — withdrawal rates
    accumulation: AccumulationResult
    config: SimConfig


def run_accumulation(config: SimConfig) -> AccumulationResult:
    """Run the deterministic accumulation phase.

    Grows assets monthly with contributions for years_to_retirement.
    """
    yrs = config.years_to_retirement
    months = yrs * 12

    # Initialize asset values
    asset_values = {a.name: a.value for a in config.assets}

    # Use a single blended return rate for all assets during accumulation
    # (matches R script approach — per-asset volatility only matters in drawdown)
    monthly_rate = (1 + config.accumulation_return) ** (1 / 12) - 1

    # Build contribution map: target_account -> list of contributions
    contrib_map: dict[str, list] = {}
    for c in config.contributions:
        contrib_map.setdefault(c.target_account, []).append(c)

    # Compute mortgage state
    mort_principal = 0.0
    mort_months_elapsed = 0
    if config.mortgage:
        mort_months_elapsed = months_since_purchase(config.mortgage)
        mort_principal = amortize_mortgage(config.mortgage, mort_months_elapsed)

        # Continue mortgage payments during accumulation
        mort_monthly_rate = config.mortgage.interest_rate / 12
        for _ in range(months):
            if mort_principal <= 0:
                break
            month_int = mort_principal * mort_monthly_rate
            principal_paid = config.mortgage.monthly_payment - month_int
            mort_principal = max(mort_principal - principal_paid, 0)
        mort_months_elapsed += months

    # Grow assets monthly with blended rate + contributions
    for month in range(months):
        for name, val in asset_values.items():
            # Apply blended monthly return
            asset_values[name] = val * (1 + monthly_rate)

            # Add contributions
            if name in contrib_map:
                for c in contrib_map[name]:
                    asset_values[name] += c.monthly_amount

    # Grow 529 plans
    plan_529_values = {}
    for plan in config.plans_529:
        monthly_rate = (1 + plan.mean_return) ** (1 / 12) - 1
        plan_529_values[plan.name] = grow_529(plan, months, monthly_rate)

    return AccumulationResult(
        asset_values=asset_values,
        mortgage_principal=mort_principal,
        plan_529_values=plan_529_values,
        months_elapsed_mortgage=mort_months_elapsed,
    )


def run(config: SimConfig) -> SimResult:
    """Run the full two-phase simulation."""
    rng = np.random.default_rng(config.seed)

    # Phase 1: Accumulation (deterministic)
    accum = run_accumulation(config)

    # Phase 2: Drawdown (Monte Carlo)
    n = config.num_simulations
    yrs = config.years_in_retirement
    num_assets = len(config.assets)

    # Initial values from accumulation
    init_values = np.array([accum.asset_values[a.name] for a in config.assets])
    init_total = init_values.sum()

    # Asset return parameters
    means = np.array([a.mean_return for a in config.assets])
    stds = np.array([a.std_dev for a in config.assets])

    # Tax rates per asset
    tax_rates = np.array([config.tax.rate_for(a.tax_type) for a in config.assets])

    # Pre-compute future costs (college) per year in today's dollars
    future_costs_by_year = resolve_future_costs(config.future_costs, yrs)

    # Pre-compute 529 values at retirement
    total_529 = sum(accum.plan_529_values.values())

    # Map 529 plans to college start years for offset calculation
    plan_529_by_college_year: dict[int, float] = {}
    for plan in config.plans_529:
        # College start year relative to retirement
        college_yr_from_retirement = plan.child_college_start_year - config.years_to_retirement
        if college_yr_from_retirement >= 0:
            val = accum.plan_529_values.get(plan.name, 0)
            plan_529_by_college_year[college_yr_from_retirement] = val

    # Result arrays
    asset_wealth = np.zeros((n, yrs + 1, num_assets))
    asset_wealth[:, 0, :] = init_values
    dist_rates = np.zeros((n, yrs))

    # Generate returns: shape (n, yrs, num_assets), clamped at -99%
    returns = rng.normal(
        means[np.newaxis, np.newaxis, :],
        stds[np.newaxis, np.newaxis, :],
        size=(n, yrs, num_assets),
    )
    returns = np.maximum(returns, -0.99)

    # Generate inflation: shape (n, yrs)
    inflation = rng.normal(config.inflation_mean, config.inflation_std, size=(n, yrs))
    cum_inflation = np.cumprod(1 + inflation, axis=1)

    # Mortgage state per simulation (same for all since accumulation is deterministic)
    mort_principal_init = accum.mortgage_principal

    # Run year-by-year drawdown
    # We need per-sim mortgage tracking, but since mortgage is deterministic,
    # we can precompute it
    mort_principals = np.full(yrs + 1, 0.0)
    mort_costs = np.zeros(yrs)
    if config.mortgage:
        mort_principals[0] = mort_principal_init
        for yr in range(yrs):
            new_p, cost = annual_mortgage_step(mort_principals[yr], config.mortgage)
            mort_principals[yr + 1] = new_p
            mort_costs[yr] = cost

    for year in range(yrs):
        current_assets = asset_wealth[:, year, :]  # (n, num_assets)
        current_total = current_assets.sum(axis=1)  # (n,)

        infl_factor = cum_inflation[:, year]  # (n,)

        # 1. Compute annual spending (non-housing categories, inflation-adjusted)
        spending_base = 0.0
        for cat, amount in config.spending.categories.items():
            if cat == "school":
                kids_in_school = sum(
                    1 for end_yr in config.spending.school_end_years
                    if (year + config.years_to_retirement) < end_yr
                )
                total_kids = len(config.spending.school_end_years)
                if total_kids > 0 and kids_in_school > 0:
                    spending_base += amount * (kids_in_school / total_kids)
            else:
                spending_base += amount

        # Spending is in today's dollars, inflate to nominal
        nominal_spending = spending_base * infl_factor  # (n,)

        # Add housing: mortgage (fixed) + non-mortgage (inflated)
        if config.mortgage:
            nominal_spending += mort_costs[year]
            nominal_spending += config.mortgage.non_mortgage_monthly * 12 * infl_factor

        # 2. Add future costs (college), inflation-adjusted, minus 529 offset
        if future_costs_by_year[year] > 0:
            nominal_costs = future_costs_by_year[year] * infl_factor
            # Check for 529 offset at this year
            if year in plan_529_by_college_year:
                # Spread 529 value over 4 years of college
                offset_annual = plan_529_by_college_year[year] / 4
                nominal_costs = np.maximum(nominal_costs - offset_annual, 0)
            nominal_spending += nominal_costs

        # 3. Compute weighted-average tax rate based on current portfolio composition
        weights = _safe_weights(current_assets)  # (n, num_assets)
        blended_tax_rate = (weights * tax_rates[np.newaxis, :]).sum(axis=1)  # (n,)

        # 4. Gross up for taxes
        gross_withdrawal = nominal_spending / np.maximum(1 - blended_tax_rate, 0.01)  # (n,)

        # 5. Compute distribution rate
        safe_total = np.where(current_total > 0, current_total, 1.0)
        dist_rates[:, year] = gross_withdrawal / safe_total

        # 6. Withdraw proportionally from assets
        new_assets = current_assets - gross_withdrawal[:, np.newaxis] * weights

        # 7. Apply returns to remaining portfolio
        new_assets = new_assets * (1 + returns[:, year, :])

        # 8. Black swan events
        for event in config.black_swans:
            hits = rng.random(n) < event.probability
            if event.cost > 0:
                cost = event.cost * infl_factor[hits]
                w = _safe_weights(new_assets[hits])
                new_assets[hits] -= cost[:, np.newaxis] * w
            if event.portfolio_impact != 0:
                new_assets[hits] *= (1 + event.portfolio_impact)

        asset_wealth[:, year + 1, :] = new_assets

    wealth = asset_wealth.sum(axis=2)
    return SimResult(
        wealth=wealth,
        asset_wealth=asset_wealth,
        dist_rates=dist_rates,
        accumulation=accum,
        config=config,
    )


def _safe_weights(assets: np.ndarray) -> np.ndarray:
    """Compute proportional weights from asset values.

    For rows where total is zero or negative, uses equal weights.
    """
    if assets.ndim == 1:
        assets = assets[np.newaxis, :]

    totals = assets.sum(axis=1, keepdims=True)
    safe = totals > 0
    num_assets = assets.shape[1]
    equal = np.full(num_assets, 1.0 / num_assets)
    weights = np.where(safe, assets / np.where(safe, totals, 1.0), equal[np.newaxis, :])
    return weights
