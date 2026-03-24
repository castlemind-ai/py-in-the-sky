"""Two-phase Monte Carlo simulation engine: accumulation + drawdown."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
from sim.returns import ReturnSampler


@dataclass
class AccumulationResult:
    """Results of the accumulation phase.

    In mc mode, asset_values are scalars (deterministic).
    In bootstrap mode, asset_values are arrays of shape (n_sims,).
    """

    asset_values: dict[str, float | np.ndarray]  # asset name -> value(s) at retirement
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


def _build_sampler(config: SimConfig) -> ReturnSampler | None:
    """Build a ReturnSampler from config if bootstrap mode with sources."""
    if config.method != "bootstrap" or not config.return_sources:
        return None
    return ReturnSampler.from_config(config.return_sources)


def run_accumulation(
    config: SimConfig,
    rng: np.random.Generator | None = None,
    sampler: ReturnSampler | None = None,
) -> AccumulationResult:
    """Run the accumulation phase.

    mc mode: deterministic, single blended return for all assets.
    bootstrap mode: stochastic, per-simulation monthly returns from historical data.
    """
    yrs = config.years_to_retirement
    months = yrs * 12
    num_assets = len(config.assets)
    n = config.num_simulations

    # Build contribution map: target_account -> list of contributions
    contrib_map: dict[str, list] = {}
    for c in config.contributions:
        contrib_map.setdefault(c.target_account, []).append(c)

    # Compute mortgage state (deterministic in both modes)
    mort_principal = 0.0
    mort_months_elapsed = 0
    if config.mortgage:
        mort_months_elapsed = months_since_purchase(config.mortgage)
        mort_principal = amortize_mortgage(config.mortgage, mort_months_elapsed)

        mort_monthly_rate = config.mortgage.interest_rate / 12
        for _ in range(months):
            if mort_principal <= 0:
                break
            month_int = mort_principal * mort_monthly_rate
            principal_paid = config.mortgage.monthly_payment - month_int
            mort_principal = max(mort_principal - principal_paid, 0)
        mort_months_elapsed += months

    # Grow 529 plans (deterministic in both modes)
    plan_529_values = {}
    for plan in config.plans_529:
        monthly_rate = (1 + plan.mean_return) ** (1 / 12) - 1
        plan_529_values[plan.name] = grow_529(plan, months, monthly_rate)

    use_bootstrap = config.method == "bootstrap" and sampler is not None and rng is not None

    if use_bootstrap:
        asset_values = _run_accumulation_bootstrap(
            config, rng, sampler, months, n, num_assets, contrib_map,
        )
    else:
        asset_values = _run_accumulation_mc(config, months, contrib_map)

    return AccumulationResult(
        asset_values=asset_values,
        mortgage_principal=mort_principal,
        plan_529_values=plan_529_values,
        months_elapsed_mortgage=mort_months_elapsed,
    )


def _run_accumulation_mc(
    config: SimConfig, months: int, contrib_map: dict[str, list],
) -> dict[str, float]:
    """Deterministic accumulation with blended return rate."""
    asset_values = {a.name: a.value for a in config.assets}
    monthly_rate = (1 + config.accumulation_return) ** (1 / 12) - 1

    for month in range(months):
        for name, val in asset_values.items():
            asset_values[name] = val * (1 + monthly_rate)
            if name in contrib_map:
                for c in contrib_map[name]:
                    asset_values[name] += c.monthly_amount

    return asset_values


def _run_accumulation_bootstrap(
    config: SimConfig,
    rng: np.random.Generator,
    sampler: ReturnSampler,
    months: int,
    n: int,
    num_assets: int,
    contrib_map: dict[str, list],
) -> dict[str, np.ndarray]:
    """Stochastic accumulation with bootstrapped monthly returns.

    Assets with a return_source get historical bootstrap returns.
    Assets without get parametric monthly returns from mean_return/std_dev.
    """
    # Generate monthly returns: shape (n, months, num_assets)
    asset_sources = [a.return_source for a in config.assets]

    # Bootstrap returns for historical-source assets
    means_monthly = np.array([(1 + a.mean_return) ** (1 / 12) - 1 for a in config.assets])
    stds_monthly = np.array([a.std_dev / np.sqrt(12) for a in config.assets])
    blend_weights = [a.bootstrap_blend for a in config.assets]

    bootstrap_returns = sampler.bootstrap_correlated(
        rng, asset_sources, n, months, months_per_period=1,
        blend_weights=blend_weights, blend_means=means_monthly, blend_stds=stds_monthly,
    )

    for i, src in enumerate(asset_sources):
        if src is None:
            parametric = rng.normal(means_monthly[i], stds_monthly[i], size=(n, months))
            bootstrap_returns[:, :, i] = parametric

    # Build contribution arrays: shape (n, months, num_assets)
    # Fixed contributions are constant; variable contributions are stochastic
    fixed_contrib = np.zeros(num_assets)
    variable_contribs = []  # list of (asset_idx, mean, std)
    for i, asset in enumerate(config.assets):
        if asset.name in contrib_map:
            for c in contrib_map[asset.name]:
                if c.is_variable and c.std_dev_pct > 0:
                    variable_contribs.append((i, c.monthly_amount, c.monthly_amount * c.std_dev_pct))
                else:
                    fixed_contrib[i] += c.monthly_amount

    # Pre-generate variable contribution amounts: shape (n, months)
    contrib_array = np.tile(fixed_contrib, (n, months, 1))  # (n, months, num_assets)
    for asset_idx, mean, std in variable_contribs:
        var_amounts = rng.normal(mean, std, size=(n, months))
        var_amounts = np.maximum(var_amounts, 0)  # RSU vesting can't be negative
        contrib_array[:, :, asset_idx] += var_amounts

    # Vectorized monthly accumulation: shape (n, num_assets)
    values = np.tile(
        np.array([a.value for a in config.assets]), (n, 1),
    )  # (n, num_assets)

    for month in range(months):
        values = values * (1 + bootstrap_returns[:, month, :]) + contrib_array[:, month, :]

    return {a.name: values[:, i] for i, a in enumerate(config.assets)}


def run(config: SimConfig) -> SimResult:
    """Run the full two-phase simulation."""
    rng = np.random.default_rng(config.seed)
    sampler = _build_sampler(config)

    # Phase 1: Accumulation
    accum = run_accumulation(config, rng, sampler)

    # Phase 2: Drawdown (Monte Carlo)
    n = config.num_simulations
    yrs = config.years_in_retirement
    num_assets = len(config.assets)

    # Initial values from accumulation
    # In mc mode: scalar per asset -> broadcast to (n, num_assets)
    # In bootstrap mode: array(n) per asset -> stack to (n, num_assets)
    first_val = next(iter(accum.asset_values.values()))
    if isinstance(first_val, np.ndarray):
        init_values = np.column_stack(
            [accum.asset_values[a.name] for a in config.assets]
        )  # (n, num_assets)
    else:
        init_values = np.tile(
            np.array([accum.asset_values[a.name] for a in config.assets]),
            (n, 1),
        )  # (n, num_assets)

    # Apply retirement diversification: sell concentrated positions, pay cap gains tax
    # Build the drawdown return sources (may differ from accumulation sources)
    drawdown_sources = [a.return_source for a in config.assets]
    for i, asset in enumerate(config.assets):
        if asset.retire_to_source is not None:
            # Compute capital gains tax on the sale
            basis = asset.cost_basis if asset.cost_basis is not None else 0.0
            gain = np.maximum(init_values[:, i] - basis, 0.0)
            tax = gain * config.tax.rate_for(asset.tax_type)
            init_values[:, i] -= tax
            # Switch return source for drawdown
            drawdown_sources[i] = asset.retire_to_source

    # Asset return parameters (for parametric fallback)
    means = np.array([a.mean_return for a in config.assets])
    stds = np.array([a.std_dev for a in config.assets])

    # Tax rates per asset
    tax_rates = np.array([config.tax.rate_for(a.tax_type) for a in config.assets])

    # Pre-compute future costs (college) per year in today's dollars
    future_costs_by_year = resolve_future_costs(config.future_costs, yrs)

    # Pre-compute 529 values at retirement
    total_529 = sum(accum.plan_529_values.values())

    # Map 529 plans to annual offsets across all 4 college years
    plan_529_offsets_by_year: dict[int, float] = {}
    for plan in config.plans_529:
        college_yr_from_retirement = plan.child_college_start_year - config.years_to_retirement
        val = accum.plan_529_values.get(plan.name, 0)
        annual_offset = val / 4
        for yr_offset in range(4):
            yr = college_yr_from_retirement + yr_offset
            if yr >= 0:
                plan_529_offsets_by_year[yr] = plan_529_offsets_by_year.get(yr, 0) + annual_offset

    # Result arrays
    asset_wealth = np.zeros((n, yrs + 1, num_assets))
    asset_wealth[:, 0, :] = init_values
    dist_rates = np.zeros((n, yrs))

    # Generate drawdown returns: shape (n, yrs, num_assets), clamped at -99%
    returns = _generate_drawdown_returns(
        config, rng, sampler, n, yrs, num_assets, means, stds, drawdown_sources,
    )
    returns = np.maximum(returns, -0.99)

    # Generate inflation: shape (n, yrs)
    inflation = rng.normal(config.inflation_mean, config.inflation_std, size=(n, yrs))
    cum_inflation = np.cumprod(1 + inflation, axis=1)

    # Precompute mortgage schedule (deterministic)
    mort_principals = np.full(yrs + 1, 0.0)
    mort_costs = np.zeros(yrs)
    if config.mortgage:
        mort_principals[0] = accum.mortgage_principal
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
            if year in plan_529_offsets_by_year:
                offset_annual = plan_529_offsets_by_year[year]
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

        # Floor individual asset values at zero: once ruined, balances stay at zero
        # rather than continuing to compound negatively.
        asset_wealth[:, year + 1, :] = np.maximum(new_assets, 0.0)

    wealth = asset_wealth.sum(axis=2)
    return SimResult(
        wealth=wealth,
        asset_wealth=asset_wealth,
        dist_rates=dist_rates,
        accumulation=accum,
        config=config,
    )


def _generate_drawdown_returns(
    config: SimConfig,
    rng: np.random.Generator,
    sampler: ReturnSampler | None,
    n: int,
    yrs: int,
    num_assets: int,
    means: np.ndarray,
    stds: np.ndarray,
    drawdown_sources: list[str | None] | None = None,
) -> np.ndarray:
    """Generate annual returns array of shape (n, yrs, num_assets).

    bootstrap mode: block-bootstrap 12-month blocks for historical assets,
                    parametric normal for the rest.
    mc mode: parametric normal for all assets.

    drawdown_sources overrides asset return_source (used for retirement diversification).
    """
    if config.method == "bootstrap" and sampler is not None:
        asset_sources = drawdown_sources or [a.return_source for a in config.assets]
        has_historical = any(s is not None for s in asset_sources)

        if has_historical:
            # Build blend weights — use drawdown source's blend weight
            # For diversified assets (retire_to_source), use the target's blend weight (1.0)
            blend_weights = []
            for i, asset in enumerate(config.assets):
                if asset.retire_to_source is not None:
                    blend_weights.append(1.0)  # diversified assets use pure bootstrap of target
                else:
                    blend_weights.append(asset.bootstrap_blend)

            # Bootstrap returns for historical-source assets
            returns = sampler.bootstrap_correlated(
                rng, asset_sources, n, yrs, months_per_period=12,
                blend_weights=blend_weights, blend_means=means, blend_stds=stds,
            )
            # Fill parametric assets (NaN slots)
            for i, src in enumerate(asset_sources):
                if src is None:
                    returns[:, :, i] = rng.normal(means[i], stds[i], size=(n, yrs))
            return returns

    # Default: all parametric normal
    return rng.normal(
        means[np.newaxis, np.newaxis, :],
        stds[np.newaxis, np.newaxis, :],
        size=(n, yrs, num_assets),
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
