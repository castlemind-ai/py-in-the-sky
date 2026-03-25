"""Two-phase Monte Carlo simulation engine: accumulation + drawdown."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sim.config import SimConfig
from sim.events import (
    amortize_mortgage,
    annual_mortgage_step,
    compute_rmd,
    gross_up_for_tax,
    grow_529,
    months_since_purchase,
    resolve_active_spending_categories,
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
    """Deterministic accumulation with blended return rate and growing contributions."""
    asset_values = {a.name: a.value for a in config.assets}
    monthly_rate = (1 + config.accumulation_return) ** (1 / 12) - 1

    for month in range(months):
        year = month // 12
        for name, val in asset_values.items():
            asset_values[name] = val * (1 + monthly_rate)
            if name in contrib_map:
                for c in contrib_map[name]:
                    # Apply annual growth to base contribution amount
                    grown_amount = c.monthly_amount * (1 + c.annual_growth_rate) ** year
                    asset_values[name] += grown_amount

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

    for i, asset in enumerate(config.assets):
        if asset.return_source is None:
            if asset.return_distribution == "lognormal":
                parametric = _draw_lognormal(rng, means_monthly[i], stds_monthly[i], (n, months))
            else:
                parametric = rng.normal(means_monthly[i], stds_monthly[i], size=(n, months))
            bootstrap_returns[:, :, i] = parametric

    # Build contribution arrays: shape (n, months, num_assets)
    # Fixed contributions may grow annually; variable contributions are stochastic
    # Build per-month contribution array accounting for annual_growth_rate
    contrib_array = np.zeros((n, months, num_assets))

    for i, asset in enumerate(config.assets):
        if asset.name not in contrib_map:
            continue
        for c in contrib_map[asset.name]:
            for month in range(months):
                year = month // 12
                grown_base = c.monthly_amount * (1 + c.annual_growth_rate) ** year
                if c.is_variable and c.std_dev_pct > 0:
                    std = grown_base * c.std_dev_pct
                    var_amounts = rng.normal(grown_base, std, size=n)
                    var_amounts = np.maximum(var_amounts, 0)
                    contrib_array[:, month, i] += var_amounts
                else:
                    contrib_array[:, month, i] += grown_base

    # Vectorized monthly accumulation: shape (n, num_assets)
    values = np.tile(
        np.array([a.value for a in config.assets]), (n, 1),
    )  # (n, num_assets)

    # Identify accumulation-phase black swans
    accum_swans = [e for e in config.black_swans if e.phase == "accumulation"]

    # Build asset name -> index mapping for target_asset lookup
    asset_name_to_idx = {a.name: i for i, a in enumerate(config.assets)}

    for month in range(months):
        values = values * (1 + bootstrap_returns[:, month, :]) + contrib_array[:, month, :]

        # Check for accumulation black swans at year boundaries
        if accum_swans and (month + 1) % 12 == 0:
            for event in accum_swans:
                triggers = rng.random(n) < event.probability
                if not triggers.any():
                    continue

                if event.target_asset and event.target_asset in asset_name_to_idx:
                    # Apply to specific asset only
                    idx = asset_name_to_idx[event.target_asset]
                    values[triggers, idx] *= (1 + event.portfolio_impact)

                    # Reduce future contributions for triggered simulations
                    if event.contribution_impact is not None:
                        remaining = slice(month + 1, months)
                        contrib_array[triggers, remaining, idx] *= event.contribution_impact
                else:
                    # Apply to entire portfolio
                    values[triggers, :] *= (1 + event.portfolio_impact)

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

    # Base tax rates per asset (used for proportional blending)
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

    # Track Roth contribution basis per asset per simulation: shape (n, num_roth_assets)
    # Maps asset_idx -> running basis array of shape (n,)
    roth_basis: dict[int, np.ndarray] = {}
    for i, asset in enumerate(config.assets):
        if asset.tax_type == "roth" and asset.roth_contribution_basis is not None:
            roth_basis[i] = np.full(n, asset.roth_contribution_basis, dtype=float)

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

    # Build asset index maps for Roth conversions, RMDs
    asset_name_to_idx = {a.name: i for i, a in enumerate(config.assets)}

    # Find the first taxable asset index for RMD reinvestment (or None)
    taxable_idx = next(
        (i for i, a in enumerate(config.assets) if a.tax_type == "taxable"), None
    )

    for year in range(yrs):
        current_assets = asset_wealth[:, year, :]  # (n, num_assets)
        current_total = current_assets.sum(axis=1)  # (n,)

        infl_factor = cum_inflation[:, year]  # (n,)
        current_age = config.retirement_age + year

        # --- Roth Conversion Ladder ---
        for rc in config.roth_conversions:
            if rc.start_age <= current_age < rc.end_age:
                src_idx = asset_name_to_idx.get(rc.source_account)
                tgt_idx = asset_name_to_idx.get(rc.target_account)
                if src_idx is not None and tgt_idx is not None:
                    # Amount to convert, capped at available balance
                    convert_amt = np.minimum(
                        np.full(n, rc.annual_amount), current_assets[:, src_idx]
                    )
                    # Tax the conversion as ordinary income
                    conv_tax = convert_amt * config.tax.effective_income_rate(rc.annual_amount)
                    after_tax_converted = convert_amt - conv_tax
                    current_assets[:, src_idx] -= convert_amt
                    current_assets[:, tgt_idx] += after_tax_converted
                    # Update Roth basis: converted amounts are penalty-free
                    if tgt_idx in roth_basis:
                        roth_basis[tgt_idx] += after_tax_converted
                    else:
                        roth_basis[tgt_idx] = after_tax_converted.copy()

        # --- Spending: apply age transitions ---
        active_categories = resolve_active_spending_categories(config.spending, current_age)

        # 1. Compute annual spending base (today's dollars)
        spending_base = 0.0
        for cat, amount in active_categories.items():
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

        # Inflate to nominal
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

        # 3. Social Security income offset (reduces portfolio withdrawal)
        ss_annual = _compute_ss_income(config, current_age, infl_factor)
        if ss_annual is not None:
            nominal_spending = np.maximum(nominal_spending - ss_annual, 0.0)

        # 4. Compute effective tax rates per asset (incorporates brackets, penalties)
        effective_tax_rates = _compute_effective_tax_rates(
            config, current_age, current_assets, n, num_assets,
        )

        # 5. Compute withdrawal weights and gross withdrawal
        if config.drawdown_strategy == "tax_optimized":
            weights, gross_withdrawal = _tax_optimized_withdrawal(
                current_assets, nominal_spending, effective_tax_rates, n, num_assets,
                tax_types=[a.tax_type for a in config.assets],
            )
        else:
            # Proportional: weighted-average tax rate
            weights = _safe_weights(current_assets)  # (n, num_assets)
            blended_tax_rate = (weights * effective_tax_rates).sum(axis=1)  # (n,)
            gross_withdrawal = nominal_spending / np.maximum(1 - blended_tax_rate, 0.01)

        # 6. Compute distribution rate
        safe_total = np.where(current_total > 0, current_total, 1.0)
        dist_rates[:, year] = gross_withdrawal / safe_total

        # 7. Withdraw from assets (update Roth basis proportionally)
        withdrawal_per_asset = gross_withdrawal[:, np.newaxis] * weights  # (n, num_assets)
        new_assets = current_assets - withdrawal_per_asset

        # Reduce Roth basis by amount withdrawn from each Roth asset
        for i in list(roth_basis.keys()):
            drawn = withdrawal_per_asset[:, i]
            roth_basis[i] = np.maximum(roth_basis[i] - drawn, 0.0)

        # 8. RMD enforcement: after spending, force-withdraw any remaining RMD shortfall
        if current_age >= 73:
            for i, asset in enumerate(config.assets):
                if asset.tax_type != "traditional":
                    continue
                divisor = _irs_ult_divisor(current_age)
                per_sim_rmd = asset_wealth[:, year, i] / divisor
                already_withdrawn = withdrawal_per_asset[:, i]
                shortfall = np.maximum(per_sim_rmd - already_withdrawn, 0.0)
                shortfall = np.minimum(shortfall, new_assets[:, i])
                new_assets[:, i] -= shortfall
                # Reinvest excess RMD into taxable account (or just count as spent)
                if taxable_idx is not None and taxable_idx != i:
                    rmd_tax = shortfall * config.tax.rate_for("traditional")
                    new_assets[:, taxable_idx] += shortfall - rmd_tax

        # 9. Apply returns to remaining portfolio
        new_assets = new_assets * (1 + returns[:, year, :])

        # 10. Black swan events (drawdown phase only)
        for event in config.black_swans:
            if event.phase != "drawdown":
                continue
            hits = rng.random(n) < event.probability
            if not hits.any():
                continue
            if event.cost > 0:
                cost = event.cost * infl_factor[hits]
                w = _safe_weights(new_assets[hits])
                new_assets[hits] -= cost[:, np.newaxis] * w
            if event.portfolio_impact != 0:
                new_assets[hits] *= (1 + event.portfolio_impact)

        # Floor individual asset values at zero
        asset_wealth[:, year + 1, :] = np.maximum(new_assets, 0.0)

    wealth = asset_wealth.sum(axis=2)
    return SimResult(
        wealth=wealth,
        asset_wealth=asset_wealth,
        dist_rates=dist_rates,
        accumulation=accum,
        config=config,
    )


def _irs_ult_divisor(age: int) -> float:
    """Return the IRS Uniform Lifetime Table divisor for a given age."""
    from sim.events import _IRS_ULT
    return _IRS_ULT.get(age, 2.0)


def _compute_ss_income(
    config: SimConfig,
    current_age: int,
    infl_factor: np.ndarray,
) -> np.ndarray | None:
    """Compute total Social Security income this year (primary + spouse), or None."""
    total = None

    def _add_ss(monthly_benefit: float, claim_age: int, reduction: float = 0.0) -> np.ndarray:
        annual = monthly_benefit * 12 * (1 - reduction)
        if config.social_security and config.social_security.inflation_adjusted:
            return annual * infl_factor
        return np.full(len(infl_factor), annual)

    if config.social_security and current_age >= config.social_security.claim_age:
        ss = _add_ss(
            config.social_security.monthly_benefit,
            config.social_security.claim_age,
            config.social_security.early_claim_reduction,
        )
        total = ss if total is None else total + ss

    if config.spouse and current_age >= config.spouse.claim_age:
        spouse_ss = config.spouse.social_security_monthly * 12 * infl_factor
        total = spouse_ss if total is None else total + spouse_ss

    return total


def _compute_effective_tax_rates(
    config: SimConfig,
    current_age: int,
    current_assets: np.ndarray,
    n: int,
    num_assets: int,
) -> np.ndarray:
    """Compute per-asset effective tax rates as a (num_assets,) array.

    Incorporates:
    - State capital gains rate for taxable assets
    - Early withdrawal penalty for traditional assets pre-59½
    - Progressive brackets (flat rate per asset, brackets applied at gross-up stage)
    """
    rates = np.empty(num_assets)
    for i, asset in enumerate(config.assets):
        base = config.tax.rate_for(asset.tax_type)

        # Early withdrawal penalty for traditional accounts pre-59½
        if (asset.tax_type == "traditional"
                and current_age < 59.5
                and not asset.rule_of_55_eligible):
            base += config.tax.early_withdrawal_penalty_rate

        rates[i] = base
    return rates


def _tax_optimized_withdrawal(
    current_assets: np.ndarray,
    nominal_spending: np.ndarray,
    tax_rates: np.ndarray,
    n: int,
    num_assets: int,
    tax_types: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute withdrawal weights using tax-optimized sequencing.

    Drain order: cash → taxable → traditional → after_tax_401k → roth.
    This preserves Roth accounts (tax-free growth) longest.
    Returns (weights, gross_withdrawal).
    """
    # Drain priority by account type (lower number = drain first)
    _PRIORITY = {"cash": 0, "taxable": 1, "traditional": 2, "after_tax_401k": 3, "roth": 4}
    if tax_types:
        priorities = [_PRIORITY.get(t, 2) for t in tax_types]
    else:
        # Fall back to sorting by tax rate descending (highest-tax drained first, Roth last)
        priorities = list(tax_rates)
    order = np.argsort(priorities)
    weights = np.zeros((n, num_assets))
    gross_withdrawal = np.zeros(n)

    remaining_need = nominal_spending.copy()

    for idx in order:
        if np.all(remaining_need <= 0):
            break
        rate = tax_rates[idx]
        available = current_assets[:, idx]
        # How much gross we need to cover remaining net need from this asset
        gross_needed = remaining_need / np.maximum(1 - rate, 0.01)
        # Actual gross taken from this asset (can't exceed balance)
        gross_taken = np.minimum(gross_needed, available)
        gross_taken = np.maximum(gross_taken, 0)
        weights[:, idx] = np.where(available > 0, gross_taken / np.maximum(available, 1e-12), 0)
        gross_withdrawal += gross_taken
        # Net amount covered
        net_covered = gross_taken * (1 - rate)
        remaining_need = np.maximum(remaining_need - net_covered, 0)

    # Normalize weights so they sum to at most 1
    total_weights = weights.sum(axis=1, keepdims=True)
    safe = total_weights > 0
    weights = np.where(safe, weights / np.where(safe, total_weights, 1.0), weights)

    return weights, gross_withdrawal


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
                    parametric for the rest.
    mc mode: parametric for all assets.

    drawdown_sources overrides asset return_source (used for retirement diversification).
    Per-asset return_distribution controls whether parametric draws are normal or lognormal.
    """
    if config.method == "bootstrap" and sampler is not None:
        asset_sources = drawdown_sources or [a.return_source for a in config.assets]
        has_historical = any(s is not None for s in asset_sources)

        if has_historical:
            blend_weights = [asset.bootstrap_blend for asset in config.assets]

            returns = sampler.bootstrap_correlated(
                rng, asset_sources, n, yrs, months_per_period=12,
                blend_weights=blend_weights, blend_means=means, blend_stds=stds,
            )
            # Fill parametric assets (NaN slots)
            for i, (src, asset) in enumerate(zip(asset_sources, config.assets)):
                if src is None:
                    if asset.return_distribution == "lognormal":
                        returns[:, :, i] = _draw_lognormal(rng, means[i], stds[i], (n, yrs))
                    else:
                        returns[:, :, i] = rng.normal(means[i], stds[i], size=(n, yrs))
            return returns

    # Default: all parametric
    result = np.empty((n, yrs, num_assets))
    for i, asset in enumerate(config.assets):
        if asset.return_distribution == "lognormal":
            result[:, :, i] = _draw_lognormal(rng, means[i], stds[i], (n, yrs))
        else:
            result[:, :, i] = rng.normal(means[i], stds[i], size=(n, yrs))
    return result


def _draw_lognormal(
    rng: np.random.Generator,
    mean_return: float,
    std_dev: float,
    size: tuple[int, ...],
) -> np.ndarray:
    """Draw lognormal returns using moment-matching from arithmetic mean and std dev.

    Converts arithmetic mean/std to log-space parameters so the resulting
    distribution has the specified first two moments. Draws are always > -1
    (no below -100% outcomes), naturally handling the -99% clamp concern.
    """
    # Moment matching: if R ~ Lognormal(mu, sigma), then
    #   E[1+R] = exp(mu + sigma^2/2) = 1 + mean_return
    #   Var[1+R] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2) = std_dev^2
    m1 = 1.0 + mean_return
    v = std_dev ** 2
    sigma2 = np.log(1.0 + v / (m1 ** 2))
    mu = np.log(m1) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return rng.lognormal(mu, sigma, size=size) - 1.0


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
