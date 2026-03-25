"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TaxBracket:
    """A single marginal income tax bracket."""

    threshold: float  # income level at which this rate begins
    rate: float       # marginal rate above threshold


@dataclass
class Asset:
    """A single holding in the portfolio."""

    name: str
    value: float
    mean_return: float
    std_dev: float
    tax_type: str = "taxable"  # "roth" | "traditional" | "taxable" | "cash" | "after_tax_401k"
    return_source: str | None = None  # key into SimConfig.return_sources
    cost_basis: float | None = None  # for capital gains on diversification
    retire_to_source: str | None = None  # switch return_source at retirement
    bootstrap_blend: float = 1.0  # 1.0 = pure bootstrap, 0.5 = 50/50 blend with parametric
    rule_of_55_eligible: bool = False  # exempt from pre-59½ early withdrawal penalty
    roth_contribution_basis: float | None = None  # tracked Roth contribution basis (None = no tracking)
    return_distribution: str = "normal"  # "normal" or "lognormal"
    is_401k_plan: bool = False  # counts toward annual 401k total contribution limit


@dataclass
class TaxConfig:
    """Tax rates by account type."""

    income_rate: float = 0.35         # flat traditional 401k/TSP rate (used if no tax_brackets)
    capital_gains_rate: float = 0.20  # federal long-term capital gains rate
    roth_rate: float = 0.0            # Roth = tax-free
    state_capital_gains_rate: float = 0.0  # state cap gains rate (added to federal)
    early_withdrawal_penalty_rate: float = 0.10  # 10% IRS penalty for pre-59½ traditional withdrawals
    roth_earnings_penalty_rate: float = 0.10     # penalty on Roth earnings withdrawn pre-59½
    tax_brackets: list[TaxBracket] = field(default_factory=list)  # progressive brackets (optional)

    def rate_for(self, tax_type: str) -> float:
        """Return the base tax rate for a given account type.

        For progressive bracket computation on traditional withdrawals,
        use effective_income_rate() instead.
        """
        if tax_type == "roth":
            return self.roth_rate
        elif tax_type == "traditional":
            return self.income_rate
        elif tax_type == "taxable":
            return self.capital_gains_rate + self.state_capital_gains_rate
        elif tax_type == "cash":
            return 0.0
        elif tax_type == "after_tax_401k":
            return 0.0  # after-tax basis; conversion already taxed at contribution
        return self.income_rate

    def effective_income_rate(self, annual_withdrawal: float) -> float:
        """Compute effective income tax rate for a given withdrawal amount.

        If tax_brackets is configured, applies progressive marginal rates.
        Otherwise falls back to flat income_rate.
        """
        if not self.tax_brackets or annual_withdrawal <= 0:
            return self.income_rate

        # Walk brackets in threshold order
        brackets = sorted(self.tax_brackets, key=lambda b: b.threshold)
        total_tax = 0.0
        remaining = annual_withdrawal
        prev_threshold = 0.0

        for bracket in brackets:
            if remaining <= 0:
                break
            band_size = bracket.threshold - prev_threshold
            taxable_in_band = min(remaining, band_size)
            total_tax += taxable_in_band * bracket.rate
            remaining -= taxable_in_band
            prev_threshold = bracket.threshold

        # Top bracket: apply to whatever is left
        if remaining > 0 and brackets:
            total_tax += remaining * brackets[-1].rate

        return total_tax / annual_withdrawal


@dataclass
class MortgageConfig:
    """Mortgage details for home (tracked separately from investable assets)."""

    purchase_price: float
    down_payment_pct: float
    interest_rate: float
    monthly_payment: float
    purchase_date: str  # "YYYY-MM-DD"
    non_mortgage_monthly: float  # property tax, insurance, maintenance


@dataclass
class Contribution:
    """A monthly savings contribution during the accumulation phase."""

    name: str
    monthly_amount: float
    target_account: str  # must match an asset name
    is_variable: bool = False
    std_dev_pct: float = 0.0  # e.g., 0.20 for 20% variance on RSUs
    annual_growth_rate: float = 0.0  # annual escalation rate for the base monthly amount


@dataclass
class Plan529:
    """A 529 college savings plan."""

    name: str
    current_value: float
    monthly_contribution: float
    mean_return: float
    child_college_start_year: int  # years from now when college starts
    start_month: int = 0  # months from now before contributions begin


@dataclass
class AgeTransition:
    """A change to spending categories at a specific age."""

    at_age: int               # age at which this transition fires (inclusive)
    remove: list[str] = field(default_factory=list)   # category keys to remove
    add: dict[str, float] = field(default_factory=dict)  # new category key -> annual amount


@dataclass
class SpendingSchedule:
    """Annual spending by category with school phase-out and age transitions."""

    categories: dict[str, float]  # name -> annual amount in today's dollars
    school_end_years: list[int] = field(default_factory=list)  # when each kid finishes school
    age_transitions: list[AgeTransition] = field(default_factory=list)


@dataclass
class FutureCost:
    """A discrete expected cost (e.g. college tuition)."""

    name: str
    amount: float  # annual amount in today's dollars
    year: int  # year offset from now
    duration: int = 1
    inflation_adjusted: bool = True


@dataclass
class BlackSwanEvent:
    """A rare, high-impact event."""

    name: str
    probability: float  # annual probability
    cost: float = 0.0  # direct cost in today's dollars
    portfolio_impact: float = 0.0  # fractional portfolio change, e.g. -0.30
    phase: str = "drawdown"  # "drawdown" or "accumulation"
    target_asset: str | None = None  # if set, only affects this asset (accumulation)
    contribution_impact: float | None = None  # multiplier on future contributions when triggered


@dataclass
class SocialSecurityConfig:
    """Social Security benefit configuration."""

    monthly_benefit: float        # in today's dollars at full retirement age
    claim_age: int = 67           # age at which benefits begin
    inflation_adjusted: bool = True
    early_claim_reduction: float = 0.0  # fractional reduction for early claiming (e.g. 0.30 at 62)


@dataclass
class SpouseConfig:
    """Minimal spouse config for dual Social Security streams."""

    social_security_monthly: float = 0.0  # spouse's monthly SS benefit in today's dollars
    claim_age: int = 67


@dataclass
class RothConversion:
    """An annual Roth conversion ladder entry."""

    start_age: int       # age at which conversions begin
    end_age: int         # age at which conversions end (exclusive)
    annual_amount: float # amount converted per year from source to target
    source_account: str  # must match an asset name with tax_type="traditional"
    target_account: str  # must match an asset name with tax_type="roth"


@dataclass
class SimConfig:
    """Top-level simulation configuration."""

    assets: list[Asset]
    spending: SpendingSchedule
    tax: TaxConfig = field(default_factory=TaxConfig)
    mortgage: MortgageConfig | None = None
    contributions: list[Contribution] = field(default_factory=list)
    plans_529: list[Plan529] = field(default_factory=list)
    future_costs: list[FutureCost] = field(default_factory=list)
    black_swans: list[BlackSwanEvent] = field(default_factory=list)
    social_security: SocialSecurityConfig | None = None
    spouse: SpouseConfig | None = None
    roth_conversions: list[RothConversion] = field(default_factory=list)
    current_age: int = 39
    retirement_age: int = 50
    life_expectancy: int = 90
    inflation_mean: float = 0.03
    inflation_std: float = 0.01
    num_simulations: int = 10_000
    accumulation_return: float = 0.065  # blended annual return (mc mode only)
    method: str = "mc"  # "mc" or "bootstrap"
    return_sources: dict[str, str] = field(default_factory=dict)  # name -> CSV path
    seed: int | None = None
    # Portfolio size considered a success threshold. Shown in fan chart and histogram.
    # Set to 0 to disable. Tip: use 25–33x annual spending for a withdrawal-rate-based target.
    target_wealth: float = 0.0
    drawdown_strategy: str = "proportional"  # "proportional" or "tax_optimized"
    # Bootstrap inflation from historical CPI data (bootstrap mode only).
    # Must reference a key in return_sources. Shares random start indices with asset returns,
    # preserving inflation-return correlation (e.g. 1970s stagflation).
    inflation_source: str | None = None
    inflation_blend: float = 1.0  # 1.0 = pure bootstrap, 0.0 = pure parametric Normal(mean,std)

    @property
    def years_to_retirement(self) -> int:
        return self.retirement_age - self.current_age

    @property
    def years_in_retirement(self) -> int:
        return self.life_expectancy - self.retirement_age


class ConfigValidationError(ValueError):
    """Raised when config values are invalid."""


def validate_config(config: SimConfig) -> None:
    """Validate a SimConfig, raising ConfigValidationError on problems."""
    errors: list[str] = []

    if config.retirement_age <= config.current_age:
        errors.append(
            f"retirement_age ({config.retirement_age}) must be greater than "
            f"current_age ({config.current_age})"
        )
    if config.life_expectancy <= config.retirement_age:
        errors.append(
            f"life_expectancy ({config.life_expectancy}) must be greater than "
            f"retirement_age ({config.retirement_age})"
        )

    # Validate return source CSV paths exist (bootstrap mode)
    for name, csv_path in config.return_sources.items():
        if not Path(csv_path).exists():
            errors.append(f"return_sources['{name}'] file not found: {csv_path}")

    # Validate contribution target_account references existing assets
    asset_names = {a.name for a in config.assets}
    for c in config.contributions:
        if c.target_account not in asset_names:
            errors.append(
                f"contribution '{c.name}' targets account '{c.target_account}' "
                f"which does not match any asset name"
            )

    # Validate asset return_source references exist in return_sources
    if config.method == "bootstrap":
        for asset in config.assets:
            if asset.return_source and asset.return_source not in config.return_sources:
                errors.append(
                    f"asset '{asset.name}' has return_source '{asset.return_source}' "
                    f"not found in return_sources"
                )
            if asset.retire_to_source and asset.retire_to_source not in config.return_sources:
                errors.append(
                    f"asset '{asset.name}' has retire_to_source '{asset.retire_to_source}' "
                    f"not found in return_sources"
                )

    # Validate combined capital gains rate is < 1
    combined_cap_gains = config.tax.capital_gains_rate + config.tax.state_capital_gains_rate
    if combined_cap_gains >= 1.0:
        errors.append(
            f"Combined capital_gains_rate + state_capital_gains_rate ({combined_cap_gains:.2%}) "
            f"must be less than 100%"
        )

    # Validate return_distribution values
    valid_distributions = {"normal", "lognormal"}
    for asset in config.assets:
        if asset.return_distribution not in valid_distributions:
            errors.append(
                f"asset '{asset.name}' has invalid return_distribution '{asset.return_distribution}'. "
                f"Must be one of: {sorted(valid_distributions)}"
            )

    # Validate inflation_source
    if config.inflation_source is not None and config.method == "bootstrap":
        if config.inflation_source not in config.return_sources:
            errors.append(
                f"inflation_source '{config.inflation_source}' not found in return_sources"
            )

    # Validate drawdown_strategy
    valid_strategies = {"proportional", "tax_optimized"}
    if config.drawdown_strategy not in valid_strategies:
        errors.append(
            f"drawdown_strategy '{config.drawdown_strategy}' is not valid. "
            f"Must be one of: {sorted(valid_strategies)}"
        )

    # Validate Roth conversion accounts
    for rc in config.roth_conversions:
        if rc.source_account not in asset_names:
            errors.append(
                f"roth_conversion source_account '{rc.source_account}' does not match any asset"
            )
        if rc.target_account not in asset_names:
            errors.append(
                f"roth_conversion target_account '{rc.target_account}' does not match any asset"
            )
        if rc.start_age >= rc.end_age:
            errors.append(
                f"roth_conversion start_age ({rc.start_age}) must be less than end_age ({rc.end_age})"
            )

    # Validate annual 401k contribution limit ($70,000 in 2026)
    _401k_limit = 70_000
    is_401k = {a.name for a in config.assets if a.is_401k_plan}
    if is_401k:
        total_annual_401k = sum(
            c.monthly_amount * 12
            for c in config.contributions
            if c.target_account in is_401k
        )
        if total_annual_401k > _401k_limit:
            errors.append(
                f"Total annual contributions to 401k-plan assets (${total_annual_401k:,.0f}) "
                f"exceeds the ${_401k_limit:,} annual 401k limit"
            )

    if errors:
        raise ConfigValidationError(
            "Invalid configuration:\n  - " + "\n  - ".join(errors)
        )


def load_config(path: str | Path) -> SimConfig:
    """Load a SimConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    assets = [Asset(**a) for a in raw["assets"]]

    # Tax config
    tax_raw = raw.get("tax", {})
    if tax_raw:
        brackets_raw = tax_raw.pop("tax_brackets", [])
        tax = TaxConfig(**tax_raw)
        tax.tax_brackets = [TaxBracket(**b) for b in (brackets_raw or [])]
    else:
        tax = TaxConfig()

    # Mortgage
    mortgage = None
    if "mortgage" in raw:
        mortgage = MortgageConfig(**raw["mortgage"])

    # Contributions
    contributions = [Contribution(**c) for c in raw.get("contributions", [])]

    # 529 plans
    plans_529 = [Plan529(**p) for p in raw.get("plans_529", [])]

    # Spending
    spending_raw = raw.get("spending", {})
    transitions_raw = spending_raw.get("age_transitions", [])
    age_transitions = []
    for t in transitions_raw:
        age_transitions.append(AgeTransition(
            at_age=t["at_age"],
            remove=t.get("remove", []),
            add=t.get("add", {}),
        ))
    spending = SpendingSchedule(
        categories=spending_raw.get("categories", {}),
        school_end_years=spending_raw.get("school_end_years", []),
        age_transitions=age_transitions,
    )

    # Future costs
    future_costs = [FutureCost(**fc) for fc in raw.get("future_costs", [])]

    # Black swans
    black_swans = [BlackSwanEvent(**bs) for bs in raw.get("black_swans", [])]

    # Social Security
    social_security = None
    if "social_security" in raw:
        social_security = SocialSecurityConfig(**raw["social_security"])

    # Spouse
    spouse = None
    if "spouse" in raw:
        spouse = SpouseConfig(**raw["spouse"])

    # Roth conversions
    roth_conversions = [RothConversion(**rc) for rc in raw.get("roth_conversions", [])]

    config = SimConfig(
        assets=assets,
        spending=spending,
        tax=tax,
        mortgage=mortgage,
        contributions=contributions,
        plans_529=plans_529,
        future_costs=future_costs,
        black_swans=black_swans,
        social_security=social_security,
        spouse=spouse,
        roth_conversions=roth_conversions,
        current_age=raw.get("current_age", 39),
        retirement_age=raw.get("retirement_age", 50),
        life_expectancy=raw.get("life_expectancy", 90),
        inflation_mean=raw.get("inflation_mean", 0.03),
        inflation_std=raw.get("inflation_std", 0.01),
        num_simulations=raw.get("num_simulations", 10_000),
        accumulation_return=raw.get("accumulation_return", 0.065),
        method=raw.get("method", "mc"),
        return_sources=raw.get("return_sources", {}),
        seed=raw.get("seed"),
        target_wealth=raw.get("target_wealth", 0.0),
        drawdown_strategy=raw.get("drawdown_strategy", "proportional"),
        inflation_source=raw.get("inflation_source"),
        inflation_blend=raw.get("inflation_blend", 1.0),
    )

    validate_config(config)
    return config
