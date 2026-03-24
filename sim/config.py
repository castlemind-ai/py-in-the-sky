"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Asset:
    """A single holding in the portfolio."""

    name: str
    value: float
    mean_return: float
    std_dev: float
    tax_type: str = "taxable"  # "roth" | "traditional" | "taxable" | "cash"
    return_source: str | None = None  # key into SimConfig.return_sources
    cost_basis: float | None = None  # for capital gains on diversification
    retire_to_source: str | None = None  # switch return_source at retirement
    bootstrap_blend: float = 1.0  # 1.0 = pure bootstrap, 0.5 = 50/50 blend with parametric


@dataclass
class TaxConfig:
    """Tax rates by account type."""

    income_rate: float = 0.35       # traditional 401k/TSP withdrawals
    capital_gains_rate: float = 0.20  # taxable brokerage
    roth_rate: float = 0.0          # Roth = tax-free

    def rate_for(self, tax_type: str) -> float:
        """Return the tax rate for a given account type."""
        if tax_type == "roth":
            return self.roth_rate
        elif tax_type == "traditional":
            return self.income_rate
        elif tax_type == "taxable":
            return self.capital_gains_rate
        elif tax_type == "cash":
            return 0.0
        return self.income_rate


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
class SpendingSchedule:
    """Annual spending by category with school phase-out."""

    categories: dict[str, float]  # name -> annual amount in today's dollars
    school_end_years: list[int] = field(default_factory=list)  # when each kid finishes school


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
    target_wealth: float = 0.0

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
    tax = TaxConfig(**tax_raw) if tax_raw else TaxConfig()

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
    spending = SpendingSchedule(
        categories=spending_raw.get("categories", {}),
        school_end_years=spending_raw.get("school_end_years", []),
    )

    # Future costs
    future_costs = [FutureCost(**fc) for fc in raw.get("future_costs", [])]

    # Black swans
    black_swans = [BlackSwanEvent(**bs) for bs in raw.get("black_swans", [])]

    config = SimConfig(
        assets=assets,
        spending=spending,
        tax=tax,
        mortgage=mortgage,
        contributions=contributions,
        plans_529=plans_529,
        future_costs=future_costs,
        black_swans=black_swans,
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
    )

    validate_config(config)
    return config
