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
    accumulation_return: float = 0.065  # blended annual return for accumulation phase
    method: str = "mc"  # "mc" or "bootstrap"
    hist_returns_path: str | None = None
    seed: int | None = None
    target_wealth: float = 0.0

    @property
    def years_to_retirement(self) -> int:
        return self.retirement_age - self.current_age

    @property
    def years_in_retirement(self) -> int:
        return self.life_expectancy - self.retirement_age


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

    return SimConfig(
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
        hist_returns_path=raw.get("hist_returns_path"),
        seed=raw.get("seed"),
        target_wealth=raw.get("target_wealth", 0.0),
    )
