"""Tests for configuration loading."""

import tempfile

import pytest

from sim.config import (
    Asset,
    ConfigValidationError,
    SimConfig,
    SpendingSchedule,
    TaxConfig,
    load_config,
    validate_config,
)


def test_load_example_config():
    config = load_config("example_config.yaml")
    assert isinstance(config, SimConfig)
    assert len(config.assets) == 5
    assert config.retirement_age == 50
    assert config.num_simulations == 10000
    assert config.mortgage is not None
    assert config.mortgage.purchase_price == 800000


def test_load_minimal_config():
    yaml_content = """
assets:
  - name: "Stocks"
    value: 100000
    mean_return: 0.07
    std_dev: 0.15
spending:
  categories:
    living: 40000
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        config = load_config(f.name)

    assert len(config.assets) == 1
    assert config.assets[0].name == "Stocks"
    assert config.spending.categories["living"] == 40000
    assert config.future_costs == []
    assert config.black_swans == []


def test_tax_config_rates():
    tax = TaxConfig(income_rate=0.35, capital_gains_rate=0.20, roth_rate=0.0)
    assert tax.rate_for("roth") == 0.0
    assert tax.rate_for("traditional") == 0.35
    assert tax.rate_for("taxable") == 0.20
    assert tax.rate_for("cash") == 0.0


def test_asset_tax_type():
    a = Asset(name="Roth", value=1000, mean_return=0.05, std_dev=0.10, tax_type="roth")
    assert a.tax_type == "roth"


def test_config_properties():
    config = load_config("example_config.yaml")
    assert config.years_to_retirement == 15  # 50 - 35
    assert config.years_in_retirement == 40  # 90 - 50


def test_contributions_loaded():
    config = load_config("example_config.yaml")
    assert len(config.contributions) == 3
    rsu = [c for c in config.contributions if c.is_variable]
    assert len(rsu) == 1
    assert rsu[0].std_dev_pct == 0.20


def test_529_plans_loaded():
    config = load_config("example_config.yaml")
    assert len(config.plans_529) == 1
    assert config.plans_529[0].current_value == 10000


# --- Config validation tests ---


def _make_config(**overrides) -> SimConfig:
    defaults = dict(
        assets=[Asset(name="A", value=100, mean_return=0.05, std_dev=0.1)],
        spending=SpendingSchedule(categories={}),
        current_age=35,
        retirement_age=50,
        life_expectancy=90,
    )
    defaults.update(overrides)
    return SimConfig(**defaults)


def test_validate_retirement_age_must_exceed_current_age():
    config = _make_config(current_age=50, retirement_age=50)
    with pytest.raises(ConfigValidationError, match="retirement_age.*must be greater than.*current_age"):
        validate_config(config)


def test_validate_life_expectancy_must_exceed_retirement_age():
    config = _make_config(retirement_age=50, life_expectancy=50)
    with pytest.raises(ConfigValidationError, match="life_expectancy.*must be greater than.*retirement_age"):
        validate_config(config)


def test_validate_return_source_csv_missing():
    config = _make_config(return_sources={"foo": "/nonexistent/path.csv"})
    with pytest.raises(ConfigValidationError, match="file not found"):
        validate_config(config)


def test_validate_contribution_target_account_missing():
    from sim.config import Contribution
    config = _make_config(
        contributions=[Contribution(name="Save", monthly_amount=100, target_account="NoSuchAsset")],
    )
    with pytest.raises(ConfigValidationError, match="does not match any asset"):
        validate_config(config)


def test_validate_asset_return_source_not_in_sources():
    config = _make_config(
        assets=[Asset(name="A", value=100, mean_return=0.05, std_dev=0.1, return_source="missing")],
        method="bootstrap",
        return_sources={},
    )
    with pytest.raises(ConfigValidationError, match="not found in return_sources"):
        validate_config(config)


def test_validate_valid_config_passes():
    config = _make_config()
    validate_config(config)  # should not raise
