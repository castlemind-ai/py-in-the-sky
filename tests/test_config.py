"""Tests for configuration loading."""

import tempfile

from sim.config import Asset, SimConfig, SpendingSchedule, TaxConfig, load_config


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
