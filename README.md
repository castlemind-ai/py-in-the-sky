# py-in-the-sky

Monte Carlo wealth and retirement simulation. Models a two-phase financial life — accumulation (saving) and drawdown (spending) — with stochastic returns, taxes, mortgages, college costs, and black swan events.

## Quick start

```bash
pip install -e ".[dev]"
python run.py                          # uses example_config.yaml
python run.py my_config.yaml           # use your own config
python run.py my_config.yaml --no-plot # stats only, no charts
```

Generates four charts in `./output/` and prints a summary to stdout:

| Chart | What it shows |
|---|---|
| `fan_chart.png` | Wealth trajectory with 10th–90th percentile bands |
| `final_wealth_histogram.png` | Distribution of final portfolio value (log scale) |
| `ruin_probability.png` | Cumulative probability of portfolio depletion over time |
| `distribution_rate.png` | Annual withdrawal rate with 4% rule reference line |

## How it works

### Two-phase simulation

**Accumulation** (working years): Grows each asset monthly with contributions. Two modes:

- **`mc`** — Deterministic. All assets grow at a single blended rate (`accumulation_return`).
- **`bootstrap`** — Stochastic. Assets with a `return_source` get monthly returns sampled from historical CSV data. Each simulation draws different months, producing a distribution of retirement portfolio values. Assets without a `return_source` use parametric returns from `mean_return`/`std_dev`.

**Drawdown** (retirement years): 10,000 Monte Carlo simulations of annual withdrawals against stochastic returns and inflation. Withdrawals are proportional across assets and tax-aware.

### Return modeling

**Parametric** (`method: "mc"`): Normal distribution with per-asset `mean_return` and `std_dev`.

**Bootstrap** (`method: "bootstrap"`): Block-bootstrap sampling from historical monthly return CSVs. For annual drawdown returns, 12 consecutive months are sampled and compounded, preserving within-year volatility clustering. All assets sharing historical sources use the same random start indices, preserving cross-asset correlation (e.g., stocks crash together).

**Blended**: Per-asset `bootstrap_blend` (0.0–1.0) mixes bootstrap and parametric returns. Useful for concentrated single-stock positions where bootstrapping from survivorship-biased history (e.g., a FAANG stock's extraordinary run) would produce unrealistically low ruin probabilities.

### Retirement diversification

Assets can specify `retire_to_source` to model selling a concentrated position at retirement and switching to index returns for drawdown. Capital gains tax is computed on `value - cost_basis` at the applicable rate.

```yaml
- name: "Company Stock"
  value: 1_500_000
  return_source: "stock"
  cost_basis: 400_000
  retire_to_source: "market"   # sell stock, pay cap gains, invest in index
```

### Variable contributions

Contributions marked `is_variable: true` with a `std_dev_pct` produce stochastic monthly amounts in bootstrap mode (e.g., RSU vesting that fluctuates with stock price). Values are floored at zero.

## Configuration

Copy and edit `example_config.yaml`. Key sections:

```yaml
# Simulation parameters
current_age: 35
retirement_age: 50
life_expectancy: 90
num_simulations: 10000
method: "mc"                    # "mc" or "bootstrap"
accumulation_return: 0.065      # blended rate (mc mode only)

# Historical data for bootstrap mode
return_sources:
  market: "data/VTI_monthly.csv"
  stock: "data/GOOG_monthly.csv"

# Tax rates
tax:
  income_rate: 0.35
  capital_gains_rate: 0.20
  roth_rate: 0.0

# Portfolio
assets:
  - name: "401k"
    value: 200000
    mean_return: 0.065
    std_dev: 0.15
    tax_type: "traditional"     # roth | traditional | taxable | cash
    return_source: "market"     # key into return_sources (bootstrap mode)
    bootstrap_blend: 1.0        # 1.0 = pure bootstrap, 0.5 = 50/50 with parametric

# Monthly savings
contributions:
  - name: "401k contributions"
    monthly_amount: 1875
    target_account: "401k"
    is_variable: false

  - name: "RSU vesting"
    monthly_amount: 8000
    target_account: "Company Stock"
    is_variable: true
    std_dev_pct: 0.40

# Mortgage (tracked separately from investable assets)
mortgage:
  purchase_price: 800000
  down_payment_pct: 0.20
  interest_rate: 0.06
  monthly_payment: 3836
  purchase_date: "2022-01-01"
  non_mortgage_monthly: 1500

# 529 college savings
plans_529:
  - name: "Child 1 529"
    current_value: 10000
    monthly_contribution: 500
    mean_return: 0.065
    child_college_start_year: 14

# Annual spending (in today's dollars, inflation-adjusted automatically)
spending:
  categories:
    school: 20000
    groceries: 10000
    travel: 10000
    healthcare: 8000
  school_end_years: [14]        # school spending stops after year 14

# Large future expenses
future_costs:
  - name: "Child 1 College"
    amount: 50000               # per year, inflation-adjusted
    year: 14
    duration: 4

# Tail risk events
black_swans:
  - name: "Medical Emergency"
    probability: 0.02
    cost: 100000
  - name: "Market Crash"
    probability: 0.05
    portfolio_impact: -0.30
```

## Edge cases and modeling choices

### Return clamping

Annual drawdown returns are clamped at -99% (`engine.py:266`) to prevent mathematically impossible negative portfolio values from extreme return draws. This is a safety floor — real-world losses beyond 99% in a single year are essentially impossible for diversified portfolios.

### Black swan dual behavior

A black swan event can specify both a direct `cost` and a `portfolio_impact` simultaneously. When triggered, the cost is withdrawn proportionally from assets first, then the portfolio impact is applied multiplicatively. This models events like a serious illness (direct medical expense) that also causes income disruption (portfolio drawdown).

### Variable contribution `std_dev_pct`

The `std_dev_pct` field on contributions is a fractional multiplier of the mean monthly amount, not a percentage. For example, `std_dev_pct: 0.40` with `monthly_amount: 8000` produces monthly amounts drawn from `Normal(8000, 3200)`, floored at zero. This is useful for modeling RSU vesting that fluctuates with stock price.

### Config validation

`load_config()` validates that:
- `retirement_age > current_age`
- `life_expectancy > retirement_age`
- All `return_sources` CSV files exist on disk
- All contribution `target_account` values match an asset name
- In bootstrap mode, all asset `return_source` and `retire_to_source` keys exist in `return_sources`

## Historical return data

The `data/` directory contains monthly return CSVs. Two formats are auto-detected:

- **Pre-computed returns** (e.g., GOOG): CSV with a `Month-over-month` column containing decimal returns.
- **Price history** (e.g., VTI): CSV with an `Adj Close` column. Returns are computed as `P_t / P_{t-1} - 1`. Dividend rows (where `Open` contains "Dividend") are filtered out.

Both formats expect newest-first row ordering (reversed internally to chronological).

## Project structure

```
run.py                  CLI entry point
sim/
  config.py             Configuration dataclasses and YAML loading
  engine.py             Two-phase simulation engine (accumulation + drawdown)
  events.py             Mortgage, spending, 529, tax helpers
  output.py             Summary statistics and matplotlib charts
  returns.py            Historical return loading and block-bootstrap sampling
tests/
  test_config.py        Config parsing tests
  test_engine.py        Simulation engine tests (mc and bootstrap modes)
  test_events.py        Event resolution tests
  test_returns.py       CSV loading and bootstrap sampling tests
data/
  GOOG_monthly.csv      GOOG monthly returns (2004–2026)
  VTI_monthly.csv       VTI monthly prices (2001–2026)
```

## Tests

```bash
pytest tests/
```

39 tests covering config parsing, both simulation modes, mortgage amortization, 529 growth, tax gross-up, bootstrap sampling, and seed reproducibility.

## Requirements

- Python 3.11+
- numpy, matplotlib, pyyaml
