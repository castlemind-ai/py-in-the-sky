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

**Drawdown** (retirement years): Monte Carlo simulations of annual withdrawals against stochastic returns and inflation. Withdrawals are tax-aware and support configurable sequencing strategies.

### Return modeling

**Parametric** (`method: "mc"`): Normal or lognormal distribution with per-asset `mean_return` and `std_dev`. Set `return_distribution: "lognormal"` per asset for more realistic long-horizon modeling (no below -100% outcomes; positive skew in compounded returns).

**Bootstrap** (`method: "bootstrap"`): Block-bootstrap sampling from historical monthly return CSVs. For annual drawdown returns, 12 consecutive months are sampled and compounded, preserving within-year volatility clustering. All assets sharing historical sources use the same random start indices, preserving cross-asset correlation (e.g., stocks crash together).

**Blended**: Per-asset `bootstrap_blend` (0.0–1.0) mixes bootstrap and parametric returns.

#### Choosing `bootstrap_blend` for concentrated positions

FAANG-style stocks have extraordinary historical returns that are unlikely to repeat for any individual holding — bootstrapping purely from that history (*survivorship bias*) produces unrealistically low ruin probabilities. Recommended values:

| `bootstrap_blend` | Interpretation |
|---|---|
| `1.0` | Pure historical (optimistic for single-stock winners) |
| `0.4–0.6` | Balanced blend — recommended for concentrated single-stock positions |
| `0.0` | Pure parametric using `mean_return`/`std_dev` |

For diversified index funds, `1.0` is appropriate. For a large FAANG position, `0.4–0.5` is more defensible.

### Retirement diversification

Assets can specify `retire_to_source` to model selling a concentrated position at retirement and switching to index returns for drawdown. Capital gains tax is computed on `value - cost_basis` at the applicable rate (including `state_capital_gains_rate`).

```yaml
- name: "Company Stock"
  value: 1_500_000
  return_source: "stock"
  cost_basis: 400_000
  retire_to_source: "market"   # sell stock, pay cap gains, invest in index
```

### Variable and growing contributions

Contributions marked `is_variable: true` with a `std_dev_pct` produce stochastic monthly amounts in bootstrap mode (e.g., RSU vesting). Values are floored at zero.

Add `annual_growth_rate` to model grant refreshes, promotions, or annual raises:

```yaml
- name: "RSU vesting"
  monthly_amount: 8000
  annual_growth_rate: 0.05     # 5% annual increase in base grant value
  is_variable: true
  std_dev_pct: 0.40
```

### Tax modeling

**Flat rate**: Set `income_rate`, `capital_gains_rate`, and optionally `state_capital_gains_rate` (added to federal for taxable withdrawals — relevant for CA residents at ~13.3%).

**Progressive brackets** (optional): Replace flat `income_rate` with a `tax_brackets` list for more accurate modeling of retirement income. Falls back to `income_rate` if not set.

**Pre-59½ early withdrawal penalty**: Withdrawals from `traditional` accounts before age 59½ incur an additional `early_withdrawal_penalty_rate` (default 10%). Set `rule_of_55_eligible: true` on an asset to exempt it (for Rule of 55 scenarios — requires separation from employer at age 55+).

**Roth contribution basis**: Set `roth_contribution_basis` on a Roth asset to separately track contributed vs. earned amounts. Contributions are always penalty-free; earnings are penalized pre-59½.

**Mega backdoor Roth**: Use `tax_type: "after_tax_401k"` for after-tax 401(k) contributions auto-converted to Roth. Tax-free growth and withdrawals. Mark `is_401k_plan: true` to count toward the $70,000 annual 401(k) limit.

### Social Security

Add a `social_security` block to model the income floor from SS benefits, reducing required portfolio withdrawals once benefits begin:

```yaml
social_security:
  monthly_benefit: 3500        # in today's dollars at full retirement age
  claim_age: 67
  inflation_adjusted: true
  early_claim_reduction: 0.0   # set to ~0.30 if claiming at 62
```

A `spouse` block adds a second SS stream at the spouse's claim age.

### Age-conditioned spending

Use `age_transitions` in the spending block to model step changes — e.g., healthcare costs dropping sharply when Medicare begins at 65:

```yaml
spending:
  categories:
    healthcare_pre_medicare: 30000
  age_transitions:
    - at_age: 65
      remove: [healthcare_pre_medicare]
      add:
        healthcare_post_medicare: 8000
```

### Roth conversion ladder

Model annual traditional→Roth conversions to build a penalty-free withdrawal pool for the 55–59½ window:

```yaml
roth_conversions:
  - start_age: 55
    end_age: 59
    annual_amount: 80000
    source_account: "401k"
    target_account: "Roth IRA"
```

Conversions are taxed as ordinary income in the year executed. Converted amounts are added to Roth basis (penalty-free after the 5-year rule, modeled via basis tracking).

### Drawdown sequencing

Set `drawdown_strategy: "tax_optimized"` to drain assets in ascending tax-rate order (taxable → traditional → Roth) rather than proportionally. This reduces lifetime tax burden by preserving low-cost Roth assets longest.

### Required Minimum Distributions (RMDs)

For traditional accounts, once the simulated age reaches 73, the engine computes the IRS Uniform Lifetime Table RMD each year. If the normal portfolio withdrawal doesn't cover the RMD, the shortfall is force-withdrawn. Any excess over spending is reinvested in the first taxable account (or consumed as spending if none exists).

## Configuration reference

Copy and edit `example_config.yaml`. Key sections:

```yaml
current_age: 35
retirement_age: 50
life_expectancy: 90
num_simulations: 10000
method: "mc"                    # "mc" or "bootstrap"
accumulation_return: 0.065      # blended rate (mc mode only)
drawdown_strategy: "proportional"  # "proportional" or "tax_optimized"
target_wealth: 500000           # ~25x annual spending; 0 to disable

tax:
  income_rate: 0.35
  capital_gains_rate: 0.20
  state_capital_gains_rate: 0.133  # CA example; combined ~33%
  early_withdrawal_penalty_rate: 0.10
  # tax_brackets: [...]           # optional progressive brackets

assets:
  - name: "401k"
    value: 200000
    mean_return: 0.065
    std_dev: 0.15
    tax_type: "traditional"        # roth | traditional | taxable | cash | after_tax_401k
    rule_of_55_eligible: true
    is_401k_plan: true
    return_distribution: "lognormal"  # "normal" (default) or "lognormal"

  - name: "Roth IRA"
    value: 50000
    mean_return: 0.065
    std_dev: 0.15
    tax_type: "roth"
    roth_contribution_basis: 30000

social_security:
  monthly_benefit: 3500
  claim_age: 67
  inflation_adjusted: true

spending:
  categories:
    healthcare_pre_medicare: 30000
  age_transitions:
    - at_age: 65
      remove: [healthcare_pre_medicare]
      add:
        healthcare_post_medicare: 8000

roth_conversions:
  - start_age: 55
    end_age: 59
    annual_amount: 60000
    source_account: "401k"
    target_account: "Roth IRA"
```

## Edge cases and modeling choices

### Return clamping

Annual drawdown returns are clamped at -99% to prevent mathematically impossible negative portfolio values from extreme normal distribution draws. This is a safety floor — use `return_distribution: "lognormal"` to avoid this issue entirely (lognormal draws are always > -100%).

### Black swan dual behavior

A black swan event can specify both a direct `cost` and a `portfolio_impact` simultaneously. When triggered, the cost is withdrawn proportionally from assets first, then the portfolio impact is applied multiplicatively. This models events like a serious illness (direct medical expense) that also causes income disruption (portfolio drawdown).

### Variable contribution `std_dev_pct`

The `std_dev_pct` field on contributions is a fractional multiplier of the mean monthly amount, not a percentage. For example, `std_dev_pct: 0.40` with `monthly_amount: 8000` produces monthly amounts drawn from `Normal(8000, 3200)`, floored at zero. This is useful for modeling RSU vesting that fluctuates with stock price.

### Setting `target_wealth`

`target_wealth` is a portfolio size threshold shown in the fan chart and histogram, with probability of reaching it reported in the summary. The standard rule of thumb is 25× annual spending (based on the 4% safe withdrawal rate) — e.g., $60k/year spending → `target_wealth: 1500000`. Set to `0` to disable.

### Config validation

`load_config()` validates that:
- `retirement_age > current_age` and `life_expectancy > retirement_age`
- All `return_sources` CSV files exist on disk
- All contribution `target_account` values match an asset name
- In bootstrap mode, all asset `return_source` / `retire_to_source` keys exist in `return_sources`
- Combined capital gains rate (`capital_gains_rate + state_capital_gains_rate`) < 100%
- `drawdown_strategy` is a known value
- Roth conversion `source_account` / `target_account` match existing assets
- Total 401k-plan contributions do not exceed $70,000/year (when `is_401k_plan: true`)

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
  events.py             Mortgage, spending, 529, tax helpers, RMD computation
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

## Requirements

- Python 3.11+
- numpy, matplotlib, pyyaml
