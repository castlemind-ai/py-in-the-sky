"""Summary statistics and matplotlib visualizations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sim.engine import SimResult


def print_summary(result: SimResult) -> None:
    """Print summary statistics to the console."""
    config = result.config
    accum = result.accumulation
    wealth = result.wealth
    final = wealth[:, -1]
    n = config.num_simulations

    # Accumulation summary
    total_at_retirement = sum(accum.asset_values.values())
    print("=" * 65)
    print("ACCUMULATION PHASE")
    print("=" * 65)
    print(f"  Years of saving:        {config.years_to_retirement}")
    print(f"  Portfolio at retirement: ${total_at_retirement:,.0f}")
    for name, val in accum.asset_values.items():
        print(f"    {name:>25s}:  ${val:,.0f}")
    if accum.plan_529_values:
        print(f"\n  529 Plans at retirement:")
        for name, val in accum.plan_529_values.items():
            print(f"    {name:>25s}:  ${val:,.0f}")
    if config.mortgage:
        print(f"\n  Mortgage principal remaining: ${accum.mortgage_principal:,.0f}")
    print()

    # Drawdown summary
    ruin_mask = np.any(wealth <= 0, axis=1)
    ruin_pct = ruin_mask.sum() / n * 100
    target_met = (final >= config.target_wealth).sum() / n * 100 if config.target_wealth > 0 else None

    percentiles = np.percentile(final, [10, 25, 50, 75, 90])
    bottom_10 = final[final <= np.percentile(final, 10)]
    expected_shortfall = bottom_10.mean() if len(bottom_10) > 0 else 0.0

    print("=" * 65)
    print("DRAWDOWN PHASE — MONTE CARLO RESULTS")
    print("=" * 65)
    print(f"  Simulations:            {n:,}")
    print(f"  Retirement years:       {config.years_in_retirement}")
    print(f"  Age range:              {config.retirement_age} → {config.life_expectancy}")
    print()
    print(f"  Probability of ruin:    {ruin_pct:.1f}%")
    if target_met is not None:
        print(f"  Target met (${config.target_wealth:,.0f}):  {target_met:.1f}%")
    print()
    print("  Final wealth percentiles:")
    labels = ["10th", "25th", "50th (median)", "75th", "90th"]
    for label, val in zip(labels, percentiles):
        print(f"    {label:>16s}:  ${val:,.0f}")
    print()
    print(f"  Mean final wealth:      ${final.mean():,.0f}")
    print(f"  Expected shortfall:     ${expected_shortfall:,.0f}")
    print()

    # Distribution rate stats (exclude ruin scenarios where rates are meaningless)
    successful = ~ruin_mask
    if successful.any():
        max_dist_rates = result.dist_rates[successful].max(axis=1)
        dr_pctiles = np.percentile(max_dist_rates, [50, 90, 95, 99])
        print("  Max distribution rate (non-ruin scenarios):")
        for label, val in zip(["50th", "90th", "95th", "99th"], dr_pctiles):
            print(f"    {label:>8s}:  {val:.1%}")
        print()

    # Year 1 distribution rate
    yr1_rates = result.dist_rates[:, 0]
    print(f"  Year 1 median dist rate: {np.median(yr1_rates):.1%}")
    print("=" * 65)

    # Per-asset median at end
    if result.asset_wealth.shape[2] > 1:
        print("\n  Per-asset median final values:")
        for i, asset in enumerate(config.assets):
            median_val = np.median(result.asset_wealth[:, -1, i])
            print(f"    {asset.name:>25s}:  ${median_val:,.0f}")
        print()


def plot_fan_chart(result: SimResult, ax: plt.Axes | None = None) -> plt.Figure | None:
    """Plot wealth trajectory with confidence bands."""
    wealth = result.wealth
    config = result.config
    years = np.arange(wealth.shape[1]) + config.retirement_age

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = None

    p10, p25, p50, p75, p90 = np.percentile(wealth, [10, 25, 50, 75, 90], axis=0)

    ax.fill_between(years, p10, p90, alpha=0.15, color="steelblue", label="10th–90th")
    ax.fill_between(years, p25, p75, alpha=0.3, color="steelblue", label="25th–75th")
    ax.plot(years, p50, color="steelblue", linewidth=2, label="Median")

    if config.target_wealth > 0:
        ax.axhline(y=config.target_wealth, color="green", linestyle="--",
                    alpha=0.7, label=f"Target (${config.target_wealth:,.0f})")

    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Ruin")
    ax.set_xlabel("Age")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Retirement Wealth Trajectory — Monte Carlo Simulation")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_final_wealth_histogram(result: SimResult, ax: plt.Axes | None = None) -> plt.Figure | None:
    """Plot distribution of final wealth on a log10 x-axis.

    Ruined paths (wealth == 0) are excluded from the histogram and noted in the title.
    """
    final = result.wealth[:, -1]
    positive = final[final > 0]
    n_total = len(final)
    n_ruined = n_total - len(positive)

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None

    if len(positive) > 0:
        bins = np.logspace(np.log10(positive.min()), np.log10(positive.max()), 80)
        ax.hist(positive, bins=bins, color="steelblue", alpha=0.7, edgecolor="white")

    if result.config.target_wealth > 0:
        ax.axvline(x=result.config.target_wealth, color="green", linestyle="--",
                    linewidth=2, label=f"Target (${result.config.target_wealth:,.0f})")

    ax.set_xscale("log")

    def _dollar_fmt(x, _):
        if x >= 1e9:
            return f"${x/1e9:.0f}B"
        elif x >= 1e6:
            return f"${x/1e6:.0f}M"
        elif x >= 1e3:
            return f"${x/1e3:.0f}K"
        return f"${x:.0f}"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_dollar_fmt))

    ruin_note = f"  ({n_ruined:,} ruined paths excluded, {n_ruined/n_total:.1%})" if n_ruined else ""
    ax.set_xlabel(f"Final Wealth — log₁₀ scale{ruin_note}")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Wealth at Age {result.config.life_expectancy}")
    if result.config.target_wealth > 0:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_ruin_probability(result: SimResult, ax: plt.Axes | None = None) -> plt.Figure | None:
    """Plot cumulative probability of ruin over time."""
    wealth = result.wealth
    config = result.config
    n = wealth.shape[0]
    years = np.arange(wealth.shape[1]) + config.retirement_age

    cum_ruin = np.cumsum(wealth <= 0, axis=1) > 0
    ruin_prob = cum_ruin.sum(axis=0) / n * 100

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = None

    ax.plot(years, ruin_prob, color="firebrick", linewidth=2)
    ax.set_xlabel("Age")
    ax.set_ylabel("Cumulative Probability of Ruin (%)")
    ax.set_title("Probability of Portfolio Depletion Over Time")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_distribution_rate(result: SimResult, ax: plt.Axes | None = None) -> plt.Figure | None:
    """Plot distribution (withdrawal) rate over time."""
    config = result.config
    # Cap rates at 100% for visualization (ruin scenarios produce meaningless rates)
    rates = np.clip(result.dist_rates, 0, 1.0)
    years = np.arange(rates.shape[1]) + config.retirement_age

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = None

    p10, p25, p50, p75, p90 = np.percentile(rates, [10, 25, 50, 75, 90], axis=0)

    ax.fill_between(years, p10 * 100, p90 * 100, alpha=0.15, color="darkorange", label="10th–90th")
    ax.fill_between(years, p25 * 100, p75 * 100, alpha=0.3, color="darkorange", label="25th–75th")
    ax.plot(years, p50 * 100, color="darkorange", linewidth=2, label="Median")
    ax.axhline(y=4, color="green", linestyle="--", alpha=0.7, label="4% Rule")
    ax.set_xlabel("Age")
    ax.set_ylabel("Distribution Rate (%)")
    ax.set_title("Annual Withdrawal Rate Over Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def generate_report(result: SimResult, output_dir: str = "./output") -> None:
    """Generate full report: console output + PNG charts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print_summary(result)

    charts = [
        ("fan_chart.png", plot_fan_chart),
        ("final_wealth_histogram.png", plot_final_wealth_histogram),
        ("ruin_probability.png", plot_ruin_probability),
        ("distribution_rate.png", plot_distribution_rate),
    ]

    for filename, plot_fn in charts:
        fig = plot_fn(result)
        if fig:
            fig.savefig(out / filename, dpi=150)
            plt.close(fig)

    print(f"\nCharts saved to {out.resolve()}/")
