"""
╔══════════════════════════════════════════════════════════════════╗
║  QUANT ENGINE — Report Generator                                ║
║  Charts, tables, and HTML report from backtest results          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Dict, List
from risk_analytics import (
    RiskReport, compute_risk_report, compute_correlation_matrix,
    rolling_sharpe, rolling_drawdown,
)

# Style
plt.rcParams.update({
    "figure.facecolor": "#0a0e17",
    "axes.facecolor": "#111827",
    "axes.edgecolor": "#1e293b",
    "axes.labelcolor": "#94a3b8",
    "text.color": "#e2e8f0",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "grid.color": "#1e293b",
    "grid.alpha": 0.5,
    "font.family": "monospace",
    "font.size": 9,
    "legend.facecolor": "#111827",
    "legend.edgecolor": "#1e293b",
})

COLORS = ["#06d6a0", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#f472b6"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")


def generate_full_report(
    results: Dict[str, "BacktestResult"],
    benchmark_result: "BacktestResult" = None,
    risk_free_rate: float = 0.05,
    output_dir: str = None,
):
    """
    Generate complete visual report:
    1. Equity curves (all strategies + benchmark)
    2. Drawdown chart
    3. Rolling Sharpe ratios
    4. Return distribution
    5. Correlation heatmap
    6. Monthly returns heatmap
    7. Risk metrics summary table
    8. HTML report
    """
    out = output_dir or OUTPUT_DIR
    os.makedirs(out, exist_ok=True)

    names = list(results.keys())
    bm_returns = benchmark_result.returns if benchmark_result else None

    # Compute risk reports
    reports = {}
    for name, res in results.items():
        reports[name] = compute_risk_report(res, risk_free_rate, bm_returns)
    if benchmark_result:
        reports["Benchmark"] = compute_risk_report(benchmark_result, risk_free_rate)

    # ─── Figure 1: Master Dashboard ─────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("QUANT ALPHA ENGINE — Backtest Report", fontsize=16, fontweight="bold",
                 color="#06d6a0", y=0.98)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1a. Equity curves
    ax1 = fig.add_subplot(gs[0, :])
    for i, (name, res) in enumerate(results.items()):
        eq = res.equity_curve / res.equity_curve.iloc[0]
        ax1.plot(eq.index, eq.values, label=name, color=COLORS[i % len(COLORS)], linewidth=1.5)
    if benchmark_result:
        eq_bm = benchmark_result.equity_curve / benchmark_result.equity_curve.iloc[0]
        ax1.plot(eq_bm.index, eq_bm.values, label="Benchmark", color="#64748b",
                 linewidth=2, linestyle="--")
    ax1.set_title("Cumulative Returns (Normalized to $1)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color="#475569", linestyle=":", linewidth=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 1b. Drawdowns
    ax2 = fig.add_subplot(gs[1, :2])
    for i, (name, res) in enumerate(results.items()):
        dd = rolling_drawdown(res.returns)
        ax2.fill_between(dd.index, dd.values, alpha=0.3, color=COLORS[i % len(COLORS)], label=name)
        ax2.plot(dd.index, dd.values, color=COLORS[i % len(COLORS)], linewidth=0.5)
    ax2.set_title("Underwater Equity (Drawdowns)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.legend(loc="lower left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 1c. Return distribution
    ax3 = fig.add_subplot(gs[1, 2])
    for i, (name, res) in enumerate(results.items()):
        ret = res.returns.dropna()
        ax3.hist(ret, bins=80, alpha=0.5, color=COLORS[i % len(COLORS)],
                 label=name, density=True)
    ax3.set_title("Return Distribution", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Daily Return")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color="#475569", linestyle=":", linewidth=0.5)

    # 1d. Rolling Sharpe
    ax4 = fig.add_subplot(gs[2, :2])
    for i, (name, res) in enumerate(results.items()):
        rs = rolling_sharpe(res.returns, window=63, rf=risk_free_rate)
        ax4.plot(rs.index, rs.values, label=name, color=COLORS[i % len(COLORS)], linewidth=1)
    ax4.axhline(y=0, color="#ef4444", linestyle="--", linewidth=0.5)
    ax4.axhline(y=1, color="#06d6a0", linestyle="--", linewidth=0.5, alpha=0.5)
    ax4.set_title("Rolling 63-Day Sharpe Ratio", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Sharpe")
    ax4.legend(loc="upper left", fontsize=7)
    ax4.grid(True, alpha=0.3)

    # 1e. Correlation heatmap
    ax5 = fig.add_subplot(gs[2, 2])
    corr = compute_correlation_matrix(results)
    im = ax5.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax5.set_xticks(range(len(corr.columns)))
    ax5.set_xticklabels([n[:10] for n in corr.columns], rotation=45, ha="right", fontsize=7)
    ax5.set_yticks(range(len(corr.index)))
    ax5.set_yticklabels([n[:10] for n in corr.index], fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax5.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "#94a3b8")
    ax5.set_title("Strategy Correlation", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax5, fraction=0.046)

    plt.savefig(os.path.join(out, "dashboard.png"), dpi=150, bbox_inches="tight",
                facecolor="#0a0e17", edgecolor="none")
    plt.close()
    print(f"  📊 Saved dashboard → {os.path.join(out, 'dashboard.png')}")

    # ─── Figure 2: Monthly Returns Heatmap ───────────────────
    for name, res in results.items():
        _plot_monthly_heatmap(res, out, name)

    # ─── Generate HTML Report ────────────────────────────────
    _generate_html_report(reports, results, out)

    # ─── Print summary table ─────────────────────────────────
    _print_summary_table(reports)

    return reports


def _plot_monthly_heatmap(result, out_dir, name):
    """Plot monthly returns heatmap for a strategy."""
    ret = result.returns.dropna()
    if len(ret) < 30:
        return

    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    pivot = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    }).pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.4 + 1)))
    im = ax.imshow(pivot.values * 100, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(len(pivot)):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.03 else "#94a3b8")
    ax.set_title(f"Monthly Returns — {name}", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Return (%)")
    safe_name = name.replace(" ", "_").replace("/", "_")[:30]
    plt.savefig(os.path.join(out_dir, f"monthly_{safe_name}.png"), dpi=150,
                bbox_inches="tight", facecolor="#0a0e17")
    plt.close()


def _print_summary_table(reports: Dict[str, RiskReport]):
    """Print a formatted summary table to terminal."""
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()

        table = Table(title="═══ BACKTEST RESULTS ═══", style="bold cyan",
                      border_style="dim", show_lines=True)
        table.add_column("Metric", style="bold", width=24)

        for name in reports:
            table.add_column(name[:16], justify="right", width=14)

        def add_row(label, key, fmt="{:.2%}", color_positive=True):
            row = [label]
            for name, r in reports.items():
                val = getattr(r, key)
                s = fmt.format(val)
                if color_positive:
                    if val > 0:
                        s = f"[green]{s}[/green]"
                    elif val < 0:
                        s = f"[red]{s}[/red]"
                row.append(s)
            table.add_row(*row)

        add_row("Total Return", "total_return")
        add_row("CAGR", "cagr")
        add_row("Annual Vol", "annualized_vol", color_positive=False)
        add_row("Sharpe Ratio", "sharpe_ratio", fmt="{:.3f}")
        add_row("Sortino Ratio", "sortino_ratio", fmt="{:.3f}")
        add_row("Calmar Ratio", "calmar_ratio", fmt="{:.3f}")
        add_row("Info Ratio", "information_ratio", fmt="{:.3f}")
        add_row("Max Drawdown", "max_drawdown")
        add_row("Max DD Duration", "max_drawdown_duration_days", fmt="{:.0f} days", color_positive=False)
        add_row("VaR (95%)", "var_95", fmt="{:.4%}", color_positive=False)
        add_row("CVaR (95%)", "cvar_95", fmt="{:.4%}", color_positive=False)
        add_row("Win Rate", "win_rate")
        add_row("Profit Factor", "profit_factor", fmt="{:.2f}")
        add_row("Kelly f*", "kelly_fraction", fmt="{:.3f}")
        add_row("Skewness", "skewness", fmt="{:.3f}", color_positive=False)
        add_row("Kurtosis", "kurtosis", fmt="{:.3f}", color_positive=False)
        add_row("Best Day", "best_day")
        add_row("Worst Day", "worst_day")
        add_row("Avg Turnover", "avg_daily_turnover", fmt="{:.4f}", color_positive=False)
        add_row("Total Costs", "total_costs", fmt="${:,.0f}", color_positive=False)
        add_row("% Time in Mkt", "pct_time_in_market")

        console.print(table)

    except ImportError:
        # Fallback: plain print
        from tabulate import tabulate
        headers = ["Metric"] + list(reports.keys())
        rows = []
        for label, key, fmt in [
            ("Total Return", "total_return", "{:.2%}"),
            ("CAGR", "cagr", "{:.2%}"),
            ("Sharpe", "sharpe_ratio", "{:.3f}"),
            ("Max DD", "max_drawdown", "{:.2%}"),
            ("Win Rate", "win_rate", "{:.2%}"),
        ]:
            row = [label] + [fmt.format(getattr(r, key)) for r in reports.values()]
            rows.append(row)
        print(tabulate(rows, headers=headers, tablefmt="grid"))


def _build_explainer_section(reports: Dict[str, RiskReport]) -> str:
    """Build a plain-English explainer section for beginners."""

    # ── Strategy plain-English descriptions ──────────────────────────────────
    strategy_info = {
        "mean_reversion": {
            "emoji": "🔄",
            "title": "Mean Reversion",
            "plain": (
                "Think of a rubber band. When a stock price gets stretched too far "
                "in one direction, this strategy bets it'll snap back. It buys stocks "
                "that have fallen unusually far and sells (shorts) stocks that have "
                "climbed unusually high. Works best in calm, sideways markets."
            ),
        },
        "momentum": {
            "emoji": "🚀",
            "title": "Momentum / Trend Following",
            "plain": (
                "Surf the wave. If a stock has been rising, keep riding it up. "
                "If it's been falling, avoid it (or bet against it). Like a train "
                "that tends to keep moving once it gets going. Works best during "
                "strong market trends."
            ),
        },
        "pairs_trading": {
            "emoji": "⚖️",
            "title": "Pairs Trading",
            "plain": (
                "Play two stocks off each other. Find two companies that tend to "
                "move together (e.g., two big banks). When one gets ahead of the "
                "other, bet they'll realign — buy the laggard, sell the leader. "
                "This strategy doesn't need the market to go up or down — it just "
                "needs the gap to close."
            ),
        },
        "stat_arb": {
            "emoji": "🧮",
            "title": "Statistical Arbitrage (PCA)",
            "plain": (
                "Advanced math-based mispricing detector. Uses a technique called "
                "PCA (Principal Component Analysis) to strip out broad market moves "
                "from each stock's price. Whatever's left — the 'unexplained' part — "
                "is traded when it drifts too far. Think of it as finding tiny "
                "mispricings across dozens of stocks simultaneously."
            ),
        },
        "multi_factor": {
            "emoji": "📊",
            "title": "Multi-Factor Model",
            "plain": (
                "A stock report card. Every stock gets graded on four qualities: "
                "trend (is it going up?), value (is it cheap?), stability (does it "
                "move smoothly?), and profitability (is the company healthy?). "
                "The top-ranked stocks are bought; the bottom-ranked are sold. "
                "Inspired by academic research from Fama and French."
            ),
        },
        "trend_200": {
            "emoji": "📈",
            "title": "200-Day MA Trend Filter",
            "plain": (
                "The simplest strategy that actually has a long track record. "
                "Rule: if a stock's price is above its 200-day average, hold it. "
                "If it drops below, sell and hold cash. That's it. During a bear "
                "market most stocks fall below their averages, so the strategy "
                "automatically moves toward cash and avoids the worst crashes."
            ),
        },
        "sector_rotation": {
            "emoji": "🔁",
            "title": "Sector Rotation",
            "plain": (
                "The economy runs in cycles — tech booms, then energy leads, then "
                "healthcare is defensive. This strategy ranks the five sectors "
                "(tech, finance, healthcare, energy, consumer) by their 3-month "
                "return, then puts all money into the top two. If all sectors are "
                "losing, it stays in cash. Rebalances every month."
            ),
        },
        "breakout": {
            "emoji": "💥",
            "title": "52-Week High Breakout",
            "plain": (
                "Buy stocks making new yearly highs — they tend to keep going. "
                "Psychology: when a stock hits a price it hasn't been at in a year, "
                "it often has strong momentum behind it. Entry: new 52-week high "
                "with high volume. Exit: either an 8% stop-loss (if it turns), or "
                "after 20 trading days (harvest the gain)."
            ),
        },
    }

    # ── Dynamic plain-English results commentary ──────────────────────────────
    strategy_reports = {k: v for k, v in reports.items() if k != "Benchmark"}
    benchmark = reports.get("Benchmark")

    best_name = max(strategy_reports, key=lambda k: strategy_reports[k].total_return)
    best = strategy_reports[best_name]
    safest_name = min(strategy_reports, key=lambda k: strategy_reports[k].max_drawdown)
    safest = strategy_reports[safest_name]

    beat_benchmark = [k for k, r in strategy_reports.items()
                      if benchmark and r.total_return > benchmark.total_return]

    bm_return_str = f"{benchmark.total_return:.1%}" if benchmark else "N/A"
    bm_cagr_str   = f"{benchmark.cagr:.1%}"         if benchmark else "N/A"

    verdict_color = "#06d6a0" if best.total_return > 0 else "#f59e0b"
    verdict = (
        f"The best-performing strategy was <b>{best_name.replace('_', ' ').title()}</b> "
        f"with a total return of <b>{best.total_return:.1%}</b> "
        f"({best.cagr:.1%}/year on average). "
    )
    if beat_benchmark:
        verdict += (
            f"It outperformed the S&P 500 benchmark ({bm_return_str} total). "
        )
    else:
        verdict += (
            f"However, none of the strategies beat simply holding the S&P 500 index "
            f"(SPY), which returned <b>{bm_return_str}</b> total "
            f"({bm_cagr_str}/year). That's a useful reminder: beating the market "
            f"is genuinely hard — even for sophisticated algorithms."
        )

    safest_note = (
        f"The most stable strategy was <b>{safest_name.replace('_', ' ').title()}</b>, "
        f"whose worst losing streak from peak to trough was only "
        f"<b>{safest.max_drawdown:.1%}</b> — much smaller than the others."
    )

    # ── Metric glossary ───────────────────────────────────────────────────────
    glossary = [
        ("Total Return",
         "How much money the strategy made (or lost) over the entire test period. "
         "+35% means $100k grew to $135k. -28% means it shrank to $72k."),
        ("CAGR",
         "Compound Annual Growth Rate — your average yearly return. Like an interest "
         "rate that compounds. 10% CAGR means your money roughly doubles every 7 years."),
        ("Annual Volatility",
         "How wildly the portfolio swings day-to-day, measured per year. Low = smooth "
         "ride. High = stomach-churning ups and downs. The S&P 500 is typically ~15–20%."),
        ("Sharpe Ratio",
         "Return per unit of risk. Above 1.0 = good. Above 2.0 = excellent. Below 0 "
         "means you'd have been better off in a savings account. It's the single most "
         "commonly used measure of risk-adjusted performance."),
        ("Sortino Ratio",
         "Like the Sharpe Ratio, but only penalises downside swings (bad volatility). "
         "A strategy with big gains but few big losses scores better here than on Sharpe."),
        ("Max Drawdown",
         "The worst peak-to-trough loss you would have experienced. If it's -50%, "
         "your portfolio once fell by half before recovering. Think: could you sleep "
         "through that without panic-selling?"),
        ("Max Drawdown Duration",
         "How many days the portfolio spent underwater — below its previous peak. "
         "A 2-year duration means you'd have waited 2 years just to get back to even."),
        ("Win Rate",
         "The percentage of trading days (or trades) that made money. 50% is a coin "
         "flip. Profitable systems can win with less than 50% if their winners are "
         "much bigger than their losers."),
        ("Profit Factor",
         "Total profits divided by total losses. 1.5 means for every $1 lost, "
         "$1.50 was made. Above 1.0 = profitable overall. Below 1.0 = losing money."),
        ("VaR 95%",
         "Value at Risk — on a typical bad day (the worst 5% of days), how much "
         "might you lose? A VaR of 1% means there's a 5% chance of losing more "
         "than 1% in a single day."),
        ("Skewness",
         "Whether bad days are worse than good days are good, or vice versa. "
         "Negative skew (common in trading) means rare but severe crashes. "
         "Positive skew means occasional big wins."),
        ("Kurtosis",
         "How often extreme moves happen. Higher kurtosis = more 'fat tails' — "
         "more frequent dramatic crashes or spikes than a normal distribution "
         "would predict."),
        ("Kelly Fraction",
         "A formula for how much of your capital to bet per trade for optimal growth. "
         "In practice, most traders use half-Kelly or less. A negative Kelly means "
         "don't bet at all — the edge is negative."),
        ("Information Ratio",
         "How consistently the strategy outperforms the benchmark. Higher = more "
         "reliably beating the market rather than just occasionally."),
    ]

    # ── Build HTML ────────────────────────────────────────────────────────────
    html = """
<div class="explainer-section">
<h2>📖 What Does All This Mean? (Beginner's Guide)</h2>

<div class="card">
  <h3 style="color:#06d6a0;margin-top:0;">What is a Backtest?</h3>
  <p>
    A <b>backtest</b> is a simulation. We take a trading strategy, feed it 10 years of
    real historical stock prices, and watch what would have happened if we'd followed
    that strategy every day — automatically, with no emotions. The goal is to see
    whether the strategy would have made money before risking real cash on it.
  </p>
  <p>
    Think of it like a flight simulator for trading. It's not a guarantee of future
    results, but it's a much safer way to evaluate an idea than just jumping in.
  </p>
  <p style="color:#f59e0b;">
    ⚠️ <b>Important caveat:</b> Real trading involves psychological pressure, taxes,
    larger transaction costs, and the fact that once everyone knows about a strategy,
    it stops working. Past backtested results are <i>not</i> a promise of future returns.
  </p>
</div>

<h3 style="color:#3b82f6;">The Strategies, Explained Simply</h3>
<div class="strategy-grid">
"""

    for key, info in strategy_info.items():
        if key in reports:
            r = reports[key]
            return_color = "#06d6a0" if r.total_return > 0 else "#ef4444"
            sharpe_note = (
                "solid" if r.sharpe_ratio > 0.5
                else ("breakeven" if r.sharpe_ratio > 0 else "negative")
            )
            html += f"""
  <div class="card strategy-card">
    <div class="strategy-header">
      <span class="strategy-emoji">{info['emoji']}</span>
      <span class="strategy-title">{info['title']}</span>
      <span class="strategy-return" style="color:{return_color};">{r.total_return:.1%} total</span>
    </div>
    <p style="color:#94a3b8;margin:8px 0;">{info['plain']}</p>
    <div class="strategy-verdict">
      Result: <b style="color:{return_color};">{r.total_return:.1%}</b> over the period
      &nbsp;|&nbsp; Sharpe: <b>{r.sharpe_ratio:.2f}</b> ({sharpe_note})
      &nbsp;|&nbsp; Worst dip: <b style="color:#f59e0b;">{r.max_drawdown:.1%}</b>
    </div>
  </div>"""

    html += f"""
</div>

<div class="card" style="border-color:#3b82f6;">
  <h3 style="color:#3b82f6;margin-top:0;">🏆 Plain-English Summary of Results</h3>
  <p style="color:{verdict_color};">{verdict}</p>
  <p style="color:#94a3b8;">{safest_note}</p>
"""
    if benchmark:
        html += f"""
  <p style="color:#94a3b8;">
    <b>The SPY benchmark</b> is the simplest possible strategy: just buy and hold
    the S&P 500 index. It returned <b style="color:#06d6a0;">{benchmark.total_return:.1%}</b>
    ({benchmark.cagr:.1%}/year) with a Sharpe of <b>{benchmark.sharpe_ratio:.2f}</b>.
    Beating it consistently is the bar that even most professional fund managers fail to clear.
  </p>"""

    iren_bh = reports.get("IREN (B&H)")
    if iren_bh:
        iren_color = "#06d6a0" if iren_bh.total_return > 0 else "#ef4444"
        html += f"""
  <p style="color:#94a3b8;">
    <b>The IREN benchmark</b> is a buy-and-hold of IREN (Iris Energy), a Bitcoin mining
    company listed since late 2021. It returned
    <b style="color:{iren_color};">{iren_bh.total_return:.1%}</b>
    ({iren_bh.cagr:.1%}/year) with a Sharpe of <b>{iren_bh.sharpe_ratio:.2f}</b>
    and a max drawdown of <b style="color:#f59e0b;">{iren_bh.max_drawdown:.1%}</b>.
    Crypto-adjacent stocks like IREN can produce dramatic gains or losses —
    high reward but extremely high risk.
  </p>"""

    html += """
</div>

<h3 style="color:#3b82f6;">📚 Metric Glossary</h3>
<table>
  <thead><tr><th style="width:200px;">Term</th><th>What it means in plain English</th></tr></thead>
  <tbody>
"""
    for term, explanation in glossary:
        html += f"<tr><td><b>{term}</b></td><td style='color:#94a3b8;'>{explanation}</td></tr>\n"

    html += """
  </tbody>
</table>
</div>
"""
    return html


def _generate_html_report(reports: Dict[str, RiskReport], results, out_dir: str):
    """Generate a standalone HTML report."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Quant Engine — Backtest Report</title>
<style>
body { background: #0a0e17; color: #e2e8f0; font-family: 'Courier New', monospace; padding: 20px; max-width: 1400px; margin: 0 auto; }
h1 { color: #06d6a0; } h2 { color: #3b82f6; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }
h3 { color: #e2e8f0; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; }
th { background: #1e293b; color: #94a3b8; padding: 8px 12px; text-align: left; font-size: 11px; }
td { padding: 6px 12px; border-bottom: 1px solid #1e293b; font-size: 12px; }
tr:hover td { background: #111827; }
.positive { color: #06d6a0; } .negative { color: #ef4444; }
img { max-width: 100%; border-radius: 8px; margin: 12px 0; }
.card { background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 16px; margin: 12px 0; }
p { line-height: 1.7; }
.strategy-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 12px; }
.strategy-card { display: flex; flex-direction: column; }
.strategy-header { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.strategy-emoji { font-size: 20px; }
.strategy-title { font-weight: bold; color: #e2e8f0; flex: 1; }
.strategy-return { font-weight: bold; font-size: 13px; }
.strategy-verdict { font-size: 11px; color: #64748b; margin-top: 8px; padding-top: 8px; border-top: 1px solid #1e293b; }
.explainer-section { margin-bottom: 40px; }
</style></head><body>
<h1>Σ QUANT ALPHA ENGINE</h1>
<p style="color:#64748b;">Systematic Multi-Strategy Backtest Report</p>
"""

    # Beginner explainer section
    html += _build_explainer_section(reports)

    # Summary table
    html += "<h2>📈 Technical Performance Summary</h2><table><thead><tr><th>Metric</th>"
    for name in reports:
        html += f"<th>{name}</th>"
    html += "</tr></thead><tbody>"

    metrics = [
        ("Total Return", "total_return", "{:.2%}"),
        ("CAGR", "cagr", "{:.2%}"),
        ("Annual Vol", "annualized_vol", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.3f}"),
        ("Calmar Ratio", "calmar_ratio", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("VaR 95%", "var_95", "{:.4%}"),
        ("CVaR 95%", "cvar_95", "{:.4%}"),
        ("Win Rate", "win_rate", "{:.2%}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Kelly f*", "kelly_fraction", "{:.3f}"),
        ("Skewness", "skewness", "{:.3f}"),
        ("Kurtosis", "kurtosis", "{:.3f}"),
    ]

    for label, key, fmt in metrics:
        html += f"<tr><td><b>{label}</b></td>"
        for r in reports.values():
            val = getattr(r, key)
            css = "positive" if val > 0 and key in ("total_return", "cagr", "sharpe_ratio") else ""
            css = "negative" if val < 0 and key in ("total_return", "cagr", "max_drawdown") else css
            html += f'<td class="{css}">{fmt.format(val)}</td>'
        html += "</tr>"
    html += "</tbody></table>"

    # Charts
    html += '<h2>Dashboard</h2><img src="dashboard.png" alt="Dashboard">'

    for name in results:
        safe = name.replace(" ", "_").replace("/", "_")[:30]
        monthly_file = f"monthly_{safe}.png"
        if os.path.exists(os.path.join(out_dir, monthly_file)):
            html += f'<h2>Monthly Returns — {name}</h2><img src="{monthly_file}">'

    html += "<p style='color:#475569;margin-top:40px;'>Generated by Quant Alpha Engine • For educational/research purposes only</p>"
    html += "</body></html>"

    path = os.path.join(out_dir, "report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"  📄 Saved HTML report → {path}")
