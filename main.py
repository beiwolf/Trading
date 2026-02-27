#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   Σ  QUANT ALPHA ENGINE  v1.0                                        ║
║                                                                      ║
║   Multi-Strategy Systematic Backtesting Platform                     ║
║   Free • Local • Real Market Data via Yahoo Finance                  ║
║                                                                      ║
║   Strategies:                                                        ║
║   ├─ Mean Reversion (Ornstein-Uhlenbeck / Bollinger)                 ║
║   ├─ Momentum / Trend Following (Dual MA + Vol Scaling)              ║
║   ├─ Pairs Trading (Cointegration + Half-Life Filter)                ║
║   ├─ Statistical Arbitrage (PCA Factor Decomposition)                ║
║   └─ Multi-Factor Model (Momentum + Value + Quality + Low Vol)       ║
║                                                                      ║
║   Usage:                                                             ║
║     python main.py                    # Run all strategies           ║
║     python main.py --strategies mom   # Run momentum only            ║
║     python main.py --help             # Show all options             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from config import *
from data_fetcher import fetch_prices, compute_indicators
from strategies import STRATEGIES
from engine import run_backtest, run_benchmark, combine_strategies
from report_generator import generate_full_report


STRATEGY_ALIASES = {
    "mr": "mean_reversion",
    "mean_rev": "mean_reversion",
    "mom": "momentum",
    "trend": "momentum",
    "pairs": "pairs_trading",
    "pt": "pairs_trading",
    "sa": "stat_arb",
    "pca": "stat_arb",
    "factor": "multi_factor",
    "mf": "multi_factor",
    "ff": "multi_factor",
    # New strategies
    "t200": "trend_200",
    "ma200": "trend_200",
    "sr": "sector_rotation",
    "rotate": "sector_rotation",
    "bo": "breakout",
    "new52": "breakout",
    # Wave 2 strategies
    "lv": "low_volatility",
    "lowvol": "low_volatility",
    "on": "overnight_drift",
    "overnight": "overnight_drift",
    "am": "adaptive_momentum",
    "adapt": "adaptive_momentum",
}

# Keys treated as benchmarks — excluded from portfolio combining
BENCHMARK_KEYS = {"IREN (B&H)"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quant Alpha Engine — Systematic Backtesting Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy shortcuts:
  mr, mean_rev    → Mean Reversion
  mom, trend      → Momentum / Trend Following
  pairs, pt       → Pairs Trading
  sa, pca         → Statistical Arbitrage (PCA)
  factor, mf, ff  → Multi-Factor Model

Examples:
  python main.py                              # Run all strategies
  python main.py --strategies mom mr          # Run momentum + mean reversion
  python main.py --start 2020-01-01           # Custom date range
  python main.py --capital 500000             # Custom capital
  python main.py --tickers AAPL MSFT GOOGL    # Custom ticker universe
  python main.py --optimize max_sharpe        # Portfolio optimization method
  python main.py --no-cache                   # Force fresh data download
        """,
    )
    parser.add_argument(
        "--strategies", "-s", nargs="+", default=None,
        help="Strategies to run (default: all). Use shortcuts like 'mom', 'mr', etc.",
    )
    parser.add_argument("--start", default=DATA_START, help=f"Start date (default: {DATA_START})")
    parser.add_argument("--end", default=DATA_END, help=f"End date (default: {DATA_END})")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help=f"Starting capital (default: ${INITIAL_CAPITAL:,})")
    parser.add_argument("--benchmark", default=BENCHMARK, help=f"Benchmark ticker (default: {BENCHMARK})")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Custom ticker universe (default: uses config.py universe)",
    )
    parser.add_argument(
        "--optimize", default=OPTIMIZATION["method"],
        choices=["equal_weight", "risk_parity", "max_sharpe", "half_kelly"],
        help="Portfolio optimization method",
    )
    parser.add_argument("--no-cache", action="store_true", help="Force fresh data download")
    parser.add_argument("--output", "-o", default="reports", help="Output directory for reports")
    return parser.parse_args()


def resolve_strategies(strategy_args):
    """Resolve strategy names/aliases to full keys."""
    if strategy_args is None:
        return list(STRATEGIES.keys())

    resolved = []
    for s in strategy_args:
        s_lower = s.lower()
        if s_lower in STRATEGIES:
            resolved.append(s_lower)
        elif s_lower in STRATEGY_ALIASES:
            resolved.append(STRATEGY_ALIASES[s_lower])
        else:
            print(f"  ⚠️  Unknown strategy: '{s}'. Available: {list(STRATEGIES.keys())}")
            print(f"       Shortcuts: {list(STRATEGY_ALIASES.keys())}")
            sys.exit(1)
    return list(set(resolved))


def get_strategy_params(key):
    """Get params dict for a strategy from config."""
    param_map = {
        "mean_reversion": MEAN_REV,
        "momentum": MOMENTUM,
        "pairs_trading": PAIRS,
        "stat_arb": STAT_ARB,
        "multi_factor": FACTOR_MODEL,
        "trend_200": TREND_200,
        "sector_rotation": SECTOR_ROTATION,
        "breakout": BREAKOUT,
        "low_volatility": LOW_VOLATILITY,
        "overnight_drift": OVERNIGHT_DRIFT,
        "adaptive_momentum": ADAPTIVE_MOMENTUM,
    }
    return param_map.get(key, {})


def main():
    args = parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║  Σ  QUANT ALPHA ENGINE  v1.0                        ║")
    print("  ║  Multi-Strategy Systematic Backtesting               ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()

    t0 = time.time()

    # ─── Resolve strategies ──────────────────────────────────
    strat_keys = resolve_strategies(args.strategies)
    print(f"  📋 Strategies: {', '.join(strat_keys)}")
    print(f"  📅 Period: {args.start} → {args.end}")
    print(f"  💰 Capital: ${args.capital:,.0f}")
    print(f"  📈 Benchmark: {args.benchmark}")
    print(f"  🔧 Optimizer: {args.optimize}")
    print()

    # ─── Fetch market data ───────────────────────────────────
    tickers = args.tickers or (ALL_TICKERS + [args.benchmark])
    tickers = list(set(tickers))

    print("  ━━━ STEP 1: Fetching Market Data ━━━")
    data = fetch_prices(
        tickers=tickers,
        start=args.start,
        end=args.end,
        use_cache=not args.no_cache,
    )
    data = compute_indicators(data)
    print()

    # ─── Run strategies ──────────────────────────────────────
    print("  ━━━ STEP 2: Running Strategies ━━━")
    backtest_results = {}

    for key in strat_keys:
        strat = STRATEGIES[key]
        params = get_strategy_params(key)
        print(f"\n  ▶ {strat['name']}")
        print(f"    {strat['description']}")

        try:
            weights = strat["fn"](data, params)

            result = run_backtest(
                weights=weights,
                data=data,
                initial_capital=args.capital,
                commission_bps=COMMISSION_BPS,
                slippage_bps=SLIPPAGE_BPS,
                max_position_pct=MAX_POSITION_PCT,
                max_gross_leverage=MAX_GROSS_LEVERAGE,
                strategy_name=strat["name"],
            )
            backtest_results[key] = result

            # Quick stats
            ret = result.returns
            total_ret = (1 + ret).prod() - 1
            sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
            max_dd = (((1 + ret).cumprod() / (1 + ret).cumprod().cummax()) - 1).min()
            print(f"    ✅ Return: {total_ret:.2%} | Sharpe: {sharpe:.3f} | Max DD: {max_dd:.2%}")
            print(f"    📊 {len(result.trade_log)} trades | ${result.total_costs:,.0f} costs")

        except Exception as e:
            print(f"    ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    if not backtest_results:
        print("\n  ❌ No strategies produced results. Exiting.")
        sys.exit(1)

    # ─── Run benchmarks ──────────────────────────────────────
    print(f"\n  ▶ Benchmark ({args.benchmark})")
    try:
        benchmark_result = run_benchmark(data, args.benchmark, args.capital)
        bm_ret = benchmark_result.returns
        bm_total = (1 + bm_ret).prod() - 1
        bm_sharpe = bm_ret.mean() / bm_ret.std() * np.sqrt(252) if bm_ret.std() > 0 else 0
        print(f"    ✅ Return: {bm_total:.2%} | Sharpe: {bm_sharpe:.3f}")
    except Exception as e:
        print(f"    ⚠️  Could not compute benchmark: {e}")
        benchmark_result = None

    # ─── IREN buy-and-hold benchmark ─────────────────────────
    print(f"\n  ▶ IREN (Buy & Hold — crypto miner benchmark)")
    try:
        iren_result = run_benchmark(data, "IREN", args.capital)
        ir_ret = iren_result.returns
        ir_total = (1 + ir_ret).prod() - 1
        ir_sharpe = ir_ret.mean() / ir_ret.std() * np.sqrt(252) if ir_ret.std() > 0 else 0
        print(f"    ✅ Return: {ir_total:.2%} | Sharpe: {ir_sharpe:.3f}")
        backtest_results["IREN (B&H)"] = iren_result
    except Exception as e:
        print(f"    ⚠️  Could not compute IREN benchmark: {e}")

    # ─── Combine strategies (exclude benchmark buy-and-holds) ─
    strategy_results_only = {
        k: v for k, v in backtest_results.items() if k not in BENCHMARK_KEYS
    }
    if len(strategy_results_only) > 1:
        print(f"\n  ━━━ STEP 3: Portfolio Optimization ({args.optimize}) ━━━")
        try:
            combined = combine_strategies(
                strategy_results_only,
                method=args.optimize,
                lookback=OPTIMIZATION["lookback_for_weights"],
                rebalance_days=OPTIMIZATION["rebalance_days"],
            )
            backtest_results["combined"] = combined

            comb_ret = combined.returns
            comb_total = (1 + comb_ret).prod() - 1
            comb_sharpe = comb_ret.mean() / comb_ret.std() * np.sqrt(252) if comb_ret.std() > 0 else 0
            print(f"    ✅ Combined Return: {comb_total:.2%} | Sharpe: {comb_sharpe:.3f}")
        except Exception as e:
            print(f"    ⚠️  Could not combine: {e}")

    # ─── Generate reports ────────────────────────────────────
    print(f"\n  ━━━ STEP 4: Generating Reports ━━━")
    reports = generate_full_report(
        results=backtest_results,
        benchmark_result=benchmark_result,
        risk_free_rate=RISK_FREE_RATE,
        output_dir=args.output,
    )

    # ─── Done ────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  ✅ Complete in {elapsed:.1f}s")
    print(f"  📁 Reports saved to: ./{args.output}/")
    print(f"     ├─ dashboard.png       (master chart)")
    print(f"     ├─ monthly_*.png       (monthly heatmaps)")
    print(f"     └─ report.html         (full HTML report)")
    print(f"\n  Open report:  open ./{args.output}/report.html")
    print()
    print("  ⚠️  DISCLAIMER: Educational/research only. Not financial advice.")
    print("  Past performance does not indicate future results.")
    print()


if __name__ == "__main__":
    import numpy as np
    main()
