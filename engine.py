"""
╔══════════════════════════════════════════════════════════════════╗
║  QUANT ENGINE — Backtesting Core                                ║
║  Simulates strategy execution with realistic friction           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    strategy_name: str
    weights: pd.DataFrame           # Target weights (date × ticker)
    returns: pd.Series              # Daily portfolio returns (after costs)
    equity_curve: pd.Series         # Cumulative equity ($)
    gross_returns: pd.Series        # Returns before costs
    turnover: pd.Series             # Daily turnover (single-sided)
    total_costs: float              # Total transaction costs ($)
    positions: pd.DataFrame         # Actual dollar positions over time
    trade_log: list                 # List of trade events


def run_backtest(
    weights: pd.DataFrame,
    data: dict,
    initial_capital: float = 1_000_000,
    commission_bps: float = 5,
    slippage_bps: float = 3,
    max_position_pct: float = 0.10,
    max_gross_leverage: float = 2.0,
    strategy_name: str = "Strategy",
) -> BacktestResult:
    """
    Execute a backtest given target weights and market data.

    This is an event-driven simulation that:
    1. Applies target weights at each rebalance
    2. Marks positions to market daily
    3. Deducts transaction costs (commission + slippage)
    4. Enforces position limits and leverage constraints

    Args:
        weights:  DataFrame of target portfolio weights (date × ticker)
        data:     Market data dict with 'close', 'returns' keys
        initial_capital: Starting capital
        commission_bps:  Commission per trade in basis points
        slippage_bps:    Slippage per trade in basis points
        max_position_pct: Max weight per position
        max_gross_leverage: Max sum of |weights|

    Returns:
        BacktestResult with all outputs
    """
    close = data["close"]
    returns = data["returns"]

    # Align dates
    common_dates = weights.index.intersection(returns.index)
    weights = weights.loc[common_dates]
    returns = returns.loc[common_dates]
    close = close.loc[common_dates]

    tickers = weights.columns.tolist()
    n_days = len(common_dates)

    cost_rate = (commission_bps + slippage_bps) / 10000

    # ─── Position & PnL tracking ─────────────────────────────
    equity = np.zeros(n_days)
    equity[0] = initial_capital
    daily_returns = np.zeros(n_days)
    gross_daily_returns = np.zeros(n_days)
    daily_turnover = np.zeros(n_days)
    total_costs = 0.0
    trade_log = []

    prev_weights = pd.Series(0.0, index=tickers)
    positions = pd.DataFrame(0.0, index=common_dates, columns=tickers)

    for i in range(1, n_days):
        # Current target weights
        target_w = weights.iloc[i].fillna(0)

        # ─── Apply constraints ───────────────────────────────
        # Position limit
        target_w = target_w.clip(-max_position_pct, max_position_pct)

        # Gross leverage limit
        gross = target_w.abs().sum()
        if gross > max_gross_leverage:
            target_w = target_w * (max_gross_leverage / gross)

        # ─── Compute turnover and costs ──────────────────────
        weight_change = (target_w - prev_weights).abs()
        turnover = weight_change.sum() / 2  # Single-sided
        cost = turnover * cost_rate * equity[i - 1]
        total_costs += cost

        # ─── Portfolio return ────────────────────────────────
        stock_returns = returns.iloc[i].fillna(0)
        port_return_gross = (prev_weights * stock_returns).sum()
        port_return_net = port_return_gross - (cost / equity[i - 1] if equity[i - 1] > 0 else 0)

        equity[i] = equity[i - 1] * (1 + port_return_net)
        daily_returns[i] = port_return_net
        gross_daily_returns[i] = port_return_gross
        daily_turnover[i] = turnover

        # Log significant trades
        for ticker in tickers:
            wc = abs(target_w.get(ticker, 0) - prev_weights.get(ticker, 0))
            if wc > 0.01:
                trade_log.append({
                    "date": common_dates[i],
                    "ticker": ticker,
                    "old_weight": prev_weights.get(ticker, 0),
                    "new_weight": target_w.get(ticker, 0),
                    "direction": "BUY" if target_w.get(ticker, 0) > prev_weights.get(ticker, 0) else "SELL",
                })

        # Record positions
        positions.iloc[i] = target_w * equity[i]
        prev_weights = target_w.copy()

    return BacktestResult(
        strategy_name=strategy_name,
        weights=weights,
        returns=pd.Series(daily_returns, index=common_dates, name=strategy_name),
        equity_curve=pd.Series(equity, index=common_dates, name=strategy_name),
        gross_returns=pd.Series(gross_daily_returns, index=common_dates),
        turnover=pd.Series(daily_turnover, index=common_dates),
        total_costs=total_costs,
        positions=positions,
        trade_log=trade_log,
    )


def run_benchmark(data: dict, benchmark: str, initial_capital: float) -> BacktestResult:
    """
    Run a simple buy-and-hold benchmark.
    """
    close = data["close"]
    if benchmark not in close.columns:
        # Need to fetch benchmark separately
        import yfinance as yf
        bm_data = yf.download(benchmark, start=close.index[0], end=close.index[-1], progress=False)
        bm_close = bm_data["Close"]
        bm_returns = np.log(bm_close / bm_close.shift(1)).dropna()
        common = close.index.intersection(bm_returns.index)
        bm_ret = bm_returns.loc[common]
    else:
        bm_ret = data["returns"][benchmark]
        common = bm_ret.index

    equity = initial_capital * (1 + bm_ret).cumprod()
    equity = pd.concat([pd.Series([initial_capital], index=[common[0] - pd.Timedelta(days=1)]), equity])

    return BacktestResult(
        strategy_name=f"Benchmark ({benchmark})",
        weights=pd.DataFrame(),
        returns=bm_ret,
        equity_curve=equity,
        gross_returns=bm_ret,
        turnover=pd.Series(0, index=common),
        total_costs=0,
        positions=pd.DataFrame(),
        trade_log=[],
    )


# ═══════════════════════════════════════════════════════════════
# PORTFOLIO COMBINER (Multi-Strategy Allocation)
# ═══════════════════════════════════════════════════════════════

def combine_strategies(
    results: Dict[str, BacktestResult],
    method: str = "risk_parity",
    lookback: int = 60,
    rebalance_days: int = 5,
) -> BacktestResult:
    """
    Combine multiple strategy return streams into an optimized portfolio.

    Methods:
        equal_weight: 1/N allocation
        risk_parity:  Inverse-volatility weighting
        max_sharpe:   Maximize Sharpe ratio (Monte Carlo)
        half_kelly:   Kelly criterion × 0.5
    """
    names = list(results.keys())
    ret_df = pd.DataFrame({n: results[n].returns for n in names}).dropna()

    if ret_df.empty:
        raise ValueError("No overlapping return data across strategies")

    n_strats = len(names)
    combined_returns = pd.Series(0.0, index=ret_df.index)
    allocation_weights = {}

    for i in range(lookback, len(ret_df), rebalance_days):
        window = ret_df.iloc[max(0, i - lookback):i]

        if method == "equal_weight":
            w = np.ones(n_strats) / n_strats

        elif method == "risk_parity":
            vols = window.std().values
            inv_vol = np.where(vols > 0, 1.0 / vols, 0)
            w = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else np.ones(n_strats) / n_strats

        elif method == "max_sharpe":
            best_sharpe = -np.inf
            best_w = np.ones(n_strats) / n_strats
            mu = window.mean().values * 252
            cov = window.cov().values * 252
            for _ in range(1000):
                rw = np.random.dirichlet(np.ones(n_strats))
                port_ret = rw @ mu
                port_vol = np.sqrt(rw @ cov @ rw)
                sharpe = port_ret / port_vol if port_vol > 0 else 0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_w = rw
            w = best_w

        elif method == "half_kelly":
            mu = window.mean().values
            var = window.var().values
            kelly = np.where(var > 0, mu / var, 0) * 0.5
            kelly = np.clip(kelly, -2, 2)
            total = np.abs(kelly).sum()
            w = kelly / total if total > 0 else np.ones(n_strats) / n_strats

        else:
            w = np.ones(n_strats) / n_strats

        # Apply weights for this period
        end_idx = min(i + rebalance_days, len(ret_df))
        for j in range(i, end_idx):
            combined_returns.iloc[j] = ret_df.iloc[j].values @ w

        allocation_weights[ret_df.index[i]] = dict(zip(names, w))

    equity = (1 + combined_returns).cumprod() * 1_000_000

    return BacktestResult(
        strategy_name=f"Combined ({method})",
        weights=pd.DataFrame(),
        returns=combined_returns,
        equity_curve=equity,
        gross_returns=combined_returns,
        turnover=pd.Series(0, index=ret_df.index),
        total_costs=0,
        positions=pd.DataFrame(),
        trade_log=[],
    )
