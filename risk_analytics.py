"""
╔══════════════════════════════════════════════════════════════════╗
║  QUANT ENGINE — Risk Analytics                                  ║
║  Comprehensive risk metrics & performance attribution           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict


@dataclass
class RiskReport:
    """Full risk analytics for a strategy."""
    name: str
    # Returns
    total_return: float
    cagr: float
    annualized_vol: float
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int
    avg_drawdown: float
    # Tail risk
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    # Distribution
    skewness: float
    kurtosis: float
    # Trading
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    best_day: float
    worst_day: float
    # Kelly
    kelly_fraction: float
    # Turnover
    avg_daily_turnover: float
    total_costs: float
    # Time
    n_days: int
    pct_time_in_market: float


def compute_risk_report(
    result,  # BacktestResult
    risk_free_rate: float = 0.05,
    benchmark_returns: pd.Series = None,
) -> RiskReport:
    """Compute comprehensive risk analytics from backtest results."""

    ret = result.returns.dropna()
    if len(ret) < 10:
        return _empty_report(result.strategy_name)

    n_days = len(ret)
    n_years = n_days / 252

    # ─── Basic returns ───────────────────────────────────────
    total_return = (1 + ret).prod() - 1
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    daily_rf = risk_free_rate / 252

    # ─── Risk-adjusted metrics ───────────────────────────────
    excess = ret - daily_rf
    sharpe = (excess.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0

    # Sortino: downside deviation
    downside = ret[ret < daily_rf] - daily_rf
    downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) > 0 else 0.001
    sortino = (cagr - risk_free_rate) / downside_dev if downside_dev > 0 else 0

    # Max drawdown
    cum = (1 + ret).cumprod()
    running_max = cum.cummax()
    drawdowns = cum / running_max - 1
    max_dd = drawdowns.min()

    # Max drawdown duration
    underwater = drawdowns < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_lengths = underwater.groupby(groups).sum()
        max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
    else:
        max_dd_duration = 0

    avg_dd = drawdowns[drawdowns < 0].mean() if (drawdowns < 0).any() else 0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    # Information ratio (vs benchmark)
    if benchmark_returns is not None:
        common = ret.index.intersection(benchmark_returns.index)
        if len(common) > 10:
            active = ret.loc[common] - benchmark_returns.loc[common]
            te = active.std() * np.sqrt(252)
            ir = (active.mean() * 252) / te if te > 0 else 0
        else:
            ir = 0
    else:
        ir = 0

    # ─── Tail risk ───────────────────────────────────────────
    sorted_ret = ret.sort_values()
    n = len(sorted_ret)
    var_95 = -sorted_ret.iloc[int(n * 0.05)] if n > 20 else 0
    cvar_95 = -sorted_ret.iloc[:int(n * 0.05)].mean() if n > 20 else 0
    var_99 = -sorted_ret.iloc[int(n * 0.01)] if n > 100 else 0
    cvar_99 = -sorted_ret.iloc[:int(n * 0.01)].mean() if n > 100 else 0

    # ─── Distribution ────────────────────────────────────────
    skew = float(stats.skew(ret.dropna()))
    kurt = float(stats.kurtosis(ret.dropna()))

    # ─── Trading metrics ─────────────────────────────────────
    wins = ret[ret > 0]
    losses = ret[ret < 0]
    win_rate = len(wins) / n_days if n_days > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Kelly criterion
    sigma = ret.std()
    kelly = (ret.mean() / (sigma ** 2)) if sigma > 0 else 0
    kelly = np.clip(kelly, -5, 5)

    # Turnover
    avg_turnover = result.turnover.mean() if hasattr(result, "turnover") and len(result.turnover) > 0 else 0

    # Time in market
    if hasattr(result, "weights") and not result.weights.empty:
        active_days = (result.weights.abs().sum(axis=1) > 0.01).sum()
        pct_in_market = active_days / n_days if n_days > 0 else 0
    else:
        pct_in_market = 1.0

    return RiskReport(
        name=result.strategy_name,
        total_return=total_return,
        cagr=cagr,
        annualized_vol=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        information_ratio=ir,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        avg_drawdown=avg_dd,
        var_95=var_95,
        cvar_95=cvar_95,
        var_99=var_99,
        cvar_99=cvar_99,
        skewness=skew,
        kurtosis=kurt,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        best_day=float(ret.max()),
        worst_day=float(ret.min()),
        kelly_fraction=kelly,
        avg_daily_turnover=avg_turnover,
        total_costs=result.total_costs,
        n_days=n_days,
        pct_time_in_market=pct_in_market,
    )


def _empty_report(name):
    return RiskReport(
        name=name, total_return=0, cagr=0, annualized_vol=0,
        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, information_ratio=0,
        max_drawdown=0, max_drawdown_duration_days=0, avg_drawdown=0,
        var_95=0, cvar_95=0, var_99=0, cvar_99=0,
        skewness=0, kurtosis=0, win_rate=0, profit_factor=0,
        avg_win=0, avg_loss=0, best_day=0, worst_day=0,
        kelly_fraction=0, avg_daily_turnover=0, total_costs=0,
        n_days=0, pct_time_in_market=0,
    )


def compute_correlation_matrix(results: Dict[str, "BacktestResult"]) -> pd.DataFrame:
    """Compute return correlation matrix across strategies."""
    ret_df = pd.DataFrame({n: r.returns for n, r in results.items()}).dropna()
    return ret_df.corr()


def rolling_sharpe(returns: pd.Series, window: int = 63, rf: float = 0.05) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    daily_rf = rf / 252
    excess = returns - daily_rf
    rolling_mean = excess.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    return (rolling_mean / rolling_std * np.sqrt(252)).dropna()


def rolling_drawdown(returns: pd.Series) -> pd.Series:
    """Compute rolling drawdown series."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return cum / running_max - 1
