"""
╔══════════════════════════════════════════════════════════════════╗
║  QUANT ENGINE — Configuration                                   ║
║  Edit this file to customize your backtesting universe          ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─── DATA SETTINGS ───────────────────────────────────────────────
DATA_START = "2015-01-01"       # Backtest start date
DATA_END = "2025-01-01"        # Backtest end date (or "today")
BENCHMARK = "SPY"               # Benchmark ticker for comparison

# Universe of tickers to trade
# Diversified across sectors for stat arb and pairs strategies
UNIVERSE = {
    "tech":       ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "finance":    ["JPM", "GS", "BAC", "MS", "C"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "energy":     ["XOM", "CVX", "COP", "SLB", "EOG"],
    "consumer":   ["AMZN", "WMT", "COST", "PG", "KO"],
}

# Flatten for convenience
ALL_TICKERS = [t for sector in UNIVERSE.values() for t in sector]

# ─── PORTFOLIO SETTINGS ─────────────────────────────────────────
INITIAL_CAPITAL = 1_000_000     # Starting capital ($)
COMMISSION_BPS = 5              # Commission in basis points (5 bps = 0.05%)
SLIPPAGE_BPS = 3                # Estimated slippage in basis points
MAX_POSITION_PCT = 0.10         # Max single position as % of portfolio
MAX_GROSS_LEVERAGE = 2.0        # Max gross exposure (long + |short|)
MAX_NET_LEVERAGE = 0.3          # Max net exposure (|long - short|)
RISK_FREE_RATE = 0.05           # Annual risk-free rate for Sharpe calc

# ─── MEAN REVERSION STRATEGY ────────────────────────────────────
MEAN_REV = {
    "lookback": 20,             # Rolling window for mean/std
    "entry_z": 2.0,             # Z-score threshold to enter
    "exit_z": 0.5,              # Z-score threshold to exit
    "max_holding_days": 10,     # Force exit after N days
    "use_volume_filter": True,  # Only trade when volume > 1.5x avg
}

# ─── MOMENTUM / TREND FOLLOWING ─────────────────────────────────
MOMENTUM = {
    "fast_period": 10,          # Fast moving average period
    "slow_period": 50,          # Slow moving average period
    "atr_period": 14,           # ATR period for volatility filter
    "trend_strength_min": 0.01, # Min MA divergence to signal
    "use_adx_filter": True,     # Require ADX > 25 for entry
    "adx_period": 14,           # ADX calculation period
}

# ─── PAIRS TRADING ──────────────────────────────────────────────
PAIRS = {
    "lookback": 60,             # Cointegration lookback window
    "entry_z": 2.0,             # Spread Z-score entry threshold
    "exit_z": 0.5,              # Spread Z-score exit threshold
    "min_correlation": 0.7,     # Min correlation to qualify as pair
    "max_half_life": 30,        # Max mean-reversion half-life (days)
    "reestimate_every": 20,     # Re-fit hedge ratio every N days
}

# ─── STATISTICAL ARBITRAGE (PCA) ────────────────────────────────
STAT_ARB = {
    "lookback": 60,             # Rolling PCA window
    "n_components": 5,          # Number of PCA factors
    "entry_z": 1.5,             # Residual Z-score entry
    "exit_z": 0.3,              # Residual Z-score exit
    "sector_neutral": True,     # Force sector neutrality
}

# ─── MULTI-FACTOR MODEL ─────────────────────────────────────────
FACTOR_MODEL = {
    "momentum_window": 252,     # 12-month momentum lookback
    "skip_recent": 21,          # Skip most recent month (reversal)
    "value_metric": "pe_ratio", # Placeholder — uses price/book proxy
    "volatility_window": 60,    # Realized vol lookback
    "rebalance_frequency": 21,  # Rebalance every N trading days
    "long_pct": 0.20,           # Go long top 20%
    "short_pct": 0.20,          # Go short bottom 20%
}

# ─── PORTFOLIO OPTIMIZATION ─────────────────────────────────────
OPTIMIZATION = {
    "method": "risk_parity",    # equal_weight | risk_parity | max_sharpe | half_kelly
    "rebalance_days": 5,        # Rebalance combined portfolio every N days
    "lookback_for_weights": 60, # Days of strategy returns to estimate weights
}

# ─── RISK LIMITS ────────────────────────────────────────────────
RISK_LIMITS = {
    "max_drawdown_pct": 0.15,   # Halt trading if DD exceeds 15%
    "max_daily_loss_pct": 0.03, # Reduce exposure if daily loss > 3%
    "var_confidence": 0.95,     # VaR confidence level
    "position_vol_target": 0.15,# Target annualized vol per strategy
}
