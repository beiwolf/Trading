# Σ Quant Alpha Engine

**Multi-strategy systematic backtesting platform with real market data.**  
Free, local, no API keys needed.

---

## Quick Start

```bash
# 1. Clone or copy this folder
cd quant-engine

# 2. Install dependencies (all free/open-source)
pip install -r requirements.txt

# 3. Run all strategies with default settings
python main.py

# 4. Open the report
open reports/report.html          # macOS
xdg-open reports/report.html     # Linux
start reports/report.html         # Windows
```

That's it. First run downloads ~10 years of data for 25+ stocks from Yahoo Finance (free) and caches it locally.

---

## What's Inside

### 5 Hedge Fund-Grade Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| **Mean Reversion** | Market-neutral | Ornstein-Uhlenbeck model. Trades when Z-score exceeds ±2σ from rolling mean. Volume filter + max hold constraint. |
| **Momentum** | Directional | Dual moving average crossover (10/50) with volatility-scaled position sizing. Inspired by AHL/AQR trend systems. |
| **Pairs Trading** | Market-neutral | Finds cointegrated pairs via correlation screening. OLS hedge ratio, half-life filter, rolling re-estimation. |
| **Stat Arb (PCA)** | Market-neutral | Decomposes returns via PCA into market factors + idiosyncratic residuals. Trades mean-reversion of residuals. DE Shaw / Two Sigma style. |
| **Multi-Factor** | Long/short | Ranks stocks on Momentum + Value + Low Vol + Quality. Longs top quintile, shorts bottom. Fama-French inspired. |

### Risk Engine (28 Metrics)

- Sharpe, Sortino, Calmar, Information Ratio
- VaR & CVaR at 95% and 99%
- Max drawdown & duration
- Win rate, profit factor, Kelly criterion
- Skewness, kurtosis (tail risk)
- Rolling analytics & correlation matrix

### Portfolio Optimization

| Method | Description |
|--------|-------------|
| `equal_weight` | 1/N allocation. Robust baseline. |
| `risk_parity` | Inverse-volatility weighting. Equalizes risk contribution. |
| `max_sharpe` | Monte Carlo search for tangency portfolio. |
| `half_kelly` | Kelly criterion × 0.5 for geometric growth with safety. |

### Reports

- **`dashboard.png`** — Master chart: equity curves, drawdowns, distributions, rolling Sharpe, correlation heatmap
- **`monthly_*.png`** — Monthly returns heatmaps per strategy
- **`report.html`** — Standalone HTML report with everything

---

## Usage Examples

```bash
# Run specific strategies
python main.py --strategies mom mr          # Momentum + Mean Reversion
python main.py -s pairs pca                 # Pairs + Stat Arb
python main.py -s factor                    # Multi-Factor only

# Custom date range
python main.py --start 2020-01-01 --end 2024-12-31

# Custom tickers
python main.py --tickers AAPL MSFT GOOGL AMZN META NVDA TSLA

# Custom capital & optimization
python main.py --capital 500000 --optimize max_sharpe

# Force fresh data (skip cache)
python main.py --no-cache

# Full options
python main.py --help
```

### Strategy Shortcuts

| Shortcut | Strategy |
|----------|----------|
| `mr`, `mean_rev` | Mean Reversion |
| `mom`, `trend` | Momentum |
| `pairs`, `pt` | Pairs Trading |
| `sa`, `pca` | Stat Arb |
| `factor`, `mf`, `ff` | Multi-Factor |

---

## Customization

### Edit `config.py` to change:

- **Ticker universe** — Add/remove stocks, change sectors
- **Strategy parameters** — Lookback windows, Z-score thresholds, MA periods
- **Risk limits** — Max drawdown, daily loss limits, VaR confidence
- **Portfolio settings** — Capital, commission, slippage, leverage limits
- **Optimization** — Method, rebalance frequency

### Add Your Own Strategy

1. Create a function in `strategies.py`:

```python
def my_strategy(data: dict, params: dict) -> pd.DataFrame:
    """
    Args:
        data: dict with 'close', 'returns', 'volume', indicators, etc.
        params: dict of strategy parameters from config

    Returns:
        pd.DataFrame of weights (date × ticker)
        Positive = long, negative = short
    """
    close = data["close"]
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    # ... your logic here ...
    return weights
```

2. Register it in the `STRATEGIES` dict at the bottom of `strategies.py`
3. Add parameters to `config.py`
4. Run: `python main.py -s my_strategy`

---

## Architecture

```
quant-engine/
├── main.py              # Entry point & CLI
├── config.py            # All parameters (edit this)
├── data_fetcher.py      # Yahoo Finance data + caching + indicators
├── strategies.py        # 5 strategy implementations
├── engine.py            # Backtesting core (PnL, costs, constraints)
├── risk_analytics.py    # 28 risk metrics + rolling analytics
├── report_generator.py  # Charts (matplotlib) + HTML report
├── requirements.txt     # Dependencies (all free)
├── .cache/              # Auto-created: cached market data
└── reports/             # Auto-created: output charts & reports
```

### Data Flow

```
Yahoo Finance (free)
       │
       ▼
  data_fetcher.py  ──→  Cache (.parquet)
       │
       ▼
  strategies.py    ──→  Target weights (date × ticker)
       │
       ▼
  engine.py        ──→  Simulated execution (costs, slippage, limits)
       │
       ▼
  risk_analytics.py ──→  28 risk metrics per strategy
       │
       ▼
  report_generator.py ──→  PNG charts + HTML report
```

---

## Cost Breakdown

| Component | Cost |
|-----------|------|
| Market data (Yahoo Finance) | **Free** |
| Python + all libraries | **Free** (open source) |
| Compute | **Your machine** |
| Total | **$0** |

---

## Limitations & Next Steps

**Current limitations:**
- Daily data only (no intraday)
- US equities focus (yfinance supports international too)
- No options/futures strategies (yet)
- Simplified slippage model (flat bps, not volume-dependent)

**Possible extensions:**
- Live paper trading via Alpaca API (free)
- Intraday data via Polygon.io ($29/mo) or IEX Cloud
- Options strategies using yfinance options chain data
- Machine learning signals (scikit-learn, PyTorch)
- Event-driven execution via Backtrader or Zipline

---

⚠️ **Disclaimer:** This is for educational and research purposes only. Not financial advice. Past performance does not indicate future results. Always do your own research.
