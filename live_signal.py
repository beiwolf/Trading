#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  SECTOR ROTATION — Live Signal Generator                        ║
║  Run on the first trading day of each month to get your         ║
║  current allocation. Tells you exactly what to buy and sell.    ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python live_signal.py                     # $100k portfolio, individual stocks
    python live_signal.py --capital 50000     # custom portfolio size
    python live_signal.py --etf               # use sector ETFs (simpler, lower cost)
    python live_signal.py --top 1             # concentrate in #1 sector only
    python live_signal.py --dry-run           # show signal without saving state
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(BASE_DIR, ".state")
STATE_FILE = os.path.join(STATE_DIR, "sector_rotation.json")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

# ─── Universe ─────────────────────────────────────────────────────────────────
SECTOR_STOCKS = {
    "tech":       ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "finance":    ["JPM", "GS", "BAC", "MS", "C"],
    "healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "energy":     ["XOM", "CVX", "COP", "SLB", "EOG"],
    "consumer":   ["AMZN", "WMT", "COST", "PG", "KO"],
}

# One ETF per sector — simpler alternative to buying 5 individual stocks
SECTOR_ETFS = {
    "tech":       "XLK",   # Technology Select Sector SPDR
    "finance":    "XLF",   # Financial Select Sector SPDR
    "healthcare": "XLV",   # Health Care Select Sector SPDR
    "energy":     "XLE",   # Energy Select Sector SPDR
    "consumer":   "XLP",   # Consumer Staples Select Sector SPDR
}

SECTOR_FULL_NAMES = {
    "tech":       "Technology",
    "finance":    "Financial",
    "healthcare": "Health Care",
    "energy":     "Energy",
    "consumer":   "Consumer",
}

LOOKBACK_DAYS = 63   # 3-month momentum signal (~63 trading days)


# ─── Data ─────────────────────────────────────────────────────────────────────

def fetch_data() -> pd.DataFrame:
    """Download last ~110 days of closing prices for the full universe."""
    all_tickers = sorted({t for tkrs in SECTOR_STOCKS.values() for t in tkrs}
                         | set(SECTOR_ETFS.values()) | {"SPY"})

    end   = datetime.today()
    start = end - timedelta(days=130)   # extra buffer for holidays/weekends

    print(f"  📡 Fetching data ({start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}) ...")
    raw = yf.download(
        all_tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError("No data from Yahoo Finance — check your internet connection.")

    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(name=close.name)

    close = close.ffill().dropna(how="all")
    print(f"  ✅ {len(close)} days loaded. Latest: {close.index[-1].strftime('%Y-%m-%d')}")
    return close


# ─── Signal ───────────────────────────────────────────────────────────────────

def rank_sectors(close: pd.DataFrame) -> list[dict]:
    """
    Score each sector by the average 3-month return of its stocks.
    Returns a list sorted from best to worst.
    """
    if len(close) < LOOKBACK_DAYS:
        raise ValueError(f"Need ≥ {LOOKBACK_DAYS} trading days of data, got {len(close)}")

    start_px = close.iloc[-LOOKBACK_DAYS]
    end_px   = close.iloc[-1]

    results = []
    for sector, tickers in SECTOR_STOCKS.items():
        valid = [t for t in tickers if t in close.columns]
        sp = start_px[valid].replace(0, np.nan)
        ep = end_px[valid]
        mask = sp.notna() & ep.notna() & (sp > 0)

        if mask.any():
            indiv = ((ep[mask] / sp[mask]) - 1).to_dict()
            avg_ret = float(np.mean(list(indiv.values())))
        else:
            indiv = {}
            avg_ret = np.nan

        # ETF return as a reference figure
        etf = SECTOR_ETFS[sector]
        etf_ret = None
        if etf in close.columns:
            s, e = start_px.get(etf, np.nan), end_px.get(etf, np.nan)
            if not (np.isnan(s) or np.isnan(e) or s == 0):
                etf_ret = float(e / s - 1)

        results.append({
            "sector":      sector,
            "name":        SECTOR_FULL_NAMES[sector],
            "avg_return":  avg_ret,
            "etf_ticker":  etf,
            "etf_return":  etf_ret,
            "stock_returns": indiv,
            "stocks":      valid,
        })

    results.sort(key=lambda x: x["avg_return"] if not np.isnan(x["avg_return"]) else -999,
                 reverse=True)
    return results


def build_allocation(ranked: list[dict], n_top: int, use_etf: bool) -> dict:
    """
    Determine what to hold.
    Only invests in sectors with positive momentum — stays in cash otherwise.
    """
    positive = [r for r in ranked[:n_top] if not np.isnan(r["avg_return"]) and r["avg_return"] > 0]

    if not positive:
        return {
            "status": "CASH",
            "sectors": {},
            "holdings": [],       # list of (ticker, weight)
            "message": "All top sectors have negative 3-month returns. Stay in cash or hold SPY/T-bills.",
        }

    if use_etf:
        w = 1.0 / len(positive)
        holdings = [(r["etf_ticker"], w) for r in positive]
    else:
        all_stocks = [t for r in positive for t in r["stocks"]]
        w = 1.0 / len(all_stocks)
        holdings = [(t, w) for t in all_stocks]

    return {
        "status": "INVESTED",
        "sectors": {r["sector"]: r["name"] for r in positive},
        "holdings": holdings,
        "message": f"Invest equally across all {'ETFs' if use_etf else 'stocks'} in the top {len(positive)} sector(s).",
    }


# ─── Trade diff ───────────────────────────────────────────────────────────────

def compute_trades(prev: dict, alloc: dict, close: pd.DataFrame, capital: float):
    """Compare new allocation to previous state and generate a trade list."""
    latest = close.iloc[-1]
    prev_h = {h["ticker"]: h["weight"] for h in prev.get("holdings", [])}
    new_h  = dict(alloc["holdings"])

    sells, buys = [], []
    for ticker in sorted(set(list(prev_h) + list(new_h))):
        pw = prev_h.get(ticker, 0.0)
        nw = new_h.get(ticker, 0.0)
        price = latest.get(ticker, np.nan)
        if pd.isna(price) or price <= 0:
            continue

        change_usd = (nw - pw) * capital
        if abs(change_usd) < 10:   # ignore rounding noise
            continue

        shares = abs(change_usd) / price
        entry = {"ticker": ticker, "shares": round(shares, 2),
                 "value": abs(change_usd), "price": float(price)}

        if change_usd < 0:
            entry["action"] = "SELL ALL" if nw == 0 else "TRIM"
            sells.append(entry)
        else:
            entry["action"] = "BUY" if pw == 0 else "ADD"
            buys.append(entry)

    return sells, buys


# ─── State persistence ────────────────────────────────────────────────────────

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(alloc: dict, close: pd.DataFrame, capital: float):
    os.makedirs(STATE_DIR, exist_ok=True)
    latest = close.iloc[-1]
    holdings = []
    for ticker, weight in alloc["holdings"]:
        price = latest.get(ticker, np.nan)
        holdings.append({
            "ticker": ticker,
            "weight": weight,
            "price":  float(price) if not pd.isna(price) else None,
            "value":  round(weight * capital, 2),
        })
    state = {
        "last_run":       datetime.today().strftime("%Y-%m-%d"),
        "status":         alloc["status"],
        "active_sectors": list(alloc["sectors"].keys()),
        "capital":        capital,
        "holdings":       holdings,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    return state


# ─── HTML report ──────────────────────────────────────────────────────────────

def generate_html(ranked, alloc, sells, buys, close, capital, use_etf, prev):
    today = datetime.today().strftime("%Y-%m-%d")
    latest = close.iloc[-1]
    max_abs = max((abs(r["avg_return"]) for r in ranked if not np.isnan(r["avg_return"])), default=1)

    lines = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Sector Rotation Signal — {today}</title>
<style>
body{{background:#0a0e17;color:#e2e8f0;font-family:'Courier New',monospace;padding:24px;max-width:960px;margin:0 auto;}}
h1{{color:#06d6a0;}} h2{{color:#3b82f6;border-bottom:1px solid #1e293b;padding-bottom:8px;margin-top:32px;}}
h3{{color:#e2e8f0;margin-top:18px;}}
.card{{background:#111827;border:1px solid #1e293b;border-radius:8px;padding:18px;margin:12px 0;}}
table{{border-collapse:collapse;width:100%;margin:10px 0;}}
th{{background:#1e293b;color:#94a3b8;padding:8px 12px;text-align:left;font-size:11px;}}
td{{padding:8px 12px;border-bottom:1px solid #1e293b;font-size:12px;}}
tr:hover td{{background:#111827;}}
.g{{color:#06d6a0;font-weight:bold;}} .r{{color:#ef4444;font-weight:bold;}} .y{{color:#f59e0b;font-weight:bold;}}
.bar{{height:14px;border-radius:3px;display:inline-block;vertical-align:middle;}}
.bg{{background:#1e293b;border-radius:3px;width:160px;display:inline-block;vertical-align:middle;}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:bold;}}
.bt{{background:#06d6a040;color:#06d6a0;border:1px solid #06d6a0;}}
p{{line-height:1.75;color:#94a3b8;}}
ol li{{line-height:2;color:#94a3b8;}}
</style></head><body>
<h1>📡 Sector Rotation — Monthly Signal</h1>
<p style="color:#64748b;">Date: <b>{today}</b> &nbsp;|&nbsp; Capital: <b>${capital:,.0f}</b>
 &nbsp;|&nbsp; Mode: <b>{'ETFs' if use_etf else 'Individual stocks'}</b>
 &nbsp;|&nbsp; Data as of: <b>{close.index[-1].strftime('%Y-%m-%d')}</b></p>"""]

    # ── Signal banner ─────────────────────────────────────────────────────────
    if alloc["status"] == "CASH":
        lines.append(f"""<div class="card" style="border-color:#f59e0b;">
<h3 style="color:#f59e0b;margin-top:0;">⚠️ Signal: HOLD CASH</h3>
<p>{alloc['message']}</p>
<p>Suggested parking: <b>SPY</b> (broad market), <b>SGOV</b> (T-bills), or just leave in a money-market account.</p>
</div>""")
    else:
        sector_names = " + ".join(f"<b>{v}</b>" for v in alloc["sectors"].values())
        lines.append(f"""<div class="card" style="border-color:#06d6a0;">
<h3 style="color:#06d6a0;margin-top:0;">✅ Signal: INVESTED</h3>
<p>This month's winning sectors: {sector_names}</p>
<p>{alloc['message']}</p>
</div>""")

    # ── Sector rankings ───────────────────────────────────────────────────────
    lines.append("<h2>📊 Sector Rankings — 3-Month Return</h2>")
    lines.append("<table><thead><tr><th>#</th><th>Sector</th><th>Avg Stock Return</th><th>Momentum Bar</th><th>ETF</th><th>ETF Return</th><th></th></tr></thead><tbody>")
    for i, r in enumerate(ranked, 1):
        ret   = r["avg_return"]
        is_top = r["sector"] in alloc["sectors"]
        r_str  = f"{ret:+.2%}" if not np.isnan(ret) else "—"
        cls    = "g" if not np.isnan(ret) and ret > 0 else "r"
        bar_px = int(abs(ret) / max_abs * 160) if not np.isnan(ret) else 0
        bar_c  = "#06d6a0" if not np.isnan(ret) and ret > 0 else "#ef4444"
        bar    = f'<span class="bg"><span class="bar" style="width:{bar_px}px;background:{bar_c};"></span></span>'
        etf_r  = f"{r['etf_return']:+.2%}" if r["etf_return"] is not None else "—"
        badge  = '<span class="badge bt">TOP</span>' if is_top else ""
        lines.append(f"<tr><td>{i}</td><td><b>{r['name']}</b></td><td class='{cls}'>{r_str}</td><td>{bar}</td>"
                     f"<td style='color:#64748b;'>{r['etf_ticker']}</td><td class='{cls}'>{etf_r}</td><td>{badge}</td></tr>")
    lines.append("</tbody></table>")

    # ── Individual stock returns within winning sectors ───────────────────────
    top_sectors = [r for r in ranked if r["sector"] in alloc["sectors"]]
    if top_sectors:
        lines.append("<h2>🔍 Stock-Level Returns Within Winning Sectors</h2>")
        for r in top_sectors:
            lines.append(f"<h3>{r['name']} ({r['etf_ticker']})</h3>")
            lines.append("<table><thead><tr><th>Stock</th><th>3-Month Return</th><th>Latest Price</th></tr></thead><tbody>")
            for ticker, ret in sorted(r["stock_returns"].items(), key=lambda x: x[1], reverse=True):
                price = latest.get(ticker, np.nan)
                p_str = f"${price:.2f}" if not pd.isna(price) else "—"
                cls   = "g" if ret > 0 else "r"
                lines.append(f"<tr><td><b>{ticker}</b></td><td class='{cls}'>{ret:+.2%}</td><td>{p_str}</td></tr>")
            lines.append("</tbody></table>")

    # ── Target allocation ─────────────────────────────────────────────────────
    if alloc["holdings"]:
        lines.append(f"<h2>📋 Target Portfolio (${capital:,.0f})</h2>")
        lines.append("<table><thead><tr><th>Ticker</th><th>Weight</th><th>$ Amount</th><th>Approx Shares</th><th>Current Price</th></tr></thead><tbody>")
        for ticker, weight in sorted(alloc["holdings"]):
            price   = latest.get(ticker, np.nan)
            dollars = weight * capital
            shares  = dollars / price if not pd.isna(price) and price > 0 else 0
            p_str   = f"${price:.2f}" if not pd.isna(price) else "—"
            lines.append(f"<tr><td><b>{ticker}</b></td><td>{weight:.1%}</td>"
                         f"<td class='g'>${dollars:,.0f}</td><td>{shares:.1f}</td><td>{p_str}</td></tr>")
        lines.append("</tbody></table>")

    # ── Trade list ────────────────────────────────────────────────────────────
    prev_date = prev.get("last_run")
    lines.append(f"<h2>🔄 Trade Actions {'vs ' + prev_date if prev_date else '(First Run)'}</h2>")

    if not prev:
        lines.append('<div class="card"><p>No previous state found — this is your first run. '
                     'Use the Target Portfolio table above as your starting allocation.</p></div>')
    elif not sells and not buys:
        lines.append('<div class="card"><p class="g">✅ No trades needed — allocation is unchanged from last month.</p></div>')
    else:
        if sells:
            lines.append("<h3>🔴 Sell / Reduce</h3>")
            lines.append("<table><thead><tr><th>Ticker</th><th>Action</th><th>Shares</th><th>Est. Proceeds</th><th>Price</th></tr></thead><tbody>")
            for t in sells:
                lines.append(f"<tr><td><b>{t['ticker']}</b></td><td class='r'>{t['action']}</td>"
                              f"<td>{t['shares']:.2f}</td><td>${t['value']:,.0f}</td><td>${t['price']:.2f}</td></tr>")
            lines.append("</tbody></table>")
        if buys:
            lines.append("<h3>🟢 Buy / Add</h3>")
            lines.append("<table><thead><tr><th>Ticker</th><th>Action</th><th>Shares</th><th>Est. Cost</th><th>Price</th></tr></thead><tbody>")
            for t in buys:
                lines.append(f"<tr><td><b>{t['ticker']}</b></td><td class='g'>{t['action']}</td>"
                              f"<td>{t['shares']:.2f}</td><td>${t['value']:,.0f}</td><td>${t['price']:.2f}</td></tr>")
            lines.append("</tbody></table>")

    # ── How this works ────────────────────────────────────────────────────────
    lines.append("""<h2>📖 How to Use This</h2>
<div class="card">
  <p><b>When to run:</b> The first trading day of each month (or every ~21 trading days).
     The strategy rebalances monthly — running it more often won't help.</p>
  <p><b>Step-by-step:</b></p>
  <ol>
    <li>Look at the Sector Rankings table. The strategy picks the top sectors with positive momentum.</li>
    <li>Execute any SELL orders first (to free up cash).</li>
    <li>Then execute BUY orders using available cash.</li>
    <li>Use limit orders close to the current price — avoid market orders for larger positions.</li>
    <li>Come back next month, run this script again, and follow the updated trades.</li>
  </ol>
  <p><b>ETF shortcut:</b> Instead of 5 individual stocks per sector, just buy the single
     sector ETF (<code>XLK</code> for tech, <code>XLF</code> for finance, etc.).
     Use <code>python live_signal.py --etf</code> to see this simplified view.
     Slightly lower returns historically, but far fewer trades and lower cost.</p>
  <p><b>If the signal says CASH:</b> All sectors have negative 3-month returns — the model
     says to wait. Park money in <b>SPY</b>, <b>SGOV</b> (3-month T-bills), or a
     high-yield savings account. Resume investing when a sector turns positive.</p>
  <p class="y">⚠️ Past backtested results do not guarantee future returns.
     This is an educational tool, not financial advice. Always do your own research.</p>
</div>""")

    lines.append(f"<p style='color:#334155;margin-top:40px;'>Sector Rotation Live Signal &nbsp;•&nbsp; {today} &nbsp;•&nbsp; Educational purposes only</p>")
    lines.append("</body></html>")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sector Rotation — Monthly Live Signal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python live_signal.py                    # default: $100k, individual stocks
  python live_signal.py --capital 50000    # $50k portfolio
  python live_signal.py --etf              # use sector ETFs (XLK, XLF, etc.)
  python live_signal.py --top 1            # concentrate in top 1 sector only
  python live_signal.py --dry-run          # show signal without saving state
        """
    )
    parser.add_argument("--capital",  type=float, default=100_000, help="Portfolio size in USD (default: $100,000)")
    parser.add_argument("--top",      type=int,   default=2,       help="Number of top sectors to invest in (default: 2)")
    parser.add_argument("--etf",      action="store_true",         help="Use one sector ETF per sector instead of 5 stocks")
    parser.add_argument("--dry-run",  action="store_true",         help="Print signal without saving state file")
    parser.add_argument("--output",   default=REPORT_DIR,          help="Output directory for HTML report")
    args = parser.parse_args()

    today = datetime.today().strftime("%Y-%m-%d")

    print()
    print("  ╔════════════════════════════════════════════════════════╗")
    print("  ║  SECTOR ROTATION — Live Signal Generator              ║")
    print(f"  ║  {today}                                        ║")
    print("  ╚════════════════════════════════════════════════════════╝")
    print(f"  💰 Capital: ${args.capital:,.0f}  |  Top sectors: {args.top}  |  Mode: {'ETFs' if args.etf else 'Individual stocks'}")
    print()

    try:
        close = fetch_current_data()
    except Exception as e:
        print(f"  ❌ Could not fetch data: {e}", file=sys.stderr)
        sys.exit(1)

    # Compute signal
    ranked = rank_sectors(close)
    alloc  = build_allocation(ranked, n_top=args.top, use_etf=args.etf)
    prev   = load_state()
    sells, buys = compute_trades(prev, alloc, close, args.capital)

    # ── Terminal output ───────────────────────────────────────────────────────
    print("  ━━━ SECTOR RANKINGS (3-Month Return) ━━━")
    for i, r in enumerate(ranked, 1):
        ret     = r["avg_return"]
        is_top  = r["sector"] in alloc["sectors"]
        ret_str = f"{ret:+.2%}" if not np.isnan(ret) else "   N/A"
        bar_len = min(20, int(abs(ret) * 200)) if not np.isnan(ret) else 0
        bar     = ("█" if ret > 0 else "░") * bar_len
        marker  = "  ← THIS MONTH" if is_top else ""
        print(f"  {i}. {r['name']:<20} {ret_str:>9}  {bar:<20}{marker}")

    print()
    if alloc["status"] == "CASH":
        print("  ⚠️  SIGNAL: HOLD CASH — no sector has positive momentum")
        print(f"  → {alloc['message']}")
    else:
        top_str = " + ".join(alloc["sectors"].values())
        print(f"  ✅ SIGNAL: INVEST in {top_str}")
        latest = close.iloc[-1]
        print()
        print(f"  ━━━ TARGET ALLOCATION (${args.capital:,.0f}) ━━━")
        for ticker, weight in sorted(alloc["holdings"]):
            price   = latest.get(ticker, np.nan)
            dollars = weight * args.capital
            shares  = dollars / price if not pd.isna(price) and price > 0 else 0
            p_str   = f"${price:>8.2f}" if not pd.isna(price) else "       N/A"
            print(f"  {ticker:<8}  {weight:.1%}  ${dollars:>10,.0f}  ≈ {shares:>7.1f} shares  @ {p_str}")

    if prev:
        prev_date = prev.get("last_run", "last run")
        print()
        print(f"  ━━━ TRADES vs {prev_date} ━━━")
        if not sells and not buys:
            print("  ✅ No trades needed — same allocation as last month.")
        for t in sorted(sells, key=lambda x: x["ticker"]):
            print(f"  SELL {t['action']:<10} {t['ticker']:<8}  {t['shares']:>7.2f} shares  ≈ ${t['value']:>10,.0f}  @ ${t['price']:.2f}")
        for t in sorted(buys, key=lambda x: x["ticker"]):
            print(f"  BUY  {t['action']:<10} {t['ticker']:<8}  {t['shares']:>7.2f} shares  ≈ ${t['value']:>10,.0f}  @ ${t['price']:.2f}")
    else:
        print()
        print("  ℹ️  No previous state found — this is your first run.")
        print("  Use the Target Allocation above as your starting portfolio.")

    # ── Save state & HTML ─────────────────────────────────────────────────────
    if not args.dry_run:
        save_state(alloc, close, args.capital)
        print(f"\n  💾 State saved → {STATE_FILE}")
    else:
        print("\n  [dry-run] State not saved.")

    os.makedirs(args.output, exist_ok=True)
    html = generate_html(ranked, alloc, sells, buys, close, args.capital, args.etf, prev)
    report_path = os.path.join(args.output, f"live_signal_{today}.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"  📄 Report saved → {report_path}")
    print(f"\n  Open: open {report_path}")
    print()
    print("  ⚠️  DISCLAIMER: Educational/research use only. Not financial advice.")
    print()


def fetch_current_data() -> pd.DataFrame:
    """Thin alias kept for test/import compatibility."""
    return fetch_data()


if __name__ == "__main__":
    main()
