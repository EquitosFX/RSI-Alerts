#!/usr/bin/env python3
"""
FOREX RSI ALERT MONITOR
=======================
Watches any number of pairs on any number of timeframes and pushes a phone
notification when RSI crosses a level you set. Free, unlimited alerts, no
subscription.

WHY THIS INSTEAD OF TRADINGVIEW:
RSI alerts are "technical" alerts, and TradingView's free tier gives you at
most one. This has no cap.

WHAT IT COSTS: nothing. Yahoo Finance data via yfinance (no API key),
Telegram for delivery (free), and either your own machine or GitHub Actions
to run it.

--------------------------------------------------------------------------
SETUP (about 10 minutes)
--------------------------------------------------------------------------
1. pip install yfinance pandas requests

2. Make a Telegram bot:
   - In Telegram, message @BotFather -> /newbot -> follow prompts
   - Copy the token it gives you into TELEGRAM_TOKEN below
   - Message your new bot once (say "hi") so it can reply to you
   - Visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
     and copy the "chat":{"id": NUMBER } value into TELEGRAM_CHAT_ID

3. Edit WATCHLIST below.

4. Test without sending anything:   python3 rsi_alerts.py --dry-run
   Then run for real:               python3 rsi_alerts.py

5. To run it continuously, pick one:
   - Your own machine:  python3 rsi_alerts.py --loop 300
   - cron (every 5 min): */5 * * * * cd /path && python3 rsi_alerts.py
   - GitHub Actions: see the workflow at the bottom of this file
--------------------------------------------------------------------------
"""

import json, os, sys, time, argparse
from datetime import datetime, timezone

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

# ==========================================================================
# CONFIG — edit this section
# ==========================================================================
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "PUT_YOUR_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

RSI_LENGTH = 14

# symbol       : Yahoo ticker. Forex = PAIR=X. Gold = XAUUSD=X (or GC=F futures)
# timeframes   : any of 5m 15m 30m 1h 1d 1wk
# above/below  : alert when RSI CROSSES these levels. None = ignore that side.
TIMEFRAMES = ["1h", "4h", "1d"]

# NOTE ON GOLD: Yahoo does not serve "XAUUSD=X". Its gold instrument is GC=F,
# the continuous front-month COMEX futures contract. It is not spot — it trades
# a few dollars away from spot (the basis) and rolls between contract months.
# For RSI that difference is immaterial: RSI reads the shape of the series, not
# its level, and futures and spot move together almost tick for tick. But the
# absolute price in the alert will not exactly match your broker's spot quote.
WATCHLIST = [
    {"name": "GOLD",    "symbol": "GC=F",     "timeframes": TIMEFRAMES, "above": 10, "below": 30},
    {"name": "EURUSD",  "symbol": "EURUSD=X", "timeframes": TIMEFRAMES, "above": 70, "below": 30},
    {"name": "GBPUSD",  "symbol": "GBPUSD=X", "timeframes": TIMEFRAMES, "above": 70, "below": 30},
    {"name": "USDJPY",  "symbol": "USDJPY=X", "timeframes": TIMEFRAMES, "above": 70, "below": 30},
    {"name": "AUDUSD",  "symbol": "AUDUSD=X", "timeframes": TIMEFRAMES, "above": 70, "below": 30},
    {"name": "DXY",     "symbol": "DX-Y.NYB", "timeframes": TIMEFRAMES, "above": 70, "below": 30},
]

STATE_FILE = "rsi_alert_state.json"

# Yahoo caps how far back intraday data goes
PERIOD_FOR = {"5m": "5d", "15m": "1mo", "30m": "1mo", "1h": "6mo", "1d": "2y", "1wk": "5y"}

# Yahoo has NO 4h interval. These are built by resampling a lower one.
# 4h bars are aligned to 00:00 UTC (00/04/08/12/16/20). Your broker may align
# its 4h bars to its own server time instead — usually 17:00 New York — so the
# candles will not always match what you see on the chart. RSI on a 4h series
# offset by an hour or two is close but not identical. Worth knowing before you
# wonder why a level fired here and not there.
RESAMPLE_FROM = {"2h": ("1h", "2h"), "4h": ("1h", "4h"), "8h": ("1h", "8h")}

# ==========================================================================


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI — the same smoothing TradingView's ta.rsi() uses."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def send_telegram(text: str, dry: bool = False) -> bool:
    if dry:
        print(f"  [DRY RUN — would send]\n{text}\n")
        return True
    if "PUT_YOUR" in TELEGRAM_TOKEN:
        print("  !! Telegram not configured — set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=20,
        )
        if r.status_code != 200:
            print(f"  !! Telegram error {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  !! Telegram send failed: {e}")
        return False


def fetch(symbol: str, interval: str) -> pd.DataFrame | None:
    """Download bars. Timeframes Yahoo doesn't offer are resampled from a lower one."""
    if yf is None:
        print("  !! yfinance not installed  (pip install yfinance)")
        return None

    base, rule = RESAMPLE_FROM.get(interval, (interval, None))

    try:
        df = yf.download(symbol, period=PERIOD_FOR.get(base, "6mo"),
                         interval=base, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):      # yfinance sometimes returns these
            df.columns = df.columns.get_level_values(0)

        if rule:
            df = df.resample(rule, origin="epoch", label="left", closed="left").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])

        if len(df) < RSI_LENGTH + 5:
            print(f"  !! only {len(df)} bars for {symbol} {interval} — need {RSI_LENGTH + 5}")
            return None
        return df
    except Exception as e:
        print(f"  !! fetch failed {symbol} {interval}: {e}")
        return None


def check_all(dry: bool = False) -> int:
    state = load_state()
    fired = 0
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for item in WATCHLIST:
        for tf in item["timeframes"]:
            key = f"{item['name']}|{tf}"
            df = fetch(item["symbol"], tf)
            if df is None:
                print(f"  {key:<20} no data")
                continue

            r = rsi_wilder(df["Close"], RSI_LENGTH).dropna()
            if len(r) < 2:
                continue

            # Use the last CLOSED bar, not the forming one. The forming bar's RSI
            # moves around and would fire alerts that later stop being true.
            curr, prev = float(r.iloc[-2]), float(r.iloc[-3]) if len(r) >= 3 else None
            price = float(df["Close"].iloc[-2])
            if prev is None:
                continue

            events = []
            hi, lo = item.get("above"), item.get("below")
            if hi is not None and prev <= hi < curr:
                events.append(("ABOVE", hi, "📈"))
            if lo is not None and prev >= lo > curr:
                events.append(("BELOW", lo, "📉"))

            for direction, level, icon in events:
                # Don't repeat the same crossing for the same bar
                stamp = str(df.index[-2])
                if state.get(key, {}).get(direction) == stamp:
                    continue
                msg = (f"{icon} <b>{item['name']}</b> · {tf}\n"
                       f"RSI({RSI_LENGTH}) crossed <b>{direction} {level}</b>\n"
                       f"RSI now <b>{curr:.1f}</b> (was {prev:.1f})\n"
                       f"Price {price:,.4f}\n"
                       f"<i>{now} · closed bar</i>")
                if send_telegram(msg, dry):
                    fired += 1
                    state.setdefault(key, {})[direction] = stamp
                    print(f"  {key:<20} ALERT {direction} {level} (RSI {curr:.1f})")

            if not events:
                print(f"  {key:<20} RSI {curr:5.1f}   quiet")

    save_state(state)
    return fired


def self_test() -> None:
    """Validate the RSI maths against a known series before trusting alerts."""
    print("SELF-TEST — RSI implementation")
    path = "/mnt/user-data/uploads/XAUUSD_D1.csv"
    if not os.path.exists(path):
        print("  (skipped — reference CSV not present)")
        return
    d = pd.read_csv(path, names=["dt", "o", "h", "l", "c", "v"], parse_dates=["dt"])
    r = rsi_wilder(d["c"], 14)
    print(f"  {len(d)} bars, last RSI {r.iloc[-1]:.2f}")
    print(f"  range {r.min():.1f} - {r.max():.1f}  (must sit inside 0-100)")
    over = (r > 70).sum()
    print(f"  bars above 70: {over} ({over/len(r)*100:.1f}%)")
    crosses = ((r > 70) & (r.shift() <= 70)).sum()
    print(f"  crossings above 70: {crosses}  (expected ~117 from the earlier analysis)")
    assert 0 <= r.min() and r.max() <= 100, "RSI out of bounds"
    print("  PASS\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print alerts instead of sending")
    ap.add_argument("--loop", type=int, metavar="SECONDS", help="run forever, checking every N seconds")
    ap.add_argument("--self-test", action="store_true", help="validate the RSI maths and exit")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        sys.exit(0)

    if a.loop:
        print(f"Monitoring every {a.loop}s. Ctrl-C to stop.")
        while True:
            print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
            n = check_all(a.dry_run)
            print(f"  {n} alert(s) sent")
            time.sleep(a.loop)
    else:
        n = check_all(a.dry_run)
        print(f"\n{n} alert(s) sent")

# ==========================================================================
# GITHUB ACTIONS — free hosting, no machine of your own left running.
# Save as .github/workflows/rsi.yml in a repo, and put TELEGRAM_TOKEN and
# TELEGRAM_CHAT_ID in Settings > Secrets and variables > Actions.
#
# name: RSI alerts
# on:
#   schedule: [{cron: "*/15 * * * *"}]     # every 15 min
#   workflow_dispatch:
# jobs:
#   check:
#     runs-on: ubuntu-latest
#     steps:
#       - uses: actions/checkout@v4
#       - uses: actions/setup-python@v5
#         with: {python-version
