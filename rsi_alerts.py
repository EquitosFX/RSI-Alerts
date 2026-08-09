#!/usr/bin/env python3
"""
FOREX RSI ALERT MONITOR
=======================
Watches any number of pairs across multiple timeframes and pushes Telegram
notifications on RSI crossings, plus a periodic digest of what is already
extended.

TWO KINDS OF MESSAGE:
  1. CROSSING alerts  - fire once, the moment RSI passes through a level.
  2. EXTENDED digest  - a periodic summary of everything currently beyond
                        70/30, so you can see standing conditions rather than
                        only the moment they began.

WHY THE DIGEST IS PERIODIC AND NOT PER-RUN:
A pair sitting at 72 for three days would otherwise notify you ~72 times.
Alert systems die from volume, not from missing signals - once you start
ignoring the phone, the good alerts are lost with the noise.
Set DIGEST_EVERY_HOURS = 0 to switch it off.

SETUP: see README notes - Telegram bot via @BotFather, secrets in GitHub.
Test delivery any time with:  python3 rsi_alerts.py --test-message
"""

import json, os, sys, time, argparse
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

# ==========================================================================
# CONFIG
# ==========================================================================
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "PUT_YOUR_TOKEN_HERE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "PUT_YOUR_CHAT_ID_HERE")

RSI_LENGTH = 14
TIMEFRAMES = ["1h", "4h", "1d"]

# Crossing levels. Add or remove freely - each fires its own message.
LEVELS_ABOVE = [70, 75, 80]
LEVELS_BELOW = [30, 25, 20]

# Digest of standing conditions. 0 = off. 4 = roughly six times a day.
DIGEST_EVERY_HOURS = 4
DIGEST_ABOVE = 70          # what counts as "extended" for the digest
DIGEST_BELOW = 30

# ---- Instruments ---------------------------------------------------------
# Yahoo tickers. Currency pairs use PAIR=X. Metals are futures (=F) because
# Yahoo does not serve spot XAU/XAG.
#
# START SMALL. Every name here costs bandwidth and adds messages. The full
# 28-pair set below is more than most people can actually act on - comment
# out what you do not trade. You can always add later; trimming back after
# you have started ignoring the phone is much harder.

MAJORS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X", "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
}

EUR_CROSSES = {
    "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", "EURCHF": "EURCHF=X",
    "EURCAD": "EURCAD=X", "EURAUD": "EURAUD=X", "EURNZD": "EURNZD=X",
}

GBP_CROSSES = {
    "GBPJPY": "GBPJPY=X", "GBPCHF": "GBPCHF=X", "GBPCAD": "GBPCAD=X",
    "GBPAUD": "GBPAUD=X", "GBPNZD": "GBPNZD=X",
}

OTHER_CROSSES = {
    "AUDJPY": "AUDJPY=X", "AUDCHF": "AUDCHF=X", "AUDCAD": "AUDCAD=X",
    "AUDNZD": "AUDNZD=X", "NZDJPY": "NZDJPY=X", "NZDCHF": "NZDCHF=X",
    "NZDCAD": "NZDCAD=X", "CADJPY": "CADJPY=X", "CADCHF": "CADCHF=X",
    "CHFJPY": "CHFJPY=X",
}

EXOTICS = {
    "USDMXN": "USDMXN=X", "USDZAR": "USDZAR=X", "USDTRY": "USDTRY=X",
    "USDSEK": "USDSEK=X", "USDNOK": "USDNOK=X", "USDSGD": "USDSGD=X",
    "USDPLN": "USDPLN=X", "USDHUF": "USDHUF=X", "USDCNH": "USDCNH=X",
    "USDINR": "USDINR=X",
}

METALS_AND_INDEX = {
    "GOLD":   "GC=F",       # continuous front-month COMEX futures, not spot
    "SILVER": "SI=F",
    "DXY":    "DX-Y.NYB",
}

# Edit this line to choose what runs. Merge whichever groups you want.
# WATCHLIST = {**MAJORS, **METALS_AND_INDEX}
WATCHLIST = {**MAJORS, **EUR_CROSSES, **GBP_CROSSES, **OTHER_CROSSES, **METALS_AND_INDEX}
# WATCHLIST = {**MAJORS, **EUR_CROSSES, **GBP_CROSSES, **OTHER_CROSSES, **EXOTICS, **METALS_AND_INDEX}

STATE_FILE = "rsi_alert_state.json"
PERIOD_FOR = {"5m": "5d", "15m": "1mo", "30m": "1mo", "1h": "6mo", "1d": "2y", "1wk": "5y"}

# Yahoo has no 4h interval - these are resampled from a lower one.
# 4h bars align to 00:00 UTC. Your broker likely aligns to its own server
# time (often 17:00 New York), so candles will not always match your chart.
RESAMPLE_FROM = {"2h": ("1h", "2h"), "4h": ("1h", "4h"), "8h": ("1h", "8h")}
# ==========================================================================


def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI - the same smoothing TradingView's ta.rsi() uses."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


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
        print(f"  [DRY RUN - would send]\n{text}\n")
        return True
    if "PUT_YOUR" in TELEGRAM_TOKEN:
        print("  !! Telegram not configured - set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=20)
        if r.status_code != 200:
            print(f"  !! Telegram error {r.status_code}: {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"  !! Telegram send failed: {e}")
        return False


def fetch_batch(tickers: list[str], interval: str) -> dict[str, pd.DataFrame]:
    """
    One request for ALL tickers on a timeframe instead of one each.
    With 28 pairs that is 3 requests per run rather than 84 - the difference
    between working and being rate-limited by Yahoo.
    """
    if yf is None:
        print("  !! yfinance not installed  (pip install yfinance)")
        return {}

    base, rule = RESAMPLE_FROM.get(interval, (interval, None))
    out: dict[str, pd.DataFrame] = {}

    try:
        raw = yf.download(tickers, period=PERIOD_FOR.get(base, "6mo"),
                          interval=base, progress=False, auto_adjust=False,
                          group_by="ticker", threads=True)
    except Exception as e:
        print(f"  !! batch fetch failed for {interval}: {e}")
        return {}

    if raw is None or raw.empty:
        return {}

    for tk in tickers:
        try:
            df = raw[tk].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            df = df.dropna(subset=["Close"])
            if rule:
                df = df.resample(rule, origin="epoch", label="left", closed="left").agg({
                    "Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
            if len(df) >= RSI_LENGTH + 5:
                out[tk] = df
        except Exception:
            continue
    return out


def check_all(dry: bool = False) -> int:
    state = load_state()
    sent = 0
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    extended: list[tuple] = []

    for tf in TIMEFRAMES:
        data = fetch_batch(list(WATCHLIST.values()), tf)
        print(f"\n  --- {tf} ({len(data)}/{len(WATCHLIST)} fetched) ---")

        for name, ticker in WATCHLIST.items():
            df = data.get(ticker)
            if df is None:
                print(f"  {name}|{tf:<3}  no data")
                continue

            r = rsi_wilder(df["Close"], RSI_LENGTH).dropna()
            if len(r) < 3:
                continue

            # Last CLOSED bar, not the forming one - a forming bar's RSI moves
            # around and would fire alerts that stop being true minutes later.
            curr, prev = float(r.iloc[-2]), float(r.iloc[-3])
            price = float(df["Close"].iloc[-2])
            bar_id = str(df.index[-2])

            if curr >= DIGEST_ABOVE or curr <= DIGEST_BELOW:
                extended.append((name, tf, curr))

            events = []
            for lv in sorted(LEVELS_ABOVE):
                if prev <= lv < curr:
                    events.append(("ABOVE", lv))
            for lv in sorted(LEVELS_BELOW, reverse=True):
                if prev >= lv > curr:
                    events.append(("BELOW", lv))

            for direction, lv in events:
                key = f"{name}|{tf}|{direction}|{lv}"
                if state.get(key) == bar_id:
                    continue
                if direction == "ABOVE":
                    icon = "🔴🔴" if lv >= 80 else "🔴" if lv >= 75 else "📈"
                    word = "overbought"
                else:
                    icon = "🔵🔵" if lv <= 20 else "🔵" if lv <= 25 else "📉"
                    word = "oversold"
                msg = (f"{icon} <b>{name}</b> · {tf}\n"
                       f"RSI({RSI_LENGTH}) crossed <b>{direction} {lv}</b> ({word})\n"
                       f"RSI now <b>{curr:.1f}</b> (was {prev:.1f})\n"
                       f"Price {price:,.4f}\n"
                       f"<i>{stamp} · closed bar</i>")
                if send_telegram(msg, dry):
                    sent += 1
                    state[key] = bar_id
                    print(f"  {name}|{tf:<3}  ALERT {direction} {lv}  (RSI {curr:.1f})")

            if not events:
                flag = "  <<" if (curr >= DIGEST_ABOVE or curr <= DIGEST_BELOW) else ""
                print(f"  {name}|{tf:<3}  RSI {curr:5.1f}{flag}")

    # ---- Periodic digest of standing conditions --------------------------
    if DIGEST_EVERY_HOURS > 0 and extended:
        last = state.get("_last_digest")
        due = True
        if last:
            try:
                due = now - datetime.fromisoformat(last) >= timedelta(hours=DIGEST_EVERY_HOURS)
            except Exception:
                due = True
        if due:
            ob = sorted([e for e in extended if e[2] >= DIGEST_ABOVE], key=lambda x: -x[2])
            os_ = sorted([e for e in extended if e[2] <= DIGEST_BELOW], key=lambda x: x[2])
            lines = [f"📊 <b>Currently extended</b>", ""]
            if ob:
                lines.append("<b>Overbought (RSI ≥ 70)</b>")
                lines += [f"  {n} · {t} · <b>{v:.1f}</b>" for n, t, v in ob]
                lines.append("")
            if os_:
                lines.append("<b>Oversold (RSI ≤ 30)</b>")
                lines += [f"  {n} · {t} · <b>{v:.1f}</b>" for n, t, v in os_]
                lines.append("")
            lines.append(f"<i>{stamp} · standing conditions, not new crossings</i>")
            if send_telegram("\n".join(lines), dry):
                sent += 1
                state["_last_digest"] = now.isoformat()
                print(f"\n  DIGEST sent ({len(ob)} overbought, {len(os_)} oversold)")

    save_state(state)
    return sent


def self_test() -> None:
    print("SELF-TEST - RSI implementation")
    path = "/mnt/user-data/uploads/XAUUSD_D1.csv"
    if not os.path.exists(path):
        print("  (skipped - reference CSV not present)")
        return
    d = pd.read_csv(path, names=["dt", "o", "h", "l", "c", "v"], parse_dates=["dt"])
    r = rsi_wilder(d["c"], 14)
    print(f"  {len(d)} bars, last RSI {r.iloc[-1]:.2f}, range {r.min():.1f}-{r.max():.1f}")
    for lv in LEVELS_ABOVE:
        print(f"  crossings above {lv}: {((r > lv) & (r.shift() <= lv)).sum()}")
    for lv in LEVELS_BELOW:
        print(f"  crossings below {lv}: {((r < lv) & (r.shift() >= lv)).sum()}")
    assert 0 <= r.min() and r.max() <= 100, "RSI out of bounds"
    print("  PASS\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print alerts instead of sending")
    ap.add_argument("--loop", type=int, metavar="SECONDS", help="run forever, checking every N seconds")
    ap.add_argument("--self-test", action="store_true", help="validate the RSI maths and exit")
    ap.add_argument("--test-message", action="store_true", help="send one Telegram message and exit")
    a = ap.parse_args()

    if a.test_message:
        tok, cid = TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
        print("Telegram configuration:")
        print(f"  token   : {'MISSING' if 'PUT_YOUR' in tok else f'set, {len(tok)} chars, ends ...{tok[-4:]}'}")
        print(f"  chat id : {'MISSING' if 'PUT_YOUR' in cid else f'set, value = {cid}'}")
        if "PUT_YOUR" in tok or "PUT_YOUR" in cid:
            print("\n  -> Secrets are not reaching the script. Check the names in")
            print("     Settings > Secrets and variables > Actions.")
            sys.exit(1)
        ok = send_telegram("✅ <b>Test message</b>\nYour RSI alert bot can reach you.")
        print("\n  -> SENT. Check your phone." if ok else "\n  -> FAILED. See error above.")
        sys.exit(0 if ok else 1)

    if a.self_test:
        self_test(); sys.exit(0)

    print(f"Watching {len(WATCHLIST)} instruments x {len(TIMEFRAMES)} timeframes "
          f"= {len(WATCHLIST) * len(TIMEFRAMES)} checks, in {len(TIMEFRAMES)} requests")

    if a.loop:
        while True:
            print(f"\n=== {datetime.now().strftime('%H:%M:%S')} ===")
            print(f"\n{check_all(a.dry_run)} message(s) sent")
            time.sleep(a.loop)
    else:
        print(f"\n{check_all(a.dry_run)} message(s) sent")
