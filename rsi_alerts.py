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

import numpy as np
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

# ---- Forward-outcome logging --------------------------------------------
# Records flagged conditions and scores them N days later, so you accumulate
# genuine forward-test data instead of re-slicing history. Nothing here has
# been fitted to these observations - that is the entire point.
LOG_OUTCOMES = True
OUTCOME_HORIZONS = [10, 20]        # trading days to score at
OUTCOME_REPORT_EVERY_DAYS = 30     # send a summary this often
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
WATCHLIST = {**MAJORS, **METALS_AND_INDEX}
# WATCHLIST = {**MAJORS, **EUR_CROSSES, **GBP_CROSSES, **OTHER_CROSSES, **METALS_AND_INDEX}
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


def true_range(df: pd.DataFrame) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def adx_dmi(df: pd.DataFrame, n: int = 14):
    """Wilder's ADX with directional indicators. ADX measures trend STRENGTH."""
    h, l = df["High"], df["Low"]
    tr = true_range(df)
    up, dn = h.diff(), -l.diff()
    plus  = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    dip = 100 * plus.ewm(alpha=1/n, adjust=False).mean() / atr
    dim = 100 * minus.ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (dip - dim).abs() / (dip + dim)
    return dx.ewm(alpha=1/n, adjust=False).mean(), dip, dim


def choppiness(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Low = a directional move is underway. High = grinding sideways."""
    tr_sum = true_range(df).rolling(n).sum()
    rng = df["High"].rolling(n).max() - df["Low"].rolling(n).min()
    return 100 * np.log10(tr_sum / rng) / np.log10(n)


def parabolic_sar(df: pd.DataFrame, af0=0.02, step=0.02, afmax=0.2):
    """Standard Wilder PSAR. Returns (sar values, direction +1/-1)."""
    h, l = df["High"].values, df["Low"].values
    n = len(df)
    sar = np.zeros(n); trend = np.ones(n)
    ep = h[0]; af = af0; sar[0] = l[0]
    for i in range(1, n):
        sar[i] = sar[i-1] + af * (ep - sar[i-1])
        if trend[i-1] > 0:
            sar[i] = min(sar[i], l[i-1], l[max(i-2, 0)])
            if l[i] < sar[i]:
                trend[i] = -1; sar[i] = ep; ep = l[i]; af = af0
            else:
                trend[i] = 1
                if h[i] > ep: ep = h[i]; af = min(af + step, afmax)
        else:
            sar[i] = max(sar[i], h[i-1], h[max(i-2, 0)])
            if h[i] > sar[i]:
                trend[i] = 1; sar[i] = ep; ep = h[i]; af = af0
            else:
                trend[i] = -1
                if l[i] < ep: ep = l[i]; af = min(af + step, afmax)
    return sar, trend


def direction_panel(df: pd.DataFrame, dip_up: bool) -> tuple[str, int]:
    """
    The four classic direction reads, as CONTEXT.
    Measured on 5,216 daily gold bars, forward 5 days vs a +0.152% baseline:
      Price > EMA50   +0.067pp (fails split)   below  -0.086pp
      SAR dots below  +0.068pp (holds split)   above  -0.078pp
      MACD > signal   -0.008pp  (nothing)
      ADX             strength, not direction - both up AND down trends
                      showed positive forward returns
    The vote count is NOT a strength ranking. 3-of-4 agreement scored +0.191pp
    while 4-of-4 scored only +0.042pp. A real consensus effect would rise
    monotonically; this does not, which is the signature of noise. Displayed
    because seeing the board at a glance is useful, not because more ticks
    means a better trade.
    """
    c = df["Close"]
    ema = c.ewm(span=50, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    _, sar_dir = parabolic_sar(df)

    ema_up  = bool(c.iloc[-2] > ema.iloc[-2])
    macd_up = bool(macd.iloc[-2] > sig.iloc[-2])
    sar_up  = bool(sar_dir[-2] > 0)
    votes = sum([ema_up, macd_up, sar_up, dip_up])
    a = lambda b: "↑" if b else "↓"
    # NOTE: shown as a board, deliberately NOT as a score. Across 18 instruments
    # the bullish states of every one of these measured slightly NEGATIVE on FX
    # and their bearish states slightly positive. More arrows up is not better.
    line = (f"EMA{a(ema_up)} MACD{a(macd_up)} SAR{a(sar_up)} DI{a(dip_up)}"
            f"  ·  {votes}/4 up")
    return line, votes


def ichimoku(df: pd.DataFrame):
    """
    Ichimoku cloud, standard 9/26/52 settings.

    MEASURED ACROSS 18 INSTRUMENTS (17 FX pairs + gold), daily, 5-day forward,
    each against its own baseline, and confirmed in BOTH halves of history:

        Price BELOW cloud   +0.027pp   (15/18 pairs, p<0.0001)
        Price ABOVE cloud   -0.018pp   ( 4/18 pairs, p=0.003)
        Thick cloud + below +0.127pp   (15/18 pairs, p<0.0001)  <- strongest found
        Inside cloud        smaller absolute moves (1/18, p<0.0001)

    So the textbook reading is INVERTED on FX. "Price above a green cloud is
    bullish" measures slightly negative; price below the cloud measures
    positive. The consolidation claim, by contrast, is TRUE - inside the cloud
    really does mean smaller subsequent moves.

    Effect sizes are small. +0.127pp over five days clears a major's ~0.02%
    spread and does NOT clear a cross's ~0.05%. This is context, not a trade.
    """
    h, l, c = df["High"], df["Low"], df["Close"]
    tenkan = (h.rolling(9).max()  + l.rolling(9).min())  / 2
    kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
    spanA  = ((tenkan + kijun) / 2).shift(26)
    spanB  = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    hi = pd.concat([spanA, spanB], axis=1).max(axis=1)
    lo = pd.concat([spanA, spanB], axis=1).min(axis=1)
    thick = (hi - lo) / c * 100
    i = -2                                    # last closed bar
    if pd.isna(hi.iloc[i]) or pd.isna(lo.iloc[i]):
        return "", False
    px = float(c.iloc[i])
    pos = "above" if px > hi.iloc[i] else "below" if px < lo.iloc[i] else "inside"
    tq = thick.dropna()
    pct = float((tq < thick.iloc[i]).mean() * 100) if len(tq) > 50 else 50.0
    band = "thick" if pct >= 75 else "thin" if pct <= 25 else "mid"
    flag = (pos == "below" and band == "thick")
    note = {"above": "", "below": "", "inside": " · smaller moves expected"}[pos]
    return f"Cloud {pos} ({band}){note}", flag


def regime_tag(adx_v: float, chop_v: float) -> str:
    """
    CONTEXT ONLY - never a gate on whether an alert is sent.
    Tested on 5,216 daily gold bars: Choppiness < 38 was the most robust regime
    reading found (+0.249pp over baseline, 60.2% win, holding across both halves
    of the sample). But it says a move is HAPPENING, not which way it goes - a
    confirmed downtrend showed positive forward returns too. And as a FILTER on
    the RSI signal it made things worse, not better. So it is printed, not acted on.
    """
    if np.isnan(adx_v) or np.isnan(chop_v):
        return ""
    if chop_v < 38 and adx_v > 25:  return "strong move underway"
    if chop_v < 38:                 return "move underway"
    if chop_v > 62:                 return "choppy"
    if adx_v > 25:                  return "trending"
    return "neutral"


def stochastic(df: pd.DataFrame, k: int = 14, smooth: int = 3) -> float:
    ll = df["Low"].rolling(k).min(); hh = df["High"].rolling(k).max()
    return float((100*(df["Close"]-ll)/(hh-ll)).rolling(smooth).mean().iloc[-2])


def setup_grade(cloud_pos: str, thick_band: str, rsi_v: float, stoch_v: float) -> tuple[str, str]:
    """
    The counter-trend stack, measured on 17 FX pairs, daily, 10-day hold,
    spread charged (0.02% majors / 0.05% crosses):

        below thick cloud alone              +0.125%/trade  13/17 pairs  p=0.006
        + Stochastic %K < 20                 +0.256%/trade  15/17 pairs  p=0.0002
        + RSI < 40                           +0.175%/trade  13/17 pairs  p=0.003
        + both                               +0.307%/trade  15/17 pairs  p<0.0001

    These STACK, unlike the momentum filters tested earlier, because every one
    of them is a counter-trend read. Confirming a momentum signal with more
    momentum just means entering later; confirming a stretched condition with
    another stretched condition genuinely narrows the set.

    HELD-OUT RESULT — THE STACKING DID NOT REPLICATE. Retested on H4, a dataset
    that played no part in finding it (60 H4 bars ~ 10 trading days):

        below thick cloud alone   daily +0.125% (p=0.006) -> H4 +0.050% (p=0.18)
        + Stoch<20                daily +0.256% (p=0.0002)-> H4 +0.022% (p=0.60)
        + RSI<40 + Stoch<20       daily +0.307% (p<0.0001)-> H4 +0.028% (p=0.54)

    The base condition survives directionally but weakened. The incremental
    benefit of stacking vanished entirely - on H4 the full stack does no better
    than the base. That was in-sample optimisation.

    What DID hold: the short side is symmetric (+0.107% shorting above a thick
    cloud with RSI>60 and Stoch>80, 13/17 pairs). Symmetry is what separates a
    real mean-reversion effect from drift wearing a costume.

    The tiers below are therefore DESCRIPTIVE labels, not a ranking. Do not read
    "full stack" as better than "base".
    """
    if thick_band != "thick":
        return "", "", 0
    if cloud_pos == "below":
        parts = ["below thick cloud"]
        if not np.isnan(stoch_v) and stoch_v < 20: parts.append("Stoch<20")
        if not np.isnan(rsi_v) and rsi_v < 40:     parts.append("RSI<40")
        side = 1
    elif cloud_pos == "above":
        parts = ["above thick cloud"]
        if not np.isnan(stoch_v) and stoch_v > 80: parts.append("Stoch>80")
        if not np.isnan(rsi_v) and rsi_v > 60:     parts.append("RSI>60")
        side = -1
    else:
        return "", "", 0
    tier = {1: "base", 2: "stacked", 3: "full stack"}[min(len(parts), 3)]
    return " + ".join(parts), tier, side


def plain_read(side: int, tier: str, cloud_pos: str, band: str,
               rsi_v: float, stoch_v: float, adx_v: float, chop_v: float,
               direction: str) -> str:
    """
    One plain sentence saying what the whole board adds up to.

    Written to be read in two seconds, and deliberately worded as a measured
    TENDENCY rather than an instruction. Nothing tested in this project earns
    the word "buy". The honest phrasing is which way the historical lean sat
    and how weak it was.
    """
    # regime clause, from the two readings that measure regime rather than direction
    if chop_v < 38 and adx_v > 25:   regime = "a strong move is underway"
    elif chop_v < 38:                regime = "a move is underway"
    elif chop_v > 62:                regime = "the market is chopping"
    elif adx_v > 25:                 regime = "trending"
    else:                            regime = "no clear regime"

    if side > 0:
        strength = {"full stack": "the strongest setup measured",
                    "stacked":    "a moderately supported setup",
                    "base":       "a weakly supported setup"}.get(tier, "")
        return (f"📖 <b>READ — leans UP.</b> Stretched low inside a thick cloud: "
                f"{strength}. Over the next 10–20 days this configuration has "
                f"historically drifted higher. The trend board below reads bearish, "
                f"and on FX that is the favourable side, not a contradiction. "
                f"Context: {regime}.")
    if side < 0:
        strength = {"full stack": "the mirror of the strongest setup",
                    "stacked":    "a moderately supported setup",
                    "base":       "a weakly supported setup"}.get(tier, "")
        return (f"📖 <b>READ — leans DOWN.</b> Stretched high inside a thick cloud: "
                f"{strength}. The short side tested positive but weaker than the "
                f"long side and did not reach significance. Context: {regime}.")

    # No cloud flag: say so plainly rather than manufacturing a view.
    where = {"above": "above the cloud", "below": "below the cloud",
             "inside": "inside the cloud"}.get(cloud_pos, "")
    if cloud_pos == "inside":
        return (f"📖 <b>READ — no lean.</b> Price is {where}, which measured as "
                f"consolidation: smaller moves than usual follow. RSI is stretched "
                f"but nothing here has a measured direction. Context: {regime}.")
    return (f"📖 <b>READ — no lean.</b> RSI is stretched, but price is {where} and "
            f"the cloud is {band} — the conditions that measured anything are absent. "
            f"Treat as awareness only. Context: {regime}.")


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

            # Regime context. Read on the same closed bar as the RSI value.
            try:
                a_s, dip_s, dim_s = adx_dmi(df)
                adx_v = float(a_s.iloc[-2])
                chop_v = float(choppiness(df).iloc[-2])
                dir_v = "+DI" if float(dip_s.iloc[-2]) > float(dim_s.iloc[-2]) else "-DI"
            except Exception:
                adx_v = chop_v = float("nan"); dir_v = "?"
            tag = regime_tag(adx_v, chop_v)
            try:
                dirline, votes = direction_panel(df, dir_v == "+DI")
            except Exception:
                dirline, votes = "", -1
            try:
                cloud, cloud_flag = ichimoku(df)
                cpos = cloud.split()[1] if cloud else ""
                cband = cloud.split("(")[1].split(")")[0] if "(" in cloud else ""
            except Exception:
                cloud, cloud_flag, cpos, cband = "", False, "", ""
            try:
                stoch_v = stochastic(df)
            except Exception:
                stoch_v = float("nan")
            stack_label, stack_tier, stack_side = setup_grade(cpos, cband, curr, stoch_v)
            read = plain_read(stack_side, stack_tier, cpos, cband,
                              curr, stoch_v, adx_v, chop_v, "")
            ctx = ("" if not tag else
                   f"{read}\n\n"
                   f"{dirline}\n"
                   + (f"{cloud}\n" if cloud else "")
                   + (f"⚑ <b>{stack_label}</b> ({stack_tier}, "
                      f"{'long' if stack_side > 0 else 'short'} side)\n" if stack_label else "")
                   + f"ADX {adx_v:.0f} · Chop {chop_v:.0f} · <i>{tag}</i>")

            if curr >= DIGEST_ABOVE or curr <= DIGEST_BELOW:
                extended.append((name, tf, curr))

            if LOG_OUTCOMES and stack_label and tf == "1d":
                pend = state.setdefault("_pending", [])
                for hz in OUTCOME_HORIZONS:
                    key = f"{name}|{bar_id}|{hz}"
                    if any(r.get("key") == key for r in pend):
                        continue
                    pend.append({"key": key, "pair": name, "ticker": ticker, "tf": tf,
                                 "cond": stack_tier, "side": stack_side,
                                 "price": price, "h": hz,
                                 "due": (now + timedelta(days=int(hz*1.45))).isoformat()})

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
                       + (f"{ctx}\n" if ctx else "")
                       + f"<i>{stamp} · closed bar · context, not confirmation</i>")
                if send_telegram(msg, dry):
                    sent += 1
                    state[key] = bar_id
                    print(f"  {name}|{tf:<3}  ALERT {direction} {lv}  (RSI {curr:.1f})")

            if not events:
                flag = "  <<" if (curr >= DIGEST_ABOVE or curr <= DIGEST_BELOW) else ""
                print(f"  {name}|{tf:<3}  RSI {curr:5.1f}  ADX {adx_v:4.0f}  Chop {chop_v:4.0f}{flag}")

    # ---- Forward-outcome logging -----------------------------------------
    # Score anything whose horizon has elapsed, then log today's flags.
    if LOG_OUTCOMES:
        pend = state.setdefault("_pending", [])
        tally = state.setdefault("_tally", {})
        still = []
        for rec in pend:
            try:
                due = datetime.fromisoformat(rec["due"])
            except Exception:
                continue
            if now < due:
                still.append(rec); continue
            df = fetch_batch([rec["ticker"]], rec["tf"])
            d2 = df.get(rec["ticker"])
            if d2 is None or d2.empty:
                still.append(rec); continue
            side = rec.get("side", 1)
            ret = (float(d2["Close"].iloc[-2]) / rec["price"] - 1) * 100 * side
            k = f"{rec['cond']}|{rec['h']}d"
            t = tally.setdefault(k, {"n": 0, "sum": 0.0, "wins": 0,
                                     "long_n": 0, "long_w": 0, "short_n": 0, "short_w": 0,
                                     "streak": 0, "maxw": 0, "maxl": 0,
                                     "best": None, "worst": None})
            t["n"] += 1; t["sum"] += ret; t["wins"] += (ret > 0)
            if side > 0: t["long_n"] += 1;  t["long_w"] += (ret > 0)
            else:        t["short_n"] += 1; t["short_w"] += (ret > 0)
            # streaks, as on any honest stats panel
            t["streak"] = (t["streak"] + 1) if ret > 0 and t["streak"] >= 0 else \
                          (t["streak"] - 1) if ret <= 0 and t["streak"] <= 0 else \
                          (1 if ret > 0 else -1)
            t["maxw"] = max(t["maxw"], t["streak"]); t["maxl"] = min(t["maxl"], t["streak"])
            t["best"]  = ret if t["best"]  is None else max(t["best"], ret)
            t["worst"] = ret if t["worst"] is None else min(t["worst"], ret)
            print(f"  SCORED {rec['pair']} {rec['cond']} {rec['h']}d "
                  f"{'long' if side>0 else 'short'} -> {ret:+.2f}%")
        state["_pending"] = still

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

    # ---- Periodic forward-test report ------------------------------------
    if LOG_OUTCOMES:
        tally = state.get("_tally", {})
        last = state.get("_last_report")
        due = True
        if last:
            try: due = now - datetime.fromisoformat(last) >= timedelta(days=OUTCOME_REPORT_EVERY_DAYS)
            except Exception: due = True
        if due and tally and sum(v["n"] for v in tally.values()) >= 10:
            lines = ["🧪 <b>Forward test — live results</b>", ""]
            for k in sorted(tally):
                v = tally[k]
                if not v["n"]: continue
                lw = f"{v['long_w']/v['long_n']*100:.0f}%" if v.get("long_n") else "—"
                sw = f"{v['short_w']/v['short_n']*100:.0f}%" if v.get("short_n") else "—"
                lines.append(f"<b>{k}</b>")
                lines.append(f"  n={v['n']}  avg <b>{v['sum']/v['n']:+.3f}%</b>  "
                             f"win {v['wins']/v['n']*100:.0f}%")
                lines.append(f"  long {lw} ({v.get('long_n',0)}) / "
                             f"short {sw} ({v.get('short_n',0)})")
                lines.append(f"  streak {v['streak']:+d} · best run {v['maxw']} · "
                             f"worst run {abs(v['maxl'])}")
                if v.get("best") is not None:
                    lines.append(f"  best {v['best']:+.2f}% · worst {v['worst']:+.2f}%")
                lines.append("")
            lines += [f"<i>Logged live since this bot started — nothing here was fitted "
                      f"to these observations.</i>",
                      f"<i>Long vs short win rates matter more than the headline: if longs "
                      f"far outperform shorts, that is drift, not edge.</i>",
                      f"<i>Backtest for comparison: base condition +0.125%/trade on daily, "
                      f"but only +0.050% on held-out H4.</i>",
                      f"<i>{len(state.get('_pending', []))} still pending.</i>"]
            if send_telegram("\n".join(lines), dry):
                sent += 1
                state["_last_report"] = now.isoformat()

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
