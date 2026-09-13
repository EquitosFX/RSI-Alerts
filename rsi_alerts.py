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
OUTCOME_HORIZONS = [10, 20]        # trading days to score the STACK signal at.
                                     # Only [0] (10d) is the real, validated horizon -
                                     # it drives Kelly sizing and the "time to close"
                                     # reminder. [1] (20d) is exploratory only: logged
                                     # to the tally for research, never sized, never
                                     # reminded on. Don't confuse this with
                                     # REVERSION_HORIZON_DAYS below (also 20) - that
                                     # one IS the real, live-traded reversion hold.
OUTCOME_REPORT_EVERY_DAYS = 30     # send a summary this often

# FX has no single universal "market close" the way stocks do, but most
# data feeds (and this bot's own daily bars) roll the trading day over
# around 5pm New York time - 21:00 UTC in winter (EST), 22:00 UTC in
# summer (EDT). A 3-hour window spanning both covers the DST shift without
# needing to track it explicitly. The final "time to close" reminder waits
# for this window even once its horizon is technically due, so it arrives
# at roughly the same time every day instead of at whatever hour the cron
# happened to run after the countdown elapsed. The midpoint check-in is
# NOT windowed - it's informational only, not an action point, so its
# exact timing matters far less.
CLOSE_WINDOW_START_UTC = 20
CLOSE_WINDOW_END_UTC = 23

# Append a short plain-English explainer to every alert. Off by default now
# that the wording ("leans") is familiar - flip back to True any time you
# want it back, e.g. after adding a new pair/instrument to the watchlist.
SHOW_EXPLAINER = False
# What counts as "extended" for the DIGEST only. Crossing alerts are unaffected
# by these - those use LEVELS_ABOVE / LEVELS_BELOW above.
# Measured on 85,255 daily RSI readings across 17 pairs:
#   70/30 -> 9.6% of readings qualify -> ~9 lines per digest at 31 instruments
#   75/25 -> 3.4%                     -> ~3 lines
#   80/20 -> 1.0%                     -> ~1 line
DIGEST_ABOVE = 75
DIGEST_BELOW = 25

# ---- Layer 7/8: portfolio exposure & position sizing ---------------------
# Layer 7: currency-level and pairwise-correlation exposure warnings across
# whatever is flagged THIS run. Layer 8: fractional-Kelly position sizing
# computed from the bot's OWN live _tally data above (never from a backtest
# figure - only forward-tested numbers reflect what has actually happened
# since deployment). Both are printed as CONTEXT, same philosophy as
# regime_tag(): never a hard gate on whether an alert fires, because this
# project already found hard-gating the RSI signal on a second indicator
# made results worse, not better.
try:
    ACCOUNT_EQUITY = float(os.environ.get("ACCOUNT_EQUITY") or "10000")  # for $ sizing only
except ValueError:
    # Guards against e.g. the ACCOUNT_EQUITY secret accidentally holding the
    # literal string "***" - GitHub masks secret VALUES as *** in log output;
    # if that masked text is ever copied back into the secret field instead
    # of the real number, this is what gets read at runtime. One bad config
    # value should never take down the whole bot - see fetch_cot() etc. for
    # the same never-crash-the-run philosophy applied elsewhere.
    print(f"  !! ACCOUNT_EQUITY env var isn't a valid number "
          f"({os.environ.get('ACCOUNT_EQUITY')!r}) - using $10,000 default. "
          f"Check Settings > Secrets and variables > Actions.")
    ACCOUNT_EQUITY = 10000.0
MAX_NET_CCY_EXPOSURE  = 2.0     # net same-direction stacked setups on one currency before warning
CORR_WINDOW           = 20      # bars for the rolling correlation check
CORR_THRESHOLD        = 0.75    # |rho| above this between two ACTIVE setups gets flagged
ATR_STOP_MULT         = 2.0     # stop distance used for sizing, in ATR multiples
ATR_TARGET_MULT       = 3.0     # target distance for the entry-plan line, in ATR multiples
                                 # (2.0/3.0 = 1.5:1 reward:risk, matching this project's
                                 # established ATR-entry-plan convention). NOTE: this specific
                                 # target level has NOT been backtested as an exit rule for any
                                 # signal here - Phase 2 validated a fixed-HORIZON exit (10/20
                                 # trading days), not a stop/target exit. Treat the target price
                                 # as a sizing/planning convenience, not a validated exit signal.
KELLY_FRACTION        = 0.5     # half-Kelly - full Kelly is well-documented as too aggressive
                                 # given real-world estimation error on a live sample
MAX_RISK_PCT          = 0.02    # hard cap regardless of what Kelly says
MIN_TRADES_FOR_KELLY  = 50      # below this the tally is noise, not edge - see kelly_from_tally()
DEFAULT_RISK_PCT      = 0.005   # fixed, conservative fallback while the sample builds

# ---- Extra context: z-score / HV percentile -------------------------------
# Phase 1 survey candidates (z-score mean reversion, historical-volatility
# percentile) - pure OHLC, nothing new to fetch. UNVALIDATED on this project;
# shown as one extra compact context line, same footing as ADX/Chop.
ZSCORE_WINDOW  = 20
HV_WINDOW      = 20
HV_LOOKBACK    = 100

# ---- Z-score / Keltner reversion alerts (Phase 2 validated, daily-only) --
# Both cleared the project's pooled/spread-charged/split-half protocol on
# daily with a 20-trading-day hold:
#   Z-score |z|>=2   PF 1.15  n=7,732  t=4.13  both halves PF 1.16/1.14
#   Keltner reversion PF 1.14  n=5,844  t=3.32  both halves PF 1.13/1.16
# NEITHER cleared the held-out H4 check (excess over baseline ~0 there:
# z-score -0.013%, Keltner -0.001%) - so these fire on 1d ONLY, never 1h/4h.
# They also do NOT stack with each other - tested WORSE combined (avg
# +0.111%, PF 1.11) than either alone, because both measure essentially the
# same thing (distance from mean in vol-adjusted units). Kept as two
# separate, independently-tallied alert types - never OR'd or AND'd, and
# deliberately NOT folded into stack_tier/quality_score above, which stay
# scoped to the cloud+Stoch+RSI combination they were validated on.
ZREV_THRESH          = 2.0    # |z| >= this triggers a reversion entry
KELTNER_WINDOW        = 20
KELTNER_MULT          = 2.0
REVERSION_HORIZON_DAYS = 20   # the only horizon that cleared Phase 2 for these

# ---- COT (Commitments of Traders) overlay ---------------------------------
# CFTC Legacy Futures-Only report, free, Socrata Open Data API, no key
# required. Weekly cadence (Tuesday positions, released the following
# Friday ~3:30pm ET) - cached so the bot doesn't hit it every hourly run
# for data that only changes once a week. UNVALIDATED as a signal here;
# shown as one context line, same footing as everything above.
COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
# WATCHLIST name -> distinguishing substring of the CFTC contract name,
# matched with SoQL LIKE (so exact CFTC spacing/capitalization doesn't need
# to be exact - only present in market_and_exchange_names).
COT_CONTRACTS = {
    "EURUSD": "EURO FX",
    "GBPUSD": "BRITISH POUND",
    "USDJPY": "JAPANESE YEN",       # net-long yen speculators = USDJPY headwind
    "USDCHF": "SWISS FRANC",
    "USDCAD": "CANADIAN DOLLAR",
    "AUDUSD": "AUSTRALIAN DOLLAR",
    "NZDUSD": "NZ DOLLAR",
    "GOLD":   "GOLD",
    "SILVER": "SILVER",
}
COT_CACHE_DAYS = 3   # the underlying data only updates weekly - no reason
                      # to re-fetch more often than this

# ---- NFP proximity guardrail (z-score reversion ONLY) ---------------------
# Phase 2 found z-score reversion specifically underperforms when entered
# within +/-2 calendar days of an NFP release: PF 1.03 near vs 1.11 far,
# split-half CONSISTENT in both groups (see signal_vs_nfp.py) - the near
# figure doesn't even clear this project's own PF 1.10 validation bar.
# Keltner reversion showed the OPPOSITE (better near NFP, also split-half
# consistent), so this gate applies ONLY to zscore_rev - never keltner_rev,
# never the stack.
#
# FRED is free and needs its own key (separate signup from anything else
# here): https://fred.stlouisfed.org/docs/api/api_key.html
# Falls back to a hardcoded "first Friday of the month" approximation if
# FRED is unreachable AND no cached dates exist at all - a network hiccup
# on this must never silently disable or crash the whole bot. NFP shifts
# off the first Friday only rarely (holiday collisions), so the fallback
# is a deliberate approximation, not a guess pulled from nowhere.
FRED_API_KEY   = os.environ.get("FRED_API_KEY", "")
NFP_WINDOW_DAYS = 2    # matches the tested window exactly - not re-tuned here
NFP_CACHE_DAYS  = 14   # NFP dates are announced months ahead and don't change -
                        # no need to refetch often

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
# Currently: the 28 standard FX pairs plus gold, silver and DXY = 31 instruments.
WATCHLIST = {**MAJORS, **EUR_CROSSES, **GBP_CROSSES, **OTHER_CROSSES, **METALS_AND_INDEX}
# WATCHLIST = {**MAJORS, **METALS_AND_INDEX}                                          # quieter: 10
# WATCHLIST = {**MAJORS, **EUR_CROSSES, **GBP_CROSSES, **OTHER_CROSSES, **EXOTICS, **METALS_AND_INDEX}  # 41

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


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder-smoothed ATR in price units. adx_dmi() computes the same thing
    inline for its own normalization; kept as its own function too because
    Layer 8 sizing below needs an ATR value directly."""
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


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


def zscore(close: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """
    Rolling z-score: (price - rolling mean) / rolling std. A continuous
    mean-reversion read in the same conceptual family as RSI, but not
    bounded 0-100 - large |z| means price is stretched relative to its own
    recent range in raw standard-deviation terms.

    Phase 1 survey candidate - UNVALIDATED on this project. Shown as one
    extra context line, same footing as ADX/Chop: never a signal or a gate
    on its own.
    """
    m = close.rolling(window).mean()
    s = close.rolling(window).std()
    return (close - m) / s


def hv_percentile(close: pd.Series, window: int = HV_WINDOW,
                   lookback: int = HV_LOOKBACK) -> pd.Series:
    """
    Where today's realized volatility (annualized stdev of log returns over
    `window` bars) ranks against its own trailing `lookback` history, 0-100.
    High = volatility itself is stretched, regardless of direction - a
    different read than ADX/Choppiness, which measure trend strength and
    directional persistence rather than the magnitude of movement.

    Phase 1 survey candidate - UNVALIDATED on this project. Context only.
    """
    log_ret = np.log(close).diff()
    real_vol = log_ret.rolling(window).std() * np.sqrt(252)
    return real_vol.rolling(lookback).rank(pct=True) * 100


def keltner(df: pd.DataFrame, window: int = KELTNER_WINDOW, mult: float = KELTNER_MULT):
    """
    Keltner Channels: EMA(window) +/- mult*ATR(window).

    Phase 2 VALIDATED as a daily mean-reversion signal (PF 1.14, n=5,844,
    both split-halves consistent) - did NOT clear the held-out H4 check
    (excess over baseline ~0 there). See REVERSION_HORIZON_DAYS in CONFIG
    and the alert-firing block in check_all() for the entry logic (close
    crossing back inside the bands after being outside them).
    """
    ema = df["Close"].ewm(span=window, adjust=False).mean()
    a = atr(df, window)
    return ema + mult * a, ema - mult * a


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


def quality_score(stack_tier: str, adx_v: float, chop_v: float) -> int:
    """
    A 0-5 at-a-glance score built by EXTENDING stack_tier above, not
    replacing it. stack_tier already IS a 1-3 quality read (base=1,
    stacked=2, full stack=3 - how many of {thick cloud, Stoch, RSI} fired),
    and that 1-3 part is the piece this project has actually validated
    (pooled/spread-charged/split-half - see setup_grade()'s docstring).

    The extra two points come from data the bot already computes and add
    NOTHING new to fetch or store:
      +1  ADX >= 25   (a trend is actually present)
      +1  Chop < 38   (a directional move is underway)

    IMPORTANT CAVEAT: those two extra points are NOT validated as quality
    signals. This project already tested ADX/Choppiness as a hard GATE on
    the RSI signal and found it made results worse, not better (see
    regime_tag()'s docstring). Using them here as additive, non-gating
    context is a different claim than gating - but it is an UNTESTED one
    until it goes through the same split-half protocol everything else
    here has been through. Read "5/5" as "more context lines up," not as
    "more likely to work" - the same warning setup_grade() gives about not
    reading "full stack" as better than "base" applies here too.

    Returns 0 when there is no stack at all (stack_tier == "").
    """
    base_pts = {"": 0, "base": 1, "stacked": 2, "full stack": 3}.get(stack_tier, 0)
    if base_pts == 0:
        return 0
    extra = 0
    if not np.isnan(adx_v) and adx_v >= 25:  extra += 1
    if not np.isnan(chop_v) and chop_v < 38: extra += 1
    return base_pts + extra


def board_note(votes: int) -> str:
    """
    What the four-indicator board actually implies, which is the opposite of
    what it looks like.

    Measured across 17 FX pairs, daily, 10-day forward, excess over each pair's
    own baseline. The BULLISH state of every one of these measured negative:

        price > EMA50   -0.023pp     price < EMA50   +0.024pp
        MACD > signal   -0.011pp     MACD < signal   +0.011pp
        SAR below price +0.000pp     SAR above price -0.000pp
        +DI > -DI       -0.041pp     -DI > +DI       +0.057pp   (with ADX>25)

    So 4/4 up is the LEAST favourable reading on the board, not the most.
    Roughly 35 standard indicators have now been tested on these pairs and the
    result is the same every time: FX daily mean-reverts at this horizon.
    """
    if votes < 0:
        return ""                      # panel unavailable - say nothing rather than guess
    if votes >= 4:
        return ("all four trend reads bullish — on FX that has been the "
                "<i>least</i> favourable state, not the most")
    if votes == 3:
        return "mostly bullish — historically a mild headwind on FX"
    if votes == 2:
        return "split — the board says nothing either way"
    if votes == 1:
        return "mostly bearish — historically the mildly favourable side on FX"
    return ("all four trend reads bearish — historically the favourable "
            "side on FX, which is why this is not a warning")


def explainer() -> str:
    """One short footer so the wording never has to be decoded from memory."""
    return ("ℹ️ <i>“Leans” means a measured historical tendency, not advice. "
            "Built from 17 FX pairs, 2010–2026: price stretched low inside a thick "
            "cloud averaged +0.13% over 10 days on daily data — but only +0.05% on "
            "held-out H4, and that was not significant. Trend indicators read "
            "backwards on FX because these pairs mean-revert. Small effects, "
            "never forward-tested. Treat every alert as a place to look, not a "
            "reason to trade.</i>")


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
        strength = {"full stack": "the strongest version of this setup, by what's been tested",
                    "stacked":    "a moderately-supported version",
                    "base":       "a weakly-supported version"}.get(tier, "")
        return (f"📖 <b>READ — leans UP.</b> Price has dropped sharply while still inside "
                f"a well-established trend zone (the cloud) — {strength}. Setups like this "
                f"have historically drifted higher over the next 10–20 days. ({regime.capitalize()}.)")
    if side < 0:
        strength = {"full stack": "the mirror-image of the strongest setup measured",
                    "stacked":    "a moderately-supported version",
                    "base":       "a weakly-supported version"}.get(tier, "")
        return (f"📖 <b>READ — leans DOWN.</b> Price has risen sharply while still inside "
                f"a well-established trend zone (the cloud) — {strength}. Historically this "
                f"has drifted lower, though the short side tested weaker than the long side "
                f"and didn't reach significance on its own. ({regime.capitalize()}.)")

    # No cloud flag: say so plainly rather than manufacturing a view.
    where = {"above": "above the cloud", "below": "below the cloud",
             "inside": "inside the cloud"}.get(cloud_pos, "")
    if cloud_pos == "inside":
        return (f"📖 <b>READ — no lean.</b> Price is {where} (consolidating) — smaller "
                f"moves than usual tend to follow. RSI is stretched, but nothing here "
                f"points a direction. ({regime.capitalize()}.)")
    return (f"📖 <b>READ — no lean.</b> RSI is stretched, but price is {where} and the "
            f"cloud is {band} — the conditions that give this signal its edge aren't "
            f"present right now. Treat as background awareness only. ({regime.capitalize()}.)")


# ==========================================================================
# LAYER 7 — PORTFOLIO CORRELATION / EXPOSURE GUARDRAILS
# ==========================================================================
# Neither function below blocks an alert from firing. This project already
# found that hard-gating the RSI signal on a second indicator (regime) made
# results worse, not better - so these are printed alongside the digest for
# a human to act on, not wired in as a filter. What they catch: several
# "independent" setups this run that are actually one leveraged bet wearing
# different tickets.

_FX_SPECIAL_LEGS = {"GOLD": ("XAU", "USD"), "SILVER": ("XAG", "USD"),
                     "DXY": ("USD", "DXY_BASKET")}


def currency_legs(name: str) -> tuple[str, str] | None:
    """Decompose a WATCHLIST name into (base, quote) legs. Gold/silver get a
    synthetic USD leg so a XAU long still nets against other USD exposure.
    DXY gets its own synthetic basket leg rather than raw USD, since DXY is
    a basket instrument, not a bilateral pair - folding it into USD directly
    would misstate the netting."""
    if name in _FX_SPECIAL_LEGS:
        return _FX_SPECIAL_LEGS[name]
    if len(name) == 6:
        return name[:3], name[3:]
    return None


def net_currency_exposure(active: list[tuple[str, int]]) -> dict[str, float]:
    """
    Net exposure per currency across every setup flagged THIS run, not just
    per-pair risk. Two long stack signals on EURUSD and EURGBP look like two
    independent 1%-risk trades; both are actually +EUR, so taking both is one
    ~2x EUR bet wearing two tickets.

    active: [(instrument_name, side), ...], side = stack_side (+1/-1) from
    setup_grade() - i.e. only instruments with a live stack_label this run.

    Returns net exposure per currency in units of "number of stacked same-
    direction setups" (a count, not a risk %) - deliberately not converted
    to % of equity here, since that would conflate the correlation problem
    with the separate sizing decision made in kelly_size() below.
    """
    net: dict[str, float] = {}
    for name, side in active:
        legs = currency_legs(name)
        if legs is None:
            net[name] = net.get(name, 0.0) + side
            continue
        base, quote = legs
        net[base] = net.get(base, 0.0) + side
        net[quote] = net.get(quote, 0.0) - side
    return net


def exposure_warnings(net: dict[str, float], max_net: float = MAX_NET_CCY_EXPOSURE) -> list[str]:
    """Currencies where stacked same-direction setups reach/exceed max_net."""
    warn = []
    for ccy, val in sorted(net.items(), key=lambda x: -abs(x[1])):
        if abs(val) >= max_net:
            warn.append(f"{ccy}: {val:+.1f} net {'long' if val > 0 else 'short'} across active setups")
    return warn


def pair_correlation(closes: dict[str, pd.Series], window: int = CORR_WINDOW) -> pd.DataFrame:
    """
    Rolling correlation of daily log returns between currently-ACTIVE pairs
    only (not all 31 - a full matrix is expensive and mostly irrelevant to
    exposure you don't actually hold). Catches co-movement currency-netting
    can't see - e.g. AUDUSD and NZDUSD share no currency leg but both trade
    as commodity-bloc/risk-sentiment currencies.

    closes: {instrument_name: recent Close series}, already aligned.
    NOTE: this reads a CALM trailing window. Crisis periods push correlated
    pairs toward +/-1 - if this matters for sizing, re-run it against a
    stressed historical window too, not just the live trailing one.
    """
    if len(closes) < 2:
        return pd.DataFrame()
    rets = pd.DataFrame({k: np.log(v).diff() for k, v in closes.items()}).dropna()
    if len(rets) < window:
        return pd.DataFrame()
    return rets.tail(window).corr()


def correlation_warnings(corr: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> list[str]:
    warn = []
    if corr.empty:
        return warn
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho = corr.iloc[i, j]
            if abs(rho) >= threshold:
                warn.append(f"{cols[i]} / {cols[j]}: ρ={rho:+.2f}")
    return warn


# ==========================================================================
# LAYER 8 — POSITION SIZING (fractional Kelly, from the bot's OWN live data)
# ==========================================================================
def kelly_from_tally(tally_entry: dict) -> tuple[float, str] | None:
    """
    Full Kelly fraction f* = p - q/b from one entry of state["_tally"] - the
    bot's own live forward-outcome log (see the LOG_OUTCOMES block in
    check_all()), NOT a backtest figure. p = win rate, q = 1-p, b = payoff
    ratio (avg win / avg loss).

    b is computed from the tally's running win_sum/loss_sum when present -
    the correct avg-win/avg-loss definition. Tally entries recorded before
    that split existed (or that haven't scored a full outcome under it yet)
    fall back to |best|/|worst|, a coarser approximation from the extremes
    only. Returns (f*, method) so the caller can say which one was used
    rather than silently presenting an approximation as precise.

    Returns None below MIN_TRADES_FOR_KELLY (sample too thin to trust the
    win rate at all - see this project's own ~1,100-trade significance
    calculations elsewhere), or if the sample has no losses/no wins (b
    undefined).
    """
    n = tally_entry.get("n", 0)
    if n < MIN_TRADES_FOR_KELLY:
        return None
    wins = tally_entry.get("wins", 0)
    losses = n - wins
    if wins == 0 or losses == 0:
        return None
    p = wins / n
    q = 1 - p

    win_sum, loss_sum = tally_entry.get("win_sum"), tally_entry.get("loss_sum")
    if win_sum is not None and loss_sum is not None and win_sum > 0 and loss_sum < 0:
        avg_win, avg_loss = win_sum / wins, abs(loss_sum / losses)
        b = avg_win / avg_loss if avg_loss > 0 else None
        method = "avg win/loss"
    else:
        best, worst = tally_entry.get("best"), tally_entry.get("worst")
        b = abs(best / worst) if best and worst and worst != 0 else None
        method = "best/worst approx"

    if not b or b <= 0:
        return None
    return p - q / b, method


def kelly_size(tally_entry: dict | None, equity: float = ACCOUNT_EQUITY,
               stop_pct: float | None = None) -> dict:
    """
    Turn one tally entry into a concrete size: a dollar risk amount and,
    given a stop distance, a notional position size - same spirit as an ATR
    stop, a number instead of a vibe.

    Falls back to a small FIXED risk % (never scaled up) whenever the sample
    is too thin for Kelly to mean anything - an untested condition has no
    measured edge to size against, so the fallback stays flat rather than
    guessing at one.

    stop_pct: stop distance as a fraction of price (e.g. ATR_STOP_MULT *
    atr / price). Pass None to get the risk verdict without a notional size.
    """
    result = kelly_from_tally(tally_entry) if tally_entry else None
    if result is None or result[0] <= 0:
        risk_pct = DEFAULT_RISK_PCT
        basis = "fallback — sample below the trust threshold or no measured edge yet"
    else:
        f_raw, method = result
        risk_pct = min(f_raw * KELLY_FRACTION, MAX_RISK_PCT)
        basis = f"half-Kelly of live f*={f_raw:.3f} ({method}, n={tally_entry.get('n')})"
    risk_amount = equity * risk_pct
    notional = (risk_amount / stop_pct) if stop_pct else None
    return {"risk_pct": risk_pct, "risk_amount": round(risk_amount, 2),
            "notional": round(notional, 2) if notional else None, "basis": basis}


def entry_plan(price: float, atr_v: float, side: int) -> dict | None:
    """
    Concrete stop and target PRICE levels from ATR - a number to put in an
    order, not just a risk percentage. Defaults to 2.0/3.0x ATR (1.5:1
    reward:risk), matching this project's established ATR-entry-plan
    convention.

    IMPORTANT - for z-score/Keltner reversion specifically, this has been
    TESTED as an early-exit rule and found WORSE than the validated
    fixed-horizon exit: PF fell from 1.15/1.14 to 0.99/1.03 (see
    exit_comparison_backtest.py). The stop alone gets hit ~46% of the time,
    before the slow drift these signals actually run on has room to play
    out - these are small, patient effects, not fast moves a tight ATR stop
    is suited to protect. Use the levels for sizing / a real order to place,
    but hold the full validated horizon rather than exiting early on them.

    For the main cloud/Stoch/RSI stack, this specific question (early exit
    vs its 10d horizon) has NOT been tested - don't assume the same finding
    carries over; it's a different signal on a different horizon.

    Returns None if atr_v is unusable (NaN) or price is zero/missing -
    never fabricate a level from missing data.
    """
    if np.isnan(atr_v) or not price:
        return None
    if side > 0:
        stop, target = price - ATR_STOP_MULT * atr_v, price + ATR_TARGET_MULT * atr_v
    else:
        stop, target = price + ATR_STOP_MULT * atr_v, price - ATR_TARGET_MULT * atr_v
    return {"entry": price, "stop": stop, "target": target,
            "stop_dist_pct": ATR_STOP_MULT * atr_v / price * 100,
            "target_dist_pct": ATR_TARGET_MULT * atr_v / price * 100}


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


def fetch_cot(contract_substring: str) -> dict | None:
    """
    Latest CFTC Legacy Futures-Only row for a currency/metal futures
    contract, matched by substring against market_and_exchange_names (SoQL
    LIKE, case- and spacing-tolerant). Returns speculative (non-commercial)
    net position and week-over-week change, or None on any failure - a COT
    hiccup must never break the RSI checks everything else here depends on.

    Free, no API key. Data itself is weekly (Tuesday positions, released
    the following Friday ~3:30pm ET) - see COT_CACHE_DAYS in CONFIG for why
    this isn't called every run.
    """
    try:
        params = {
            "$limit": 1,
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$where": (f"market_and_exchange_names like '%{contract_substring}%' "
                       f"AND futonly_or_combined = 'FutOnly'"),
        }
        r = requests.get(COT_URL, params=params, timeout=20)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        row = rows[0]
        long_ = float(row["noncomm_positions_long_all"])
        short_ = float(row["noncomm_positions_short_all"])
        chg_long = float(row.get("change_in_noncomm_long_all") or 0)
        chg_short = float(row.get("change_in_noncomm_short_all") or 0)
        oi = float(row.get("open_interest_all") or 0)
        net = long_ - short_
        return {
            "date": row["report_date_as_yyyy_mm_dd"][:10],
            "net": net,
            "net_chg": chg_long - chg_short,
            "pct_oi": (net / oi * 100) if oi else None,
        }
    except Exception as e:
        print(f"  !! COT fetch failed for {contract_substring}: {e}")
        return None


def cot_context(name: str, cache: dict) -> str:
    """One compact COT context line for `name`, or "" if not covered or not
    yet cached. `cache` is state["_cot_cache"] - refreshed in check_all()
    at most once every COT_CACHE_DAYS."""
    if name not in COT_CONTRACTS:
        return ""
    c = cache.get(name)
    if not c:
        return ""
    arrow = "↑" if c["net_chg"] > 0 else "↓" if c["net_chg"] < 0 else "→"
    side = "net long" if c["net"] > 0 else "net short"
    pct = f" ({c['pct_oi']:+.0f}% OI)" if c.get("pct_oi") is not None else ""
    return f"COT spec {side} {abs(c['net']):,.0f}{pct} {arrow} · {c['date']}"


def fetch_nfp_dates_fred(api_key: str) -> list[str] | None:
    """
    Real NFP (Employment Situation) release dates from FRED, most recent
    first. Returns None on ANY failure (no key, network error, bad
    response) - never raises - so a FRED hiccup falls through to the
    cache-or-fallback logic in check_all() instead of breaking the run.

    limit=24 (~2 years of monthly releases) is generous on purpose: cheap
    to fetch, and comfortably covers "is today near a release" regardless
    of exactly how far into the future FRED's list extends.
    """
    if not api_key:
        return None
    try:
        r = requests.get("https://api.stlouisfed.org/fred/series/release",
                          params={"series_id": "PAYEMS", "api_key": api_key, "file_type": "json"},
                          timeout=15)
        r.raise_for_status()
        releases = r.json().get("releases", [])
        if not releases:
            return None
        release_id = releases[0]["id"]
        r2 = requests.get("https://api.stlouisfed.org/fred/release/dates",
                           params={"release_id": release_id, "api_key": api_key,
                                   "file_type": "json", "limit": 24, "sort_order": "desc"},
                           timeout=15)
        r2.raise_for_status()
        dates = [d["date"] for d in r2.json().get("release_dates", [])]
        return dates if dates else None
    except Exception as e:
        print(f"  !! NFP date fetch from FRED failed: {e}")
        return None


def nfp_dates_fallback(around: datetime, months: int = 2) -> list[str]:
    """
    Hardcoded approximation: NFP as the first Friday of the month, for the
    month of `around` plus a few months either side. Used ONLY when FRED
    is unreachable and there is no existing cache at all (see check_all())
    - NFP shifts off the first Friday a few times a year around holidays,
    so this is a deliberate approximation, not a substitute for the real
    schedule.
    """
    out = []
    base = around.replace(day=1)
    for offset in range(-months, months + 1):
        year = base.year + (base.month - 1 + offset) // 12
        month = (base.month - 1 + offset) % 12 + 1
        month_start = datetime(year, month, 1)
        days_to_friday = (4 - month_start.weekday()) % 7   # Mon=0 ... Fri=4
        first_friday = month_start + timedelta(days=days_to_friday)
        out.append(first_friday.strftime("%Y-%m-%d"))
    return out


def is_near_nfp(check_date, nfp_cache: dict) -> bool:
    """True if check_date is within NFP_WINDOW_DAYS of any date in
    nfp_cache["dates"] (source: FRED when available, first-Friday
    approximation otherwise - see the refresh logic in check_all()).
    Returns False (never gates) if there's no date info at all - missing
    data should never silently suppress a signal."""
    dates = nfp_cache.get("dates") or []
    if not dates:
        return False
    check_ts = pd.Timestamp(check_date)
    for d in dates:
        if abs((pd.Timestamp(d) - check_ts).days) <= NFP_WINDOW_DAYS:
            return True
    return False


def check_all(dry: bool = False) -> int:
    state = load_state()
    sent = 0
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d %H:%M UTC")
    extended: list[tuple] = []
    active_setups: list[tuple[str, int]] = []   # Layer 7: (name, side) for every live stack this run
    daily_closes: dict[str, pd.Series] = {}      # Layer 7: 1d closes, for the correlation check

    # ---- COT refresh (weekly data - cache it, don't hit it every run) ----
    cot_cache = state.setdefault("_cot_cache", {})
    last_cot = state.get("_cot_last_fetch")
    need_cot_refresh = True
    if last_cot:
        try:
            need_cot_refresh = (now - datetime.fromisoformat(last_cot)) >= timedelta(days=COT_CACHE_DAYS)
        except Exception:
            need_cot_refresh = True
    wanted = [nm for nm in COT_CONTRACTS if nm in WATCHLIST]
    if need_cot_refresh:
        fetched = 0
        for nm in wanted:
            c = fetch_cot(COT_CONTRACTS[nm])
            if c:
                cot_cache[nm] = c
                fetched += 1
        state["_cot_last_fetch"] = now.isoformat()
        print(f"\n  --- COT refresh: {fetched}/{len(wanted)} contracts fetched "
              f"({', '.join(sorted(cot_cache.keys())) or 'none cached'}) ---")
    else:
        age_days = (now - datetime.fromisoformat(last_cot)).days
        print(f"\n  --- COT refresh skipped (cached {age_days}d ago, refreshes every "
              f"{COT_CACHE_DAYS}d; {len(cot_cache)}/{len(wanted)} contracts in cache) ---")

    # ---- NFP dates refresh (for the z-score reversion guardrail) ---------
    nfp_cache = state.setdefault("_nfp_cache", {})
    last_nfp = state.get("_nfp_last_fetch")
    need_nfp_refresh = True
    if last_nfp:
        try:
            need_nfp_refresh = (now - datetime.fromisoformat(last_nfp)) >= timedelta(days=NFP_CACHE_DAYS)
        except Exception:
            need_nfp_refresh = True
    if need_nfp_refresh:
        fetched = fetch_nfp_dates_fred(FRED_API_KEY)
        if fetched:
            nfp_cache["dates"] = fetched
            nfp_cache["source"] = "fred"
            state["_nfp_last_fetch"] = now.isoformat()
            print(f"  --- NFP dates refreshed from FRED ({len(fetched)} dates, "
                  f"most recent {fetched[0]}) ---")
        elif not nfp_cache.get("dates"):
            fallback_dates = nfp_dates_fallback(now)
            nfp_cache["dates"] = fallback_dates
            nfp_cache["source"] = "fallback"
            state["_nfp_last_fetch"] = now.isoformat()
            print(f"  --- NFP dates: FRED unavailable, using first-Friday fallback "
                  f"({len(fallback_dates)} dates) ---")
        else:
            print(f"  --- NFP dates: FRED unavailable, keeping existing "
                  f"{nfp_cache.get('source', '?')}-sourced cache ---")
    else:
        print(f"  --- NFP dates: cached ({nfp_cache.get('source', '?')}), "
              f"{len(nfp_cache.get('dates', []))} dates on file ---")

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
                atr_v = float(atr(df).iloc[-2])
                dir_v = "+DI" if float(dip_s.iloc[-2]) > float(dim_s.iloc[-2]) else "-DI"
            except Exception:
                adx_v = chop_v = atr_v = float("nan"); dir_v = "?"
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
            try:
                z_series = zscore(df["Close"])
                z_v = float(z_series.iloc[-2])
                z_prev = float(z_series.iloc[-3])
                hv_v = float(hv_percentile(df["Close"]).iloc[-2])
            except Exception:
                z_v = z_prev = hv_v = float("nan")
            try:
                kelt_upper, kelt_lower = keltner(df)
                ku_curr, kl_curr = float(kelt_upper.iloc[-2]), float(kelt_lower.iloc[-2])
                ku_prev, kl_prev = float(kelt_upper.iloc[-3]), float(kelt_lower.iloc[-3])
                c_prev = float(df["Close"].iloc[-3])
            except Exception:
                ku_curr = kl_curr = ku_prev = kl_prev = c_prev = float("nan")
            stack_label, stack_tier, stack_side = setup_grade(cpos, cband, curr, stoch_v)
            if tf != "1d":
                # The cloud/Stoch/RSI stack was only ever validated on daily
                # bars - its own held-out H4 retest already showed the edge
                # weakening there, and live signal-scorecard results (n=23,
                # 22/23 on 1h/4h, PF 0.88 vs PF 0.75 on "full stack" 1h/4h
                # specifically) are consistent with that. Same discipline
                # already applied to z-score/Keltner reversion: no verdict,
                # no sizing, no entry plan outside 1d.
                stack_label, stack_tier, stack_side = "", "", 0
            qscore = quality_score(stack_tier, adx_v, chop_v)
            read = plain_read(stack_side, stack_tier, cpos, cband,
                              curr, stoch_v, adx_v, chop_v, "")

            # Layer 7 tracking: every live stack this run feeds the portfolio
            # exposure/correlation check that runs once, after the tf loop.
            kelly = None
            plan = None
            if stack_label:
                active_setups.append((name, stack_side))
                tk = f"{stack_tier}|{OUTCOME_HORIZONS[0]}d"
                stop_pct = (ATR_STOP_MULT * atr_v / price) if not np.isnan(atr_v) and price else None
                kelly = kelly_size(state.get("_tally", {}).get(tk), stop_pct=stop_pct)
                plan = entry_plan(price, atr_v, stack_side)
            if tf == "1d":
                daily_closes[name] = df["Close"]

            # One-line verdict up top - direction, tier, quality, size - so
            # the decision-relevant bit is visible even in a truncated phone
            # notification preview, before the fuller narrative below it.
            verdict = ("" if not stack_label else
                       f"🎯 <b>{'LONG' if stack_side > 0 else 'SHORT'}</b> · {stack_tier} "
                       f"· Quality {qscore}/5"
                       + (f" · risk {kelly['risk_pct']*100:.2f}% (${kelly['risk_amount']:,.0f})"
                          if kelly else "")
                       + "\n"
                       + (f"Stop {plan['stop']:,.4f} (-{plan['stop_dist_pct']:.1f}%) · "
                          f"Target {plan['target']:,.4f} (+{plan['target_dist_pct']:.1f}%)\n"
                          if plan else "")
                       + "\n")

            # Trend board + cloud + regime, compacted from 4 lines to 2 -
            # same information, faster to scan.
            board_line = ""
            if dirline:
                note = board_note(votes)
                board_line = (dirline + (f" · <i>{note}</i>" if note else "") + "\n"
                              + "<i>  ↳ 4 trend reads: EMA=price vs its 50-day average · "
                                "MACD=momentum vs its own signal line · "
                                "SAR=stop-and-reverse dots · DI=directional strength (part of ADX)</i>\n")

            direction_word = {"+DI": "up", "-DI": "down", "?": "n/a"}.get(dir_v, "n/a")
            cloud_note = ""
            if cloud:
                pos_word = cloud.split()[1] if len(cloud.split()) > 1 else ""
                band_word = cloud.split("(")[1].split(")")[0] if "(" in cloud else ""
                cloud_note = (f"{cloud} — price is {pos_word} the Ichimoku trend zone "
                              f"({band_word}-thickness)\n")
            detail_line = (cloud_note
                            + f"ADX {adx_v:.0f} <i>(trend strength, 0-100)</i> · "
                              f"Chop {chop_v:.0f} <i>(choppiness, 0-100 - lower means a real "
                              f"move is underway)</i> · direction: {direction_word} · <i>{tag}</i>")
            extra_bits = []
            if not np.isnan(z_v):
                extra_bits.append(f"Z {z_v:+.1f} <i>(std. deviations from its 20-day average)</i>")
            if not np.isnan(hv_v):
                vol_read = "unusually volatile" if hv_v >= 80 else "unusually calm" if hv_v <= 20 else "normal range"
                extra_bits.append(f"HVpct {hv_v:.0f} <i>(today's volatility is higher than "
                                   f"{hv_v:.0f}% of the past year - {vol_read})</i>")
            cot_line = cot_context(name, cot_cache)
            if cot_line: extra_bits.append(cot_line)
            extra_line = (" · ".join(extra_bits) + "\n") if extra_bits else ""

            ctx = ("" if not tag else
                   verdict
                   + f"{read}\n\n"
                   + "🔬 <i>Technical detail (not needed to act on this):</i>\n"
                   + board_line
                   + detail_line + "\n"
                   + extra_line)

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
                                 "price": price, "h": hz, "created": now.isoformat(),
                                 "due": (now + timedelta(days=int(hz*1.45))).isoformat()})

            events = []
            for lv in sorted(LEVELS_ABOVE):
                if prev <= lv < curr:
                    events.append(("ABOVE", lv))
            for lv in sorted(LEVELS_BELOW, reverse=True):
                if prev >= lv > curr:
                    events.append(("BELOW", lv))

            for direction, lv in events:
                if not (stack_label and plan):
                    # No validated trade call attached to this crossing (no
                    # stack confirmation, or not on 1d where the stack is
                    # actually validated) - the bot now only sends messages
                    # that are an actual trade call with a stop and target,
                    # never a bare RSI reading on its own.
                    continue
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
                       f"RSI({RSI_LENGTH}) crossed <b>{direction} {lv}</b> ({word}) "
                       f"→ <b>{curr:.1f}</b> (was {prev:.1f}) · {price:,.4f}\n"
                       + (f"{ctx}\n" if ctx else "")
                       + f"<i>{stamp} · closed bar</i>"
                       + (f"\n\n{explainer()}" if SHOW_EXPLAINER else ""))
                if send_telegram(msg, dry):
                    sent += 1
                    state[key] = bar_id
                    print(f"  {name}|{tf:<3}  ALERT {direction} {lv}  (RSI {curr:.1f})")

            # ---- Z-score / Keltner reversion alerts --------------------------
            # Phase 2 validated, daily-only (see REVERSION_HORIZON_DAYS in
            # CONFIG) - independent of the RSI-crossing signal above, never
            # combined with each other. Two separate signal types, each with
            # its own dedup key, tally key, and Kelly sizing lookup.
            if tf == "1d":
                zrev_long = (not np.isnan(z_prev) and z_prev > -ZREV_THRESH and z_v <= -ZREV_THRESH)
                zrev_short = (not np.isnan(z_prev) and z_prev < ZREV_THRESH and z_v >= ZREV_THRESH)
                keltrev_long = (not np.isnan(kl_prev) and c_prev >= kl_prev and price < kl_curr)
                keltrev_short = (not np.isnan(ku_prev) and c_prev <= ku_prev and price > ku_curr)

                for sig_name, icon, label, fired, side in [
                    ("zscore_rev", "🔄", "Z-score reversion", zrev_long, 1),
                    ("zscore_rev", "🔄", "Z-score reversion", zrev_short, -1),
                    ("keltner_rev", "〰️", "Keltner reversion", keltrev_long, 1),
                    ("keltner_rev", "〰️", "Keltner reversion", keltrev_short, -1),
                ]:
                    if not fired:
                        continue
                    if sig_name == "zscore_rev" and is_near_nfp(df.index[-2], nfp_cache):
                        # Validated finding (signal_vs_nfp.py): z-score
                        # reversion entered within +/-2 days of an NFP
                        # release scores PF 1.03 vs 1.11 far-from-NFP,
                        # split-half consistent in both groups - doesn't
                        # even clear this project's own 1.10 bar. Keltner
                        # reversion showed the OPPOSITE and is NOT gated.
                        print(f"  {name}|{tf:<3}  SKIPPED {sig_name} {('long' if side>0 else 'short')} "
                              f"- within {NFP_WINDOW_DAYS}d of an NFP release")
                        continue
                    rkey = f"{name}|{tf}|{sig_name}|{'long' if side > 0 else 'short'}"
                    if state.get(rkey) == bar_id:
                        continue
                    stop_pct = (ATR_STOP_MULT * atr_v / price) if not np.isnan(atr_v) and price else None
                    plan_r = entry_plan(price, atr_v, side)
                    if plan_r is None:
                        # No valid ATR to build a stop/target from - can't
                        # alert a trade without one, so skip entirely rather
                        # than send a message (or track exposure) for a
                        # call that has no risk levels attached.
                        continue
                    active_setups.append((name, side))
                    tk = f"{sig_name}|{REVERSION_HORIZON_DAYS}d"
                    kelly_r = kelly_size(state.get("_tally", {}).get(tk), stop_pct=stop_pct)
                    dirword = "LONG" if side > 0 else "SHORT"
                    pf_note = "1.15" if sig_name == "zscore_rev" else "1.14"
                    exit_pf_note = "0.99" if sig_name == "zscore_rev" else "1.03"
                    rmsg = (f"{icon} <b>{name}</b> · 1d · {label}\n"
                            f"{dirword} · {price:,.4f}\n"
                            f"🎯 <b>{dirword}</b> · risk {kelly_r['risk_pct']*100:.2f}% "
                            f"(${kelly_r['risk_amount']:,.0f}"
                            + (f", notional ${kelly_r['notional']:,.0f}" if kelly_r['notional'] else "")
                            + f") · <i>{kelly_r['basis']}</i>\n"
                            + (f"Stop {plan_r['stop']:,.4f} (-{plan_r['stop_dist_pct']:.1f}%) · "
                               f"Target {plan_r['target']:,.4f} (+{plan_r['target_dist_pct']:.1f}%)\n"
                               if plan_r else "")
                            + f"📖 <b>READ:</b> price has moved unusually far from its recent "
                              f"average - this has historically drifted back over about a "
                              f"month. <b>Hold the full {REVERSION_HORIZON_DAYS} days</b> - "
                              f"exiting early at the stop/target above tested worse, even if "
                              f"price dips first.\n"
                            f"🔬 <i>Why we trust this: pooled/spread-charged/split-half PF "
                            f"{pf_note} on {REVERSION_HORIZON_DAYS}d daily hold. The stop/target "
                            f"above are for sizing only - tested as an early-exit rule and came "
                            f"out worse (PF {exit_pf_note}; the stop alone gets hit ~46% of the "
                            f"time before the drift has room to play out). Didn't clear the "
                            f"held-out H4 check - daily only, doesn't stack with the other "
                            f"reversion alert.</i>\n"
                            f"ADX {adx_v:.0f} · Chop {chop_v:.0f} · Z {z_v:+.1f}\n"
                            f"<i>{stamp} · closed bar</i>")
                    if send_telegram(rmsg, dry):
                        sent += 1
                        state[rkey] = bar_id
                        print(f"  {name}|{tf:<3}  ALERT {sig_name} {dirword}")
                    if LOG_OUTCOMES:
                        pend = state.setdefault("_pending", [])
                        pkey = f"{name}|{bar_id}|{sig_name}|{REVERSION_HORIZON_DAYS}"
                        if not any(r.get("key") == pkey for r in pend):
                            pend.append({"key": pkey, "pair": name, "ticker": ticker, "tf": tf,
                                         "cond": sig_name, "side": side, "price": price,
                                         "h": REVERSION_HORIZON_DAYS, "created": now.isoformat(),
                                         "due": (now + timedelta(days=int(REVERSION_HORIZON_DAYS*1.45))).isoformat()})

            if not events:
                flag = "  <<" if (curr >= DIGEST_ABOVE or curr <= DIGEST_BELOW) else ""
                print(f"  {name}|{tf:<3}  RSI {curr:5.1f}  ADX {adx_v:4.0f}  Chop {chop_v:4.0f}{flag}")

    # ---- Layer 7: portfolio exposure across everything flagged this run --
    active_names = {n for n, _ in active_setups}
    net_exposure = net_currency_exposure(active_setups)
    exp_warn = exposure_warnings(net_exposure)
    corr = pair_correlation({n: daily_closes[n] for n in active_names if n in daily_closes})
    corr_warn = correlation_warnings(corr)
    # Standing snapshot for the digest - the top few currencies by |exposure|
    # regardless of whether they cross MAX_NET_CCY_EXPOSURE, so the digest
    # shows "here's where things stand" every time, not only when something
    # trips the warning threshold.
    top_exposure = [(c, v) for c, v in
                     sorted(net_exposure.items(), key=lambda x: -abs(x[1]))[:5] if v]
    if exp_warn or corr_warn:
        print(f"\n  --- portfolio guardrails ---")
        for w in exp_warn:  print(f"  exposure: {w}")
        for w in corr_warn: print(f"  corr:     {w}")

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
            # Same "does this record actually notify the user" rule used for
            # the final close-out reminder below - the stack's secondary 20d
            # research horizon gets neither a midpoint check-in nor a final
            # reminder, just silent tallying, so the two stay consistent.
            is_reversion_signal = rec["cond"] in ("zscore_rev", "keltner_rev")
            is_primary_stack_horizon = (not is_reversion_signal) and rec["h"] == OUTCOME_HORIZONS[0]
            notifies = is_reversion_signal or is_primary_stack_horizon

            if now < due:
                if notifies and not rec.get("midpoint_sent"):
                    try:
                        created = datetime.fromisoformat(rec["created"])
                    except Exception:
                        # older pending record from before "created" was tracked -
                        # reconstruct it the same way it would have been computed
                        created = due - timedelta(days=int(rec["h"] * 1.45))
                    midpoint = created + (due - created) / 2
                    if now >= midpoint:
                        dfm = fetch_batch([rec["ticker"]], rec["tf"])
                        d2m = dfm.get(rec["ticker"])
                        if d2m is not None and not d2m.empty:
                            side_m = rec.get("side", 1)
                            current_price = float(d2m["Close"].iloc[-2])
                            unrealized = (current_price / rec["price"] - 1) * 100 * side_m
                            elapsed_frac = (now - created) / (due - created)
                            days_elapsed = round(elapsed_frac * rec["h"])
                            days_left = max(rec["h"] - days_elapsed, 0)
                            dirword_m = "LONG" if side_m > 0 else "SHORT"
                            mid_msg = (f"📊 <b>Halfway check-in: {rec['pair']} {dirword_m}</b>\n"
                                       f"{rec['cond']} · day {days_elapsed} of {rec['h']} "
                                       f"({days_left} to go)\n"
                                       f"Entry {rec['price']:,.4f} → now {current_price:,.4f}\n"
                                       f"<b>Unrealized: {unrealized:+.2f}%</b>\n\n"
                                       f"<i>Just a check-in, not an action - the validated "
                                       f"exit is still at day {rec['h']}, not now.</i>")
                            if send_telegram(mid_msg, dry):
                                sent += 1
                            rec["midpoint_sent"] = True
                still.append(rec); continue

            # Due, but wait for the close window so scoring/reminders happen
            # at a consistent time of day rather than whatever hour the
            # cron happened to run after the countdown elapsed. Applies to
            # every due record, not just ones that notify the user, so the
            # internal tally stays close-to-close consistently too.
            if not (CLOSE_WINDOW_START_UTC <= now.hour < CLOSE_WINDOW_END_UTC):
                still.append(rec); continue

            df = fetch_batch([rec["ticker"]], rec["tf"])
            d2 = df.get(rec["ticker"])
            if d2 is None or d2.empty:
                still.append(rec); continue
            side = rec.get("side", 1)
            exit_price = float(d2["Close"].iloc[-2])
            ret = (exit_price / rec["price"] - 1) * 100 * side
            k = f"{rec['cond']}|{rec['h']}d"
            t = tally.setdefault(k, {"n": 0, "sum": 0.0, "wins": 0,
                                     "long_n": 0, "long_w": 0, "short_n": 0, "short_w": 0,
                                     "streak": 0, "maxw": 0, "maxl": 0,
                                     "best": None, "worst": None,
                                     "win_sum": 0.0, "loss_sum": 0.0})
            # setdefault upgrades tally entries created before win_sum/loss_sum
            # existed, so old live data isn't discarded by this change.
            t.setdefault("win_sum", 0.0); t.setdefault("loss_sum", 0.0)
            t["n"] += 1; t["sum"] += ret; t["wins"] += (ret > 0)
            if ret > 0: t["win_sum"] += ret
            else:       t["loss_sum"] += ret
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

            # ---- Per-trade "time to close" reminder --------------------------
            # Validated finding (exit_comparison_backtest.py, exit_width_sweep.py):
            # fixed-horizon holding beats stop/target exit at EVERY width tested,
            # not just the original one - so the right action here is always
            # "close now", regardless of where price sits vs the stop/target
            # shown in the original alert. Only fires for the PRIMARY validated
            # horizon per signal type - the stack's secondary 20d research
            # horizon (OUTCOME_HORIZONS[1]) is tallied above but doesn't also
            # send a second, redundant reminder for the same original trade.
            # (notifies was already computed above, same rule as the midpoint check.)
            if notifies:
                entry_date_approx = due - timedelta(days=int(rec["h"] * 1.45))
                dirword = "LONG" if side > 0 else "SHORT"
                remind_msg = (f"⏰ <b>Time to close: {rec['pair']} {dirword}</b>\n"
                              f"{rec['cond']} · entered ~{entry_date_approx.strftime('%Y-%m-%d')} "
                              f"({rec['h']} trading days ago) at {rec['price']:,.4f}\n"
                              f"Exit price now: {exit_price:,.4f} <i>(today's close)</i>\n"
                              f"<b>Result: {ret:+.2f}%</b>\n\n"
                              f"<i>This is the validated exit point - close now regardless of "
                              f"where price sits versus the stop/target shown in the original "
                              f"alert. Exiting early OR holding past this point both tested "
                              f"worse than closing exactly here. This message is timed to "
                              f"arrive near each day's close (~5pm New York time), not at a "
                              f"random hour.</i>")
                if send_telegram(remind_msg, dry):
                    sent += 1
        state["_pending"] = still

    # ---- Periodic digest of standing conditions --------------------------
    if DIGEST_EVERY_HOURS > 0 and (extended or top_exposure):
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
            lines = [f"📊 <b>Currently extended</b>",
                     "<i>Background only - not new trade alerts, just which pairs have an "
                     "unusually stretched RSI reading right now.</i>", ""]
            if ob:
                lines.append("<b>Overbought (RSI ≥ 70)</b>")
                lines += [f"  {n} · {t} · <b>{v:.1f}</b>" for n, t, v in ob]
                lines.append("")
            if os_:
                lines.append("<b>Oversold (RSI ≤ 30)</b>")
                lines += [f"  {n} · {t} · <b>{v:.1f}</b>" for n, t, v in os_]
                lines.append("")
            if top_exposure:
                # Standing snapshot every time the digest fires, not just
                # when something crosses MAX_NET_CCY_EXPOSURE - context,
                # never a gate.
                lines.append("<b>📐 Portfolio snapshot</b> <i>(context - never blocks a trade)</i>")
                lines.append("<i>Net exposure per currency across every currently-active setup: "
                              "positive = net long that currency, negative = net short. A bigger "
                              "number means several open trades are really the same underlying "
                              "bet, not genuine diversification, even if they're on different pairs.</i>")
                warn_ccys = {w.split(":")[0] for w in exp_warn}
                lines += [f"  {c}: {v:+.1f}" + (" ⚠️ several trades, same bet" if c in warn_ccys else "")
                          for c, v in top_exposure]
                if corr_warn:
                    lines.append("<i>ρ (correlation) = how closely two currently-active pairs have "
                                  "moved together over the last 20 days. Near +1 or -1 means holding "
                                  "both isn't really two separate bets, even with no shared currency.</i>")
                    lines += [f"  {w}" for w in corr_warn]
                lines.append("")
            lines.append(f"<i>{stamp} · standing conditions, not new crossings</i>")
            if send_telegram("\n".join(lines), dry):
                sent += 1
                state["_last_digest"] = now.isoformat()
                print(f"\n  DIGEST sent ({len(ob)} overbought, {len(os_)} oversold, "
                      f"{len(exp_warn)} exposure warn, {len(corr_warn)} corr warn)")

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
