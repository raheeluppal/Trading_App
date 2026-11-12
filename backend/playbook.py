"""Observable, closed-candle checks for the Visual Day-Trading Playbook.

These are conservative operational definitions of the guide's visual setups.
They are not a trained probability model and do not infer hand-drawn levels.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


ET = ZoneInfo("America/New_York")
REQUIRED = {"open", "high", "low", "close", "volume"}
ENTRY_START = time(9, 30)
ENTRY_END = time(11, 0)


def in_entry_window(when: datetime) -> bool:
    """Accept new playbook entries only from 09:30 through 10:59 ET."""
    return ENTRY_START <= when.astimezone(ET).time() < ENTRY_END


@dataclass(frozen=True)
class Setup:
    signal: str = "WAIT"
    name: str | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    reason: str = "No confirmed playbook setup"
    signal_bar: str | None = None

    def as_dict(self):
        return asdict(self)


def _closed(bars: pd.DataFrame | None, minutes: int, now: datetime) -> pd.DataFrame:
    if bars is None or bars.empty or not REQUIRED.issubset(bars.columns):
        return pd.DataFrame()
    frame = bars.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame.index = frame.index.get_level_values(-1)
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(ET)
    frame = frame.sort_index()
    end = frame.index + pd.Timedelta(minutes=minutes)
    frame = frame[end <= now.astimezone(ET)]
    return frame[~frame.index.duplicated(keep="last")]


def _daily_closed(bars: pd.DataFrame | None, now: datetime) -> pd.DataFrame:
    if bars is None or bars.empty or not REQUIRED.issubset(bars.columns):
        return pd.DataFrame()
    frame = bars.copy()
    if isinstance(frame.index, pd.MultiIndex):
        frame.index = frame.index.get_level_values(-1)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    return frame[frame.index.date < now.astimezone(ET).date()]


def _trend(bars: pd.DataFrame) -> int:
    if len(bars) < 22:
        return 0
    close = bars["close"]
    fast = close.ewm(span=9, adjust=False).mean().iloc[-1]
    slow = close.ewm(span=21, adjust=False).mean().iloc[-1]
    if close.iloc[-1] > fast > slow:
        return 1
    if close.iloc[-1] < fast < slow:
        return -1
    return 0


def evaluate_playbook(minute_bars, five_bars, hourly_bars, daily_bars, now=None) -> Setup:
    """Return a setup only after its 2m candle, 5m confirmation and entry break.

    The newest completed one-minute close is the observed entry price. Historical
    levels exclude the signal candle, so break/retest comparisons are reachable.
    """
    now = now or datetime.now(ET)
    if not in_entry_window(now):
        return Setup(reason="Outside the 09:30-11:00 ET entry window")
    one = _closed(minute_bars, 1, now)
    five = _closed(five_bars, 5, now)
    hour = _closed(hourly_bars, 60, now)
    day = _daily_closed(daily_bars, now)
    one = one.between_time("09:30", "15:59")
    five = five.between_time("09:30", "15:55")
    hour = hour.between_time("09:30", "15:59")
    if len(one) < 60 or len(five) < 30 or len(hour) < 22 or len(day) < 22:
        return Setup(reason="Insufficient completed 1m/5m/1h/daily bars")

    # Resample within each ET trading day, never bridging overnight sessions.
    two = one.resample("2min", origin="start_day", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    counts = one["close"].resample("2min", origin="start_day").count()
    two = two[counts.reindex(two.index).eq(2).to_numpy()]
    if len(two) < 22:
        return Setup(reason="Insufficient complete 2-minute candles")

    signal = two.iloc[-1]
    price = float(one.iloc[-1]["close"])
    if one.index[-1] <= two.index[-1] + timedelta(minutes=1):
        return Setup(reason="Waiting for price to break the closed signal candle")
    if (now.astimezone(ET) - one.index[-1]) > timedelta(minutes=3):
        return Setup(reason="Intraday price is stale")
    if two.index[-1].date() != now.astimezone(ET).date():
        return Setup(reason="No signal candle in the current session")

    vol_base = two["volume"].iloc[-21:-1].mean()
    if vol_base <= 0 or signal["volume"] < vol_base * 1.2:
        return Setup(reason="Signal candle lacks relative volume")

    five_last = five.iloc[-1]
    five_up = five_last["close"] > five_last["open"]
    five_down = five_last["close"] < five_last["open"]
    daily_trend, hourly_trend, five_trend = _trend(day), _trend(hour), _trend(five)
    ema9 = five["close"].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = five["close"].ewm(span=21, adjust=False).mean().iloc[-1]
    session = one[one.index.date == now.astimezone(ET).date()]
    vwap = ((session["close"] * session["volume"]).sum() / session["volume"].sum())
    # Prior 5m boundaries are not contaminated by the current candle.
    support = float(five["low"].iloc[-22:-2].min())
    resistance = float(five["high"].iloc[-22:-2].max())
    major_support = [float(hour["low"].iloc[-21:-1].min()), float(day["low"].iloc[-1])]
    major_resistance = [float(hour["high"].iloc[-21:-1].max()), float(day["high"].iloc[-1])]
    buffer = max(price * 0.0002, 0.01)
    near_support = abs(float(signal["low"]) - support) / support <= 0.003
    near_resistance = abs(float(signal["high"]) - resistance) / resistance <= 0.003
    at_major_support = any(abs(float(signal["low"]) - level) / level <= 0.003 for level in major_support)
    at_major_resistance = any(abs(float(signal["high"]) - level) / level <= 0.003 for level in major_resistance)
    prior_five_volume = five["volume"].iloc[-22:-2]
    break_volume = five["volume"].iloc[-2] >= prior_five_volume.mean() * 1.2
    support_tests = (abs(five["low"].iloc[-22:-2] - support) / support <= 0.003).sum()
    body = abs(float(signal["close"] - signal["open"]))
    bullish_rejection = signal["close"] > signal["open"] and signal["open"] - signal["low"] >= body
    bearish_rejection = signal["close"] < signal["open"] and signal["high"] - signal["open"] >= body
    bullish_break = price > signal["high"] and five_up
    bearish_break = price < signal["low"] and five_down

    name = None
    side = None
    stop = None
    target = None
    if daily_trend == hourly_trend == five_trend == 1 and bullish_break and price > vwap:
        if signal["low"] <= ema9 * 1.003 and signal["close"] >= ema9 and signal["low"] >= ema21 * 0.997:
            pullback_low = min(float(signal["low"]), float(five["low"].iloc[-3:].min()))
            name, side, stop = "EMA/VWAP pullback", "BUY", pullback_low - buffer
        elif break_volume and five["close"].iloc[-2] > resistance and signal["low"] <= resistance * 1.003 and signal["close"] > resistance:
            name, side, stop = "Break and retest", "BUY", float(signal["low"] - buffer)
    if name is None and daily_trend >= 0 and hourly_trend >= 0 and bullish_break and bullish_rejection and near_support and at_major_support:
        name, side, stop = "Support rejection", "BUY", float(signal["low"] - buffer)
        target = resistance
    if name is None and daily_trend == hourly_trend == five_trend == -1 and bearish_break and price < vwap:
        if signal["high"] >= ema9 * 0.997 and signal["close"] <= ema9 and signal["high"] <= ema21 * 1.003:
            pullback_high = max(float(signal["high"]), float(five["high"].iloc[-3:].max()))
            name, side, stop = "EMA/VWAP pullback", "PUT", pullback_high + buffer
        elif break_volume and support_tests >= 3 and five["close"].iloc[-2] < support and signal["high"] >= support * 0.997 and signal["close"] < support:
            name, side, stop = "Major level breakdown", "PUT", max(float(signal["high"]), support) + buffer
    if name is None and daily_trend <= 0 and hourly_trend <= 0 and bearish_break and bearish_rejection and near_resistance and at_major_resistance:
        name, side, stop = "Resistance rejection", "PUT", float(signal["high"] + buffer)
        target = support
    if name is None and abs(ema9 - ema21) / price < 0.0015:
        if near_support and bullish_rejection and bullish_break:
            name, side, stop, target = "Range bounce", "BUY", support - buffer, (support + resistance) / 2
        elif near_resistance and bearish_rejection and bearish_break:
            name, side, stop, target = "Range bounce", "PUT", resistance + buffer, (support + resistance) / 2
    if name is None:
        return Setup(reason="No location, context, candle and 5m confirmation aligned")

    risk = abs(price - stop)
    if risk <= 0 or (side == "BUY" and stop >= price) or (side == "PUT" and stop <= price):
        return Setup(reason="Invalid stop location")
    if target is None:
        mapped = [resistance, *major_resistance] if side == "BUY" else [support, *major_support]
        beyond = [level for level in mapped if level > price] if side == "BUY" else [level for level in mapped if level < price]
        if not beyond:
            return Setup(reason="No mapped target beyond entry")
        target = min(beyond) if side == "BUY" else max(beyond)
    reward = target - price if side == "BUY" else price - target
    if reward < 2 * risk:
        return Setup(reason="First mapped target offers less than 2R")
    return Setup(side, name, price, stop, float(target), "Closed 2m signal, 5m confirmation, volume and entry break", two.index[-1].isoformat())
