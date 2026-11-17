"""Observable, closed-candle checks for the Visual Day-Trading Playbook.

These are conservative operational definitions of the guide's visual setups.
They are not a trained probability model and do not infer hand-drawn levels.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


ET = ZoneInfo("America/New_York")
REQUIRED = {"open", "high", "low", "close", "volume"}
ENTRY_START = time(9, 30)
ENTRY_END = time(11, 0)
CONTINUATION_START = time(11, 0)
CONTINUATION_END = time(14, 30)


def in_entry_window(when: datetime) -> bool:
    """Accept new playbook entries only from 09:30 through 10:59 ET."""
    return ENTRY_START <= when.astimezone(ET).time() < ENTRY_END


def in_continuation_window(when: datetime) -> bool:
    """Allow only the dedicated trend-continuation setup after 11 ET."""
    local_time = when.astimezone(ET).time()
    return CONTINUATION_START <= local_time < CONTINUATION_END


def entry_eligible(when: datetime) -> bool:
    return in_entry_window(when) or in_continuation_window(when)


@dataclass(frozen=True)
class Setup:
    signal: str = "WAIT"
    name: str | None = None
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    reason: str = "No confirmed playbook setup"
    signal_bar: str | None = None
    location: str | None = None
    zones: list[dict] = field(default_factory=list)
    quality_score: int = 0

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


def evaluate_playbook(minute_bars, five_bars, hourly_bars, daily_bars, now=None, marked_zones=None,
                      market_five_bars=None) -> Setup:
    """Return a setup only after its 2m candle, 5m confirmation and entry break.

    The newest completed one-minute close is the observed entry price. Historical
    levels exclude the signal candle, so break/retest comparisons are reachable.
    """
    now = now or datetime.now(ET)
    if not entry_eligible(now):
        return Setup(reason="Outside the 09:30-11:00 ET or 11:00-14:30 ET continuation window")
    one = _closed(minute_bars, 1, now)
    five = _closed(five_bars, 5, now)
    hour = _closed(hourly_bars, 60, now)
    day = _daily_closed(daily_bars, now)
    market = _closed(market_five_bars, 5, now) if market_five_bars is not None else pd.DataFrame()
    one = one.between_time("09:30", "15:59")
    five = five.between_time("09:30", "15:55")
    if not market.empty:
        market = market.between_time("09:30", "15:55")
    hour = hour.between_time("09:30", "15:59")
    if len(one) < 4 or len(five) < 22 or len(hour) < 22 or len(day) < 22:
        return Setup(reason="Insufficient completed 1m/5m/1h/daily bars")
    if five.index[-1].date() != now.astimezone(ET).date():
        return Setup(reason="Waiting for current-session 5-minute confirmation")
    market_trend = _trend(market) if len(market) >= 22 else 0

    # Resample within each ET trading day, never bridging overnight sessions.
    two = one.resample("2min", origin="start_day", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    counts = one["close"].resample("2min", origin="start_day").count()
    two = two[counts.reindex(two.index).eq(2).to_numpy()]
    if len(two) < 2:
        return Setup(reason="Insufficient complete 2-minute candles")

    signal = two.iloc[-1]
    price = float(one.iloc[-1]["close"])
    if one.index[-1] <= two.index[-1] + timedelta(minutes=1):
        return Setup(reason="Waiting for price to break the closed signal candle")
    if (now.astimezone(ET) - one.index[-1]) > timedelta(minutes=3):
        return Setup(reason="Intraday price is stale")
    if two.index[-1].date() != now.astimezone(ET).date():
        return Setup(reason="No signal candle in the current session")
    if two.index[-2].date() != two.index[-1].date():
        return Setup(reason="Waiting for a preceding 2-minute candle this session")

    # A valid reclaim often has steady participation rather than a single
    # climax bar.  Compare with the recent 2m baseline instead of requiring
    # the signal bar to beat the immediately preceding bar.
    recent_two_volume = two["volume"].iloc[-8:-1]
    volume_baseline = float(recent_two_volume.median()) if not recent_two_volume.empty else 0.0
    if volume_baseline <= 0 or signal["volume"] < volume_baseline * 0.75:
        return Setup(reason="Signal candle volume is below the recent 2-minute baseline")

    five_last = five.iloc[-1]
    five_up = five_last["close"] > five_last["open"]
    five_down = five_last["close"] < five_last["open"]
    daily_trend, hourly_trend, five_trend = _trend(day), _trend(hour), _trend(five)
    ema9 = five["close"].ewm(span=9, adjust=False).mean().iloc[-1]
    ema21 = five["close"].ewm(span=21, adjust=False).mean().iloc[-1]
    session = one[one.index.date == now.astimezone(ET).date()]
    cumulative_volume = session["volume"].cumsum()
    if cumulative_volume.iloc[-1] <= 0:
        return Setup(reason="Current-session volume is unavailable")
    session_vwap = (session["close"] * session["volume"]).cumsum() / cumulative_volume.where(cumulative_volume.ne(0))
    vwap = float(session_vwap.iloc[-1])
    opening = session.iloc[:15]
    opening_range_high = float(opening["high"].max()) if len(opening) >= 15 else None
    opening_range_low = float(opening["low"].min()) if len(opening) >= 15 else None
    # Prior 5m boundaries are not contaminated by the current candle.
    # _closed() already removes the still-forming 5m candle.  Excluding an
    # additional bar made the replay blind to the most recent confirmed
    # opening-range boundary and its retest.
    prior_five = five.iloc[-22:-1]
    support = float(prior_five["low"].min())
    resistance = float(prior_five["high"].max())
    major_support = [float(hour["low"].iloc[-21:-1].min()), float(day["low"].iloc[-1])]
    major_resistance = [float(hour["high"].iloc[-21:-1].max()), float(day["high"].iloc[-1])]
    buffer = max(price * 0.0002, 0.01)
    signal_range = max(float(signal["high"] - signal["low"]), 0.01)
    zones = [
        {"kind": "prior_day_low", "low": major_support[1], "high": major_support[1]},
        {"kind": "prior_day_high", "low": major_resistance[1], "high": major_resistance[1]},
        {"kind": "hourly_support", "low": major_support[0], "high": major_support[0]},
        {"kind": "hourly_resistance", "low": major_resistance[0], "high": major_resistance[0]},
        {"kind": "range_support", "low": support, "high": support},
        {"kind": "range_resistance", "low": resistance, "high": resistance},
        {"kind": "vwap", "low": float(vwap), "high": float(vwap)},
        {"kind": "ema9", "low": float(ema9), "high": float(ema9)},
        {"kind": "ema21", "low": float(ema21), "high": float(ema21)},
    ]
    if opening_range_high is not None:
        zones.extend([
            {"kind": "opening_range_high", "low": opening_range_high, "high": opening_range_high},
            {"kind": "opening_range_low", "low": opening_range_low, "high": opening_range_low},
        ])
    prior_close = float(day["close"].iloc[-1])
    session_open = float(session["open"].iloc[0])
    if abs(session_open - prior_close) >= 0.01:
        zones.append({"kind": "daily_gap", "low": min(prior_close, session_open), "high": max(prior_close, session_open)})
    for zone in marked_zones or []:
        if zone["low"] <= 0 or zone["high"] < zone["low"]:
            raise ValueError("Marked zones require positive, ordered price bounds")
        zones.append({"kind": zone["kind"], "low": float(zone["low"]), "high": float(zone["high"])})
    bullish_zone_kinds = {"demand", "support", "trendline_support", "ath_retest", "daily_gap", "opening_range_low"}
    bearish_zone_kinds = {"supply", "resistance", "trendline_resistance", "daily_gap", "opening_range_high"}
    def touches(zone, candle_price):
        return zone["low"] - signal_range <= candle_price <= zone["high"] + signal_range
    bullish_zones = [z for z in zones if z["kind"] in bullish_zone_kinds
                     and (z["kind"] != "daily_gap" or price >= z["high"])
                     and touches(z, float(signal["low"]))]
    bearish_zones = [z for z in zones if z["kind"] in bearish_zone_kinds
                     and (z["kind"] != "daily_gap" or price <= z["low"])
                     and touches(z, float(signal["high"]))]
    near_support = abs(float(signal["low"]) - support) <= signal_range
    near_resistance = abs(float(signal["high"]) - resistance) <= signal_range
    at_major_support = any(abs(float(signal["low"]) - level) <= signal_range for level in major_support) or bool(bullish_zones)
    at_major_resistance = any(abs(float(signal["high"]) - level) <= signal_range for level in major_resistance) or bool(bearish_zones)
    near_support = near_support or bool(bullish_zones)
    near_resistance = near_resistance or bool(bearish_zones)
    break_volume = five["volume"].iloc[-1] >= five["volume"].iloc[-4:-1].median()
    support_tests = (abs(prior_five["low"] - support) <= (prior_five["high"] - prior_five["low"])).sum()
    body = abs(float(signal["close"] - signal["open"]))
    bullish_rejection = signal["close"] > signal["open"] and signal["open"] - signal["low"] >= body
    bearish_rejection = signal["close"] < signal["open"] and signal["high"] - signal["open"] >= body
    bullish_break = price > signal["high"] and five_up
    bearish_break = price < signal["low"] and five_down
    vwap_side = session["close"].iloc[-10:] > session_vwap.iloc[-10:]
    vwap_crosses = vwap_side.ne(vwap_side.shift()).iloc[1:].sum() >= 2
    vwap_flat = len(session_vwap) >= 10 and abs(session_vwap.iloc[-1] - session_vwap.iloc[-10]) <= signal_range
    ranging_vwap = vwap_crosses or vwap_flat

    name = None
    side = None
    stop = None
    target = None
    late_window = in_continuation_window(now)
    # A neutral hourly trend is acceptable when the completed 5m structure is
    # directional.  Strongly opposing daily context is still rejected.
    bullish_context = five_trend == 1 and hourly_trend >= 0 and daily_trend != -1
    bearish_context = five_trend == -1 and hourly_trend <= 0 and daily_trend != 1

    # The chart examples frequently show the touch/rejection before the
    # trigger candle.  Keep the window short and closed so this cannot look
    # ahead, but allow the preceding three 2m candles to establish the setup.
    pullback_bars = two.iloc[-4:-1]
    def prior_reclaim(levels, bullish):
        for level in levels:
            touched = ((pullback_bars["low"] <= level) & (pullback_bars["high"] >= level)).any()
            if touched and ((signal["close"] >= level) if bullish else (signal["close"] <= level)):
                return True
        return False

    bullish_signal_touch = any(
        signal["low"] <= level <= signal["high"] and signal["close"] >= level
        for level in (ema9, ema21, vwap)
    )
    bearish_signal_touch = any(
        signal["low"] <= level <= signal["high"] and signal["close"] <= level
        for level in (ema9, ema21, vwap)
    )
    bullish_pullback = bullish_signal_touch or prior_reclaim((ema9, ema21, vwap), True) or any(
        z["kind"] in {"trendline_support", "demand", "support", "ath_retest", "opening_range_low"}
        and ((pullback_bars["low"] <= z["high"]) & (signal["close"] >= z["high"])).any()
        for z in bullish_zones)
    bearish_pullback = bearish_signal_touch or prior_reclaim((ema9, ema21, vwap), False) or any(
        z["kind"] in {"trendline_resistance", "supply", "resistance", "opening_range_high"}
        and ((pullback_bars["high"] >= z["low"]) & (signal["close"] <= z["low"])).any()
        for z in bearish_zones)
    if not late_window and bullish_context and bullish_break and price > vwap:
        if bullish_pullback:
            pullback_low = min(float(signal["low"]), float(five["low"].iloc[-3:].min()))
            name, side, stop = "EMA/VWAP pullback", "BUY", pullback_low - buffer
    if name is None and not late_window and bullish_break and bullish_rejection and near_support and at_major_support:
        name, side, stop = ("ATH retest" if any(z["kind"] == "ath_retest" for z in bullish_zones) else "Support rejection"), "BUY", float(signal["low"] - buffer)
        target = resistance
    if name is None and not late_window and bearish_context and bearish_break and price < vwap:
        if bearish_pullback:
            pullback_high = max(float(signal["high"]), float(five["high"].iloc[-3:].max()))
            name, side, stop = "EMA/VWAP pullback", "PUT", pullback_high + buffer
    if name is None and not late_window and bearish_break and bearish_rejection and near_resistance and at_major_resistance:
        name, side, stop = "Resistance rejection", "PUT", float(signal["high"] + buffer)
        target = support
    if name is None and not late_window and break_volume and bullish_break and five["close"].iloc[-2] > resistance and signal["low"] <= resistance <= signal["high"] and signal["close"] > resistance:
        name, side, stop = "Break and retest", "BUY", float(signal["low"] - buffer)
    if name is None and not late_window and break_volume and bearish_break and five["close"].iloc[-2] < support and signal["low"] <= support <= signal["high"] and signal["close"] < support:
        if support_tests >= 2 and signal["low"] < two.iloc[-2]["low"]:
            name = "Major level breakdown"
        else:
            name = "Break and retest"
        side, stop = "PUT", max(float(signal["high"]), support) + buffer
    if name is None and not late_window and five_trend == 0 and ranging_vwap:
        if near_support and bullish_rejection and bullish_break:
            name, side, stop, target = "Range bounce", "BUY", support - buffer, (support + resistance) / 2
        elif near_resistance and bearish_rejection and bearish_break:
            name, side, stop, target = "Range bounce", "PUT", resistance + buffer, (support + resistance) / 2
    if name is None and late_window and bearish_context and bearish_break and price < vwap \
            and ema9 < ema21 and break_volume and five_last["close"] < support and bearish_pullback:
        name, side = "Gap-and-go continuation", "PUT"
        stop = max(float(signal["high"]), float(five["high"].iloc[-3:].max())) + buffer
    if name is None:
        return Setup(reason="No location, context, candle and 5m confirmation aligned")

    risk = abs(price - stop)
    if risk <= 0 or (side == "BUY" and stop >= price) or (side == "PUT" and stop <= price):
        return Setup(reason="Invalid stop location")
    if target is None:
        mapped = [resistance, *major_resistance] if side == "BUY" else [support, *major_support]
        mapped.extend(z["low"] if side == "BUY" else z["high"] for z in zones if z["kind"] in {"daily_gap", "demand", "supply", "support", "resistance", "ath_retest", "trendline_support", "trendline_resistance", "opening_range_low", "opening_range_high"})
        beyond = [level for level in mapped if level > price] if side == "BUY" else [level for level in mapped if level < price]
        if not beyond:
            return Setup(reason="No mapped target beyond entry")
        target = min(beyond) if side == "BUY" else max(beyond)
    reward = target - price if side == "BUY" else price - target
    if reward <= 0:
        return Setup(reason="No room to the first mapped target")
    if reward / risk < 1.0:
        return Setup(reason="Insufficient room to mapped target relative to stop risk")
    touched = bullish_zones if side == "BUY" else bearish_zones
    quality_score = sum([
        bool(touched or (near_support if side == "BUY" else near_resistance)),
        bool((bullish_context if side == "BUY" else bearish_context)),
        bool((bullish_break if side == "BUY" else bearish_break)),
        bool(signal["volume"] >= volume_baseline),
        bool(reward / risk >= 1.5),
        bool(not market_trend or (market_trend == 1 and side == "BUY") or (market_trend == -1 and side == "PUT")),
    ])
    location = None
    if touched:
        location = touched[-1]["kind"]
    elif bullish_pullback if side == "BUY" else bearish_pullback:
        location = "EMA/VWAP"
    elif near_support if side == "BUY" else near_resistance:
        location = "mapped support" if side == "BUY" else "mapped resistance"
    return Setup(side, name, price, stop, float(target), "Closed 2m signal, 5m confirmation, volume and entry break", two.index[-1].isoformat(), location, zones, quality_score)
