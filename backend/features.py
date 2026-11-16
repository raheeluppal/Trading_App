"""Chart values used by the visual intraday playbook."""

import pandas as pd


def get_chart_data(bars):
    frame = bars.copy().sort_index()
    if isinstance(frame.index, pd.MultiIndex):
        frame.index = frame.index.get_level_values(-1)
    frame["ema9"] = frame["close"].ewm(span=9, adjust=False).mean()
    frame["ema21"] = frame["close"].ewm(span=21, adjust=False).mean()
    timestamp = pd.to_datetime(frame.index, utc=True)
    session = timestamp.tz_convert("America/New_York").date
    volume = frame["volume"].clip(lower=0)
    cumulative_volume = volume.groupby(session).cumsum()
    frame["vwap"] = (frame["close"] * volume).groupby(session).cumsum() / cumulative_volume.where(cumulative_volume.ne(0))
    return [
        {
            "timestamp": pd.Timestamp(index).isoformat(),
            "open": float(row.open), "high": float(row.high),
            "low": float(row.low), "close": float(row.close),
            "volume": int(row.volume),
            "ema9": float(row.ema9), "ema21": float(row.ema21),
            "vwap": float(row.vwap) if pd.notna(row.vwap) else None,
        }
        for index, row in frame.iterrows()
    ]
