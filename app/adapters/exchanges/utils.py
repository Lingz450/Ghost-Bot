from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CANONICAL_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"]


@dataclass(frozen=True)
class TimeframeResolution:
    request_tf: str
    fetch_tf: str
    needs_resample: bool


def timeframe_to_minutes(timeframe: str) -> int:
    tf = (timeframe or "").strip()
    if not tf:
        return 0
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    if tf.endswith("w"):
        return int(tf[:-1]) * 10080
    return 0


def normalize_timeframe(raw: str) -> str:
    v = (raw or "").strip().lower().replace(" ", "")
    aliases = {
        "1min": "1m",
        "3min": "3m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "60m": "1h",
        "1hr": "1h",
        "1hour": "1h",
        "2hr": "2h",
        "2hour": "2h",
        "4hr": "4h",
        "4hour": "4h",
        "6hr": "6h",
        "6hour": "6h",
        "12hr": "12h",
        "12hour": "12h",
        "daily": "1d",
        "day": "1d",
        "weekly": "1w",
        "week": "1w",
    }
    return aliases.get(v, v)


def resolve_timeframe(request_tf: str, supported: set[str]) -> TimeframeResolution | None:
    request = normalize_timeframe(request_tf)
    if request in supported:
        return TimeframeResolution(request_tf=request, fetch_tf=request, needs_resample=False)

    req_minutes = timeframe_to_minutes(request)
    if req_minutes <= 0:
        return None

    candidates: list[tuple[int, str]] = []
    for tf in supported:
        mins = timeframe_to_minutes(tf)
        if mins <= 0:
            continue
        if mins <= req_minutes and req_minutes % mins == 0:
            candidates.append((mins, tf))
    if not candidates:
        return None

    best = sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]
    return TimeframeResolution(request_tf=request, fetch_tf=best, needs_resample=True)


def resample_ohlcv(
    candles: list[dict],
    *,
    source_tf: str,
    target_tf: str,
    limit: int,
) -> list[dict]:
    if not candles:
        return []
    source_minutes = timeframe_to_minutes(source_tf)
    target_minutes = timeframe_to_minutes(target_tf)
    if source_minutes <= 0 or target_minutes <= 0 or target_minutes <= source_minutes:
        return candles[-limit:]

    df = pd.DataFrame(candles)
    if df.empty:
        return []
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("dt")
    df = df.set_index("dt")
    rule = f"{target_minutes}min"
    agg = df.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    agg = agg.dropna(subset=["open", "high", "low", "close"])
    if agg.empty:
        return []

    out: list[dict] = []
    source = str(candles[0].get("source", "resampled"))
    for idx, row in agg.tail(limit).iterrows():
        out.append(
            {
                "ts": int(idx.timestamp() * 1000),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0.0) or 0.0),
                "source": source,
                "resampled_from": source_tf,
            }
        )
    return out


def candles_are_sane(candles: list[dict]) -> bool:
    if not candles:
        return False
    prev_ts = 0
    for row in candles:
        ts = int(row.get("ts", 0) or 0)
        o = float(row.get("open", 0) or 0)
        h = float(row.get("high", 0) or 0)
        l = float(row.get("low", 0) or 0)
        c = float(row.get("close", 0) or 0)
        if ts <= 0 or o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return False
        if h < l:
            return False
        if prev_ts and ts <= prev_ts:
            return False
        prev_ts = ts
    return True
