from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from app.adapters.ohlcv import OHLCVAdapter


class TradeVerifyService:
    def __init__(self, ohlcv_adapter: OHLCVAdapter) -> None:
        self.ohlcv_adapter = ohlcv_adapter

    async def verify(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        entry: float,
        stop: float,
        targets: list[float],
        mode: str = "ambiguous",
        window_hours: int = 72,
    ) -> dict:
        timeframe = timeframe or "1h"
        candles = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe=timeframe, limit=500)
        df = pd.DataFrame(candles)
        if df.empty:
            raise RuntimeError("No candles available")
        source_line = str(candles[-1].get("source_line") or "") if candles else ""

        ts_ms = int(timestamp.astimezone(timezone.utc).timestamp() * 1000)
        active = df[df["ts"] >= ts_ms].copy()
        if active.empty:
            raise RuntimeError("No candles available after the provided timestamp")

        max_ts = ts_ms + window_hours * 3600 * 1000
        active = active[active["ts"] <= max_ts]

        direction = "long" if targets[0] > entry else "short"
        filled_at = None
        outcome = "open"
        first_hit = None
        hit_price = None
        ambiguous = False

        best_target = max(targets) if direction == "long" else min(targets)

        for _, row in active.iterrows():
            low = float(row["low"])
            high = float(row["high"])

            if filled_at is None and low <= entry <= high:
                filled_at = int(row["ts"])

            if filled_at is None:
                continue

            stop_hit = (low <= stop <= high)
            target_hit = any(low <= t <= high for t in targets)

            if stop_hit and target_hit:
                if mode == "conservative":
                    outcome = "loss"
                    first_hit = int(row["ts"])
                    hit_price = stop
                    break
                if mode == "optimistic":
                    outcome = "win"
                    first_hit = int(row["ts"])
                    hit_price = best_target
                    break
                ambiguous = True
                outcome = "ambiguous"
                first_hit = int(row["ts"])
                hit_price = None
                break

            if stop_hit:
                outcome = "loss"
                first_hit = int(row["ts"])
                hit_price = stop
                break
            if target_hit:
                outcome = "win"
                first_hit = int(row["ts"])
                hit_price = best_target
                break

        if filled_at is None:
            return {
                "symbol": symbol.upper(),
                "result": "not_filled",
                "note": "Entry was never touched in the evaluation window.",
                "direction": direction,
            }

        post_fill = active[active["ts"] >= filled_at]
        mfe = float(post_fill["high"].max() - entry) if direction == "long" else float(entry - post_fill["low"].min())
        mae = float(entry - post_fill["low"].min()) if direction == "long" else float(post_fill["high"].max() - entry)

        risk = abs(entry - stop)
        if risk == 0:
            r_multiple = 0.0
        elif outcome == "win" and hit_price is not None:
            reward = (hit_price - entry) if direction == "long" else (entry - hit_price)
            r_multiple = reward / risk
        elif outcome == "loss":
            r_multiple = -1.0
        else:
            r_multiple = 0.0

        return {
            "symbol": symbol.upper(),
            "direction": direction,
            "result": outcome,
            "filled_at": filled_at,
            "first_hit": first_hit,
            "ambiguous": ambiguous,
            "mfe": round(mfe, 6),
            "mae": round(mae, 6),
            "r_multiple": round(r_multiple, 3),
            "mode": mode,
            "evaluated_until": int((timestamp + timedelta(hours=window_hours)).timestamp() * 1000),
            "source_line": source_line,
        }
