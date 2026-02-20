from __future__ import annotations

import math

import pandas as pd

from app.adapters.ohlcv import OHLCVAdapter
from app.core.ta import atr, bollinger_mid, ema, pivot_levels


def _fmt(v: float) -> str:
    """Format a price value: drop trailing zeros, max 8 sig figs."""
    if v == 0:
        return "0"
    if abs(v) >= 1:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.8f}".rstrip("0").rstrip(".")


def _nearest(val: float, candidates: list[tuple[float, str]], pct: float = 0.015) -> str | None:
    """Return the label of the closest candidate within pct of val, or None."""
    best_label = None
    best_dist = float("inf")
    for cv, label in candidates:
        dist = abs(val - cv) / max(abs(cv), 1e-12)
        if dist < pct and dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


class SetupReviewService:
    def __init__(self, ohlcv_adapter: OHLCVAdapter) -> None:
        self.ohlcv_adapter = ohlcv_adapter

    def _safe_last(self, series: pd.Series, default: float) -> float:
        cleaned = series.dropna()
        if cleaned.empty:
            return default
        value = float(cleaned.iloc[-1])
        if math.isnan(value):
            return default
        return value

    def _bollinger_bands(self, series: pd.Series, period: int = 20, std: float = 2.0) -> tuple[float, float, float]:
        mid = series.rolling(window=period).mean()
        dev = series.rolling(window=period).std()
        upper = mid + std * dev
        lower = mid - std * dev
        m = self._safe_last(mid, float("nan"))
        u = self._safe_last(upper, float("nan"))
        lo = self._safe_last(lower, float("nan"))
        return u, m, lo

    async def review(
        self,
        symbol: str,
        timeframe: str,
        entry: float,
        stop: float,
        targets: list[float],
        direction: str | None = None,
        amount_usd: float | None = None,
        leverage: float | None = None,
    ) -> dict:
        symbol = symbol.upper()
        tf = timeframe or "1h"

        candles = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe=tf, limit=260)
        df = pd.DataFrame(candles)
        if df.empty:
            raise RuntimeError("No candle data available for setup review")

        close_last = float(df["close"].iloc[-1])
        if direction is None:
            direction = "long" if (sum(targets) / max(len(targets), 1)) > entry else "short"

        risk = abs(entry - stop)
        if risk <= 0:
            raise RuntimeError("Invalid setup: entry and stop cannot be the same.")

        if direction == "long":
            rewards = [max(t - entry, 0.0) for t in targets]
        else:
            rewards = [max(entry - t, 0.0) for t in targets]

        rr_values = [r / risk for r in rewards if r > 0]
        rr_first = rr_values[0] if rr_values else 0.0
        rr_best = max(rr_values) if rr_values else 0.0

        atr_val = self._safe_last(atr(df, 14), max(close_last * 0.01, 1e-9))
        stop_atr = risk / max(atr_val, 1e-9)

        support, resistance = pivot_levels(df, lookback=80)

        # --- Compute indicators for level reasoning ---
        close = df["close"]
        ema20_val  = self._safe_last(ema(close, 20),  float("nan"))
        ema50_val  = self._safe_last(ema(close, 50),  float("nan"))
        ema200_val = self._safe_last(ema(close, 200), float("nan"))
        bb_upper, bb_mid, bb_lower = self._bollinger_bands(close, 20, 2.0)

        # Recent 24-candle swing high/low (approximates daily range)
        recent_df = df.tail(24)
        swing_high = float(recent_df["high"].max())
        swing_low  = float(recent_df["low"].min())

        # Build named level map for proximity reasoning
        named_levels: list[tuple[float, str]] = []
        for v, lbl in [
            (ema20_val,  f"{tf} ema20"),
            (ema50_val,  f"{tf} ema50"),
            (ema200_val, f"{tf} ema200"),
            (bb_mid,     f"{tf} bollinger mid"),
            (bb_upper,   f"{tf} bollinger upper"),
            (bb_lower,   f"{tf} bollinger lower"),
            (resistance, "resistance"),
            (support,    "support"),
            (swing_high, "recent swing high"),
            (swing_low,  "recent swing low"),
        ]:
            if not math.isnan(v):
                named_levels.append((v, lbl))

        def _reason(price: float, fallback: str) -> str:
            label = _nearest(price, named_levels, pct=0.025)
            return label if label else fallback

        # --- Entry context ---
        if direction == "long":
            near_level = entry <= support * 1.03
            entry_context = (
                f"entry near support ({_fmt(support)})"
                if near_level
                else f"entry above support ({_fmt(support)}) — possible chase"
            )
        else:
            near_level = entry >= resistance * 0.97
            entry_context = (
                f"entry near resistance ({_fmt(resistance)})"
                if near_level
                else f"entry below resistance ({_fmt(resistance)}) — may be late"
            )

        # --- Stop note ---
        if stop_atr < 0.8:
            stop_note = f"stop looks tight ({stop_atr:.2f} ATR) — noise could hunt it"
        elif stop_atr > 3.5:
            stop_note = f"stop is wide ({stop_atr:.2f} ATR) — safer but capital inefficient"
        else:
            stop_note = f"stop distance is reasonable ({stop_atr:.2f} ATR)"

        # --- Verdict score ---
        score = 0
        score += 1 if rr_first >= 1.5 else 0
        score += 1 if rr_best >= 3.0 else 0
        score += 1 if 0.8 <= stop_atr <= 3.5 else 0
        score += 1 if near_level else 0

        if score >= 3:
            verdict = "good"
        elif score == 2:
            verdict = "ok"
        else:
            verdict = "weak"

        # --- Suggested levels with reasoning ---
        if direction == "long":
            suggested_stop = min(stop, support - 0.35 * atr_val, entry - 1.2 * atr_val)
            suggested_tp1  = max(targets[0], entry + 1.5 * risk)
            suggested_tp2  = max(max(targets), entry + 2.8 * risk, resistance)
        else:
            suggested_stop = max(stop, resistance + 0.35 * atr_val, entry + 1.2 * atr_val)
            suggested_tp1  = min(targets[0], entry - 1.5 * risk)
            suggested_tp2  = min(min(targets), entry - 2.8 * risk, support)

        # Build human reasons for each suggested level
        if direction == "long":
            entry_reason = _reason(entry, f"near support {_fmt(support)}")
            stop_reason  = _reason(suggested_stop, f"below support {_fmt(support)} + ATR buffer")
            tp1_reason   = _reason(suggested_tp1, "1.5R from entry")
            tp2_reason   = _reason(suggested_tp2, "2.8R / near resistance")
        else:
            entry_reason = _reason(entry, f"near resistance {_fmt(resistance)}")
            stop_reason  = _reason(suggested_stop, f"above resistance {_fmt(resistance)} + ATR buffer")
            tp1_reason   = _reason(suggested_tp1, "1.5R from entry")
            tp2_reason   = _reason(suggested_tp2, f"2.8R / near support {_fmt(support)}")

        # --- Position sizing ---
        position = None
        size_note = "add `amount` and `leverage` for dollar PnL estimates"
        margin = float(amount_usd) if amount_usd is not None else None
        lev = float(leverage) if leverage is not None else None
        if margin is not None and margin <= 0:
            margin = None
        if lev is not None and lev <= 0:
            lev = None

        if margin is not None and lev is not None:
            notional = margin * lev
            qty = notional / max(entry, 1e-9)

            def _pnl(exit_price: float) -> float:
                if direction == "long":
                    return (exit_price - entry) * qty
                return (entry - exit_price) * qty

            stop_pnl = _pnl(stop)
            tp_rows = [{"tp": float(tp), "pnl_usd": round(_pnl(float(tp)), 2)} for tp in targets]
            position = {
                "margin_usd": round(margin, 2),
                "leverage": round(lev, 3),
                "notional_usd": round(notional, 2),
                "qty": round(qty, 8),
                "stop_pnl_usd": round(stop_pnl, 2),
                "tp_pnls": tp_rows,
            }
            size_note = "PnL estimates assume fixed size, no fees/funding/slippage"
        elif margin is not None and lev is None:
            size_note = "margin captured — add leverage (e.g. `10x`) for dollar PnL"

        return {
            "symbol": symbol,
            "timeframe": tf,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "targets": targets,
            "verdict": verdict,
            "rr_first": round(rr_first, 2),
            "rr_best": round(rr_best, 2),
            "atr": round(float(atr_val), 6),
            "stop_atr": round(stop_atr, 2),
            "support": support,
            "resistance": resistance,
            "entry_context": entry_context,
            "stop_note": stop_note,
            "suggested": {
                "entry":       round(entry, 8),
                "entry_why":   entry_reason,
                "stop":        round(suggested_stop, 8),
                "stop_why":    stop_reason,
                "tp1":         round(suggested_tp1, 8),
                "tp1_why":     tp1_reason,
                "tp2":         round(suggested_tp2, 8),
                "tp2_why":     tp2_reason,
            },
            "indicators": {
                "ema20":    round(ema20_val, 8)  if not math.isnan(ema20_val)  else None,
                "ema50":    round(ema50_val, 8)  if not math.isnan(ema50_val)  else None,
                "ema200":   round(ema200_val, 8) if not math.isnan(ema200_val) else None,
                "bb_upper": round(bb_upper, 8)   if not math.isnan(bb_upper)   else None,
                "bb_mid":   round(bb_mid, 8)     if not math.isnan(bb_mid)     else None,
                "bb_lower": round(bb_lower, 8)   if not math.isnan(bb_lower)   else None,
            },
            "position": position,
            "size_note": size_note,
            "risk_line": "use as a risk-planning map — execute only if structure still holds",
        }
