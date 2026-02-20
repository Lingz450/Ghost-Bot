from __future__ import annotations

import asyncio
import math

import pandas as pd

from app.adapters.derivatives import DerivativesAdapter
from app.adapters.ohlcv import BINANCE_SUPPORTED_INTERVALS, OHLCVAdapter
from app.adapters.prices import PriceAdapter
from app.core.fmt import fmt_level
from app.core.ta import atr, bollinger_mid, consolidation_zone, ema, macd, pivot_levels, rsi
from app.services.market_context import MarketContextService, format_market_context
from app.services.news import NewsService

DEFAULT_TIMEFRAME_PACK = ["15m", "1h", "4h", "1d"]
DEFAULT_EMA_PERIODS = [20, 50, 200]
DEFAULT_RSI_PERIODS = [14]
ALL_EMA_PERIODS = [9, 20, 50, 100, 200]
ALL_RSI_PERIODS = [7, 14, 21]
MAX_TIMEFRAMES = 4
MAX_EMA_PERIODS = 5
MAX_RSI_PERIODS = 3


class MarketAnalysisService:
    def __init__(
        self,
        price_adapter: PriceAdapter,
        ohlcv_adapter: OHLCVAdapter,
        deriv_adapter: DerivativesAdapter,
        news_service: NewsService,
        fast_mode: bool = True,
        default_timeframes: list[str] | None = None,
        include_derivatives_default: bool = False,
        include_news_default: bool = False,
        request_timeout_sec: float = 8.0,
    ) -> None:
        self.price_adapter = price_adapter
        self.ohlcv_adapter = ohlcv_adapter
        self.deriv_adapter = deriv_adapter
        self.news_service = news_service
        self.fast_mode = fast_mode
        self.default_timeframes = default_timeframes or ["1h"]
        self.include_derivatives_default = include_derivatives_default
        self.include_news_default = include_news_default
        self.request_timeout_sec = max(2.0, float(request_timeout_sec))
        self.market_context_service = MarketContextService(price_adapter, ohlcv_adapter)
        self._narrative_map = {
            "PHB": "AI/privacy infrastructure narrative on BNB-linked flow; usually momentum-driven around AI headlines.",
            "FET": "AI-agent narrative token; often beta to broader AI sector and BTC risk sentiment.",
            "AGIX": "AI narrative rotation coin; reacts hard to sector headlines and liquidity spikes.",
            "RNDR": "AI + compute narrative; tends to track high-beta growth sentiment.",
            "ARB": "L2 beta play; liquidity and incentive headlines can move it quickly.",
        }

    async def get_market_context(self) -> dict:
        return await self.market_context_service.get_market_context()

    def _tf_rank(self, tf: str) -> int:
        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return int(tf[:-1]) * 60
        if tf.endswith("d"):
            return int(tf[:-1]) * 1440
        if tf.endswith("w"):
            return int(tf[:-1]) * 10080
        if tf.endswith("M"):
            return int(tf[:-1]) * 43200
        return 999999

    def _safe_last(self, series: pd.Series, default: float) -> float:
        cleaned = series.dropna()
        if cleaned.empty:
            return default
        value = float(cleaned.iloc[-1])
        if math.isnan(value):
            return default
        return value

    async def _with_timeout(self, coro, timeout: float):
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            return exc

    async def analyze(
        self,
        symbol: str,
        direction: str | None = None,
        timeframe: str | None = None,
        timeframes: list[str] | None = None,
        ema_periods: list[int] | None = None,
        rsi_periods: list[int] | None = None,
        all_timeframes: bool = False,
        all_emas: bool = False,
        all_rsis: bool = False,
        notes: list[str] | None = None,
        include_derivatives: bool | None = None,
        include_news: bool | None = None,
    ) -> dict:
        symbol = symbol.upper()
        notes = list(notes or [])

        include_derivatives_flag = self.include_derivatives_default if include_derivatives is None else bool(include_derivatives)
        include_news_flag = self.include_news_default if include_news is None else bool(include_news)

        if all_timeframes:
            requested_tfs = DEFAULT_TIMEFRAME_PACK.copy()
        elif timeframes:
            requested_tfs = [tf for tf in timeframes if tf]
        elif timeframe:
            requested_tfs = [timeframe]
        else:
            requested_tfs = self.default_timeframes.copy() if self.fast_mode else ["1h", "4h"]

        requested_tfs = [tf if tf.endswith("M") else tf.lower() for tf in requested_tfs]
        requested_tfs = [tf for tf in requested_tfs if tf in BINANCE_SUPPORTED_INTERVALS]
        if not requested_tfs:
            requested_tfs = self.default_timeframes.copy() if self.fast_mode else ["1h", "4h"]

        dedup_tfs: list[str] = []
        for tf in requested_tfs:
            if tf not in dedup_tfs:
                dedup_tfs.append(tf)
        requested_tfs = dedup_tfs

        if len(requested_tfs) > MAX_TIMEFRAMES:
            requested_tfs = requested_tfs[:MAX_TIMEFRAMES]
            notes.append("Running the core timeframe pack to keep it fast.")

        if all_emas:
            ema_set = ALL_EMA_PERIODS.copy()
        elif ema_periods:
            ema_set = sorted({int(p) for p in ema_periods if 2 <= int(p) <= 500})
        else:
            ema_set = DEFAULT_EMA_PERIODS.copy()
        if len(ema_set) > MAX_EMA_PERIODS:
            ema_set = ema_set[:MAX_EMA_PERIODS]
            notes.append("Trimmed EMA list to max 5 periods for speed.")

        if all_rsis:
            rsi_set = ALL_RSI_PERIODS.copy()
        elif rsi_periods:
            rsi_set = sorted({int(p) for p in rsi_periods if 2 <= int(p) <= 50})
        else:
            rsi_set = DEFAULT_RSI_PERIODS.copy()
        if len(rsi_set) > MAX_RSI_PERIODS:
            rsi_set = rsi_set[:MAX_RSI_PERIODS]
            notes.append("Trimmed RSI list to max 3 periods for speed.")

        if all_timeframes or all_emas or all_rsis:
            tf_txt = "/".join(requested_tfs)
            ema_txt = "/".join(str(x) for x in ema_set)
            rsi_txt = "/".join(str(x) for x in rsi_set)
            notes.append(f"Core pack active for speed: TF={tf_txt}, EMA={ema_txt}, RSI={rsi_txt}.")

        price_task = self._with_timeout(self.price_adapter.get_price(symbol), self.request_timeout_sec)
        candle_tasks = [self._with_timeout(self.ohlcv_adapter.get_ohlcv(symbol, tf, 260), self.request_timeout_sec + 2) for tf in requested_tfs]
        deriv_task = (
            self._with_timeout(self.deriv_adapter.get_funding_and_oi(symbol), min(6.0, self.request_timeout_sec))
            if include_derivatives_flag
            else None
        )
        news_task = (
            self._with_timeout(self.news_service.get_asset_headlines(symbol), min(6.0, self.request_timeout_sec))
            if include_news_flag
            else None
        )

        task_list = [price_task]
        if deriv_task is not None:
            task_list.append(deriv_task)
        if news_task is not None:
            task_list.append(news_task)
        task_list.extend(candle_tasks)
        gathered = await asyncio.gather(*task_list, return_exceptions=False)

        idx = 0
        price = gathered[idx]
        idx += 1
        deriv = None
        headlines = None
        if include_derivatives_flag:
            deriv = gathered[idx]
            idx += 1
        if include_news_flag:
            headlines = gathered[idx]
            idx += 1
        candle_results = gathered[idx:]

        if isinstance(price, Exception):
            raise RuntimeError(f"Price unavailable for {symbol}: {price}")

        if deriv is None or isinstance(deriv, Exception):
            deriv = {
                "funding_rate": None,
                "open_interest": None,
                "source": "unavailable",
            }
            if include_derivatives_flag:
                notes.append("Derivatives data unavailable in fast path.")

        if headlines is None or isinstance(headlines, Exception):
            headlines = []
            if include_news_flag:
                notes.append("News catalysts unavailable in fast path.")

        frames: dict[str, pd.DataFrame] = {}
        tf_errors: list[str] = []
        for tf, result in zip(requested_tfs, candle_results, strict=False):
            if isinstance(result, Exception):
                tf_errors.append(tf)
                continue
            df = pd.DataFrame(result)
            if df.empty:
                tf_errors.append(tf)
                continue
            frames[tf] = df

        if not frames:
            raise RuntimeError("No valid OHLCV data for requested timeframe(s).")

        if tf_errors:
            notes.append(f"Skipped unavailable timeframe(s): {', '.join(tf_errors)}")

        sorted_tfs = sorted(frames.keys(), key=self._tf_rank)
        primary_tf = "1h" if "1h" in frames else sorted_tfs[0]
        higher_tf = "4h" if "4h" in frames else sorted_tfs[-1]

        current = float(price["price"])
        data_source_line = price.get("source_line")
        if not data_source_line:
            for tf in sorted_tfs:
                rows = frames[tf].to_dict("records")
                if rows and rows[-1].get("source_line"):
                    data_source_line = rows[-1].get("source_line")
                    break

        rsi_primary_period = 14 if 14 in rsi_set else rsi_set[0]
        ema_anchor_period = 20 if 20 in ema_set else ema_set[0]
        ema_secondary_period = 50 if 50 in ema_set else (ema_set[1] if len(ema_set) > 1 else ema_set[0])

        mtf_snapshot: list[str] = []
        trend_score = 0
        per_tf_metrics: dict[str, dict] = {}

        for tf in sorted_tfs:
            df = frames[tf]
            close = df["close"]
            close_last = self._safe_last(close, current)
            ema_vals = {p: self._safe_last(ema(close, p), close_last) for p in ema_set}
            rsi_vals = {p: self._safe_last(rsi(close, p), 50.0) for p in rsi_set}
            anchor_val = ema_vals[ema_anchor_period]
            relation = "above" if close_last >= anchor_val else "below"
            rsi_label = " | ".join([f"RSI{p} {rsi_vals[p]:.1f}" for p in rsi_set[:2]])
            mtf_snapshot.append(f"{tf}: {rsi_label} | {relation} EMA{ema_anchor_period}")

            trend_score += 1 if close_last > ema_vals[ema_anchor_period] else -1
            if ema_secondary_period != ema_anchor_period:
                trend_score += 1 if ema_vals[ema_anchor_period] > ema_vals[ema_secondary_period] else -1
            trend_score += 1 if rsi_vals[rsi_primary_period] >= 50 else -1

            per_tf_metrics[tf] = {
                "close": close_last,
                "ema": ema_vals,
                "rsi": rsi_vals,
            }

        df_primary = frames[primary_tf]
        df_higher = frames[higher_tf]

        support, resistance = pivot_levels(df_higher, lookback=80)
        zone_low, zone_high = consolidation_zone(df_primary, window=14)
        atr_primary = self._safe_last(atr(df_primary, 14), max(current * 0.01, 1e-8))

        close_primary = df_primary["close"]
        close_higher = df_higher["close"]
        macd_line, macd_signal = macd(close_primary)
        macd_now = self._safe_last(macd_line, 0.0)
        macd_sig_now = self._safe_last(macd_signal, 0.0)
        bb_mid_higher = self._safe_last(bollinger_mid(close_higher, 20), current)

        trend_score += 1 if macd_now > macd_sig_now else -1
        inferred = "long" if trend_score >= 0 else "short"
        side = direction or inferred

        if side == "long":
            entry_low = max(support, zone_low)
            entry_high = min(current, zone_high) if zone_high > entry_low else current
            stop = min(support, entry_low) - max(atr_primary * 0.8, current * 0.008)
            tp1 = max(resistance * 0.985, current + 1.2 * atr_primary)
            tp2 = max(resistance * 1.01, current + 2.1 * atr_primary)
            condition = f"Leaning long if {symbol} holds {fmt_level(entry_low)}"
        else:
            entry_high = min(resistance, zone_high)
            entry_low = max(zone_low, current) if current > 0 else zone_low
            stop = max(resistance, entry_high) + max(atr_primary * 0.8, current * 0.008)
            tp1 = min(support * 1.015, current - 1.2 * atr_primary)
            tp2 = min(support * 0.99, current - 2.1 * atr_primary)
            condition = f"Leaning short if {symbol} rejects {fmt_level(entry_high)}"

        ema_primary_fast = per_tf_metrics[primary_tf]["ema"][ema_anchor_period]
        ema_primary_slow = per_tf_metrics[primary_tf]["ema"][ema_secondary_period]
        ema_higher_fast = per_tf_metrics[higher_tf]["ema"][ema_anchor_period]
        ema_higher_slow = per_tf_metrics[higher_tf]["ema"][ema_secondary_period]
        rsi_primary = per_tf_metrics[primary_tf]["rsi"][rsi_primary_period]
        rsi_higher = per_tf_metrics[higher_tf]["rsi"][rsi_primary_period]

        bullets = [
            (
                f"Trend: {primary_tf} EMA{ema_anchor_period}/{ema_secondary_period} is "
                f"{'bullish' if ema_primary_fast > ema_primary_slow else 'bearish'}, "
                f"{higher_tf} is {'bullish' if ema_higher_fast > ema_higher_slow else 'bearish'}."
            ),
            (
                f"Momentum: RSI{rsi_primary_period} {primary_tf}={rsi_primary:.1f}, "
                f"{higher_tf}={rsi_higher:.1f}, MACD={'up' if macd_now > macd_sig_now else 'down'}."
            ),
            f"Structure: support {fmt_level(support)}, resistance {fmt_level(resistance)}, BB mid {higher_tf} {fmt_level(bb_mid_higher)}.",
        ]

        if include_derivatives_flag:
            bullets.append(
                f"Derivatives: funding={deriv.get('funding_rate')} OI={deriv.get('open_interest')} ({deriv.get('source')})."
            )

        if include_news_flag and headlines:
            bullets.append(f"Catalyst: {headlines[0]['title']}")

        market_context = await self._with_timeout(self.get_market_context(), min(8.0, self.request_timeout_sec + 2.0))
        if isinstance(market_context, Exception):
            market_context = {}
            notes.append("Market context was partially unavailable.")
        market_context_text = format_market_context(market_context if isinstance(market_context, dict) else {})
        btc_ctx = market_context.get("btc", {}) if isinstance(market_context, dict) else {}
        btc_1h = str(btc_ctx.get("trend_1h") or "unknown")
        btc_4h = str(btc_ctx.get("trend_4h") or "unknown")

        summary = (
            f"{symbol} is showing {'relative strength' if trend_score >= 0 else 'selling pressure'} across "
            f"{', '.join(sorted_tfs)}. {condition}. BTC backdrop is {btc_1h} on 1h and {btc_4h} on 4h."
        )

        return {
            "symbol": symbol,
            "side": side,
            "summary": summary,
            "entry": f"{fmt_level(entry_low)} - {fmt_level(entry_high)}",
            "tp1": f"{fmt_level(tp1)} ({primary_tf})",
            "tp2": f"{fmt_level(tp2)} ({higher_tf})",
            "sl": f"{fmt_level(stop)}",
            "why": bullets[:5],
            "condition": condition,
            "price": current,
            "price_source": price["source"],
            "data_source_line": data_source_line,
            "updated_at": price["ts"],
            "risk": "Do not over-size this one. If it loses structure, cut it fast.",
            "market_context": market_context,
            "market_context_text": market_context_text,
            "mtf_snapshot": mtf_snapshot,
            "input_notes": notes,
            "details": {
                "timeframes": sorted_tfs,
                "ema_periods": ema_set,
                "rsi_periods": rsi_set,
                "macd": macd_now,
                "macd_signal": macd_sig_now,
                "atr_primary": atr_primary,
                "primary_tf": primary_tf,
                "higher_tf": higher_tf,
                "derivatives": deriv,
                "headlines": headlines[:3],
                "include_derivatives": include_derivatives_flag,
                "include_news": include_news_flag,
                "market_context": market_context,
            },
        }

    async def fallback_asset_brief(self, symbol: str, reason: str | None = None) -> dict:
        symbol_u = symbol.upper()
        headlines = []
        try:
            headlines = await self.news_service.get_asset_headlines(symbol_u, limit=2)
        except Exception:  # noqa: BLE001
            headlines = []

        narrative = self._narrative_map.get(
            symbol_u,
            f"{symbol_u} likely trades as a narrative-driven alt. Without candles, execution risk is higher.",
        )
        safe_action = (
            "I would wait for a clean breakout/reclaim on your chart, then define entry + stop after confirmation."
        )

        return {
            "symbol": symbol_u,
            "reason": reason or f"I can't fetch candles for {symbol_u} right now.",
            "narrative": narrative,
            "safe_action": safe_action,
            "alternatives": ["BTC", "ETH", "SOL"],
            "headlines": headlines,
        }
