from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select

from app.adapters.market_router import MarketDataRouter
from app.adapters.ohlcv import OHLCVAdapter
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient
from app.core.ta import ema
from app.db.models import IndicatorSnapshot, UniverseSymbol

logger = logging.getLogger(__name__)


class EMAScannerService:
    PRECOMPUTED_EMAS = {20, 50, 100, 200}

    def __init__(
        self,
        http: ResilientHTTPClient,
        cache: RedisCache,
        ohlcv_adapter: OHLCVAdapter,
        market_router: MarketDataRouter,
        binance_base: str,
        db_factory,
        freshness_minutes: int = 45,
        live_fallback_universe: int = 120,
        concurrency: int = 12,
    ) -> None:
        self.http = http
        self.cache = cache
        self.ohlcv_adapter = ohlcv_adapter
        self.market_router = market_router
        self.binance_base = binance_base
        self.db_factory = db_factory
        self.freshness_minutes = max(5, min(int(freshness_minutes), 240))
        self.live_fallback_universe = max(20, min(int(live_fallback_universe), 300))
        self.concurrency = max(2, min(int(concurrency), 32))

    async def _top_usdt_symbols(self, universe_size: int) -> list[str]:
        key = f"ema:universe:{universe_size}"
        cached = await self.cache.get_json(key)
        if isinstance(cached, list) and cached:
            return [str(x) for x in cached]

        available = await self.market_router.get_market_universe("spot")
        if available:
            out = sorted(list(available))[:universe_size]
            if out:
                await self.cache.set_json(key, out, ttl=180)
                return out

        try:
            rows = await self.http.get_json(f"{self.binance_base}/api/v3/ticker/24hr")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ema_universe_fetch_failed", extra={"event": "ema_universe_error", "error": str(exc)})
            return await self._symbols_from_db(universe_size)

        scored: list[tuple[str, float]] = []
        for row in rows:
            symbol = str(row.get("symbol", ""))
            if not symbol.endswith("USDT"):
                continue
            try:
                quote_vol = float(row.get("quoteVolume", 0) or 0)
            except Exception:  # noqa: BLE001
                quote_vol = 0.0
            if quote_vol <= 0:
                continue
            scored.append((symbol[:-4].upper(), quote_vol))
        scored.sort(key=lambda x: x[1], reverse=True)
        out = [s for s, _ in scored[:universe_size]]
        await self.cache.set_json(key, out, ttl=180)
        return out

    async def _symbols_from_db(self, universe_size: int) -> list[str]:
        async with self.db_factory() as session:
            query = await session.execute(
                select(UniverseSymbol.symbol)
                .order_by(UniverseSymbol.rank.asc())
                .limit(universe_size)
            )
            return [str(x).upper() for x in query.scalars().all()]

    async def _query_precomputed(self, timeframe: str, ema_length: int, mode: str, limit: int) -> list[dict]:
        field_map = {
            20: IndicatorSnapshot.ema20,
            50: IndicatorSnapshot.ema50,
            100: IndicatorSnapshot.ema100,
            200: IndicatorSnapshot.ema200,
        }
        ema_field = field_map.get(ema_length)
        if ema_field is None:
            return []

        freshness_cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.freshness_minutes)
        async with self.db_factory() as session:
            query = await session.execute(
                select(
                    IndicatorSnapshot.symbol,
                    IndicatorSnapshot.close_price,
                    ema_field,
                    IndicatorSnapshot.computed_at,
                )
                .where(IndicatorSnapshot.timeframe == timeframe)
                .where(IndicatorSnapshot.computed_at >= freshness_cutoff)
                .where(IndicatorSnapshot.close_price.is_not(None))
                .where(ema_field.is_not(None))
                .order_by(IndicatorSnapshot.computed_at.desc())
                .limit(max(1500, limit * 20))
            )
            rows = query.all()

        latest_rows: dict[str, tuple[float, float]] = {}
        items: list[dict] = []
        for symbol, close_price, ema_val, _computed_at in rows:
            symbol_u = str(symbol).upper()
            if symbol_u in latest_rows:
                continue
            latest_rows[symbol_u] = (float(close_price), float(ema_val))

        for symbol_u, (close_f, ema_f) in latest_rows.items():
            if ema_f == 0:
                continue
            diff_pct = ((close_f - ema_f) / ema_f) * 100.0
            if mode == "above" and diff_pct < 0:
                continue
            if mode == "below" and diff_pct > 0:
                continue
            items.append(
                {
                    "symbol": symbol_u,
                    "price": round(close_f, 6),
                    "ema": round(ema_f, 6),
                    "distance_pct": round(diff_pct, 3),
                    "side": "above" if diff_pct >= 0 else "below",
                }
            )

        items.sort(key=lambda x: abs(x["distance_pct"]))
        return items[:limit]

    async def _symbol_ema(self, symbol: str, timeframe: str, ema_length: int) -> dict | None:
        try:
            candles = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe=timeframe, limit=max(ema_length * 3, 220))
        except Exception:  # noqa: BLE001
            return None
        if len(candles) < ema_length + 5:
            return None
        df = pd.DataFrame(candles)
        if df.empty or "close" not in df:
            return None
        close = df["close"].astype(float)
        ema_series = ema(close, ema_length).dropna()
        if ema_series.empty:
            return None
        price = float(close.iloc[-1])
        ema_val = float(ema_series.iloc[-1])
        if ema_val == 0:
            return None
        diff_pct = ((price - ema_val) / ema_val) * 100.0
        return {
            "symbol": symbol.upper(),
            "price": round(price, 6),
            "ema": round(ema_val, 6),
            "distance_pct": round(diff_pct, 3),
            "side": "above" if diff_pct >= 0 else "below",
            "source_line": str(candles[-1].get("source_line") or ""),
        }

    async def _scan_live(self, timeframe: str, ema_length: int, mode: str, limit: int) -> list[dict]:
        universe = await self._top_usdt_symbols(self.live_fallback_universe)
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(sym: str) -> dict | None:
            async with sem:
                return await self._symbol_ema(sym, timeframe, ema_length)

        rows = await asyncio.gather(*[_one(s) for s in universe], return_exceptions=False)
        items = [x for x in rows if x]
        if mode == "above":
            items = [x for x in items if x["distance_pct"] >= 0]
        elif mode == "below":
            items = [x for x in items if x["distance_pct"] <= 0]
        items.sort(key=lambda x: abs(x["distance_pct"]))
        return items[:limit]

    async def scan(
        self,
        *,
        timeframe: str = "4h",
        ema_length: int = 200,
        mode: str = "closest",
        limit: int = 10,
    ) -> dict:
        tf = (timeframe or "4h").lower()
        ema_len = max(2, min(int(ema_length), 500))
        mode_norm = mode if mode in {"closest", "above", "below"} else "closest"
        cap = max(1, min(int(limit), 20))
        cache_key = f"scan:ema:{tf}:{ema_len}:{mode_norm}:{cap}"
        cached = await self.cache.get_json(cache_key)
        if cached:
            return cached

        source = "precomputed"
        items: list[dict]
        if ema_len in self.PRECOMPUTED_EMAS:
            items = await self._query_precomputed(tf, ema_len, mode_norm, cap)
        else:
            items = []
        if not items:
            items = await self._scan_live(tf, ema_len, mode_norm, cap)
            source = "live_fallback"

        payload = {
            "summary": f"EMA scan on {tf} | EMA({ema_len}) | mode={mode_norm} [{source}]",
            "timeframe": tf,
            "ema_length": ema_len,
            "mode": mode_norm,
            "items": items,
            "source_line": "Data source: precomputed multi-exchange snapshots | Updated: just now",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        for row in items:
            if row.get("source_line"):
                payload["source_line"] = row.get("source_line")
                break
        await self.cache.set_json(cache_key, payload, ttl=75)
        return payload
