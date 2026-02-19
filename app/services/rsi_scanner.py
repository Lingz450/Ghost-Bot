from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from app.adapters.market_router import MarketDataRouter
from app.adapters.ohlcv import OHLCVAdapter
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient
from app.core.ta import ema, rsi
from app.db.models import IndicatorSnapshot, UniverseSymbol

logger = logging.getLogger(__name__)


class RSIScannerService:
    SUPPORTED_SCAN_TIMEFRAMES = {"15m", "1h", "4h", "1d"}
    TIMEFRAME_MINUTES = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}

    def __init__(
        self,
        http: ResilientHTTPClient,
        cache: RedisCache,
        ohlcv_adapter: OHLCVAdapter,
        market_router: MarketDataRouter,
        coingecko_base: str,
        binance_base: str,
        db_factory,
        universe_size: int = 500,
        scan_timeframes: list[str] | None = None,
        concurrency: int = 12,
        freshness_minutes: int = 45,
        live_fallback_universe: int = 120,
    ) -> None:
        self.http = http
        self.cache = cache
        self.ohlcv_adapter = ohlcv_adapter
        self.market_router = market_router
        self.coingecko_base = coingecko_base
        self.binance_base = binance_base
        self.db_factory = db_factory
        self.universe_size = max(100, min(int(universe_size), 1000))
        self.scan_timeframes = self._normalize_scan_timeframes(scan_timeframes or ["15m", "1h", "4h", "1d"])
        self.concurrency = max(2, min(int(concurrency), 32))
        self.freshness_minutes = max(5, min(int(freshness_minutes), 240))
        self.live_fallback_universe = max(20, min(int(live_fallback_universe), 300))

    def _normalize_scan_timeframes(self, values: list[str]) -> list[str]:
        out: list[str] = []
        for value in values:
            tf = str(value).strip().lower()
            if tf in self.SUPPORTED_SCAN_TIMEFRAMES and tf not in out:
                out.append(tf)
        return out or ["15m", "1h", "4h", "1d"]

    async def _top_usdt_symbols(self, universe_size: int = 80) -> list[dict]:
        cap = max(20, min(int(universe_size), 1000))
        cache_key = f"rsi:universe:{cap}"
        cached = await self.cache.get_json(cache_key)
        if cached:
            return [{"symbol": str(row["symbol"]), "quote_volume_24h": float(row["quote_volume_24h"])} for row in cached]

        available = await self.market_router.get_market_universe("spot")
        if available:
            try:
                page_size = 250
                pages = max(1, min((cap // page_size) + 2, 6))
                markets: list[dict] = []
                for page in range(1, pages + 1):
                    chunk = await self.http.get_json(
                        f"{self.coingecko_base}/coins/markets",
                        params={
                            "vs_currency": "usd",
                            "order": "market_cap_desc",
                            "per_page": page_size,
                            "page": page,
                            "sparkline": "false",
                        },
                    )
                    markets.extend(list(chunk))
                scored: list[dict] = []
                for row in markets:
                    sym = str(row.get("symbol", "")).upper()
                    if not sym or sym not in available:
                        continue
                    vol = float(row.get("total_volume", 0) or 0)
                    if vol <= 0:
                        continue
                    scored.append({"symbol": sym, "quote_volume_24h": vol})
                if scored:
                    scored.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
                    symbols = scored[:cap]
                    await self.cache.set_json(cache_key, symbols, ttl=300)
                    return symbols
            except Exception as exc:  # noqa: BLE001
                logger.warning("rsi_coingecko_universe_failed", extra={"event": "rsi_universe_error", "error": str(exc)})

        try:
            rows = await self.http.get_json(f"{self.binance_base}/api/v3/ticker/24hr")
        except Exception as exc:  # noqa: BLE001
            logger.warning("rsi_universe_fetch_failed", extra={"event": "rsi_universe_error", "error": str(exc)})
            return await self._top_symbols_from_db(cap)

        scored: list[dict] = []
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
            base = symbol[:-4]
            if available and base not in available:
                continue
            scored.append({"symbol": base, "quote_volume_24h": quote_vol})

        scored.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
        symbols = scored[:cap]
        await self.cache.set_json(cache_key, symbols, ttl=180)
        return symbols

    async def _top_symbols_from_db(self, cap: int) -> list[dict]:
        async with self.db_factory() as session:
            query = await session.execute(
                select(UniverseSymbol.symbol, UniverseSymbol.quote_volume_24h)
                .order_by(UniverseSymbol.rank.asc())
                .limit(cap)
            )
            rows = query.all()
        return [
            {"symbol": str(symbol).upper(), "quote_volume_24h": float(quote_vol or 0.0)}
            for symbol, quote_vol in rows
        ]

    async def refresh_universe(self, universe_size: int | None = None) -> dict:
        cap = max(20, min(int(universe_size or self.universe_size), 1000))
        ranked = await self._top_usdt_symbols(cap)
        if not ranked:
            return {
                "symbols": 0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "skipped": True,
                "reason": "universe_unavailable",
            }
        now = datetime.now(timezone.utc)
        rows = []
        for idx, row in enumerate(ranked, start=1):
            rows.append(
                {
                    "symbol": str(row["symbol"]).upper(),
                    "exchange": "router",
                    "rank": idx,
                    "quote_volume_24h": float(row["quote_volume_24h"]),
                    "updated_at": now,
                }
            )

        async with self.db_factory() as session:
            await session.execute(delete(UniverseSymbol))
            if rows:
                await session.execute(insert(UniverseSymbol).values(rows))
            await session.commit()

        return {"symbols": len(rows), "updated_at": now.isoformat()}

    async def _universe_symbols(self, limit: int | None = None) -> list[str]:
        cap = max(20, min(int(limit or self.universe_size), 1000))
        async with self.db_factory() as session:
            query = await session.execute(
                select(UniverseSymbol.symbol)
                .order_by(UniverseSymbol.rank.asc())
                .limit(cap)
            )
            rows = [str(x) for x in query.scalars().all()]

        if rows:
            return rows

        await self.refresh_universe(cap)
        async with self.db_factory() as session:
            query = await session.execute(
                select(UniverseSymbol.symbol)
                .order_by(UniverseSymbol.rank.asc())
                .limit(cap)
            )
            return [str(x) for x in query.scalars().all()]

    async def _symbol_rsi(self, symbol: str, timeframe: str, rsi_length: int) -> dict | None:
        try:
            candles = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe=timeframe, limit=max(180, rsi_length * 8))
        except Exception:  # noqa: BLE001
            return None
        if len(candles) < rsi_length + 5:
            return None
        df = pd.DataFrame(candles)
        if df.empty or "close" not in df:
            return None
        series = rsi(df["close"], rsi_length).dropna()
        if series.empty:
            return None
        value = float(series.iloc[-1])
        return {
            "symbol": symbol.upper(),
            "rsi": round(value, 2),
            "source_line": str(candles[-1].get("source_line") or ""),
        }

    async def _symbol_snapshot(self, symbol: str, timeframe: str) -> dict | None:
        try:
            candles = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe=timeframe, limit=260)
        except Exception:  # noqa: BLE001
            return None
        if len(candles) < 60:
            return None
        df = pd.DataFrame(candles)
        if df.empty or "close" not in df:
            return None

        close = df["close"].astype(float)
        rsi14_series = rsi(close, 14).dropna()
        if rsi14_series.empty:
            return None

        last_candle = candles[-1] if candles else {}
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "exchange": str(last_candle.get("exchange") or "router"),
            "close_price": float(close.iloc[-1]),
            "rsi14": float(rsi14_series.iloc[-1]),
            "ema20": float(ema(close, 20).iloc[-1]),
            "ema50": float(ema(close, 50).iloc[-1]),
            "ema100": float(ema(close, 100).iloc[-1]),
            "ema200": float(ema(close, 200).iloc[-1]),
            "computed_at": datetime.now(timezone.utc),
        }

    async def _upsert_snapshots(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        stmt = insert(IndicatorSnapshot).values(rows)
        update_fields = {
            "close_price": stmt.excluded.close_price,
            "rsi14": stmt.excluded.rsi14,
            "ema20": stmt.excluded.ema20,
            "ema50": stmt.excluded.ema50,
            "ema100": stmt.excluded.ema100,
            "ema200": stmt.excluded.ema200,
            "computed_at": stmt.excluded.computed_at,
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "timeframe", "exchange"],
            set_=update_fields,
        )
        async with self.db_factory() as session:
            await session.execute(stmt)
            await session.commit()
        return len(rows)

    def _due_timeframes(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now(timezone.utc)
        epoch_minute = int(now.timestamp() // 60)
        due: list[str] = []
        for tf in self.scan_timeframes:
            interval = self.TIMEFRAME_MINUTES.get(tf)
            if not interval:
                continue
            # Called every ~5 minutes from scheduler/cron; this catches timeframe closes.
            if epoch_minute % interval < 5:
                due.append(tf)
        return due

    async def refresh_indicators(
        self,
        *,
        timeframes: list[str] | None = None,
        universe_size: int | None = None,
        force: bool = False,
    ) -> dict:
        tfs = self._normalize_scan_timeframes(timeframes or self.scan_timeframes)
        if not force:
            due = set(self._due_timeframes())
            tfs = [tf for tf in tfs if tf in due]
        if not tfs:
            return {"updated": 0, "timeframes": [], "symbols": 0, "updated_at": datetime.now(timezone.utc).isoformat()}

        symbols = await self._universe_symbols(limit=universe_size or self.universe_size)
        if not symbols:
            return {"updated": 0, "timeframes": tfs, "symbols": 0, "updated_at": datetime.now(timezone.utc).isoformat()}

        total_rows = 0
        semaphore = asyncio.Semaphore(self.concurrency)

        for tf in tfs:
            async def _one(sym: str) -> dict | None:
                async with semaphore:
                    return await self._symbol_snapshot(sym, tf)

            computed = await asyncio.gather(*[_one(sym) for sym in symbols], return_exceptions=False)
            rows = [row for row in computed if row]
            total_rows += await self._upsert_snapshots(rows)

        return {
            "updated": total_rows,
            "timeframes": tfs,
            "symbols": len(symbols),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _query_precomputed(self, timeframe: str, mode: str, limit: int) -> list[dict]:
        freshness_cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.freshness_minutes)
        async with self.db_factory() as session:
            query = await session.execute(
                select(
                    IndicatorSnapshot.symbol,
                    IndicatorSnapshot.rsi14,
                    IndicatorSnapshot.computed_at,
                )
                .where(IndicatorSnapshot.timeframe == timeframe)
                .where(IndicatorSnapshot.computed_at >= freshness_cutoff)
                .where(IndicatorSnapshot.rsi14.is_not(None))
                .order_by(IndicatorSnapshot.computed_at.desc())
                .limit(max(limit * 40, 1000))
            )
            rows = query.all()

        latest_by_symbol: dict[str, float] = {}
        for sym, rsi_v, _computed_at in rows:
            symbol = str(sym).upper()
            if symbol in latest_by_symbol:
                continue
            latest_by_symbol[symbol] = float(rsi_v)

        ranked = [{"symbol": sym, "rsi": round(val, 2)} for sym, val in latest_by_symbol.items()]
        ranked.sort(key=lambda x: x["rsi"], reverse=(mode != "oversold"))
        return ranked[:limit]

    async def _scan_live_universe(self, timeframe: str, mode: str, limit: int, rsi_length: int) -> list[dict]:
        top = await self._top_usdt_symbols(universe_size=self.live_fallback_universe)
        universe = [str(row["symbol"]).upper() for row in top]
        sem = asyncio.Semaphore(self.concurrency)

        async def _one(sym: str) -> dict | None:
            async with sem:
                return await self._symbol_rsi(sym, timeframe, rsi_length)

        raw = await asyncio.gather(*[_one(sym) for sym in universe], return_exceptions=False)
        items = [x for x in raw if x]
        if mode == "oversold":
            items.sort(key=lambda x: x["rsi"])
        else:
            items.sort(key=lambda x: x["rsi"], reverse=True)
        return items[:limit]

    def _bucket(self, value: float, mode: str) -> str:
        if mode == "oversold":
            if value <= 20:
                return "extreme oversold"
            if value <= 30:
                return "oversold"
            return "neutral"
        if value >= 80:
            return "extreme overbought"
        if value >= 70:
            return "overbought"
        return "neutral"

    async def scan(
        self,
        timeframe: str = "1h",
        mode: str = "oversold",
        limit: int = 10,
        rsi_length: int = 14,
        symbol: str | None = None,
    ) -> dict:
        tf = timeframe or "1h"
        mode_norm = "overbought" if mode.lower() == "overbought" else "oversold"
        cap = max(1, min(limit, 20))
        rsi_length = max(2, min(int(rsi_length), 50))
        source = "precomputed"

        if symbol:
            row = await self._symbol_rsi(symbol.upper(), tf, rsi_length)
            items = [row] if row else []
            source = "live_symbol"
        else:
            items = []
            if rsi_length == 14 and tf in self.scan_timeframes:
                items = await self._query_precomputed(tf, mode_norm, cap)
                if len(items) < max(3, cap // 2):
                    try:
                        await self.refresh_universe(self.universe_size)
                        await self.refresh_indicators(timeframes=[tf], universe_size=self.universe_size, force=True)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "rsi_precompute_refresh_failed",
                            extra={"event": "rsi_refresh_fallback", "timeframe": tf, "error": str(exc)},
                        )
                    items = await self._query_precomputed(tf, mode_norm, cap)
            if not items:
                items = await self._scan_live_universe(tf, mode_norm, cap, rsi_length)
                source = "live_fallback"

        ranked = []
        for row in items[:cap]:
            ranked.append(
                {
                    "symbol": row["symbol"],
                    "rsi": row["rsi"],
                    "note": self._bucket(float(row["rsi"]), mode_norm),
                    "source_line": row.get("source_line", ""),
                }
            )

        source_line = "Data source: precomputed multi-exchange snapshots | Updated: just now"
        for row in ranked:
            if row.get("source_line"):
                source_line = row["source_line"]
                break

        return {
            "summary": f"RSI scan ({mode_norm}) on {tf} using RSI({rsi_length}) [{source}].",
            "timeframe": tf,
            "mode": mode_norm,
            "rsi_length": rsi_length,
            "items": ranked,
            "source_line": source_line,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
