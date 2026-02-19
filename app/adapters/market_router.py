from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from app.adapters.exchanges import (
    BinanceExchangeAdapter,
    BloFinExchangeAdapter,
    BybitExchangeAdapter,
    MEXCExchangeAdapter,
    OKXExchangeAdapter,
)
from app.adapters.exchanges.utils import candles_are_sane
from app.adapters.symbols import normalize_symbol
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


@dataclass
class PriceQuote:
    symbol: str
    price: float
    exchange: str
    market_kind: str
    instrument_id: str
    source: str
    source_line: str
    updated_at: str
    fallback_from: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OhlcvSeries:
    symbol: str
    timeframe: str
    candles: list[dict]
    exchange: str
    market_kind: str
    instrument_id: str
    source: str
    source_line: str
    updated_at: str
    fallback_from: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrderBook:
    symbol: str
    bids: list[list[float]]
    asks: list[list[float]]
    exchange: str
    market_kind: str
    instrument_id: str
    source: str
    source_line: str
    updated_at: str
    fallback_from: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FundingOI:
    symbol: str
    exchange: str
    market_kind: str
    instrument_id: str
    funding_rate: float | None
    open_interest: float | None
    source: str
    source_line: str
    updated_at: str
    fallback_from: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class MarketDataRouter:
    def __init__(
        self,
        *,
        http: ResilientHTTPClient,
        cache: RedisCache,
        binance_base_url: str,
        binance_futures_base_url: str,
        bybit_base_url: str,
        okx_base_url: str,
        mexc_base_url: str,
        blofin_base_url: str,
        enable_binance: bool = True,
        enable_bybit: bool = True,
        enable_okx: bool = True,
        enable_mexc: bool = False,
        enable_blofin: bool = False,
        exchange_priority: str = "binance,bybit,okx,mexc,blofin",
        market_prefer_spot: bool = True,
        best_source_ttl_hours: int = 12,
        instruments_ttl_min: int = 45,
    ) -> None:
        self.http = http
        self.cache = cache
        self.market_prefer_spot = bool(market_prefer_spot)
        self.best_source_ttl_sec = max(3600, int(best_source_ttl_hours) * 3600)
        self.price_ttl = 12
        self.ohlcv_ttl = 90
        self.orderbook_ttl = 20
        self.instruments_ttl_sec = max(300, int(instruments_ttl_min) * 60)

        all_adapters: dict[str, object] = {}
        if enable_binance:
            all_adapters["binance"] = BinanceExchangeAdapter(
                http=http,
                cache=cache,
                spot_base_url=binance_base_url,
                futures_base_url=binance_futures_base_url,
                instruments_ttl_sec=self.instruments_ttl_sec,
            )
        if enable_bybit:
            all_adapters["bybit"] = BybitExchangeAdapter(
                http=http,
                cache=cache,
                base_url=bybit_base_url,
                instruments_ttl_sec=self.instruments_ttl_sec,
            )
        if enable_okx:
            all_adapters["okx"] = OKXExchangeAdapter(
                http=http,
                cache=cache,
                base_url=okx_base_url,
                instruments_ttl_sec=self.instruments_ttl_sec,
            )
        if enable_mexc:
            all_adapters["mexc"] = MEXCExchangeAdapter(
                http=http,
                cache=cache,
                base_url=mexc_base_url,
                instruments_ttl_sec=self.instruments_ttl_sec,
            )
        if enable_blofin:
            all_adapters["blofin"] = BloFinExchangeAdapter()

        priority = [x.strip().lower() for x in (exchange_priority or "").split(",") if x.strip()]
        ordered = [x for x in priority if x in all_adapters]
        for name in list(all_adapters.keys()):
            if name not in ordered:
                ordered.append(name)
        self.priority = ordered
        self.adapters = {name: all_adapters[name] for name in ordered}

    def _normalize(self, symbol: str) -> str:
        return normalize_symbol(symbol).base

    async def _acquire_exchange_token(self, exchange: str) -> None:
        key = f"ex:rate:{exchange}:{int(datetime.now(timezone.utc).timestamp())}"
        count = await self.cache.incr_with_expiry(key, ttl=2)
        if count > 12:
            await asyncio.sleep(0.12)

    async def _best_source_get(self, symbol: str, market_kind: str) -> dict | None:
        return await self.cache.get_json(f"best_source:{symbol}:{market_kind}")

    async def _best_source_set(self, symbol: str, market_kind: str, exchange: str, instrument_id: str) -> None:
        await self.cache.set_json(
            f"best_source:{symbol}:{market_kind}",
            {"exchange": exchange, "instrument_id": instrument_id, "ts": datetime.now(timezone.utc).isoformat()},
            ttl=self.best_source_ttl_sec,
        )

    async def _markets(self, exchange: str, market_kind: str) -> dict[str, str]:
        adapter = self.adapters[exchange]
        if market_kind == "spot":
            return await adapter.list_spot_markets()
        return await adapter.list_perp_markets()

    async def _resolve_instrument(self, exchange: str, symbol: str, market_kind: str) -> str | None:
        try:
            markets = await self._markets(exchange, market_kind)
        except Exception:  # noqa: BLE001
            return None
        return markets.get(symbol)

    def _source_line(
        self,
        *,
        exchange: str,
        market_kind: str,
        instrument_id: str,
        updated_at: str,
        fallback_from: str | None,
    ) -> str:
        updated = self._relative_updated(updated_at)
        kind = "Spot" if market_kind == "spot" else "Perp"
        display = f"{exchange.capitalize()} {kind} ({instrument_id})"
        if fallback_from:
            return f"Data source: {display} (fallback from {fallback_from.capitalize()}) | Updated: {updated}"
        return f"Data source: {display} | Updated: {updated}"

    def _relative_updated(self, ts_iso: str) -> str:
        try:
            then = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            secs = max(0, int((now - then.astimezone(timezone.utc)).total_seconds()))
            if secs < 60:
                return f"{secs}s ago"
            mins = secs // 60
            return f"{mins}m ago"
        except Exception:  # noqa: BLE001
            return ts_iso

    def _kinds_for_market_data(self) -> list[str]:
        return ["spot", "perp"] if self.market_prefer_spot else ["perp", "spot"]

    async def _candidate_sources(self, symbol: str, market_kind: str) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        preferred = await self._best_source_get(symbol, market_kind)
        if isinstance(preferred, dict):
            ex = str(preferred.get("exchange", "")).lower()
            inst = str(preferred.get("instrument_id", ""))
            if ex in self.adapters and inst:
                out.append((ex, inst))

        for ex in self.priority:
            inst = await self._resolve_instrument(ex, symbol, market_kind)
            if not inst:
                continue
            if (ex, inst) not in out:
                out.append((ex, inst))
        return out

    async def resolve_symbol_market(self, symbol: str, market_kind: str = "spot") -> dict | None:
        base = self._normalize(symbol)
        candidates = await self._candidate_sources(base, market_kind)
        if not candidates:
            return None
        ex, inst = candidates[0]
        return {"symbol": base, "exchange": ex, "market_kind": market_kind, "instrument_id": inst}

    async def get_price(self, symbol: str) -> dict:
        base = self._normalize(symbol)
        first_exchange = self.priority[0] if self.priority else None
        errors: list[str] = []
        for kind in self._kinds_for_market_data():
            for ex, inst in await self._candidate_sources(base, kind):
                cache_key = f"quote:{ex}:{kind}:{inst}"
                cached = await self.cache.get_json(cache_key)
                if cached and float(cached.get("price", 0) or 0) > 0:
                    fallback_from = first_exchange if first_exchange and ex != first_exchange else None
                    line = self._source_line(
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        updated_at=str(cached.get("ts")),
                        fallback_from=fallback_from,
                    )
                    return PriceQuote(
                        symbol=base,
                        price=float(cached["price"]),
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        source=f"{ex}_{kind}",
                        source_line=line,
                        updated_at=str(cached.get("ts")),
                        fallback_from=fallback_from,
                    ).to_dict()

                try:
                    await self._acquire_exchange_token(ex)
                    payload = await self.adapters[ex].get_price(inst, market_kind=kind)
                    price = float(payload["price"])
                    if price <= 0:
                        raise RuntimeError("non_positive_price")
                    updated_at = str(payload.get("ts") or datetime.now(timezone.utc).isoformat())
                    await self.cache.set_json(cache_key, {"price": price, "ts": updated_at}, ttl=self.price_ttl)
                    await self._best_source_set(base, kind, ex, inst)
                    fallback_from = first_exchange if first_exchange and ex != first_exchange else None
                    line = self._source_line(
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        updated_at=updated_at,
                        fallback_from=fallback_from,
                    )
                    return PriceQuote(
                        symbol=base,
                        price=price,
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        source=f"{ex}_{kind}",
                        source_line=line,
                        updated_at=updated_at,
                        fallback_from=fallback_from,
                    ).to_dict()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{ex}:{kind}:{exc}")
                    continue
        raise RuntimeError(f"Price unavailable for {base}; tried {', '.join(errors[:6])}")

    async def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> dict:
        base = self._normalize(symbol)
        first_exchange = self.priority[0] if self.priority else None
        errors: list[str] = []
        tf = (timeframe or "1h").strip().lower()
        lim = max(20, min(int(limit), 1000))

        for kind in self._kinds_for_market_data():
            for ex, inst in await self._candidate_sources(base, kind):
                cache_key = f"ohlcv:{ex}:{kind}:{inst}:{tf}:{lim}"
                cached = await self.cache.get_json(cache_key)
                if isinstance(cached, dict) and isinstance(cached.get("candles"), list):
                    candles = cached.get("candles", [])
                    if candles_are_sane(candles):
                        updated_at = str(cached.get("updated_at") or datetime.now(timezone.utc).isoformat())
                        fallback_from = first_exchange if first_exchange and ex != first_exchange else None
                        line = self._source_line(
                            exchange=ex,
                            market_kind=kind,
                            instrument_id=inst,
                            updated_at=updated_at,
                            fallback_from=fallback_from,
                        )
                        return OhlcvSeries(
                            symbol=base,
                            timeframe=tf,
                            candles=candles,
                            exchange=ex,
                            market_kind=kind,
                            instrument_id=inst,
                            source=f"{ex}_{kind}",
                            source_line=line,
                            updated_at=updated_at,
                            fallback_from=fallback_from,
                        ).to_dict()

                try:
                    await self._acquire_exchange_token(ex)
                    candles = await self.adapters[ex].get_ohlcv(inst, tf, lim, market_kind=kind)
                    if not candles_are_sane(candles):
                        raise RuntimeError("invalid_candles")
                    updated_at = datetime.now(timezone.utc).isoformat()
                    await self.cache.set_json(
                        cache_key,
                        {"candles": candles, "updated_at": updated_at},
                        ttl=self.ohlcv_ttl,
                    )
                    await self._best_source_set(base, kind, ex, inst)
                    fallback_from = first_exchange if first_exchange and ex != first_exchange else None
                    line = self._source_line(
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        updated_at=updated_at,
                        fallback_from=fallback_from,
                    )
                    return OhlcvSeries(
                        symbol=base,
                        timeframe=tf,
                        candles=candles,
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        source=f"{ex}_{kind}",
                        source_line=line,
                        updated_at=updated_at,
                        fallback_from=fallback_from,
                    ).to_dict()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{ex}:{kind}:{exc}")
                    continue
        raise RuntimeError(f"OHLCV unavailable for {base} {tf}; tried {', '.join(errors[:6])}")

    async def get_orderbook(self, symbol: str, depth: int = 50) -> dict:
        base = self._normalize(symbol)
        first_exchange = self.priority[0] if self.priority else None
        errors: list[str] = []
        depth = max(5, min(int(depth), 500))
        for kind in self._kinds_for_market_data():
            for ex, inst in await self._candidate_sources(base, kind):
                cache_key = f"orderbook:{ex}:{kind}:{inst}:{depth}"
                cached = await self.cache.get_json(cache_key)
                if isinstance(cached, dict) and cached.get("bids") and cached.get("asks"):
                    updated_at = str(cached.get("updated_at") or datetime.now(timezone.utc).isoformat())
                    fallback_from = first_exchange if first_exchange and ex != first_exchange else None
                    line = self._source_line(
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        updated_at=updated_at,
                        fallback_from=fallback_from,
                    )
                    return OrderBook(
                        symbol=base,
                        bids=list(cached.get("bids", [])),
                        asks=list(cached.get("asks", [])),
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        source=f"{ex}_{kind}",
                        source_line=line,
                        updated_at=updated_at,
                        fallback_from=fallback_from,
                    ).to_dict()

                try:
                    await self._acquire_exchange_token(ex)
                    payload = await self.adapters[ex].get_orderbook(inst, depth=depth, market_kind=kind)
                    bids = payload.get("bids", [])
                    asks = payload.get("asks", [])
                    if not bids or not asks:
                        raise RuntimeError("empty_orderbook")
                    updated_at = str(payload.get("ts") or datetime.now(timezone.utc).isoformat())
                    await self.cache.set_json(
                        cache_key,
                        {"bids": bids, "asks": asks, "updated_at": updated_at},
                        ttl=self.orderbook_ttl,
                    )
                    await self._best_source_set(base, kind, ex, inst)
                    fallback_from = first_exchange if first_exchange and ex != first_exchange else None
                    line = self._source_line(
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        updated_at=updated_at,
                        fallback_from=fallback_from,
                    )
                    return OrderBook(
                        symbol=base,
                        bids=bids,
                        asks=asks,
                        exchange=ex,
                        market_kind=kind,
                        instrument_id=inst,
                        source=f"{ex}_{kind}",
                        source_line=line,
                        updated_at=updated_at,
                        fallback_from=fallback_from,
                    ).to_dict()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{ex}:{kind}:{exc}")
                    continue
        raise RuntimeError(f"Orderbook unavailable for {base}; tried {', '.join(errors[:6])}")

    async def get_funding_oi(self, symbol: str) -> dict | None:
        base = self._normalize(symbol)
        first_exchange = self.priority[0] if self.priority else None
        for ex, inst in await self._candidate_sources(base, "perp"):
            try:
                await self._acquire_exchange_token(ex)
                payload = await self.adapters[ex].get_funding_oi(inst)
                if not payload:
                    continue
                updated_at = str(payload.get("ts") or datetime.now(timezone.utc).isoformat())
                await self._best_source_set(base, "perp", ex, inst)
                fallback_from = first_exchange if first_exchange and ex != first_exchange else None
                line = self._source_line(
                    exchange=ex,
                    market_kind="perp",
                    instrument_id=inst,
                    updated_at=updated_at,
                    fallback_from=fallback_from,
                )
                return FundingOI(
                    symbol=base,
                    exchange=ex,
                    market_kind="perp",
                    instrument_id=inst,
                    funding_rate=(
                        float(payload.get("funding_rate")) if payload.get("funding_rate") is not None else None
                    ),
                    open_interest=(
                        float(payload.get("open_interest")) if payload.get("open_interest") is not None else None
                    ),
                    source=f"{ex}_perp",
                    source_line=line,
                    updated_at=updated_at,
                    fallback_from=fallback_from,
                ).to_dict()
            except Exception:  # noqa: BLE001
                continue
        return None

    async def get_market_universe(self, market_kind: str = "spot") -> set[str]:
        out: set[str] = set()
        for ex in self.priority:
            try:
                markets = await self._markets(ex, market_kind)
            except Exception:  # noqa: BLE001
                continue
            out.update(markets.keys())
        return out

    async def get_spot_movers(self) -> list[dict]:
        for ex in self.priority:
            adapter = self.adapters[ex]
            if not hasattr(adapter, "get_spot_tickers_24h"):
                continue
            try:
                rows = await adapter.get_spot_tickers_24h()
            except Exception:  # noqa: BLE001
                continue
            if rows:
                for row in rows:
                    row["exchange"] = ex
                    row["market_kind"] = "spot"
                return rows
        return []
