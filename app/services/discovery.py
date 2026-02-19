from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.market_router import MarketDataRouter
from app.adapters.prices import PriceAdapter
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class DiscoveryService:
    def __init__(
        self,
        http: ResilientHTTPClient,
        cache: RedisCache,
        market_router: MarketDataRouter,
        price_adapter: PriceAdapter,
        binance_base: str,
        coingecko_base: str,
    ) -> None:
        self.http = http
        self.cache = cache
        self.market_router = market_router
        self.price_adapter = price_adapter
        self.binance_base = binance_base
        self.coingecko_base = coingecko_base

    async def _binance_bases(self) -> set[str]:
        cache_key = "binance:bases"
        cached = await self.cache.get_json(cache_key)
        if cached:
            return {str(x) for x in cached}

        data = await self.http.get_json(f"{self.binance_base}/api/v3/exchangeInfo")
        bases: set[str] = set()
        for item in data.get("symbols", []):
            if item.get("quoteAsset") != "USDT":
                continue
            if item.get("status") != "TRADING":
                continue
            base = str(item.get("baseAsset", "")).upper()
            if base:
                bases.add(base)
        await self.cache.set_json(cache_key, sorted(bases), ttl=600)
        return bases

    async def find_pair(self, query: str, limit: int = 5) -> dict:
        raw = (query or "").strip()
        if not raw:
            raise RuntimeError("Missing symbol/name for pair discovery.")

        needle = raw.lstrip("$").strip()
        search = await self.http.get_json(f"{self.coingecko_base}/search", params={"query": needle})
        coins = search.get("coins", [])
        if not coins:
            return {
                "query": raw,
                "summary": "No close pair matches found.",
                "matches": [],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        bases = await self._binance_bases()
        out = []
        for coin in coins[: max(3, limit)]:
            symbol = str(coin.get("symbol", "")).upper()
            name = str(coin.get("name", "")).strip()
            if not symbol:
                continue
            tradable = symbol in bases
            price = None
            source = "coingecko_search"
            resolved = await self.market_router.resolve_symbol_market(symbol, "spot")
            try:
                price_payload = await self.price_adapter.get_price(symbol)
                price = float(price_payload["price"])
                source = price_payload.get("source", source)
            except Exception:  # noqa: BLE001
                pass
            out.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "tradable_binance": tradable,
                    "pair": (resolved.get("instrument_id") if resolved else (f"{symbol}USDT" if tradable else None)),
                    "exchange": resolved.get("exchange") if resolved else ("binance" if tradable else None),
                    "price": round(price, 8) if price is not None else None,
                    "source": source,
                }
            )

        return {
            "query": raw,
            "summary": "Closest symbol matches found.",
            "matches": out[:limit],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def guess_by_price(self, target_price: float, limit: int = 10) -> dict:
        if target_price <= 0:
            raise RuntimeError("Target price must be positive.")

        cache_key = "coingecko:markets:top500"
        markets = await self.cache.get_json(cache_key)
        if not markets:
            page1 = await self.http.get_json(
                f"{self.coingecko_base}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": 1,
                    "sparkline": "false",
                },
            )
            page2 = await self.http.get_json(
                f"{self.coingecko_base}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 250,
                    "page": 2,
                    "sparkline": "false",
                },
            )
            markets = list(page1) + list(page2)
            await self.cache.set_json(cache_key, markets, ttl=300)

        bases = await self._binance_bases()

        scored = []
        for row in markets:
            price = row.get("current_price")
            symbol = str(row.get("symbol", "")).upper()
            if price is None or not symbol:
                continue
            try:
                price_f = float(price)
            except Exception:  # noqa: BLE001
                continue
            diff_abs = abs(price_f - target_price)
            diff_rel = diff_abs / max(target_price, 1e-9)
            if diff_rel > 0.20 and diff_abs > 2.0:
                continue
            scored.append(
                {
                    "symbol": symbol,
                    "name": str(row.get("name", "")),
                    "price": round(price_f, 8),
                    "diff_abs": diff_abs,
                    "diff_rel": diff_rel,
                    "tradable_binance": symbol in bases,
                }
            )

        scored.sort(key=lambda x: (x["diff_abs"], x["diff_rel"]))
        matches = scored[: max(1, min(limit, 20))]

        return {
            "target_price": target_price,
            "summary": f"Closest coins near ${target_price:.6g}.",
            "matches": matches,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
