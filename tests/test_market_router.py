from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.adapters.exchanges.utils import normalize_timeframe, resolve_timeframe
from app.adapters.market_router import MarketDataRouter
from app.adapters.symbols import normalize_symbol


class _DummyCache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    async def get_json(self, key: str):
        return self.store.get(key)

    async def set_json(self, key: str, value, ttl: int = 0) -> None:  # noqa: ARG002
        self.store[key] = value

    async def incr_with_expiry(self, key: str, ttl: int = 0) -> int:  # noqa: ARG002
        current = int(self.store.get(key, 0) or 0) + 1
        self.store[key] = current
        return current


@dataclass
class _FakeExchange:
    name: str
    spot_markets: dict[str, str]
    perp_markets: dict[str, str]
    fail_price: bool = False
    price_calls: int = 0

    async def list_spot_markets(self) -> dict[str, str]:
        return self.spot_markets

    async def list_perp_markets(self) -> dict[str, str]:
        return self.perp_markets

    async def get_price(self, instrument_id: str, market_kind: str = "spot") -> dict:
        self.price_calls += 1
        if self.fail_price:
            raise RuntimeError(f"{self.name} down")
        base_price = {"binance": 100.0, "bybit": 101.5, "okx": 102.0}.get(self.name, 99.0)
        return {"price": base_price, "ts": "2026-02-19T00:00:00+00:00"}

    async def get_ohlcv(self, instrument_id: str, timeframe: str, limit: int, market_kind: str = "spot"):
        return [
            {
                "ts": 1700000000000 + (i * 60_000),
                "open": 1.0 + i,
                "high": 1.1 + i,
                "low": 0.9 + i,
                "close": 1.0 + i,
                "volume": 10.0 + i,
                "source": f"{self.name}_{market_kind}",
            }
            for i in range(limit)
        ]

    async def get_orderbook(self, instrument_id: str, depth: int = 50, market_kind: str = "spot"):
        return {"bids": [[1.0, 10.0]], "asks": [[1.1, 12.0]], "ts": "2026-02-19T00:00:00+00:00"}

    async def get_funding_oi(self, instrument_id: str):
        if not self.perp_markets:
            return None
        return {"funding_rate": 0.0001, "open_interest": 1000.0, "ts": "2026-02-19T00:00:00+00:00"}


def _build_router(cache: _DummyCache) -> MarketDataRouter:
    router = MarketDataRouter(
        http=None,  # type: ignore[arg-type]
        cache=cache,  # type: ignore[arg-type]
        binance_base_url="https://api.binance.com",
        binance_futures_base_url="https://fapi.binance.com",
        bybit_base_url="https://api.bybit.com",
        okx_base_url="https://www.okx.com",
        mexc_base_url="https://api.mexc.com",
        blofin_base_url="https://openapi.blofin.com",
        enable_binance=False,
        enable_bybit=False,
        enable_okx=False,
        enable_mexc=False,
        enable_blofin=False,
        exchange_priority="binance,bybit,okx",
        market_prefer_spot=True,
    )
    return router


def test_symbol_normalization_variants() -> None:
    assert normalize_symbol("btc").base == "BTC"
    assert normalize_symbol("$eth").base == "ETH"
    assert normalize_symbol("SOLUSDT").base == "SOL"
    assert normalize_symbol("sol/usdt").base == "SOL"
    assert normalize_symbol("SOL-USDT").base == "SOL"


def test_timeframe_mapping_and_resample_resolution() -> None:
    assert normalize_timeframe("4hr") == "4h"
    assert normalize_timeframe("weekly") == "1w"
    resolution = resolve_timeframe("2h", {"1h", "4h", "1d"})
    assert resolution is not None
    assert resolution.fetch_tf == "1h"
    assert resolution.needs_resample is True


@pytest.mark.asyncio
async def test_router_fallback_binance_to_bybit_and_best_source_cache() -> None:
    cache = _DummyCache()
    router = _build_router(cache)

    binance = _FakeExchange(
        name="binance",
        spot_markets={"XION": "XIONUSDT"},
        perp_markets={},
        fail_price=True,
    )
    bybit = _FakeExchange(
        name="bybit",
        spot_markets={"XION": "XIONUSDT"},
        perp_markets={},
        fail_price=False,
    )

    router.priority = ["binance", "bybit"]
    router.adapters = {"binance": binance, "bybit": bybit}  # type: ignore[assignment]

    quote = await router.get_price("xion")
    assert quote["exchange"] == "bybit"
    assert quote["market_kind"] == "spot"
    assert "fallback from Binance" in quote["source_line"]

    best_key = "best_source:XION:spot"
    assert best_key in cache.store
    assert isinstance(cache.store[best_key], dict)
    assert cache.store[best_key]["exchange"] == "bybit"  # type: ignore[index]

    # Force a fresh quote path but keep best source cache.
    for key in list(cache.store.keys()):
        if str(key).startswith("quote:"):
            cache.store.pop(key, None)

    binance.fail_price = False
    binance.price_calls = 0
    bybit.price_calls = 0

    quote2 = await router.get_price("XION")
    assert quote2["exchange"] == "bybit"
    assert bybit.price_calls == 1
    assert binance.price_calls == 0
