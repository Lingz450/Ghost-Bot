from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.market_router import MarketDataRouter
from app.adapters.symbols import coingecko_id_for, normalize_symbol
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


BINANCE_SUPPORTED_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}

COINGECKO_FALLBACK_INTERVALS = {"15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"}


class OHLCVAdapter:
    def __init__(
        self,
        http: ResilientHTTPClient,
        cache: RedisCache,
        binance_base: str,
        coingecko_base: str,
        market_router: MarketDataRouter | None = None,
    ) -> None:
        self.http = http
        self.cache = cache
        self.binance_base = binance_base
        self.coingecko_base = coingecko_base
        self.market_router = market_router

    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> list[dict]:
        meta = normalize_symbol(symbol)
        tf_raw = timeframe.strip() or "1h"
        tf = tf_raw if tf_raw.endswith("M") else tf_raw.lower()
        if tf not in BINANCE_SUPPORTED_INTERVALS:
            raise RuntimeError(f"Timeframe `{timeframe}` isn't supported on my current data source.")
        key = f"ohlcv:{meta.base}:{tf}:{limit}"
        cached = await self.cache.get_json(key)
        if cached:
            return cached

        if self.market_router:
            try:
                routed = await self.market_router.get_ohlcv(meta.base, timeframe=tf, limit=limit)
                candles = list(routed.get("candles", []))
                updated = routed.get("updated_at") or datetime.now(timezone.utc).isoformat()
                source_line = routed.get("source_line")
                exchange = routed.get("exchange")
                market_kind = routed.get("market_kind")
                instrument_id = routed.get("instrument_id")
                for row in candles:
                    row["fetched_at"] = updated
                    row["source_line"] = source_line
                    row["exchange"] = exchange
                    row["market_kind"] = market_kind
                    row["instrument_id"] = instrument_id
                await self.cache.set_json(key, candles, ttl=60)
                return candles
            except Exception:  # noqa: BLE001
                pass

        try:
            rows = await self.http.get_json(
                f"{self.binance_base}/api/v3/klines",
                params={"symbol": meta.pair, "interval": tf, "limit": limit},
            )
            candles = [
                {
                    "ts": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": float(r[5]),
                    "source": "binance_spot",
                    "exchange": "binance",
                    "market_kind": "spot",
                    "instrument_id": meta.pair,
                    "source_line": f"Data source: Binance Spot ({meta.pair}) | Updated: 0s ago",
                }
                for r in rows
            ]
            await self.cache.set_json(key, candles, ttl=60)
            return candles
        except Exception:  # noqa: BLE001
            pass

        if tf not in COINGECKO_FALLBACK_INTERVALS:
            raise RuntimeError(f"Timeframe `{timeframe}` is Binance-only and currently unavailable.")

        cg_id = coingecko_id_for(meta.base)
        if not cg_id:
            search = await self.http.get_json(f"{self.coingecko_base}/search", params={"query": meta.base})
            for item in search.get("coins", []):
                if item.get("symbol", "").upper() == meta.base:
                    cg_id = item.get("id")
                    break
        if not cg_id:
            raise RuntimeError(f"OHLCV unavailable for {meta.base}")

        days = "7" if tf in {"15m", "30m", "1h", "2h", "4h", "6h", "12h"} else "30"
        rows = await self.http.get_json(
            f"{self.coingecko_base}/coins/{cg_id}/ohlc",
            params={"vs_currency": "usd", "days": days},
        )

        candles = [
            {
                "ts": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": 0.0,
                "source": "coingecko",
                "exchange": "coingecko",
                "market_kind": "spot",
                "instrument_id": cg_id,
                "source_line": f"Data source: CoinGecko Spot ({cg_id}) | Updated: 0s ago",
            }
            for r in rows[-limit:]
        ]

        if not candles:
            raise RuntimeError(f"OHLCV unavailable for {meta.base}")

        for c in candles:
            c["fetched_at"] = datetime.now(timezone.utc).isoformat()

        await self.cache.set_json(key, candles, ttl=90)
        return candles
