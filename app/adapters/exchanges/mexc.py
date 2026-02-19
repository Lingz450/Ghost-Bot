from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.exchanges.utils import resolve_timeframe, resample_ohlcv
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class MEXCExchangeAdapter:
    name = "mexc"
    label = "MEXC"
    SPOT_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"}

    def __init__(self, http: ResilientHTTPClient, cache: RedisCache, base_url: str, instruments_ttl_sec: int = 2700) -> None:
        self.http = http
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.instruments_ttl_sec = max(300, int(instruments_ttl_sec))

    async def list_spot_markets(self) -> dict[str, str]:
        key = "ex:mexc:spot:markets"
        cached = await self.cache.get_json(key)
        if isinstance(cached, dict) and cached:
            return {str(k): str(v) for k, v in cached.items()}
        data = await self.http.get_json(f"{self.base_url}/api/v3/exchangeInfo")
        out: dict[str, str] = {}
        for row in data.get("symbols", []):
            if row.get("quoteAsset") != "USDT" or str(row.get("status", "")).upper() != "ENABLED":
                continue
            base = str(row.get("baseAsset", "")).upper()
            symbol = str(row.get("symbol", "")).upper()
            if base and symbol:
                out[base] = symbol
        await self.cache.set_json(key, out, ttl=self.instruments_ttl_sec)
        return out

    async def list_perp_markets(self) -> dict[str, str]:
        return {}

    async def get_price(self, instrument_id: str, market_kind: str = "spot") -> dict:
        if market_kind != "spot":
            raise RuntimeError("MEXC perp unavailable")
        data = await self.http.get_json(f"{self.base_url}/api/v3/ticker/price", params={"symbol": instrument_id})
        return {"price": float(data["price"]), "ts": datetime.now(timezone.utc).isoformat()}

    async def get_ohlcv(self, instrument_id: str, timeframe: str, limit: int, market_kind: str = "spot") -> list[dict]:
        if market_kind != "spot":
            raise RuntimeError("MEXC perp unavailable")
        resolution = resolve_timeframe(timeframe, self.SPOT_TIMEFRAMES)
        if not resolution:
            raise RuntimeError(f"Unsupported timeframe `{timeframe}` for MEXC spot.")
        fetch_limit = int(limit)
        if resolution.needs_resample:
            fetch_limit = min(max(limit * 6, 200), 1000)
        rows = await self.http.get_json(
            f"{self.base_url}/api/v3/klines",
            params={"symbol": instrument_id, "interval": resolution.fetch_tf, "limit": fetch_limit},
        )
        candles = [
            {
                "ts": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "source": "mexc_spot",
            }
            for r in rows
        ]
        if resolution.needs_resample:
            candles = resample_ohlcv(
                candles,
                source_tf=resolution.fetch_tf,
                target_tf=resolution.request_tf,
                limit=limit,
            )
        return candles[-limit:]

    async def get_orderbook(self, instrument_id: str, depth: int = 50, market_kind: str = "spot") -> dict:
        if market_kind != "spot":
            raise RuntimeError("MEXC perp unavailable")
        depth = max(5, min(int(depth), 1000))
        data = await self.http.get_json(f"{self.base_url}/api/v3/depth", params={"symbol": instrument_id, "limit": depth})
        bids = [[float(p), float(q)] for p, q in data.get("bids", [])]
        asks = [[float(p), float(q)] for p, q in data.get("asks", [])]
        return {"bids": bids, "asks": asks, "ts": datetime.now(timezone.utc).isoformat()}

    async def get_funding_oi(self, instrument_id: str) -> dict | None:
        return None

    async def get_spot_tickers_24h(self) -> list[dict]:
        rows = await self.http.get_json(f"{self.base_url}/api/v3/ticker/24hr")
        out: list[dict] = []
        for row in rows:
            symbol = str(row.get("symbol", ""))
            if not symbol.endswith("USDT"):
                continue
            try:
                out.append(
                    {
                        "symbol": symbol[:-4].upper(),
                        "instrument_id": symbol.upper(),
                        "change": float(row.get("priceChangePercent", 0) or 0),
                        "quote_volume": float(row.get("quoteVolume", 0) or 0),
                        "price": float(row.get("lastPrice", 0) or 0),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return out
