from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.exchanges.utils import resolve_timeframe, resample_ohlcv
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class OKXExchangeAdapter:
    name = "okx"
    label = "OKX"

    _TF_MAP = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1H",
        "2h": "2H",
        "4h": "4H",
        "6h": "6H",
        "12h": "12H",
        "1d": "1D",
        "3d": "3D",
        "1w": "1W",
    }
    SPOT_TIMEFRAMES = set(_TF_MAP.keys())
    PERP_TIMEFRAMES = SPOT_TIMEFRAMES.copy()

    def __init__(self, http: ResilientHTTPClient, cache: RedisCache, base_url: str, instruments_ttl_sec: int = 2700) -> None:
        self.http = http
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.instruments_ttl_sec = max(300, int(instruments_ttl_sec))

    async def list_spot_markets(self) -> dict[str, str]:
        key = "ex:okx:spot:markets"
        cached = await self.cache.get_json(key)
        if isinstance(cached, dict) and cached:
            return {str(k): str(v) for k, v in cached.items()}
        payload = await self.http.get_json(f"{self.base_url}/api/v5/public/instruments", params={"instType": "SPOT"})
        rows = payload.get("data", []) or []
        out: dict[str, str] = {}
        for row in rows:
            if str(row.get("quoteCcy", "")).upper() != "USDT":
                continue
            if str(row.get("state", "")).lower() not in {"live"}:
                continue
            base = str(row.get("baseCcy", "")).upper()
            inst = str(row.get("instId", ""))
            if base and inst:
                out[base] = inst
        await self.cache.set_json(key, out, ttl=self.instruments_ttl_sec)
        return out

    async def list_perp_markets(self) -> dict[str, str]:
        key = "ex:okx:perp:markets"
        cached = await self.cache.get_json(key)
        if isinstance(cached, dict) and cached:
            return {str(k): str(v) for k, v in cached.items()}
        payload = await self.http.get_json(f"{self.base_url}/api/v5/public/instruments", params={"instType": "SWAP"})
        rows = payload.get("data", []) or []
        out: dict[str, str] = {}
        for row in rows:
            if str(row.get("quoteCcy", "")).upper() != "USDT":
                continue
            if str(row.get("state", "")).lower() not in {"live"}:
                continue
            base = str(row.get("settleCcy", "")).upper() or str(row.get("baseCcy", "")).upper()
            inst = str(row.get("instId", ""))
            if base and inst:
                out[base] = inst
        await self.cache.set_json(key, out, ttl=self.instruments_ttl_sec)
        return out

    async def get_price(self, instrument_id: str, market_kind: str = "spot") -> dict:
        payload = await self.http.get_json(f"{self.base_url}/api/v5/market/ticker", params={"instId": instrument_id})
        rows = payload.get("data", []) or []
        if not rows:
            raise RuntimeError(f"OKX ticker unavailable for {instrument_id}")
        row = rows[0]
        return {"price": float(row.get("last")), "ts": datetime.now(timezone.utc).isoformat()}

    async def get_ohlcv(self, instrument_id: str, timeframe: str, limit: int, market_kind: str = "spot") -> list[dict]:
        supported = self.SPOT_TIMEFRAMES if market_kind == "spot" else self.PERP_TIMEFRAMES
        resolution = resolve_timeframe(timeframe, supported)
        if not resolution:
            raise RuntimeError(f"Unsupported timeframe `{timeframe}` for OKX {market_kind}.")

        fetch_limit = int(limit)
        if resolution.needs_resample:
            fetch_limit = min(max(limit * 6, 200), 1000)
        bar = self._TF_MAP[resolution.fetch_tf]
        payload = await self.http.get_json(
            f"{self.base_url}/api/v5/market/candles",
            params={"instId": instrument_id, "bar": bar, "limit": fetch_limit},
        )
        rows = payload.get("data", []) or []
        candles = [
            {
                "ts": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
                "source": f"{self.name}_{market_kind}",
            }
            for r in rows
        ]
        candles = sorted(candles, key=lambda x: x["ts"])
        if resolution.needs_resample:
            candles = resample_ohlcv(
                candles,
                source_tf=resolution.fetch_tf,
                target_tf=resolution.request_tf,
                limit=limit,
            )
        return candles[-limit:]

    async def get_orderbook(self, instrument_id: str, depth: int = 50, market_kind: str = "spot") -> dict:
        depth = max(1, min(int(depth), 400))
        payload = await self.http.get_json(
            f"{self.base_url}/api/v5/market/books",
            params={"instId": instrument_id, "sz": depth},
        )
        rows = payload.get("data", []) or []
        if not rows:
            raise RuntimeError(f"OKX orderbook unavailable for {instrument_id}")
        row = rows[0]
        bids = [[float(p), float(q)] for p, q, *_ in row.get("bids", [])]
        asks = [[float(p), float(q)] for p, q, *_ in row.get("asks", [])]
        return {"bids": bids, "asks": asks, "ts": datetime.now(timezone.utc).isoformat()}

    async def get_funding_oi(self, instrument_id: str) -> dict | None:
        funding_payload = await self.http.get_json(
            f"{self.base_url}/api/v5/public/funding-rate",
            params={"instId": instrument_id},
        )
        funding_rows = funding_payload.get("data", []) or []
        oi_payload = await self.http.get_json(
            f"{self.base_url}/api/v5/public/open-interest",
            params={"instType": "SWAP", "instId": instrument_id},
        )
        oi_rows = oi_payload.get("data", []) or []
        funding = None
        oi = None
        if funding_rows:
            try:
                funding = float(funding_rows[0].get("fundingRate"))
            except Exception:  # noqa: BLE001
                funding = None
        if oi_rows:
            try:
                oi = float(oi_rows[0].get("oi"))
            except Exception:  # noqa: BLE001
                oi = None
        return {"funding_rate": funding, "open_interest": oi, "ts": datetime.now(timezone.utc).isoformat()}

    async def get_spot_tickers_24h(self) -> list[dict]:
        payload = await self.http.get_json(f"{self.base_url}/api/v5/market/tickers", params={"instType": "SPOT"})
        rows = payload.get("data", []) or []
        out: list[dict] = []
        for row in rows:
            inst_id = str(row.get("instId", ""))
            if not inst_id.endswith("-USDT"):
                continue
            symbol = inst_id.replace("-USDT", "").upper()
            try:
                open24 = float(row.get("open24h", 0) or 0)
                last = float(row.get("last", 0) or 0)
                pct = ((last - open24) / open24 * 100.0) if open24 > 0 else 0.0
                out.append(
                    {
                        "symbol": symbol,
                        "instrument_id": inst_id,
                        "change": pct,
                        "quote_volume": float(row.get("volCcy24h", 0) or 0),
                        "price": last,
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return out
