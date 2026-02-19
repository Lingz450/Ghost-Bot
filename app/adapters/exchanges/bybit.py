from __future__ import annotations

from datetime import datetime, timezone

from app.adapters.exchanges.utils import resolve_timeframe, resample_ohlcv
from app.core.cache import RedisCache
from app.core.http import ResilientHTTPClient


class BybitExchangeAdapter:
    name = "bybit"
    label = "Bybit"

    _TF_MAP = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
        "1w": "W",
    }
    SPOT_TIMEFRAMES = set(_TF_MAP.keys())
    PERP_TIMEFRAMES = SPOT_TIMEFRAMES.copy()

    def __init__(self, http: ResilientHTTPClient, cache: RedisCache, base_url: str, instruments_ttl_sec: int = 2700) -> None:
        self.http = http
        self.cache = cache
        self.base_url = base_url.rstrip("/")
        self.instruments_ttl_sec = max(300, int(instruments_ttl_sec))

    async def _paged_instruments(self, category: str) -> list[dict]:
        cursor = ""
        out: list[dict] = []
        for _ in range(5):
            params = {"category": category, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            payload = await self.http.get_json(f"{self.base_url}/v5/market/instruments-info", params=params)
            result = payload.get("result", {})
            out.extend(result.get("list", []) or [])
            cursor = str(result.get("nextPageCursor") or "").strip()
            if not cursor:
                break
        return out

    async def list_spot_markets(self) -> dict[str, str]:
        key = "ex:bybit:spot:markets"
        cached = await self.cache.get_json(key)
        if isinstance(cached, dict) and cached:
            return {str(k): str(v) for k, v in cached.items()}
        rows = await self._paged_instruments("spot")
        out: dict[str, str] = {}
        for row in rows:
            symbol = str(row.get("symbol", "")).upper()
            status = str(row.get("status", "")).upper()
            if not symbol.endswith("USDT"):
                continue
            if status and status not in {"TRADING", "ONLINE"}:
                continue
            out[symbol[:-4]] = symbol
        await self.cache.set_json(key, out, ttl=self.instruments_ttl_sec)
        return out

    async def list_perp_markets(self) -> dict[str, str]:
        key = "ex:bybit:perp:markets"
        cached = await self.cache.get_json(key)
        if isinstance(cached, dict) and cached:
            return {str(k): str(v) for k, v in cached.items()}
        rows = await self._paged_instruments("linear")
        out: dict[str, str] = {}
        for row in rows:
            symbol = str(row.get("symbol", "")).upper()
            status = str(row.get("status", "")).upper()
            if not symbol.endswith("USDT"):
                continue
            if status and status not in {"TRADING", "ONLINE"}:
                continue
            out[symbol[:-4]] = symbol
        await self.cache.set_json(key, out, ttl=self.instruments_ttl_sec)
        return out

    async def get_price(self, instrument_id: str, market_kind: str = "spot") -> dict:
        category = "spot" if market_kind == "spot" else "linear"
        payload = await self.http.get_json(
            f"{self.base_url}/v5/market/tickers",
            params={"category": category, "symbol": instrument_id},
        )
        rows = payload.get("result", {}).get("list", []) or []
        if not rows:
            raise RuntimeError(f"Bybit ticker unavailable for {instrument_id}")
        row = rows[0]
        return {"price": float(row.get("lastPrice")), "ts": datetime.now(timezone.utc).isoformat()}

    async def get_ohlcv(self, instrument_id: str, timeframe: str, limit: int, market_kind: str = "spot") -> list[dict]:
        supported = self.SPOT_TIMEFRAMES if market_kind == "spot" else self.PERP_TIMEFRAMES
        resolution = resolve_timeframe(timeframe, supported)
        if not resolution:
            raise RuntimeError(f"Unsupported timeframe `{timeframe}` for Bybit {market_kind}.")

        fetch_limit = int(limit)
        if resolution.needs_resample:
            fetch_limit = min(max(limit * 6, 200), 1000)

        category = "spot" if market_kind == "spot" else "linear"
        interval = self._TF_MAP[resolution.fetch_tf]
        payload = await self.http.get_json(
            f"{self.base_url}/v5/market/kline",
            params={"category": category, "symbol": instrument_id, "interval": interval, "limit": fetch_limit},
        )
        rows = payload.get("result", {}).get("list", []) or []
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
        category = "spot" if market_kind == "spot" else "linear"
        payload = await self.http.get_json(
            f"{self.base_url}/v5/market/orderbook",
            params={"category": category, "symbol": instrument_id, "limit": max(1, min(int(depth), 200))},
        )
        result = payload.get("result", {})
        bids = [[float(p), float(q)] for p, q in result.get("b", [])]
        asks = [[float(p), float(q)] for p, q in result.get("a", [])]
        return {"bids": bids, "asks": asks, "ts": datetime.now(timezone.utc).isoformat()}

    async def get_funding_oi(self, instrument_id: str) -> dict | None:
        ticker = await self.http.get_json(
            f"{self.base_url}/v5/market/tickers",
            params={"category": "linear", "symbol": instrument_id},
        )
        rows = ticker.get("result", {}).get("list", []) or []
        if not rows:
            return None
        row = rows[0]
        open_interest = None
        oi_payload = await self.http.get_json(
            f"{self.base_url}/v5/market/open-interest",
            params={"category": "linear", "symbol": instrument_id, "intervalTime": "5min", "limit": 1},
        )
        oi_rows = oi_payload.get("result", {}).get("list", []) or []
        if oi_rows:
            try:
                open_interest = float(oi_rows[0].get("openInterest"))
            except Exception:  # noqa: BLE001
                open_interest = None
        return {
            "funding_rate": float(row.get("fundingRate")) if row.get("fundingRate") not in (None, "") else None,
            "open_interest": open_interest,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

    async def get_spot_tickers_24h(self) -> list[dict]:
        payload = await self.http.get_json(
            f"{self.base_url}/v5/market/tickers",
            params={"category": "spot"},
        )
        rows = payload.get("result", {}).get("list", []) or []
        out: list[dict] = []
        for row in rows:
            symbol = str(row.get("symbol", "")).upper()
            if not symbol.endswith("USDT"):
                continue
            try:
                out.append(
                    {
                        "symbol": symbol[:-4],
                        "instrument_id": symbol,
                        "change": float(row.get("price24hPcnt", 0) or 0) * 100.0,
                        "quote_volume": float(row.get("turnover24h", 0) or 0),
                        "price": float(row.get("lastPrice", 0) or 0),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        return out
