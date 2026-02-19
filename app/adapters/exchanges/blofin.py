from __future__ import annotations


class BloFinExchangeAdapter:
    name = "blofin"
    label = "BloFin"

    async def list_spot_markets(self) -> dict[str, str]:
        return {}

    async def list_perp_markets(self) -> dict[str, str]:
        return {}

    async def get_price(self, instrument_id: str, market_kind: str = "spot") -> dict:
        raise RuntimeError("BloFin public market adapter is not enabled in this build.")

    async def get_ohlcv(self, instrument_id: str, timeframe: str, limit: int, market_kind: str = "spot") -> list[dict]:
        raise RuntimeError("BloFin public market adapter is not enabled in this build.")

    async def get_orderbook(self, instrument_id: str, depth: int = 50, market_kind: str = "spot") -> dict:
        raise RuntimeError("BloFin public market adapter is not enabled in this build.")

    async def get_funding_oi(self, instrument_id: str) -> dict | None:
        return None

    async def get_spot_tickers_24h(self) -> list[dict]:
        return []
