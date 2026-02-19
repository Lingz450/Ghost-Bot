from __future__ import annotations

import pytest

from app.services.ema_scanner import EMAScannerService


class _DummyCache:
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    async def get_json(self, key: str):
        return self._store.get(key)

    async def set_json(self, key: str, value, ttl: int = 0) -> None:  # noqa: ARG002
        self._store[key] = value


@pytest.mark.asyncio
async def test_ema_scan_output_shape_from_precomputed() -> None:
    service = EMAScannerService(
        http=None,  # type: ignore[arg-type]
        cache=_DummyCache(),  # type: ignore[arg-type]
        ohlcv_adapter=None,  # type: ignore[arg-type]
        binance_base="https://api.binance.com",
        db_factory=None,
    )

    async def _fake_precomputed(timeframe: str, ema_length: int, mode: str, limit: int):
        assert timeframe == "4h"
        assert ema_length == 200
        assert mode == "closest"
        assert limit == 10
        return [
            {"symbol": "BTC", "price": 67000.0, "ema": 66800.0, "distance_pct": 0.299, "side": "above"},
            {"symbol": "ETH", "price": 3400.0, "ema": 3410.0, "distance_pct": -0.293, "side": "below"},
        ]

    async def _fake_scan_live(timeframe: str, ema_length: int, mode: str, limit: int):  # noqa: ARG001
        return []

    service._query_precomputed = _fake_precomputed  # type: ignore[method-assign]
    service._scan_live = _fake_scan_live  # type: ignore[method-assign]

    payload = await service.scan(timeframe="4h", ema_length=200, mode="closest", limit=10)
    assert payload["timeframe"] == "4h"
    assert payload["ema_length"] == 200
    assert payload["mode"] == "closest"
    assert isinstance(payload["items"], list)
    assert payload["items"][0]["symbol"] == "BTC"
    assert "updated_at" in payload
