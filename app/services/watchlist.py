from __future__ import annotations

from datetime import datetime

from app.adapters.market_router import MarketDataRouter
from app.adapters.news_sources import NewsSourcesAdapter
from app.core.http import ResilientHTTPClient


class WatchlistService:
    def __init__(
        self,
        http: ResilientHTTPClient,
        news_adapter: NewsSourcesAdapter,
        market_router: MarketDataRouter,
        coingecko_base: str,
        include_btc_eth: bool = True,
    ) -> None:
        self.http = http
        self.news_adapter = news_adapter
        self.market_router = market_router
        self.coingecko_base = coingecko_base
        self.include_btc_eth = include_btc_eth

    async def _top_movers(self, direction: str | None = None) -> list[dict]:
        filtered = []
        try:
            rows = await self.market_router.get_spot_movers()
            for row in rows:
                sym = str(row.get("symbol", "")).upper()
                quote_volume = float(row.get("quote_volume", 0) or 0)
                if quote_volume < 5_000_000:
                    continue
                filtered.append(
                    {
                        "symbol": sym,
                        "change": float(row.get("change", 0) or 0),
                        "volume": quote_volume,
                        "exchange": row.get("exchange"),
                        "instrument_id": row.get("instrument_id"),
                    }
                )
            if direction == "short":
                filtered.sort(key=lambda x: x["change"], reverse=True)
            elif direction == "long":
                filtered.sort(key=lambda x: x["change"])
            else:
                filtered.sort(key=lambda x: abs(x["change"]), reverse=True)
            if filtered:
                return filtered
        except Exception:  # noqa: BLE001
            pass

        try:
            trending = await self.http.get_json(f"{self.coingecko_base}/search/trending")
            for coin in trending.get("coins", []):
                item = coin.get("item", {})
                filtered.append(
                    {
                        "symbol": item.get("symbol", "").upper(),
                        "change": 0.0,
                        "volume": 0.0,
                    }
                )
        except Exception:  # noqa: BLE001
            pass

        return filtered

    async def build_watchlist(self, count: int = 5, direction: str | None = None) -> dict:
        direction = (direction or "").strip().lower() or None
        movers = await self._top_movers(direction=direction)
        news = await self.news_adapter.get_latest_news(limit=15)

        picked: list[dict] = []
        if self.include_btc_eth:
            for major in ("BTC", "ETH"):
                if len(picked) >= count:
                    break
                picked.append({"symbol": major, "change": 0.0, "volume": 0.0})

        for row in movers:
            if len(picked) >= count:
                break
            if row["symbol"] in {p["symbol"] for p in picked}:
                continue
            picked.append(row)

        items = []
        for row in picked[:count]:
            symbol = row["symbol"]
            catalyst = "volume expansion + momentum rotation"
            if direction == "short":
                catalyst = "overextended move, watch for fade setup"
            elif direction == "long":
                catalyst = "pullback candidate, watch reclaim confirmation"
            for story in news:
                if symbol in story["title"].upper():
                    catalyst = story["title"]
                    break
            items.append(f"{symbol} ({row.get('change', 0.0):+.2f}% 24h) - {catalyst}")

        if direction == "short":
            themes = [
                "Short-side watchlist: focusing on stretched movers with mean-reversion risk.",
                "Wait for rejection confirmation before entry.",
            ]
        elif direction == "long":
            themes = [
                "Long-side watchlist: focusing on weak names near reclaim zones.",
                "Prioritize confirmation over blind dip buys.",
            ]
        else:
            themes = [
                "High-beta majors are setting tone for alts.",
                "Watch volume-confirmed breakouts, not random spikes.",
            ]

        return {
            "summary": " ".join(themes),
            "items": items,
            "direction": direction,
            "source_line": "Data source: multi-exchange spot router | Updated: just now",
            "updated_at": datetime.utcnow().isoformat(),
        }
