from __future__ import annotations

import asyncio
from statistics import mean

import pandas as pd

from app.adapters.llm import LLMClient
from app.core.cache import RedisCache
from app.core.ta import ema, rsi
from app.services.market_analysis import MarketAnalysisService
from app.services.market_context import format_market_context


class BroadcastService:
    """
    Runs every 15 minutes. Checks for notable market conditions:
    - BTC/ETH/SOL hitting key EMA levels (20, 50, 100, 200)
    - RSI extremes (>75 or <30) on 1h or 4h for majors
    - Volume spikes (>2x average)
    - Significant price moves (>3% in 1h)

    If any condition triggers, generate a Ghost-style market commentary
    and send to configured broadcast channel(s).

    Rate limit: max 1 broadcast per hour to avoid spam.
    Uses cache key `last_broadcast` with 60min TTL.
    """

    def __init__(
        self,
        *,
        analysis_service: MarketAnalysisService,
        llm_client: LLMClient | None,
        cache: RedisCache,
        symbols: list[str] | None = None,
        rate_limit_minutes: int = 60,
    ) -> None:
        self.analysis_service = analysis_service
        self.llm_client = llm_client
        self.cache = cache
        self.symbols = [s.upper() for s in (symbols or ["BTC", "ETH", "SOL"])]
        self.rate_limit_seconds = max(900, int(rate_limit_minutes) * 60)

    async def _symbol_events(self, symbol: str) -> list[str]:
        price_task = self.analysis_service.price_adapter.get_price(symbol)
        candles_1h_task = self.analysis_service.ohlcv_adapter.get_ohlcv(symbol, "1h", 260)
        candles_4h_task = self.analysis_service.ohlcv_adapter.get_ohlcv(symbol, "4h", 260)
        price, candles_1h, candles_4h = await asyncio.gather(price_task, candles_1h_task, candles_4h_task)

        out: list[str] = []
        current = float(price.get("price") or 0.0)
        if current <= 0:
            return out

        df_1h = pd.DataFrame(candles_1h or [])
        df_4h = pd.DataFrame(candles_4h or [])
        if df_1h.empty:
            return out

        close_1h = df_1h["close"].astype(float)
        vol_1h = df_1h["volume"].astype(float) if "volume" in df_1h else pd.Series(dtype=float)
        close_4h = df_4h["close"].astype(float) if not df_4h.empty and "close" in df_4h else pd.Series(dtype=float)

        if len(close_1h) >= 2:
            move_1h = ((float(close_1h.iloc[-1]) / float(close_1h.iloc[-2])) - 1.0) * 100.0
            if abs(move_1h) >= 3.0:
                out.append(f"{symbol} moved {move_1h:+.2f}% in the last hour")

        if len(vol_1h) >= 22:
            baseline = mean(float(v) for v in vol_1h.iloc[-21:-1])
            if baseline > 0:
                ratio = float(vol_1h.iloc[-1]) / baseline
                if ratio >= 2.0:
                    out.append(f"{symbol} volume is spiking at {ratio:.2f}x the 1h baseline")

        for period in (20, 50, 100, 200):
            ema_series = ema(close_1h, period).dropna()
            if ema_series.empty:
                continue
            ema_val = float(ema_series.iloc[-1])
            if abs(current - ema_val) / current <= 0.0025:
                out.append(f"{symbol} is testing the 1h EMA{period} around ${ema_val:,.2f}")

        rsi_1h_series = rsi(close_1h, 14).dropna()
        if not rsi_1h_series.empty:
            r1 = float(rsi_1h_series.iloc[-1])
            if r1 >= 75 or r1 <= 30:
                out.append(f"{symbol} RSI14 on 1h is at {r1:.1f}")

        rsi_4h_series = rsi(close_4h, 14).dropna() if not close_4h.empty else pd.Series(dtype=float)
        if not rsi_4h_series.empty:
            r4 = float(rsi_4h_series.iloc[-1])
            if r4 >= 75 or r4 <= 30:
                out.append(f"{symbol} RSI14 on 4h is at {r4:.1f}")

        return out

    async def _find_triggers(self) -> list[str]:
        tasks = [self._symbol_events(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        triggers: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            triggers.extend(result)
        return triggers[:8]

    async def _build_commentary(self, triggers: list[str], market_context: dict) -> str:
        if self.llm_client:
            prompt = (
                "Turn these market triggers into one short proactive broadcast.\n"
                "Keep it 2-4 lines, trader-chat style, direct, no fluff.\n"
                f"Triggers: {triggers}\n"
                f"Market context: {format_market_context(market_context)}"
            )
            try:
                text = await self.llm_client.reply(prompt, max_output_tokens=220, temperature=0.6)
            except Exception:  # noqa: BLE001
                text = ""
            cleaned = (text or "").strip()
            if cleaned:
                return cleaned
        return (
            "market heads-up:\n"
            + "\n".join(f"- {item}" for item in triggers[:4])
            + "\nwatch for fakeouts if BTC loses momentum."
        )

    async def check_and_broadcast(self, bot, channel_ids: list[int]):
        if not channel_ids:
            return 0

        triggers = await self._find_triggers()
        if not triggers:
            return 0

        if not await self.cache.set_if_absent("last_broadcast", ttl=self.rate_limit_seconds):
            return 0

        try:
            market_context = await self.analysis_service.get_market_context()
        except Exception:  # noqa: BLE001
            market_context = {}
        commentary = await self._build_commentary(triggers, market_context if isinstance(market_context, dict) else {})

        sent = 0
        for chat_id in channel_ids:
            try:
                await bot.send_message(chat_id=chat_id, text=commentary)
                sent += 1
            except Exception:  # noqa: BLE001
                continue
        return sent
