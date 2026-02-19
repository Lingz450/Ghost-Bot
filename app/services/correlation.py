from __future__ import annotations

import pandas as pd

from app.adapters.ohlcv import OHLCVAdapter
from app.adapters.prices import PriceAdapter


class CorrelationService:
    def __init__(self, ohlcv_adapter: OHLCVAdapter, price_adapter: PriceAdapter) -> None:
        self.ohlcv_adapter = ohlcv_adapter
        self.price_adapter = price_adapter

    async def check_following(self, symbol: str, benchmark: str = "BTC", periods: int = 48) -> dict:
        symbol = symbol.upper()
        benchmark = benchmark.upper()

        a = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe="1h", limit=periods + 10)
        b = await self.ohlcv_adapter.get_ohlcv(benchmark, timeframe="1h", limit=periods + 10)

        dfa = pd.DataFrame(a).tail(periods)
        dfb = pd.DataFrame(b).tail(periods)

        ret_a = dfa["close"].pct_change().dropna()
        ret_b = dfb["close"].pct_change().dropna()

        min_len = min(len(ret_a), len(ret_b))
        ret_a = ret_a.tail(min_len)
        ret_b = ret_b.tail(min_len)

        corr = float(ret_a.corr(ret_b)) if min_len > 3 else 0.0
        var_b = float(ret_b.var()) if min_len > 3 else 0.0
        beta = float(ret_a.cov(ret_b) / var_b) if var_b > 0 else 0.0

        perf_a = float((dfa["close"].iloc[-1] / dfa["close"].iloc[0] - 1) * 100)
        perf_b = float((dfb["close"].iloc[-1] / dfb["close"].iloc[0] - 1) * 100)

        verdict = "Yes, mostly tracking BTC." if corr >= 0.65 else "Not tightly tracking BTC right now."
        source_line = ""
        if a:
            source_line = str(a[-1].get("source_line") or "")

        return {
            "summary": f"{verdict} Corr(48x1h)={corr:.2f}, beta={beta:.2f}.",
            "bullets": [
                f"{symbol} performance (48h approx): {perf_a:.2f}%",
                f"{benchmark} performance (48h approx): {perf_b:.2f}%",
                "High beta means bigger swings than BTC in both directions.",
            ],
            "corr": corr,
            "beta": beta,
            "source_line": source_line,
        }
