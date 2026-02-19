from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw
import pandas as pd

from app.adapters.ohlcv import OHLCVAdapter
from app.core.ta import ema


class ChartService:
    def __init__(self, ohlcv_adapter: OHLCVAdapter) -> None:
        self.ohlcv_adapter = ohlcv_adapter

    @staticmethod
    def _scale(value: float, vmin: float, vmax: float, y_top: int, y_bottom: int) -> int:
        if vmax <= vmin:
            return y_bottom
        ratio = (value - vmin) / (vmax - vmin)
        return int(y_bottom - ratio * (y_bottom - y_top))

    async def render_chart(
        self,
        symbol: str,
        timeframe: str = "1h",
        ema_periods: list[int] | None = None,
        limit: int = 120,
    ) -> tuple[bytes, dict]:
        periods = ema_periods or [20, 50, 200]
        candles = await self.ohlcv_adapter.get_ohlcv(symbol, timeframe=timeframe, limit=max(80, min(limit, 260)))
        df = pd.DataFrame(candles)
        if df.empty:
            raise RuntimeError("No candles available for chart rendering.")

        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        width, height = 1400, 900
        margin = 40
        price_top, price_bottom = 80, 620
        vol_top, vol_bottom = 680, 840
        bg = (12, 18, 28)
        fg = (220, 230, 240)
        up = (29, 185, 84)
        down = (255, 77, 79)
        grid = (40, 56, 78)

        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)

        # Grid and labels
        for i in range(6):
            y = int(price_top + (price_bottom - price_top) * i / 5)
            draw.line((margin, y, width - margin, y), fill=grid, width=1)
        draw.text((margin, 20), f"{symbol.upper()} {timeframe} Candles", fill=fg)
        draw.text((margin, vol_top - 24), "Volume", fill=fg)

        lows = df["low"].tolist()
        highs = df["high"].tolist()
        vmin, vmax = min(lows), max(highs)
        if vmax <= vmin:
            vmax = vmin + 1.0

        vol_max = max(df["volume"].tolist()) if len(df) else 1.0
        if vol_max <= 0:
            vol_max = 1.0

        n = len(df)
        x_start = margin + 20
        x_end = width - margin - 20
        step = max((x_end - x_start) / max(n - 1, 1), 2.5)
        candle_w = max(int(step * 0.6), 2)

        # Candles + volume
        for idx, row in enumerate(df.itertuples(index=False)):
            o = float(row.open)
            h = float(row.high)
            l = float(row.low)
            c = float(row.close)
            v = float(row.volume)
            x = int(x_start + idx * step)
            color = up if c >= o else down

            y_h = self._scale(h, vmin, vmax, price_top, price_bottom)
            y_l = self._scale(l, vmin, vmax, price_top, price_bottom)
            y_o = self._scale(o, vmin, vmax, price_top, price_bottom)
            y_c = self._scale(c, vmin, vmax, price_top, price_bottom)

            draw.line((x, y_h, x, y_l), fill=color, width=1)
            top = min(y_o, y_c)
            bottom = max(y_o, y_c)
            if bottom - top < 1:
                bottom = top + 1
            draw.rectangle((x - candle_w // 2, top, x + candle_w // 2, bottom), fill=color)

            vh = int((v / vol_max) * (vol_bottom - vol_top))
            draw.rectangle((x - candle_w // 2, vol_bottom - vh, x + candle_w // 2, vol_bottom), fill=color)

        # EMA overlays
        close = df["close"]
        ema_colors = [
            (255, 209, 102),
            (76, 201, 240),
            (181, 23, 158),
            (247, 37, 133),
            (6, 214, 160),
        ]
        for idx, period in enumerate(periods[:5]):
            if period < 2:
                continue
            series = ema(close, int(period))
            points: list[tuple[int, int]] = []
            for j, val in enumerate(series.tolist()):
                if pd.isna(val):
                    continue
                x = int(x_start + j * step)
                y = self._scale(float(val), vmin, vmax, price_top, price_bottom)
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=ema_colors[idx], width=2)
            draw.text((margin + 120 * idx, 50), f"EMA{period}", fill=ema_colors[idx])

        # Last price marker
        last_close = float(df["close"].iloc[-1])
        y_last = self._scale(last_close, vmin, vmax, price_top, price_bottom)
        draw.line((margin, y_last, width - margin, y_last), fill=(120, 140, 170), width=1)
        draw.text((width - 220, y_last - 16), f"Last: {last_close:.6f}", fill=fg)

        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        meta = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "source_line": candles[-1].get("source_line") if candles else None,
            "exchange": candles[-1].get("exchange") if candles else None,
            "market_kind": candles[-1].get("market_kind") if candles else None,
            "instrument_id": candles[-1].get("instrument_id") if candles else None,
            "updated_at": candles[-1].get("fetched_at") if candles else None,
        }
        return buf.getvalue(), meta
