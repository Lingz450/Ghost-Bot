from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from app.adapters.market_router import MarketDataRouter


class OrderbookHeatmapService:
    def __init__(self, market_router: MarketDataRouter) -> None:
        self.market_router = market_router

    @staticmethod
    def _color_scale(value: float, max_value: float) -> tuple[int, int, int]:
        if max_value <= 0:
            return (25, 40, 60)
        ratio = max(0.0, min(1.0, value / max_value))
        r = int(20 + 220 * ratio)
        g = int(40 + 160 * (1.0 - abs(0.5 - ratio) * 2))
        b = int(80 + 140 * (1.0 - ratio))
        return (r, g, b)

    async def render_heatmap(self, symbol: str, depth_limit: int = 500) -> tuple[bytes, dict]:
        depth = max(100, min(int(depth_limit), 500))
        ob = await self.market_router.get_orderbook(symbol, depth=depth)
        bids = [(float(p), float(q)) for p, q in ob.get("bids", []) if float(q) > 0]
        asks = [(float(p), float(q)) for p, q in ob.get("asks", []) if float(q) > 0]
        if not bids or not asks:
            raise RuntimeError("Orderbook depth unavailable for this symbol.")

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0

        low = mid * 0.98
        high = mid * 1.02
        bins_count = 90
        step = (high - low) / bins_count if bins_count else 1.0
        if step <= 0:
            step = max(mid * 0.0001, 1e-8)

        bid_hist = [0.0] * bins_count
        ask_hist = [0.0] * bins_count

        def _bin_index(price: float) -> int:
            idx = int((price - low) / step)
            if idx < 0:
                return -1
            if idx >= bins_count:
                return bins_count
            return idx

        for price, qty in bids:
            idx = _bin_index(price)
            if 0 <= idx < bins_count:
                bid_hist[idx] += qty
        for price, qty in asks:
            idx = _bin_index(price)
            if 0 <= idx < bins_count:
                ask_hist[idx] += qty

        width, height = 1300, 320
        margin = 40
        chart_left = margin
        chart_right = width - margin
        chart_top = 70
        row_height = 90
        row_gap = 20
        bid_top = chart_top
        ask_top = chart_top + row_height + row_gap

        image = Image.new("RGB", (width, height), (10, 16, 26))
        draw = ImageDraw.Draw(image)

        pair = str(ob.get("instrument_id") or symbol.upper())
        draw.text((margin, 18), f"{pair} Orderbook Heatmap", fill=(230, 240, 250))
        draw.text((margin, bid_top - 18), "Bids", fill=(130, 220, 170))
        draw.text((margin, ask_top - 18), "Asks", fill=(240, 150, 150))

        max_liq = max(max(bid_hist) if bid_hist else 0.0, max(ask_hist) if ask_hist else 0.0, 1.0)
        total_w = chart_right - chart_left
        cell_w = max(total_w / bins_count, 1.0)

        for idx in range(bins_count):
            x0 = int(chart_left + idx * cell_w)
            x1 = int(chart_left + (idx + 1) * cell_w)
            bid_color = self._color_scale(bid_hist[idx], max_liq)
            ask_color = self._color_scale(ask_hist[idx], max_liq)
            draw.rectangle((x0, bid_top, x1, bid_top + row_height), fill=bid_color)
            draw.rectangle((x0, ask_top, x1, ask_top + row_height), fill=ask_color)

        draw.rectangle((chart_left, bid_top, chart_right, bid_top + row_height), outline=(45, 60, 80), width=1)
        draw.rectangle((chart_left, ask_top, chart_right, ask_top + row_height), outline=(45, 60, 80), width=1)

        bid_wall_idx = max(range(len(bid_hist)), key=lambda i: bid_hist[i]) if bid_hist else 0
        ask_wall_idx = max(range(len(ask_hist)), key=lambda i: ask_hist[i]) if ask_hist else 0
        bid_wall_price = low + bid_wall_idx * step
        ask_wall_price = low + ask_wall_idx * step

        draw.text(
            (margin, height - 28),
            f"Best bid: {best_bid:.6f} | Best ask: {best_ask:.6f} | Bid wall: {bid_wall_price:.6f} | Ask wall: {ask_wall_price:.6f}",
            fill=(210, 220, 235),
        )

        buf = BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        meta = {
            "pair": pair,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "bid_wall": bid_wall_price,
            "ask_wall": ask_wall_price,
            "source_line": ob.get("source_line"),
            "exchange": ob.get("exchange"),
            "market_kind": ob.get("market_kind"),
        }
        return buf.getvalue(), meta
