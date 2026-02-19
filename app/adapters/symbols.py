from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class SymbolMeta:
    input_symbol: str
    base: str
    quote: str = "USDT"

    @property
    def pair(self) -> str:
        return f"{self.base}{self.quote}"


COINGECKO_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "TRX": "tron",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
}


def normalize_symbol(symbol: str) -> SymbolMeta:
    raw = str(symbol or "").strip().upper()
    s = raw.replace("$", "")
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "/").replace("-", "/")
    if "/" in s:
        left, right = s.split("/", 1)
        if left:
            s = left
        if right in {"USD", "USDT", "USDC", "BUSD"}:
            pass
    aliases = {
        "XBT": "BTC",
        "BITCOIN": "BTC",
        "ETHEREUM": "ETH",
        "SOLANA": "SOL",
    }
    s = aliases.get(s, s)
    if s.endswith(("USDT", "USDC", "USD", "BUSD")):
        for q in ("USDT", "USDC", "USD", "BUSD"):
            if s.endswith(q):
                s = s[: -len(q)]
                break
    s = re.sub(r"[^A-Z0-9]", "", s)
    if not s:
        s = raw.strip().upper().replace("$", "")
    if s.endswith("USDT"):
        s = s[:-4]
    return SymbolMeta(input_symbol=symbol, base=s)


def coingecko_id_for(symbol: str) -> str | None:
    return COINGECKO_MAP.get(normalize_symbol(symbol).base)
