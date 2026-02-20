from __future__ import annotations

from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def analysis_actions(symbol: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Set alert", callback_data=f"set_alert:{symbol}")
    kb.button(text="Show levels", callback_data=f"show_levels:{symbol}")
    kb.button(text="Why?", callback_data=f"why:{symbol}")
    kb.button(text="Refresh", callback_data=f"refresh:{symbol}")
    kb.button(text="More detail", callback_data=f"details:{symbol}")
    kb.button(text="Derivatives", callback_data=f"derivatives:{symbol}")
    kb.button(text="News", callback_data=f"catalysts:{symbol}")
    kb.button(text="Backtest this", callback_data=f"backtest:{symbol}")
    kb.adjust(2, 2, 2, 1, 1)
    return kb.as_markup()


def wallet_actions(chain: str, address: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Save wallet", callback_data=f"save_wallet:{chain}:{address}")
    kb.adjust(1)
    return kb.as_markup()


def settings_menu(settings: dict) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()

    anon = "ON" if settings.get("anon_mode") else "OFF"
    formal = "ON" if settings.get("formal_mode") else "OFF"
    profanity = settings.get("profanity_level", "light")

    kb.button(text=f"Anon mode: {anon}", callback_data="settings:toggle:anon_mode")
    kb.button(text=f"Formal mode: {formal}", callback_data="settings:toggle:formal_mode")

    for level in ("none", "light", "medium"):
        kb.button(text=f"Profanity: {level}", callback_data=f"settings:set:profanity_level:{level}")

    for risk in ("low", "medium", "high"):
        kb.button(text=f"Risk: {risk}", callback_data=f"settings:set:risk_profile:{risk}")

    for tf in ("15m", "1h", "4h"):
        kb.button(text=f"TF: {tf}", callback_data=f"settings:set:preferred_timeframe:{tf}")

    for ex in ("binance", "bybit", "okx"):
        kb.button(text=f"EX: {ex}", callback_data=f"settings:set:preferred_exchange:{ex}")

    for tone in ("wild", "standard"):
        kb.button(text=f"Tone: {tone}", callback_data=f"settings:set:tone_mode:{tone}")

    kb.adjust(2, 3, 3, 3, 2)
    return kb.as_markup()


def simple_followup(options: list[tuple[str, str]]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for text, value in options:
        kb.button(text=text, callback_data=value)
    kb.adjust(len(options))
    return kb.as_markup()


def smart_action_menu(symbol: str | None = None) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if symbol:
        sym = symbol.upper()
        kb.button(text=f"{sym} Analyze 1h", callback_data=f"quick:analysis_tf:{sym}:1h")
        kb.button(text=f"{sym} Analyze 4h", callback_data=f"quick:analysis_tf:{sym}:4h")
        kb.button(text=f"{sym} Chart 1h", callback_data=f"quick:chart:{sym}:1h")
        kb.button(text=f"{sym} Heatmap", callback_data=f"quick:heatmap:{sym}")
        kb.button(text=f"{sym} Set Alert", callback_data=f"set_alert:{sym}")
        kb.button(text="Top Overbought 1h", callback_data="quick:rsi:overbought:1h:5")
        kb.button(text="Top Oversold 1h", callback_data="quick:rsi:oversold:1h:5")
        kb.button(text=f"{sym} News", callback_data=f"catalysts:{sym}")
        kb.adjust(2, 2, 2, 2)
        return kb.as_markup()

    kb.button(text="BTC Analyze 1h", callback_data="quick:analysis_tf:BTC:1h")
    kb.button(text="ETH Analyze 1h", callback_data="quick:analysis_tf:ETH:1h")
    kb.button(text="SOL Analyze 1h", callback_data="quick:analysis_tf:SOL:1h")
    kb.button(text="Top Overbought 1h", callback_data="quick:rsi:overbought:1h:5")
    kb.button(text="Top Oversold 1h", callback_data="quick:rsi:oversold:1h:5")
    kb.button(text="Crypto News", callback_data="quick:news:crypto")
    kb.button(text="OpenAI Updates", callback_data="quick:news:openai")
    kb.adjust(2, 2, 1, 2)
    return kb.as_markup()


def command_center_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Alpha", callback_data="cmd:menu:alpha")
    kb.button(text="Watch", callback_data="cmd:menu:watch")
    kb.button(text="Chart", callback_data="cmd:menu:chart")
    kb.button(text="Heatmap", callback_data="cmd:menu:heatmap")
    kb.button(text="RSI Scan", callback_data="cmd:menu:rsi")
    kb.button(text="EMA Scan", callback_data="cmd:menu:ema")
    kb.button(text="News", callback_data="cmd:menu:news")
    kb.button(text="Alerts", callback_data="cmd:menu:alert")
    kb.button(text="Find Pair", callback_data="cmd:menu:findpair")
    kb.button(text="Setup Math", callback_data="cmd:menu:setup")
    kb.button(text="Scan Wallet", callback_data="cmd:menu:scan")
    kb.button(text="Giveaway", callback_data="cmd:menu:giveaway")
    kb.adjust(3, 3, 3, 3)
    return kb.as_markup()


def alpha_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="BTC 1h", callback_data="cmd:alpha:BTC:1h")
    kb.button(text="BTC 4h", callback_data="cmd:alpha:BTC:4h")
    kb.button(text="ETH 1h", callback_data="cmd:alpha:ETH:1h")
    kb.button(text="ETH 4h", callback_data="cmd:alpha:ETH:4h")
    kb.button(text="SOL 1h", callback_data="cmd:alpha:SOL:1h")
    kb.button(text="SOL 4h", callback_data="cmd:alpha:SOL:4h")
    kb.button(text="Custom", callback_data="cmd:alpha:custom")
    kb.adjust(2, 2, 2, 1)
    return kb.as_markup()


def watch_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="BTC 1h", callback_data="cmd:watch:BTC:1h")
    kb.button(text="ETH 1h", callback_data="cmd:watch:ETH:1h")
    kb.button(text="SOL 1h", callback_data="cmd:watch:SOL:1h")
    kb.button(text="BTC 4h", callback_data="cmd:watch:BTC:4h")
    kb.button(text="ETH 4h", callback_data="cmd:watch:ETH:4h")
    kb.button(text="SOL 4h", callback_data="cmd:watch:SOL:4h")
    kb.button(text="Custom", callback_data="cmd:watch:custom")
    kb.adjust(2, 2, 2, 1)
    return kb.as_markup()


def chart_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="BTC 1h", callback_data="cmd:chart:BTC:1h")
    kb.button(text="BTC 4h", callback_data="cmd:chart:BTC:4h")
    kb.button(text="ETH 1h", callback_data="cmd:chart:ETH:1h")
    kb.button(text="SOL 1h", callback_data="cmd:chart:SOL:1h")
    kb.button(text="Custom", callback_data="cmd:chart:custom")
    kb.adjust(2, 2, 1)
    return kb.as_markup()


def heatmap_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="BTC", callback_data="cmd:heatmap:BTC")
    kb.button(text="ETH", callback_data="cmd:heatmap:ETH")
    kb.button(text="SOL", callback_data="cmd:heatmap:SOL")
    kb.button(text="BNB", callback_data="cmd:heatmap:BNB")
    kb.button(text="Custom", callback_data="cmd:heatmap:custom")
    kb.adjust(2, 2, 1)
    return kb.as_markup()


def rsi_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Oversold 1h Top 10", callback_data="cmd:rsi:1h:oversold:10:14")
    kb.button(text="Overbought 1h Top 10", callback_data="cmd:rsi:1h:overbought:10:14")
    kb.button(text="Oversold 4h Top 10", callback_data="cmd:rsi:4h:oversold:10:14")
    kb.button(text="Overbought 4h Top 10", callback_data="cmd:rsi:4h:overbought:10:14")
    kb.button(text="Custom", callback_data="cmd:rsi:custom")
    kb.adjust(1, 1, 1, 1, 1)
    return kb.as_markup()


def ema_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="EMA200 4h Top 10", callback_data="cmd:ema:200:4h:10")
    kb.button(text="EMA200 1d Top 10", callback_data="cmd:ema:200:1d:10")
    kb.button(text="EMA50 1h Top 10", callback_data="cmd:ema:50:1h:10")
    kb.button(text="EMA20 15m Top 10", callback_data="cmd:ema:20:15m:10")
    kb.button(text="Custom", callback_data="cmd:ema:custom")
    kb.adjust(1, 1, 1, 1, 1)
    return kb.as_markup()


def news_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Crypto News", callback_data="cmd:news:crypto:6")
    kb.button(text="OpenAI Updates", callback_data="cmd:news:openai:6")
    kb.button(text="CPI News", callback_data="cmd:news:cpi:6")
    kb.button(text="FOMC/Macro", callback_data="cmd:news:fomc:6")
    kb.button(text="Top 8", callback_data="cmd:news:crypto:8")
    kb.adjust(2, 2, 1)
    return kb.as_markup()


def alert_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Create Alert", callback_data="cmd:alert:create")
    kb.button(text="My Alerts", callback_data="cmd:alert:list")
    kb.button(text="Clear All", callback_data="cmd:alert:clear")
    kb.button(text="Delete by ID", callback_data="cmd:alert:delete")
    kb.button(text="Pause All", callback_data="cmd:alert:pause")
    kb.button(text="Resume All", callback_data="cmd:alert:resume")
    kb.adjust(2, 2, 2)
    return kb.as_markup()


def findpair_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Find by Price", callback_data="cmd:findpair:price")
    kb.button(text="Find by Name", callback_data="cmd:findpair:query")
    kb.adjust(2)
    return kb.as_markup()


def setup_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Paste Setup", callback_data="cmd:setup:freeform")
    kb.button(text="Tradecheck Wizard", callback_data="cmd:setup:wizard")
    kb.adjust(2)
    return kb.as_markup()


def scan_quick_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="Solana Address", callback_data="cmd:scan:solana")
    kb.button(text="Tron Address", callback_data="cmd:scan:tron")
    kb.adjust(2)
    return kb.as_markup()


def giveaway_menu(is_admin: bool) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if is_admin:
        kb.button(text="Start Giveaway", callback_data="gw:start")
        kb.button(text="Status", callback_data="gw:status")
        kb.button(text="End", callback_data="gw:end")
        kb.button(text="Reroll", callback_data="gw:reroll")
        kb.adjust(2, 2)
        return kb.as_markup()
    kb.button(text="Join Active Giveaway", callback_data="gw:join")
    kb.button(text="Status", callback_data="gw:status")
    kb.adjust(1, 1)
    return kb.as_markup()


def giveaway_duration_menu() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="5m", callback_data="gw:dur:300")
    kb.button(text="10m", callback_data="gw:dur:600")
    kb.button(text="30m", callback_data="gw:dur:1800")
    kb.button(text="1h", callback_data="gw:dur:3600")
    kb.button(text="6h", callback_data="gw:dur:21600")
    kb.button(text="1d", callback_data="gw:dur:86400")
    kb.adjust(3, 3)
    return kb.as_markup()


def giveaway_winners_menu(duration_seconds: int) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for winners in (1, 2, 3):
        kb.button(text=f"{winners} winner", callback_data=f"gw:win:{duration_seconds}:{winners}")
    kb.adjust(3)
    return kb.as_markup()
