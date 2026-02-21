from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from datetime import datetime, timezone
import logging

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.bot.keyboards import (
    alert_quick_menu,
    alpha_quick_menu,
    analysis_actions,
    chart_quick_menu,
    command_center_menu,
    ema_quick_menu,
    findpair_quick_menu,
    giveaway_duration_menu,
    giveaway_menu,
    giveaway_winners_menu,
    heatmap_quick_menu,
    news_quick_menu,
    rsi_quick_menu,
    scan_quick_menu,
    settings_menu,
    setup_quick_menu,
    simple_followup,
    smart_action_menu,
    wallet_actions,
    watch_quick_menu,
)
from app.bot.templates import (
    asset_unsupported_template,
    correlation_template,
    cycle_template,
    giveaway_status_template,
    help_text,
    news_template,
    pair_find_template,
    price_guess_template,
    rsi_scan_template,
    setup_review_template,
    settings_text,
    smalltalk_reply,
    trade_math_template,
    trade_plan_template,
    trade_verification_template,
    unknown_prompt,
    wallet_scan_template,
    watchlist_template,
)
from app.core.config import get_settings
from app.core.container import ServiceHub
from app.core.fred_persona import fred
from app.services.market_context import format_market_context
from app.core.nlu import COMMON_WORDS_NOT_TICKERS, Intent, is_likely_english_phrase, parse_message, parse_timestamp
from app.db.models import TradeCheck
from app.db.session import AsyncSessionLocal

router = Router()
_settings = get_settings()
_hub: ServiceHub | None = None
_ALLOWED_OPENAI_CHAT_MODES = {"hybrid", "tool_first", "llm_first", "chat_only"}
_CHAT_LOCKS: dict[int, asyncio.Lock] = {}
logger = logging.getLogger(__name__)
SOURCE_QUERY_RE = re.compile(
    r"\b(where\s+is\s+this\s+from|what(?:'s| is)\s+the\s+source|which\s+exchange|source\??|exchange\??)\b",
    re.IGNORECASE,
)
SOURCE_QUERY_STOPWORDS = {
    "source",
    "exchange",
    "where",
    "is",
    "this",
    "from",
    "what",
    "whats",
    "the",
    "for",
    "of",
    "result",
    "last",
}
ACTION_SYMBOL_STOPWORDS = {
    "what",
    "who",
    "hwo",
    "how",
    "are",
    "you",
    "doing",
    "coin",
    "coins",
    "overbought",
    "oversold",
    "list",
    "top",
    "news",
    "alert",
    "scan",
    "chart",
    "heatmap",
    "short",
    "long",
}
ACTION_SYMBOL_STOPWORDS.update(COMMON_WORDS_NOT_TICKERS)
FOLLOWUP_RE = re.compile(
    r"\b("
    r"how about|what if|too risky|risky|is\s+[0-9.]+\s+(a\s+)?good\s+entry|good entry|"
    r"sl|stop(?:\s+loss)?|entry|target|tp\d*|take profit|"
    r"leverage|[0-9]+x|risk|rr|send it|thoughts?|better|worse"
    r")\b",
    re.IGNORECASE,
)
FOLLOWUP_VALUE_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\b")


_CHAT_LOCKS_MAX = 2000


def _safe_exc(exc: Exception) -> str:
    """Return exception message safe for Telegram HTML parse_mode (no raw < > & chars)."""
    return (
        str(exc)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# Valid Telegram HTML tags (self-closing not needed)
_TELEGRAM_ALLOWED_TAGS = {"b", "i", "u", "s", "code", "pre", "a", "blockquote", "tg-spoiler"}
_HTML_TAG_RE = re.compile(r"<(/?)(\w[\w\-]*)(\s[^>]*)?>", re.IGNORECASE)


def _sanitize_llm_html(text: str) -> str:
    """
    Clean LLM-generated HTML so Telegram won't reject it.
    - Strips unsupported tags (keeps their text content)
    - Closes any unclosed valid tags
    """
    if not text:
        return text

    result: list[str] = []
    open_stack: list[str] = []  # tags opened but not yet closed
    pos = 0

    for m in _HTML_TAG_RE.finditer(text):
        # Append the literal text before this tag
        result.append(text[pos:m.start()])
        pos = m.end()

        closing = m.group(1) == "/"
        tag = m.group(2).lower()
        attrs = m.group(3) or ""

        if tag not in _TELEGRAM_ALLOWED_TAGS:
            # Unsupported tag — drop the tag but keep nothing (content will still flow through)
            continue

        if closing:
            # Only emit the closing tag if we actually opened this tag
            if tag in open_stack:
                # Close any tags opened after this one (auto-close nested unclosed tags)
                while open_stack and open_stack[-1] != tag:
                    result.append(f"</{open_stack.pop()}>")
                if open_stack:
                    open_stack.pop()
                result.append(f"</{tag}>")
        else:
            result.append(f"<{tag}{attrs}>")
            # <a> and block-level tags need tracking; void-like usage is rare in LLM output
            open_stack.append(tag)

    # Append remaining text
    result.append(text[pos:])

    # Close any still-open tags in reverse order
    for tag in reversed(open_stack):
        result.append(f"</{tag}>")

    return "".join(result)


async def _send_llm_reply(message: Message, reply: str) -> None:
    """Send an LLM reply with HTML. If Telegram rejects the HTML, retry as plain text."""
    from aiogram.exceptions import TelegramBadRequest

    cleaned = _sanitize_llm_html(reply)
    try:
        await message.answer(cleaned)
    except TelegramBadRequest:
        # Strip all remaining tags and send as plain text
        plain = re.sub(r"<[^>]+>", "", reply)
        with suppress(Exception):
            await message.answer(plain)


def _chat_lock(chat_id: int) -> asyncio.Lock:
    lock = _CHAT_LOCKS.get(chat_id)
    if lock is None:
        # Prune idle locks when dict grows too large to prevent unbounded memory growth
        if len(_CHAT_LOCKS) >= _CHAT_LOCKS_MAX:
            idle = [k for k, v in list(_CHAT_LOCKS.items()) if not v.locked()]
            for k in idle[:len(idle) // 2 + 1]:
                _CHAT_LOCKS.pop(k, None)
        lock = asyncio.Lock()
        _CHAT_LOCKS[chat_id] = lock
    return lock


async def _acquire_message_once(message: Message, ttl: int = 60 * 60 * 6) -> bool:
    hub = _require_hub()
    key = f"seen:message:{message.chat.id}:{message.message_id}"
    try:
        return await hub.cache.set_if_absent(key, ttl=ttl)
    except Exception:  # noqa: BLE001
        logger.exception("dedupe_cache_error", extra={"event": "dedupe_cache_error", "chat_id": message.chat.id})
        return True


async def _acquire_callback_once(callback: CallbackQuery, ttl: int = 60 * 30) -> bool:
    hub = _require_hub()
    cb_id = (callback.id or "").strip()
    if not cb_id:
        return True
    key = f"seen:callback:{cb_id}"
    return await hub.cache.set_if_absent(key, ttl=ttl)


async def _typing_loop(bot, chat_id: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with suppress(Exception):
            await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        try:
            await asyncio.wait_for(stop.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            pass


async def _run_with_typing_lock(bot, chat_id: int, runner) -> None:
    stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(bot, chat_id, stop))
    lock = _chat_lock(chat_id)
    try:
        async with lock:
            await runner()
    finally:
        stop.set()
        typing_task.cancel()
        with suppress(Exception):
            await typing_task


def init_handlers(hub: ServiceHub) -> None:
    global _hub
    _hub = hub


def _parse_int_list(value, fallback: list[int]) -> list[int]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        return fallback
    out: list[int] = []
    for item in items:
        try:
            out.append(int(item))
        except Exception:  # noqa: BLE001
            continue
    return out or fallback


def _extract_source_symbol_hint(text: str) -> str | None:
    for token in re.findall(r"\b[A-Za-z]{2,12}\b", text):
        low = token.lower()
        if low in SOURCE_QUERY_STOPWORDS:
            continue
        return token.upper().lstrip("$")
    return None


def _is_source_query(text: str) -> bool:
    return bool(SOURCE_QUERY_RE.search(text or ""))


def _extract_action_symbol_hint(text: str) -> str | None:
    if is_likely_english_phrase(text):
        return None
    for token in re.findall(r"\b[A-Za-z]{2,12}\b", text):
        low = token.lower()
        if low in ACTION_SYMBOL_STOPWORDS:
            continue
        return token.upper().lstrip("$")
    return None


async def _remember_source_context(
    chat_id: int,
    *,
    source_line: str | None = None,
    exchange: str | None = None,
    market_kind: str | None = None,
    instrument_id: str | None = None,
    updated_at: str | None = None,
    symbol: str | None = None,
    context: str | None = None,
) -> None:
    if not any([source_line, exchange, market_kind, instrument_id]):
        return
    hub = _require_hub()
    payload = {
        "source_line": source_line or "",
        "exchange": exchange or "",
        "market_kind": market_kind or "",
        "instrument_id": instrument_id or "",
        "updated_at": updated_at or "",
        "symbol": symbol.upper() if isinstance(symbol, str) and symbol else "",
        "context": context or "",
    }
    await hub.cache.set_json(f"last_source:{chat_id}", payload, ttl=60 * 60 * 12)
    if payload["symbol"]:
        await hub.cache.set_json(f"last_source:{chat_id}:{payload['symbol']}", payload, ttl=60 * 60 * 12)


def _format_source_response(payload: dict | None) -> str:
    if not payload:
        return "I do not have a recent source to report yet."
    line = str(payload.get("source_line") or "").strip()
    if line:
        return f"Source: {line}"
    exchange = str(payload.get("exchange") or "").strip()
    market_kind = str(payload.get("market_kind") or "").strip()
    instrument_id = str(payload.get("instrument_id") or "").strip()
    updated = str(payload.get("updated_at") or "").strip()
    context = str(payload.get("context") or "").strip()
    parts = [p for p in [exchange, market_kind, instrument_id] if p]
    if not parts:
        return "I do not have a recent source to report yet."
    base = " ".join(parts)
    suffix = f" | Updated: {updated}" if updated else ""
    prefix = f"{context} source: " if context else "Source: "
    return f"{prefix}{base}{suffix}"


async def _source_reply_for_chat(chat_id: int, query_text: str) -> str:
    hub = _require_hub()
    symbol = _extract_source_symbol_hint(query_text)
    if symbol:
        per_symbol = await hub.cache.get_json(f"last_source:{chat_id}:{symbol}")
        if isinstance(per_symbol, dict):
            return _format_source_response(per_symbol)
        analysis = await hub.cache.get_json(f"last_analysis:{chat_id}:{symbol}")
        if isinstance(analysis, dict):
            line = str(analysis.get("data_source_line") or "").strip()
            if line:
                return f"{symbol} source: {line}"
    payload = await hub.cache.get_json(f"last_source:{chat_id}")
    if isinstance(payload, dict):
        return _format_source_response(payload)
    return "I do not have a recent source to report yet. Ask for analysis/chart first, then ask `source?`."


def _parse_tf_list(value, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = [x.strip() for x in value.split(",")]
    else:
        return fallback
    out = [x for x in items if x]
    return out or fallback


def _analysis_timeframes_from_settings(settings: dict) -> list[str]:
    if _settings.analysis_fast_mode:
        return _parse_tf_list(settings.get("preferred_timeframe", "1h"), ["1h"])
    return _parse_tf_list(settings.get("preferred_timeframes", settings.get("preferred_timeframe", "1h,4h")), ["1h", "4h"])


def _require_hub() -> ServiceHub:
    if _hub is None:
        raise RuntimeError("Handlers not initialized")
    return _hub


def _openai_chat_mode() -> str:
    mode = str(_settings.openai_chat_mode or "hybrid").strip().lower()
    if mode not in _ALLOWED_OPENAI_CHAT_MODES:
        return "hybrid"
    return mode


async def _get_chat_history(chat_id: int) -> list[dict[str, str]]:
    hub = _require_hub()
    payload = await hub.cache.get_json(f"llm:history:{chat_id}")
    if not isinstance(payload, list):
        return []
    out: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            out.append({"role": role, "content": content})
    return out


async def _append_chat_history(chat_id: int, role: str, content: str) -> None:
    role = role.strip().lower()
    content = content.strip()
    if role not in {"user", "assistant"} or not content:
        return
    history = await _get_chat_history(chat_id)
    history.append({"role": role, "content": content})
    turns = max(int(_settings.openai_chat_history_turns), 1)
    history = history[-(turns * 2) :]
    hub = _require_hub()
    await hub.cache.set_json(f"llm:history:{chat_id}", history, ttl=60 * 60 * 24 * 7)


def _analysis_context_payload(symbol: str, direction: str | None, payload: dict) -> dict:
    return {
        "symbol": symbol.upper(),
        "direction": (direction or payload.get("side") or "").strip().lower() or None,
        "analysis_summary": str(payload.get("summary") or "").strip(),
        "key_levels": {
            "entry": str(payload.get("entry") or "").strip(),
            "tp1": str(payload.get("tp1") or "").strip(),
            "tp2": str(payload.get("tp2") or "").strip(),
            "sl": str(payload.get("sl") or "").strip(),
            "price": payload.get("price"),
        },
        "market_context": payload.get("market_context", {}),
        "market_context_text": str(payload.get("market_context_text") or "").strip(),
    }


async def _remember_analysis_context(chat_id: int, symbol: str, direction: str | None, payload: dict) -> None:
    hub = _require_hub()
    context = _analysis_context_payload(symbol, direction, payload)
    await hub.cache.set_json(f"last_analysis_context:{chat_id}", context, ttl=300)
    await hub.cache.set_json(f"last_analysis_context:{chat_id}:{symbol.upper()}", context, ttl=300)


async def _recent_analysis_context(chat_id: int) -> dict | None:
    hub = _require_hub()
    payload = await hub.cache.get_json(f"last_analysis_context:{chat_id}")
    return payload if isinstance(payload, dict) else None


def _looks_like_analysis_followup(text: str, context: dict | None) -> bool:
    cleaned = (text or "").strip()
    if not cleaned or not context:
        return False
    lower = cleaned.lower()
    symbol = str(context.get("symbol") or "").lower()
    if symbol and re.search(rf"\b{re.escape(symbol)}\b", lower) and FOLLOWUP_VALUE_RE.search(lower):
        return True
    if FOLLOWUP_RE.search(lower):
        return True
    if lower.endswith("?") and FOLLOWUP_VALUE_RE.search(lower):
        return True
    return False


async def _llm_analysis_reply(
    *,
    payload: dict,
    symbol: str,
    direction: str | None,
    chat_id: int | None,
) -> str | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None

    market_context_text = str(payload.get("market_context_text") or "").strip()
    direction_label = (direction or payload.get("side") or "").strip().lower() or "none"
    prompt = (
        f"Analysis data for {symbol.upper()} ({direction_label} bias):\n"
        f"{json.dumps(payload, ensure_ascii=True, default=str)}\n\n"
        f"BTC/market backdrop: {market_context_text or 'not available'}\n\n"
        "Write this as Ghost would — start with current price and % change, weave in key levels "
        "(EMA200, order blocks, bollinger bands, RSI) naturally in prose, mention macro if relevant, "
        "then give entry range / 3 targets / stop as simple plain lines. "
        "End with one sharp observation. All lowercase, casual trader voice. No HTML tags."
    )
    history = await _get_chat_history(chat_id) if chat_id is not None else []
    try:
        reply = await hub.llm_client.reply(
            prompt,
            history=history,
            max_output_tokens=min(max(int(_settings.openai_max_output_tokens), 400), 700),
            temperature=max(0.6, float(_settings.openai_temperature)),
        )
    except Exception:  # noqa: BLE001
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await _append_chat_history(chat_id, "user", f"{symbol.upper()} {(direction or payload.get('side') or '').strip()} analysis")
        await _append_chat_history(chat_id, "assistant", final)
    return final


async def _llm_followup_reply(
    user_text: str,
    context: dict,
    *,
    chat_id: int,
) -> str | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None

    cleaned = (user_text or "").strip()
    if not cleaned:
        return None

    prompt = (
        "You are replying to a follow-up message about a recent trade setup.\n"
        f"Last analysis context JSON: {json.dumps(context, ensure_ascii=True, default=str)}\n"
        f"User follow-up: {cleaned}\n"
        "Treat this as continuation of the same setup, not a fresh full report.\n"
        "If the proposed SL/entry/leverage is weak, say it directly and suggest a better level.\n"
        "Keep it conversational and concise."
    )
    history = await _get_chat_history(chat_id)
    try:
        reply = await hub.llm_client.reply(prompt, history=history, max_output_tokens=220)
    except Exception:  # noqa: BLE001
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final:
        await _append_chat_history(chat_id, "user", cleaned)
        await _append_chat_history(chat_id, "assistant", final)
    return final


async def _render_analysis_text(
    *,
    payload: dict,
    symbol: str,
    direction: str | None,
    settings: dict,
    chat_id: int,
    detailed: bool = False,
) -> str:
    try:
        return await fred.format_as_ghost(payload)
    except Exception:  # noqa: BLE001
        llm_text = await _llm_analysis_reply(
            payload=payload,
            symbol=symbol,
            direction=direction,
            chat_id=chat_id,
        )
        if llm_text:
            return llm_text
        return trade_plan_template(payload, settings, detailed=detailed)


async def _send_ghost_analysis(message: Message, symbol: str, text: str) -> None:
    with suppress(Exception):
        await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    await asyncio.sleep(1.3)
    await message.answer(text, reply_markup=analysis_actions(symbol))


def _define_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="DEFINE Analyze 1h", callback_data="define:analyze:1h"),
                InlineKeyboardButton(text="DEFINE Analyze 4h", callback_data="define:analyze:4h"),
            ],
            [
                InlineKeyboardButton(text="DEFINE Chart 1h", callback_data="define:chart:1h"),
                InlineKeyboardButton(text="DEFINE Heatmap", callback_data="define:heatmap"),
            ],
            [
                InlineKeyboardButton(text="DEFINE Set Alert", callback_data="define:alert"),
                InlineKeyboardButton(text="Top Overbought 1h", callback_data="top:overbought:1h"),
            ],
            [
                InlineKeyboardButton(text="Top Oversold 1h", callback_data="top:oversold:1h"),
                InlineKeyboardButton(text="DEFINE News", callback_data="define:news"),
            ],
        ]
    )


def _mentions_bot(text: str, bot_username: str | None) -> bool:
    if not bot_username:
        return False
    return f"@{bot_username.lower()}" in text.lower()


def _strip_bot_mention(text: str, bot_username: str | None) -> str:
    if not bot_username:
        return text
    return re.sub(rf"@{re.escape(bot_username)}", "", text, flags=re.IGNORECASE).strip()


def _is_reply_to_bot(message: Message, hub: ServiceHub) -> bool:
    reply = message.reply_to_message
    if not reply or not reply.from_user:
        return False
    return bool(reply.from_user.id == hub.bot.id)


async def _group_free_talk_enabled(chat_id: int) -> bool:
    hub = _require_hub()
    payload = await hub.cache.get_json(f"group:free_talk:{chat_id}")
    return bool(payload and payload.get("enabled"))


async def _set_group_free_talk(chat_id: int, enabled: bool) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"group:free_talk:{chat_id}", {"enabled": bool(enabled)}, ttl=60 * 60 * 24 * 365)


def _looks_like_clear_intent(text: str) -> bool:
    parsed = parse_message(text)
    return parsed.intent not in {Intent.UNKNOWN, Intent.SMALLTALK, Intent.HELP, Intent.START}


async def _is_group_admin(message: Message) -> bool:
    hub = _require_hub()
    if not message.from_user:
        return False
    if int(message.from_user.id) in set(_settings.admin_ids_list()):
        return True
    try:
        member = await hub.bot.get_chat_member(message.chat.id, message.from_user.id)
        return getattr(member, "status", "") in {"administrator", "creator"}
    except Exception:  # noqa: BLE001
        return False


async def _check_req_limit(chat_id: int) -> bool:
    hub = _require_hub()
    try:
        result = await hub.rate_limiter.check(
            key=f"rl:req:{chat_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
            limit=_settings.request_rate_limit_per_minute,
            window_seconds=60,
        )
        return result.allowed
    except Exception:  # noqa: BLE001
        logger.exception("rate_limit_check_error", extra={"event": "rate_limit_check_error", "chat_id": chat_id})
        return True


async def _llm_fallback_reply(user_text: str, settings: dict | None = None, chat_id: int | None = None) -> str | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None

    cleaned = (user_text or "").strip()
    if not cleaned:
        return None

    style = "wild"
    if settings:
        if settings.get("formal_mode"):
            style = "formal"
        else:
            style = str(settings.get("tone_mode", "wild")).lower()

    prompt = (
        f"User message: {cleaned}\n\n"
        "Answer this directly and helpfully. Never ask the user for clarification — give your best answer now.\n"
        "BOT CAPABILITIES (use this if they ask how to use features):\n"
        "- Alerts: 'alert BTC 100000 above' or tap Create Alert button\n"
        "- Analysis: 'BTC long' or 'ETH short 4h'\n"
        "- Watchlist: 'coins to watch', 'top movers'\n"
        "- News: 'latest crypto news'\n"
        "- Price: /price BTC\n"
    )
    history = await _get_chat_history(chat_id) if chat_id is not None else []
    try:
        reply = await hub.llm_client.reply(prompt, history=history)
    except Exception:  # noqa: BLE001
        return None
    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await _append_chat_history(chat_id, "user", cleaned)
        await _append_chat_history(chat_id, "assistant", final)
    return final


_MARKET_QUESTION_RE = re.compile(
    r"\b(pump|dump|moon|rug|rekt|bleed|crash|rally|bull|bear|market|btc|bitcoin|eth|ethereum|"
    r"crypto|price|move|movement|run|drop|dip|bounce|trend|happening|why|explain|think|feel|"
    r"going|direction|outlook|setup|narrative|sentiment|vibe|catalys|news|macro|tariff|"
    r"inflation|rate|fed|fomc|cpi|pce|blackrock|etf|liquidat|funding|dominan|"
    # expanded: general "which coin / what to watch / what to buy" questions
    r"coin|coins|token|tokens|alt|alts|altcoin|gem|gems|pick|picks|"
    r"watch|looking|look|buy|sell|trade|play|plays|long|short|"
    r"which|what|best|top|good|strong|weak|hot|cold|"
    r"portfolio|invest|hold|accumulate|dca|entry|exit|"
    r"solana|sol|bnb|xrp|matic|avax|ada|dot|link|doge|shib|pepe|"
    r"layer|defi|nft|meme|perp|spot|futures|leverage|"
    r"dominance|volume|liquidity|whale|orderbook|ob|candle|chart|"
    r"resistance|support|ema|rsi|macd|bollinger)\b",
    re.IGNORECASE,
)

# Questions about the bot itself — must never be treated as market questions
_BOT_META_RE = re.compile(
    r"\b(alert\s+creat|creat\s+alert|alert\s+not|button\s+not|command\s+not|"
    r"bot\s+not|bot\s+down|not\s+working|isn'?t\s+work|not\s+respond|"
    r"why\s+is\s+(the\s+)?(alert|button|command|bot|feature)|"
    r"how\s+do\s+i\s+(create|set|use|make)|what\s+commands|what\s+can\s+you|"
    r"how\s+to\s+(create|set|use)|are\s+you\s+working|still\s+(not\s+)?work|"
    r"failing|broken|feature\s+not|doesn'?t\s+work)\b",
    re.IGNORECASE,
)


def _looks_like_market_question(text: str) -> bool:
    if not text.strip():
        return False
    # Bot-meta questions must never be routed to the market chat handler
    if _BOT_META_RE.search(text):
        return False
    # Single-word checks that strongly indicate crypto intent without needing a keyword match
    lower = text.lower()
    crypto_intent_phrases = (
        "which coin", "what coin", "best coin", "top coin",
        "what to buy", "what should i buy", "should i buy",
        "what to watch", "what should i watch", "worth watching",
        "worth buying", "worth trading", "what to trade",
        "good trade", "good play", "where is the market",
        "how is the market", "what is happening", "what's happening",
        "what happened", "what do you think", "give me a call",
        "market update", "quick update", "anything good",
    )
    if any(phrase in lower for phrase in crypto_intent_phrases):
        return True
    return bool(_MARKET_QUESTION_RE.search(text))


async def _llm_market_chat_reply(
    user_text: str,
    settings: dict | None = None,
    chat_id: int | None = None,
) -> str | None:
    """Answer open-ended market questions by injecting live price + news context."""
    hub = _require_hub()
    if not hub.llm_client:
        return None

    cleaned = (user_text or "").strip()
    if not cleaned:
        return None

    style = "wild"
    if settings:
        style = "formal" if settings.get("formal_mode") else str(settings.get("tone_mode", "wild")).lower()

    # Fetch live market snapshot + recent headlines in parallel
    mkt_ctx: dict = {}
    news_headlines: list[dict] = []
    try:
        mkt_ctx, news_payload = await asyncio.gather(
            hub.analysis_service.get_market_context(),
            hub.news_service.get_digest(mode="crypto", limit=6),
            return_exceptions=True,
        )
        if isinstance(mkt_ctx, Exception):
            mkt_ctx = {}
        if isinstance(news_payload, Exception):
            news_payload = {}
        if isinstance(news_payload, dict):
            news_headlines = news_payload.get("headlines") or []
    except Exception:  # noqa: BLE001
        pass

    mkt_text = format_market_context(mkt_ctx) if mkt_ctx else ""
    news_lines = "\n".join(
        f"- {h.get('title', '')} ({h.get('source', '')})"
        for h in news_headlines[:6]
        if h.get("title")
    )

    context_block = ""
    if mkt_text:
        context_block += f"Live market snapshot: {mkt_text}\n"
    if news_lines:
        context_block += f"Recent crypto news:\n{news_lines}\n"

    prompt = (
        f"{context_block}"
        f"User question: {cleaned}\n\n"
        "Answer directly and comprehensively using the live data above. "
        "Give specific coins, prices, catalysts, and your actual directional take. "
        "If asked which coins to watch — name them with prices and reasons. "
        "Never ask the user for clarification. Never start with filler phrases. "
        "Use Telegram HTML formatting: <b>bold</b> for coin names and key levels, <i>italic</i> for closing line."
    )

    history = await _get_chat_history(chat_id) if chat_id is not None else []
    try:
        reply = await hub.llm_client.reply(
            prompt,
            history=history,
            max_output_tokens=min(max(int(_settings.openai_max_output_tokens), 500), 800),
            temperature=max(0.5, float(_settings.openai_temperature)),
        )
    except Exception:  # noqa: BLE001
        return None

    final = reply.strip() if reply and reply.strip() else None
    if final and chat_id is not None:
        await _append_chat_history(chat_id, "user", cleaned)
        await _append_chat_history(chat_id, "assistant", final)
    return final


def _parse_duration_to_seconds(raw: str) -> int | None:
    m = re.match(r"^\s*(\d+)\s*([smhd])\s*$", raw.lower())
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return value * mult


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default


def _as_float(value, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default


def _as_float_list(value) -> list[float]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = re.findall(r"[0-9]+(?:\.[0-9]+)?", value)
    else:
        raw = []
    out: list[float] = []
    for item in raw:
        v = _as_float(item)
        if v is not None:
            out.append(float(v))
    return out


def _infer_direction(entry: float, targets: list[float], explicit: str | None) -> str:
    side = (explicit or "").strip().lower()
    if side in {"long", "short"}:
        return side
    if not targets:
        return "long"
    return "long" if float(targets[0]) >= float(entry) else "short"


def _trade_math_payload(
    *,
    entry: float,
    stop: float,
    targets: list[float],
    direction: str | None,
    margin_usd: float | None,
    leverage: float | None,
    symbol: str | None = None,
) -> dict:
    e = float(entry)
    s = float(stop)
    tps = [float(x) for x in targets if float(x) > 0]
    if not tps:
        raise RuntimeError("Need at least one target.")
    risk = abs(e - s)
    if risk <= 0:
        raise RuntimeError("Entry and stop cannot be the same.")
    side = _infer_direction(e, tps, direction)

    rows: list[dict] = []
    for tp in tps:
        reward = (tp - e) if side == "long" else (e - tp)
        r_mult = reward / risk
        rows.append({"tp": round(tp, 8), "r_multiple": round(r_mult, 3)})
    best_r = max(row["r_multiple"] for row in rows)

    payload: dict = {
        "symbol": symbol or "",
        "direction": side,
        "entry": round(e, 8),
        "stop": round(s, 8),
        "targets": [round(x, 8) for x in tps],
        "risk_per_unit": round(risk, 8),
        "rows": rows,
        "best_r": round(best_r, 3),
    }

    if margin_usd and leverage and margin_usd > 0 and leverage > 0:
        notional = float(margin_usd) * float(leverage)
        qty = notional / e

        def _pnl(exit_price: float) -> float:
            if side == "long":
                return (exit_price - e) * qty
            return (e - exit_price) * qty

        payload["position"] = {
            "margin_usd": round(float(margin_usd), 2),
            "leverage": round(float(leverage), 2),
            "notional_usd": round(notional, 2),
            "qty": round(qty, 8),
            "stop_pnl_usd": round(_pnl(s), 2),
            "tp_pnls": [{"tp": round(tp, 8), "pnl_usd": round(_pnl(tp), 2)} for tp in tps],
        }

    return payload


def _extract_symbol(params: dict) -> str | None:
    for key in ("symbol", "asset", "ticker", "coin"):
        val = params.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().upper().lstrip("$")
    return None


async def _llm_route_message(user_text: str) -> dict | None:
    hub = _require_hub()
    if not hub.llm_client:
        return None
    try:
        payload = await hub.llm_client.route_message(user_text)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    return payload


_ALERT_SHORTCUT_RE = re.compile(
    r"^alert\s+([A-Za-z0-9]{2,20})\s+([\d.,]+[kKmM]?)\s*(above|below|cross|crosses|over|under)?\s*$",
    re.IGNORECASE,
)


async def _dispatch_command_text(message: Message, synthetic_text: str) -> bool:
    hub = _require_hub()
    chat_id = message.chat.id
    settings = await hub.user_service.get_settings(chat_id)

    # Fast-path: "alert {symbol} {price} [condition]" — bypass NLU to avoid LLM confusion
    _am = _ALERT_SHORTCUT_RE.match(synthetic_text.strip())
    if _am:
        from app.core.nlu import _extract_prices  # already imported elsewhere but safe to reimport
        sym = _am.group(1).upper()
        raw_price_str = _am.group(2)
        raw_cond = (_am.group(3) or "cross").lower()
        prices = _extract_prices(raw_price_str)
        px = prices[0] if prices else None
        if sym and px is not None:
            cond = "above" if raw_cond in ("above", "over") else ("below" if raw_cond in ("below", "under") else "cross")
            try:
                alert = await hub.alerts_service.create_alert(chat_id, sym, cond, float(px))
                await message.answer(
                    f"alert set for <b>{alert.symbol}</b> at <b>${float(px):,.2f}</b>. "
                    "i'll ping you when we hit it. don't get liquidated"
                )
            except RuntimeError as exc:
                await message.answer(f"couldn't set that alert — {_safe_exc(exc)}")
            except Exception:  # noqa: BLE001
                await message.answer("alert creation failed. try again.")
            return True

    parsed = parse_message(synthetic_text)

    if parsed.requires_followup:
        if parsed.intent == Intent.ANALYSIS and not parsed.entities.get("symbol"):
            kb = simple_followup(
                [
                    ("BTC", "quick:analysis:BTC"),
                    ("ETH", "quick:analysis:ETH"),
                    ("SOL", "quick:analysis:SOL"),
                ]
            )
            await message.answer(parsed.followup_question or "Need one detail.", reply_markup=kb)
            return True
        await message.answer(parsed.followup_question or unknown_prompt(), reply_markup=smart_action_menu())
        return True

    if await _handle_parsed_intent(message, parsed, settings):
        return True

    llm_reply = await _llm_fallback_reply(synthetic_text, settings, chat_id=chat_id)
    if llm_reply:
        await _send_llm_reply(message, llm_reply)
        return True
    return False


async def _handle_routed_intent(message: Message, settings: dict, route: dict) -> bool:
    hub = _require_hub()
    intent = str(route.get("intent", "")).strip().lower()
    try:
        confidence = float(route.get("confidence", 0.0) or 0.0)
    except Exception:  # noqa: BLE001
        confidence = 0.0
    params = route.get("params") if isinstance(route.get("params"), dict) else {}
    chat_id = message.chat.id
    raw_text = message.text or ""

    if confidence < _settings.openai_router_min_confidence:
        return False

    if intent in {"smalltalk", "market_chat", "general_chat"}:
        with suppress(Exception):
            await message.bot.send_chat_action(chat_id, ChatAction.TYPING)
        # Always use live-data path — Claude/Grok with market context answers everything better
        llm_reply = await _llm_market_chat_reply(raw_text, settings, chat_id=chat_id)
        if llm_reply:
            await _send_llm_reply(message, llm_reply)
            return True
        # Bot-meta questions (how-to, features) fall back to plain reply
        if _BOT_META_RE.search(raw_text):
            llm_reply = await _llm_fallback_reply(raw_text, settings, chat_id=chat_id)
            await _send_llm_reply(message, llm_reply or smalltalk_reply(settings))
            return True
        return False

    if intent == "news_digest":
        limit = max(3, min(_as_int(params.get("limit"), 6), 10))
        topic = params.get("topic")
        mode = str(params.get("mode") or "crypto").strip().lower() or "crypto"
        if isinstance(topic, str) and topic.strip().lower() == "openai" and mode == "crypto":
            mode = "openai"
        payload = await hub.news_service.get_digest(
            topic=topic if isinstance(topic, str) else None,
            mode=mode,
            limit=limit,
        )
        heads = payload.get("headlines") if isinstance(payload, dict) else None
        head = heads[0] if isinstance(heads, list) and heads else {}
        await _remember_source_context(
            chat_id,
            source_line=f"{head.get('source', 'news feed')} | {head.get('url', '')}".strip(),
            context="news",
        )
        await message.answer(news_template(payload))
        return True

    if intent in {"watch_asset", "market_analysis"}:
        symbol = _extract_symbol(params)
        if not symbol:
            await message.answer("Which coin should I analyze?")
            return True
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        side = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        settings_tfs = _analysis_timeframes_from_settings(settings)
        settings_emas = _parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200])
        settings_rsis = _parse_int_list(settings.get("preferred_rsi_periods", [14]), [14])
        payload = await hub.analysis_service.analyze(
            symbol,
            direction=side if side in {"long", "short"} else None,
            timeframe=timeframe,
            timeframes=[timeframe] if timeframe else settings_tfs,
            ema_periods=settings_emas,
            rsi_periods=settings_rsis,
            include_derivatives=bool(params.get("include_derivatives") or params.get("derivatives")),
            include_news=bool(params.get("include_news") or params.get("news") or params.get("catalysts")),
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, side, payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=side,
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(message, symbol, analysis_text)
        return True

    if intent == "rsi_scan":
        symbol = _extract_symbol(params)
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        mode_raw = str(params.get("mode", "oversold")).strip().lower()
        mode = "overbought" if mode_raw == "overbought" else "oversold"
        limit = max(1, min(_as_int(params.get("limit"), 10), 20))
        rsi_length = max(2, min(_as_int(params.get("rsi_length"), 14), 50))
        payload = await hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode=mode,
            limit=limit,
            rsi_length=rsi_length,
            symbol=symbol,
        )
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=symbol,
            context="rsi scan",
        )
        await message.answer(rsi_scan_template(payload))
        return True

    if intent == "ema_scan":
        timeframe = str(params.get("timeframe", "4h")).strip() or "4h"
        ema_length = max(2, min(_as_int(params.get("ema_length"), 200), 500))
        mode_raw = str(params.get("mode", "closest")).strip().lower()
        mode = mode_raw if mode_raw in {"closest", "above", "below"} else "closest"
        limit = max(1, min(_as_int(params.get("limit"), 10), 20))
        payload = await hub.ema_scanner_service.scan(
            timeframe=timeframe,
            ema_length=ema_length,
            mode=mode,
            limit=limit,
        )
        lines = [payload["summary"], ""]
        for idx, row in enumerate(payload.get("items", []), start=1):
            lines.append(
                f"{idx}. {row['symbol']} price {row['price']} | EMA{payload['ema_length']} {row['ema']} | "
                f"dist {row['distance_pct']}% ({row['side']})"
            )
        if not payload.get("items"):
            lines.append("No EMA matches right now.")
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="ema scan",
        )
        await message.answer("\n".join(lines))
        return True

    if intent == "chart":
        symbol = _extract_symbol(params)
        if not symbol:
            await message.answer("Which symbol should I chart?")
            return True
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        img, meta = await hub.chart_service.render_chart(symbol=symbol, timeframe=timeframe)
        caption = f"{symbol} {timeframe} chart."
        await _remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("instrument_id") or ""),
            updated_at=str(meta.get("updated_at") or ""),
            symbol=symbol,
            context="chart",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_{timeframe}_chart.png"),
            caption=caption,
        )
        return True

    if intent == "heatmap":
        symbol = _extract_symbol(params) or "BTC"
        img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
        caption = (
            f"{meta['pair']} orderbook heatmap\n"
            f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
            f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
        )
        await _remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("pair") or ""),
            symbol=symbol,
            context="heatmap",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_heatmap.png"),
            caption=caption,
        )
        return True

    if intent == "watchlist":
        count = max(1, min(_as_int(params.get("count"), 5), 20))
        direction_raw = str(params.get("direction") or "").strip().lower()
        direction = direction_raw if direction_raw in {"long", "short"} else None
        payload = await hub.watchlist_service.build_watchlist(count=count, direction=direction)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="watchlist",
        )
        await message.answer(watchlist_template(payload))
        return True

    if intent == "alert_create":
        symbol = _extract_symbol(params)
        price = _as_float(params.get("price") or params.get("target_price"))

        # Fallback: parse symbol/price directly from the raw text if router missed them
        if not symbol or price is None:
            from app.core.nlu import _extract_symbols, _extract_prices
            if not symbol:
                syms = _extract_symbols(raw_text)
                symbol = syms[0] if syms else None
            if price is None:
                pxs = _extract_prices(raw_text)
                price = pxs[0] if pxs else None

        if not symbol or price is None:
            await message.answer("Need symbol and price — e.g. <code>alert BTC 66k</code> or <code>set alert for SOL 200</code>.")
            return True
        op = str(params.get("operator") or params.get("condition") or "cross").strip().lower()
        if op in {">", ">=", "above", "gt", "gte", "crosses above", "cross above"}:
            condition = "above"
        elif op in {"<", "<=", "below", "lt", "lte", "crosses below", "cross below"}:
            condition = "below"
        else:
            condition = "cross"
        try:
            alert = await hub.alerts_service.create_alert(chat_id, symbol, condition, float(price))
        except RuntimeError as exc:
            await message.answer(f"couldn't set that alert — {_safe_exc(exc)}")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("alert_create_failed", extra={"chat_id": chat_id, "symbol": symbol, "price": price})
            await message.answer(f"alert creation failed. try again in a sec.")
            return True
        await _remember_source_context(
            chat_id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        await message.answer(
            f"alert set for <b>{alert.symbol}</b> at <b>${float(price):,.2f}</b>. "
            "i'll ping you when we hit it. don't get liquidated"
        )
        return True

    if intent == "alert_list":
        alerts = await hub.alerts_service.list_alerts(chat_id)
        if not alerts:
            await message.answer("No active alerts.")
        else:
            lines = ["<b>Active Alerts</b>", ""]
            for a in alerts:
                lines.append(f"<code>#{a.id}</code>  <b>{a.symbol}</b>  {a.condition}  {a.target_price}  <i>[{a.status}]</i>")
            first = alerts[0]
            await _remember_source_context(
                chat_id,
                exchange=first.source_exchange,
                market_kind=first.market_kind,
                instrument_id=first.instrument_id,
                symbol=first.symbol,
                context="alerts list",
            )
            await message.answer("\n".join(lines))
        return True

    if intent == "alert_delete":
        symbol = _extract_symbol(params)
        if symbol:
            count = await hub.alerts_service.delete_alerts_by_symbol(chat_id, symbol)
            await message.answer(f"Removed {count} alert(s) for {symbol}.")
            return True
        alert_id = _as_int(params.get("id") or params.get("alert_id"), 0)
        if alert_id <= 0:
            await message.answer("Which alert id should I delete?")
            return True
        ok = await hub.alerts_service.delete_alert(chat_id, alert_id)
        await message.answer("Deleted." if ok else "Alert not found.")
        return True

    if intent == "alert_clear":
        count = await hub.alerts_service.clear_user_alerts(chat_id)
        await message.answer(f"Cleared {count} alerts.")
        return True

    if intent == "pair_find":
        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            query = _extract_symbol(params)
        if not isinstance(query, str) or not query.strip():
            await message.answer("Which coin should I resolve to a pair?")
            return True
        payload = await hub.discovery_service.find_pair(query.strip())
        await message.answer(pair_find_template(payload))
        return True

    if intent == "price_guess":
        price = _as_float(params.get("price") or params.get("target_price"))
        if price is None:
            await message.answer("What price should I search around?")
            return True
        limit = max(1, min(_as_int(params.get("limit"), 10), 20))
        payload = await hub.discovery_service.guess_by_price(price, limit=limit)
        await message.answer(price_guess_template(payload))
        return True

    if intent == "setup_review":
        symbol = _extract_symbol(params)
        entry = _as_float(params.get("entry"))
        stop = _as_float(params.get("stop") or params.get("sl"))
        targets = _as_float_list(params.get("targets") or params.get("tp"))
        leverage = _as_float(params.get("leverage"))

        # "Market Price" / "market order" / "MP" as entry → fetch live price
        if entry is None and symbol and re.search(
            r"\bmarket\s*(?:price|order)?\b|\bmp\b|\bat\s+market\b", raw_text, re.IGNORECASE
        ):
            with suppress(Exception):
                price_data = await hub.market_router.get_price(symbol)
                entry = _as_float(price_data.get("price") or price_data.get("last"))

        if not symbol or entry is None or stop is None or not targets:
            await message.answer(
                "need <b>symbol</b>, <b>entry</b>, <b>stop</b>, and at least one <b>target</b>.\n"
                "e.g. <code>SNXUSDT entry 0.028 stop 0.036 tp 0.022</code>"
            )
            return True
        direction = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        amount_usd = _as_float(params.get("amount") or params.get("amount_usd") or params.get("margin"))
        payload = await hub.setup_review_service.review(
            symbol=symbol,
            timeframe=timeframe,
            entry=float(entry),
            stop=float(stop),
            targets=[float(x) for x in targets],
            direction=direction if direction in {"long", "short"} else None,
            amount_usd=amount_usd,
            leverage=leverage,
        )
        await message.answer(setup_review_template(payload, settings))
        return True

    if intent == "trade_math":
        entry = _as_float(params.get("entry"))
        stop = _as_float(params.get("stop") or params.get("sl"))
        targets = _as_float_list(params.get("targets") or params.get("tp"))
        if entry is None or stop is None or not targets:
            await message.answer("Send entry, stop, and target(s), e.g. `entry 100 sl 95 tp 110`.")
            return True
        symbol = _extract_symbol(params)
        side = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        margin_usd = _as_float(params.get("amount") or params.get("amount_usd") or params.get("margin"))
        leverage = _as_float(params.get("leverage"))
        payload = _trade_math_payload(
            entry=float(entry),
            stop=float(stop),
            targets=[float(x) for x in targets],
            direction=side,
            margin_usd=margin_usd,
            leverage=leverage,
            symbol=symbol,
        )
        await message.answer(trade_math_template(payload, settings))
        return True

    if intent == "giveaway_join":
        if not message.from_user:
            await message.answer("Could not identify user for giveaway join.")
            return True
        payload = await hub.giveaway_service.join_active(chat_id, message.from_user.id)
        await message.answer(f"Joined giveaway #{payload['giveaway_id']}. Participants: {payload['participants']}")
        return True

    if intent == "giveaway_status":
        payload = await hub.giveaway_service.status(chat_id)
        await message.answer(giveaway_status_template(payload))
        return True

    if intent == "giveaway_end":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.end_giveaway(chat_id, message.from_user.id)
        if payload.get("winner_user_id"):
            await message.answer(
                f"Giveaway #{payload['giveaway_id']} ended.\nWinner: {payload['winner_user_id']}\nPrize: {payload['prize']}"
            )
        else:
            await message.answer(f"Giveaway ended with no winner. {payload.get('note')}")
        return True

    if intent == "giveaway_reroll":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.reroll(chat_id, message.from_user.id)
        await message.answer(
            f"Reroll complete for giveaway #{payload['giveaway_id']}.\nNew winner: {payload['winner_user_id']}"
        )
        return True

    if intent == "giveaway_cancel":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.end_giveaway(chat_id, message.from_user.id)
        if payload.get("winner_user_id"):
            await message.answer(
                f"Giveaway #{payload['giveaway_id']} ended.\nWinner: {payload['winner_user_id']}\nPrize: {payload['prize']}"
            )
        else:
            await message.answer(f"Giveaway ended with no winner. {payload.get('note')}")
        return True

    if intent == "giveaway_start":
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        duration = params.get("duration") or params.get("duration_text") or "10m"
        duration_seconds = None
        if isinstance(duration, (int, float)):
            duration_seconds = max(30, int(duration))
        elif isinstance(duration, str):
            duration_seconds = _parse_duration_to_seconds(duration)
        if duration_seconds is None:
            await message.answer("Give a duration like 10m or 1h for giveaway start.")
            return True
        prize = str(params.get("prize") or "Prize").strip() or "Prize"
        payload = await hub.giveaway_service.start_giveaway(
            group_chat_id=chat_id,
            admin_chat_id=message.from_user.id,
            duration_seconds=duration_seconds,
            prize=prize,
        )
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\nEnds at: {payload['end_time']}\nUsers enter with /join"
        )
        return True

    return False


async def _get_pending_alert(chat_id: int) -> str | None:
    hub = _require_hub()
    payload = await hub.cache.get_json(f"pending_alert:{chat_id}")
    return payload.get("symbol") if payload else None


async def _set_pending_alert(chat_id: int, symbol: str) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"pending_alert:{chat_id}", {"symbol": symbol.upper()}, ttl=300)


async def _clear_pending_alert(chat_id: int) -> None:
    hub = _require_hub()
    await hub.cache.redis.delete(f"pending_alert:{chat_id}")


async def _wizard_get(chat_id: int) -> dict | None:
    hub = _require_hub()
    return await hub.cache.get_json(f"wizard:tradecheck:{chat_id}")


async def _wizard_set(chat_id: int, payload: dict, ttl: int = 900) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"wizard:tradecheck:{chat_id}", payload, ttl=ttl)


async def _wizard_clear(chat_id: int) -> None:
    hub = _require_hub()
    await hub.cache.redis.delete(f"wizard:tradecheck:{chat_id}")


async def _cmd_wizard_get(chat_id: int) -> dict | None:
    hub = _require_hub()
    return await hub.cache.get_json(f"wizard:cmd:{chat_id}")


async def _cmd_wizard_set(chat_id: int, payload: dict, ttl: int = 900) -> None:
    hub = _require_hub()
    await hub.cache.set_json(f"wizard:cmd:{chat_id}", payload, ttl=ttl)


async def _cmd_wizard_clear(chat_id: int) -> None:
    hub = _require_hub()
    await hub.cache.redis.delete(f"wizard:cmd:{chat_id}")


async def _save_trade_check(chat_id: int, data: dict, result: dict) -> None:
    hub = _require_hub()
    user = await hub.user_service.ensure_user(chat_id)
    async with AsyncSessionLocal() as session:
        row = TradeCheck(
            user_id=user.id,
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            timestamp=data["timestamp"],
            entry=float(data["entry"]),
            stop=float(data["stop"]),
            targets_json=[float(x) for x in data["targets"]],
            mode=data.get("mode", "ambiguous"),
            result_json=result,
        )
        session.add(row)
        await session.commit()


async def _handle_parsed_intent(message: Message, parsed, settings: dict) -> bool:
    hub = _require_hub()
    chat_id = message.chat.id
    raw_text = message.text or ""

    if parsed.intent == Intent.ANALYSIS:
        symbol = parsed.entities["symbol"]
        direction = parsed.entities.get("direction")
        parsed_tfs = parsed.entities.get("timeframes")
        parsed_emas = parsed.entities.get("ema_periods")
        parsed_rsis = parsed.entities.get("rsi_periods")

        settings_tfs = _analysis_timeframes_from_settings(settings)
        settings_emas = _parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200])
        settings_rsis = _parse_int_list(settings.get("preferred_rsi_periods", [14]), [14])

        try:
            payload = await hub.analysis_service.analyze(
                symbol,
                direction=direction,
                timeframe=parsed.entities.get("timeframe"),
                timeframes=parsed_tfs or settings_tfs,
                ema_periods=parsed_emas or settings_emas,
                rsi_periods=parsed_rsis or settings_rsis,
                all_timeframes=bool(parsed.entities.get("all_timeframes")),
                all_emas=bool(parsed.entities.get("all_emas")),
                all_rsis=bool(parsed.entities.get("all_rsis")),
                include_derivatives=bool(parsed.entities.get("include_derivatives")),
                include_news=bool(parsed.entities.get("include_news")),
                notes=parsed.entities.get("notes", []),
            )
        except Exception as exc:  # noqa: BLE001
            err = str(exc).lower()
            if any(
                marker in err
                for marker in (
                    "price unavailable",
                    "no valid ohlcv",
                    "isn't supported",
                    "binance-only",
                    "unavailable",
                )
            ):
                fallback = await hub.analysis_service.fallback_asset_brief(symbol, reason=str(exc))
                await message.answer(asset_unsupported_template(fallback, settings))
                return True
            raise
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, direction, payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=direction,
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(message, symbol, analysis_text)
        return True

    if parsed.intent == Intent.SETUP_REVIEW:
        timeframe = parsed.entities.get("timeframe", "1h")
        tfs = parsed.entities.get("timeframes") or []
        if tfs:
            timeframe = tfs[0]
        symbol = parsed.entities.get("symbol")
        entry = _as_float(parsed.entities.get("entry"))
        stop = _as_float(parsed.entities.get("stop"))
        targets = [float(x) for x in (parsed.entities.get("targets") or [])]

        # "Market Price" / "market order" as entry → fetch live price
        raw = message.text or ""
        if entry is None and symbol and re.search(
            r"\bmarket\s*(?:price|order)?\b|\bmp\b|\bat\s+market\b", raw, re.IGNORECASE
        ):
            with suppress(Exception):
                price_data = await hub.market_router.get_price(symbol)
                entry = _as_float(price_data.get("price") or price_data.get("last"))

        if not symbol or entry is None or stop is None or not targets:
            await message.answer(
                "need <b>symbol</b>, <b>entry</b>, <b>stop</b>, and at least one <b>target</b>.\n"
                "e.g. <code>SNXUSDT entry 0.028 stop 0.036 tp 0.022</code>"
            )
            return True
        payload = await hub.setup_review_service.review(
            symbol=symbol,
            timeframe=timeframe,
            entry=float(entry),
            stop=float(stop),
            targets=targets,
            direction=parsed.entities.get("direction"),
            amount_usd=parsed.entities.get("amount_usd"),
            leverage=parsed.entities.get("leverage"),
        )
        await message.answer(setup_review_template(payload, settings))
        return True

    if parsed.intent == Intent.TRADE_MATH:
        payload = _trade_math_payload(
            entry=float(parsed.entities["entry"]),
            stop=float(parsed.entities["stop"]),
            targets=[float(x) for x in parsed.entities["targets"]],
            direction=parsed.entities.get("direction"),
            margin_usd=parsed.entities.get("amount_usd"),
            leverage=parsed.entities.get("leverage"),
            symbol=parsed.entities.get("symbol"),
        )
        await message.answer(trade_math_template(payload, settings))
        return True

    if parsed.intent == Intent.RSI_SCAN:
        payload = await hub.rsi_scanner_service.scan(
            timeframe=parsed.entities.get("timeframe", "1h"),
            mode=parsed.entities.get("mode", "oversold"),
            limit=int(parsed.entities.get("limit", 10)),
            rsi_length=int(parsed.entities.get("rsi_length", 14)),
            symbol=parsed.entities.get("symbol"),
        )
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=parsed.entities.get("symbol"),
            context="rsi scan",
        )
        await message.answer(rsi_scan_template(payload))
        return True

    if parsed.intent == Intent.EMA_SCAN:
        payload = await hub.ema_scanner_service.scan(
            timeframe=parsed.entities.get("timeframe", "4h"),
            ema_length=int(parsed.entities.get("ema_length", 200)),
            mode=parsed.entities.get("mode", "closest"),
            limit=int(parsed.entities.get("limit", 10)),
        )
        lines = [payload["summary"], ""]
        for idx, row in enumerate(payload.get("items", []), start=1):
            lines.append(
                f"{idx}. {row['symbol']} price {row['price']} | EMA{payload['ema_length']} {row['ema']} | "
                f"dist {row['distance_pct']}% ({row['side']})"
            )
        if not payload.get("items"):
            lines.append("No EMA matches right now.")
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="ema scan",
        )
        await message.answer("\n".join(lines))
        return True

    if parsed.intent == Intent.CHART:
        img, meta = await hub.chart_service.render_chart(
            symbol=parsed.entities["symbol"],
            timeframe=parsed.entities.get("timeframe", "1h"),
        )
        symbol = str(parsed.entities["symbol"]).upper()
        timeframe = str(parsed.entities.get("timeframe", "1h"))
        caption = f"{symbol} {timeframe} chart."
        await _remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("instrument_id") or ""),
            updated_at=str(meta.get("updated_at") or ""),
            symbol=symbol,
            context="chart",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_{timeframe}_chart.png"),
            caption=caption,
        )
        return True

    if parsed.intent == Intent.HEATMAP:
        symbol = str(parsed.entities.get("symbol", "BTC"))
        img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
        await _remember_source_context(
            chat_id,
            source_line=str(meta.get("source_line") or ""),
            exchange=str(meta.get("exchange") or ""),
            market_kind=str(meta.get("market_kind") or ""),
            instrument_id=str(meta.get("pair") or ""),
            symbol=symbol,
            context="heatmap",
        )
        await message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol}_heatmap.png"),
            caption=(
                f"{meta['pair']} orderbook heatmap\n"
                f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
                f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
            ),
        )
        return True

    if parsed.intent == Intent.PAIR_FIND:
        payload = await hub.discovery_service.find_pair(parsed.entities["query"])
        await message.answer(pair_find_template(payload))
        return True

    if parsed.intent == Intent.PRICE_GUESS:
        payload = await hub.discovery_service.guess_by_price(
            target_price=float(parsed.entities["target_price"]),
            limit=int(parsed.entities.get("limit", 10)),
        )
        await message.answer(price_guess_template(payload))
        return True

    if parsed.intent == Intent.SMALLTALK:
        llm_reply = await _llm_fallback_reply(raw_text, settings, chat_id=chat_id)
        await message.answer(llm_reply or smalltalk_reply(settings))
        return True

    if parsed.intent == Intent.ASSET_UNSUPPORTED:
        await message.answer("Send the ticker + context and I'll give a safe fallback brief.")
        return True

    if parsed.intent == Intent.ALERT_CREATE:
        sym = parsed.entities["symbol"]
        price_val = float(parsed.entities["target_price"])
        cond = parsed.entities.get("condition", "cross")
        try:
            alert = await hub.alerts_service.create_alert(chat_id, sym, cond, price_val)
        except RuntimeError as exc:
            await message.answer(f"couldn't set that alert — {_safe_exc(exc)}")
            return True
        except Exception:  # noqa: BLE001
            logger.exception("alert_create_nlu_failed", extra={"chat_id": chat_id, "symbol": sym})
            await message.answer("alert creation failed. try again in a sec.")
            return True
        await _remember_source_context(
            chat_id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=alert.symbol,
            context="alert",
        )
        await message.answer(
            f"alert set for <b>{alert.symbol}</b> at <b>${price_val:,.2f}</b>. "
            "i'll ping you when we hit it. don't get liquidated"
        )
        return True

    if parsed.intent == Intent.ALERT_LIST:
        alerts = await hub.alerts_service.list_alerts(chat_id)
        if not alerts:
            await message.answer("No active alerts.")
        else:
            lines = ["<b>Active Alerts</b>", ""]
            for a in alerts:
                lines.append(f"<code>#{a.id}</code>  <b>{a.symbol}</b>  {a.condition}  {a.target_price}  <i>[{a.status}]</i>")
            first = alerts[0]
            await _remember_source_context(
                chat_id,
                exchange=first.source_exchange,
                market_kind=first.market_kind,
                instrument_id=first.instrument_id,
                symbol=first.symbol,
                context="alerts list",
            )
            await message.answer("\n".join(lines))
        return True

    if parsed.intent == Intent.ALERT_CLEAR:
        count = await hub.alerts_service.clear_user_alerts(chat_id)
        await message.answer(f"Cleared {count} alerts.")
        return True

    if parsed.intent == Intent.ALERT_PAUSE:
        count = await hub.alerts_service.pause_user_alerts(chat_id)
        await message.answer(f"Paused {count} alerts.")
        return True

    if parsed.intent == Intent.ALERT_RESUME:
        count = await hub.alerts_service.resume_user_alerts(chat_id)
        await message.answer(f"Resumed {count} alerts.")
        return True

    if parsed.intent == Intent.ALERT_DELETE:
        symbol = parsed.entities.get("symbol")
        if symbol:
            count = await hub.alerts_service.delete_alerts_by_symbol(chat_id, str(symbol))
            await message.answer(f"Removed {count} alert(s) for {str(symbol).upper()}.")
            return True
        ok = await hub.alerts_service.delete_alert(chat_id, int(parsed.entities["alert_id"]))
        await message.answer("Deleted." if ok else "Alert not found.")
        return True

    if parsed.intent == Intent.GIVEAWAY_JOIN:
        if not message.from_user:
            await message.answer("Could not identify user for join.")
            return True
        payload = await hub.giveaway_service.join_active(chat_id, message.from_user.id)
        await message.answer(f"Joined giveaway #{payload['giveaway_id']}. Participants: {payload['participants']}")
        return True

    if parsed.intent == Intent.GIVEAWAY_STATUS:
        payload = await hub.giveaway_service.status(chat_id)
        await message.answer(giveaway_status_template(payload))
        return True

    if parsed.intent == Intent.GIVEAWAY_START:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        duration_seconds = _parse_duration_to_seconds(str(parsed.entities.get("duration", "10m")))
        if duration_seconds is None:
            await message.answer("Duration format should look like 10m, 1h, or 1d.")
            return True
        payload = await hub.giveaway_service.start_giveaway(
            group_chat_id=chat_id,
            admin_chat_id=message.from_user.id,
            duration_seconds=duration_seconds,
            prize=str(parsed.entities.get("prize", "Prize")),
        )
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\nEnds at: {payload['end_time']}\nUsers enter with /join"
        )
        return True

    if parsed.intent == Intent.GIVEAWAY_CANCEL:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.end_giveaway(chat_id, message.from_user.id)
        if payload.get("winner_user_id"):
            await message.answer(
                f"Giveaway #{payload['giveaway_id']} ended.\nWinner: {payload['winner_user_id']}\nPrize: {payload['prize']}"
            )
        else:
            await message.answer(f"Giveaway ended with no winner. {payload.get('note')}")
        return True

    if parsed.intent == Intent.GIVEAWAY_REROLL:
        if not message.from_user:
            await message.answer("Could not identify sender.")
            return True
        payload = await hub.giveaway_service.reroll(chat_id, message.from_user.id)
        await message.answer(
            f"Reroll complete for giveaway #{payload['giveaway_id']}.\n"
            f"New winner: {payload['winner_user_id']} (prev: {payload.get('previous_winner_user_id')})"
        )
        return True

    if parsed.intent == Intent.WATCHLIST:
        payload = await hub.watchlist_service.build_watchlist(
            count=parsed.entities.get("count", 5),
            direction=parsed.entities.get("direction"),
        )
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            context="watchlist",
        )
        await message.answer(watchlist_template(payload))
        return True

    if parsed.intent == Intent.NEWS:
        payload = await hub.news_service.get_digest(
            topic=parsed.entities.get("topic"),
            mode=parsed.entities.get("mode", "crypto"),
            limit=int(parsed.entities.get("limit", 6)),
        )
        heads = payload.get("headlines") if isinstance(payload, dict) else None
        head = heads[0] if isinstance(heads, list) and heads else {}
        await _remember_source_context(
            chat_id,
            source_line=f"{head.get('source', 'news feed')} | {head.get('url', '')}".strip(),
            context="news",
        )
        await message.answer(news_template(payload))
        return True

    if parsed.intent == Intent.SCAN_WALLET:
        limiter = await hub.rate_limiter.check(
            key=f"rl:scan:{chat_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H')}",
            limit=_settings.wallet_scan_limit_per_hour,
            window_seconds=3600,
        )
        if not limiter.allowed:
            await message.answer("Wallet scan limit reached for this hour.")
            return True

        result = await hub.wallet_service.scan(parsed.entities["chain"], parsed.entities["address"], chat_id=chat_id)
        await message.answer(
            wallet_scan_template(result),
            reply_markup=wallet_actions(parsed.entities["chain"], parsed.entities["address"]),
        )
        return True

    if parsed.intent == Intent.CYCLE:
        payload = await hub.cycles_service.cycle_check()
        await message.answer(cycle_template(payload))
        return True

    if parsed.intent == Intent.TRADECHECK:
        ts = parsed.entities["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        data = {
            "symbol": parsed.entities["symbol"],
            "timeframe": parsed.entities.get("timeframe", "1h"),
            "timestamp": ts,
            "entry": float(parsed.entities["entry"]),
            "stop": float(parsed.entities["stop"]),
            "targets": [float(x) for x in parsed.entities["targets"]],
            "mode": "ambiguous",
        }
        result = await hub.trade_verify_service.verify(**data)
        await _save_trade_check(chat_id, data, result)
        await _remember_source_context(
            chat_id,
            source_line=str(result.get("source_line") or ""),
            symbol=data["symbol"],
            context="trade check",
        )
        await message.answer(trade_verification_template(result))
        return True

    if parsed.intent == Intent.CORRELATION:
        payload = await hub.correlation_service.check_following(
            parsed.entities["symbol"],
            benchmark=parsed.entities.get("benchmark", "BTC"),
        )
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            symbol=parsed.entities["symbol"],
            context="correlation",
        )
        await message.answer(correlation_template(payload))
        return True

    if parsed.intent == Intent.SETTINGS:
        st = await hub.user_service.get_settings(chat_id)
        await message.answer(settings_text(st), reply_markup=settings_menu(st))
        return True

    if parsed.intent == Intent.HELP:
        await message.answer(help_text())
        return True

    return False


@router.message(Command("start"))
async def start_cmd(message: Message) -> None:
    hub = _require_hub()
    await hub.user_service.ensure_user(message.chat.id)
    name = message.from_user.first_name if message.from_user else "fren"
    await message.answer(
        f"gm <b>{name}</b> 👋\n\n"
        "i'm <b>ghost</b> — your on-chain trading assistant. i live in the market 24/7 so you don't have to.\n\n"
        "try something like:\n"
        "· <code>BTC 4h</code> — full technical analysis\n"
        "· <code>ping me when ETH hits 2000</code> — price alert\n"
        "· <code>coins to watch</code> — top movers watchlist\n"
        "· <code>why is BTC pumping</code> — live market read\n\n"
        "<i>or tap a button below to get started.</i>",
        reply_markup=smart_action_menu(),
    )


@router.message(Command("help"))
async def help_cmd(message: Message) -> None:
    await message.answer(help_text(), reply_markup=command_center_menu())


@router.message(Command("admins"))
async def admins_cmd(message: Message) -> None:
    admin_ids = sorted(set(_settings.admin_ids_list()))
    if not admin_ids:
        await message.answer("no admin IDs configured.")
        return
    lines = ["<b>bot admins</b>\n"]
    lines.extend(f"· <code>{admin_id}</code>" for admin_id in admin_ids)
    await message.answer("\n".join(lines))


@router.message(Command("id"))
async def id_cmd(message: Message) -> None:
    if not message.from_user:
        await message.answer("couldn't read your user id from this update.")
        return
    await message.answer(
        f"your user id  <code>{message.from_user.id}</code>\n"
        f"this chat id  <code>{message.chat.id}</code>"
    )


@router.message(Command("settings"))
async def settings_cmd(message: Message) -> None:
    hub = _require_hub()
    settings = await hub.user_service.get_settings(message.chat.id)
    await message.answer(settings_text(settings), reply_markup=settings_menu(settings))


@router.message(Command("alpha"))
async def alpha_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick an analysis shortcut or tap Custom.", reply_markup=alpha_quick_menu())
        return
    tokens = text.split()
    if len(tokens) == 1:
        text = f"watch {tokens[0]}"
    await _dispatch_command_text(message, text)


@router.message(Command("watch"))
async def watch_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a watch shortcut or tap Custom.", reply_markup=watch_quick_menu())
        return
    await _dispatch_command_text(message, f"watch {text}")


@router.message(Command("price"))
async def price_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("send symbol to get price.\nexample: <code>/price SOL</code> or <code>/price BTC 1h</code>")
        return
    await _dispatch_command_text(message, f"watch {text}")


@router.message(Command("chart"))
async def chart_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a chart shortcut or tap Custom.", reply_markup=chart_quick_menu())
        return
    await _dispatch_command_text(message, f"chart {text}")


@router.message(Command("heatmap"))
async def heatmap_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Pick a symbol for heatmap or tap Custom.", reply_markup=heatmap_quick_menu())
        return
    await _dispatch_command_text(message, f"heatmap {text}")


@router.message(Command("rsi"))
async def rsi_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    parts = raw.split()
    if len(parts) < 3:
        await message.answer("Pick an RSI scanner preset or tap Custom.", reply_markup=rsi_quick_menu())
        return
    timeframe = parts[1].lower()
    mode = parts[2].lower()
    if mode not in {"overbought", "oversold"}:
        await message.answer("Mode must be `overbought` or `oversold`.")
        return
    top_n = max(1, min(_as_int(parts[3], 10), 20)) if len(parts) >= 4 else 10
    rsi_len = max(2, min(_as_int(parts[4], 14), 50)) if len(parts) >= 5 else 14
    await _dispatch_command_text(message, f"rsi top {top_n} {timeframe} {mode} rsi{rsi_len}")


@router.message(Command("ema"))
async def ema_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    parts = raw.split()
    if len(parts) < 3:
        await message.answer("Pick an EMA scanner preset or tap Custom.", reply_markup=ema_quick_menu())
        return
    ema_len = max(2, min(_as_int(parts[1], 200), 500))
    timeframe = parts[2].lower()
    top_n = max(1, min(_as_int(parts[3], 10), 20)) if len(parts) >= 4 else 10
    await _dispatch_command_text(message, f"ema {ema_len} {timeframe} top {top_n}")


@router.message(Command("watchlist"))
async def watchlist_cmd(message: Message) -> None:
    hub = _require_hub()
    n_match = re.search(r"/watchlist\s+(\d+)", message.text or "")
    n = int(n_match.group(1)) if n_match else 5
    direction = None
    if re.search(r"\blong\b", message.text or "", re.IGNORECASE):
        direction = "long"
    elif re.search(r"\bshort\b", message.text or "", re.IGNORECASE):
        direction = "short"
    payload = await hub.watchlist_service.build_watchlist(count=max(1, min(n, 20)), direction=direction)
    await _remember_source_context(
        message.chat.id,
        source_line=str(payload.get("source_line") or ""),
        context="watchlist",
    )
    await message.answer(watchlist_template(payload))


@router.message(Command("news"))
async def news_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()
    topic: str | None = None
    mode = "crypto"
    limit = 6
    parts = text.split()
    if len(parts) == 1:
        await message.answer("Pick a news mode.", reply_markup=news_quick_menu())
        return
    if len(parts) > 1:
        raw_topic = parts[1].strip()
        if raw_topic.isdigit():
            limit = max(3, min(int(raw_topic), 10))
        else:
            topic = raw_topic
    if len(parts) > 2 and parts[2].isdigit():
        limit = max(3, min(int(parts[2]), 10))

    if topic:
        lowered = topic.lower().strip()
        if lowered in {"crypto", "openai", "cpi", "fomc"}:
            topic = lowered
        if re.search(r"\b(openai|chatgpt|gpt|codex)\b", lowered):
            mode = "openai"
            topic = "openai"
        elif re.search(r"\b(cpi|inflation)\b", lowered):
            mode = "macro"
            topic = "cpi"
        elif re.search(r"\b(fomc|fed|powell|macro|rates?)\b", lowered):
            mode = "macro"
            topic = "macro"
    payload = await hub.news_service.get_digest(topic=topic, mode=mode, limit=limit)
    heads = payload.get("headlines") if isinstance(payload, dict) else None
    head = heads[0] if isinstance(heads, list) and heads else {}
    await _remember_source_context(
        message.chat.id,
        source_line=f"{head.get('source', 'news feed')} | {head.get('url', '')}".strip(),
        context="news",
    )
    await message.answer(news_template(payload))


@router.message(Command("cycle"))
async def cycle_cmd(message: Message) -> None:
    hub = _require_hub()
    payload = await hub.cycles_service.cycle_check()
    await message.answer(cycle_template(payload))


@router.message(Command("scan"))
async def scan_cmd(message: Message) -> None:
    hub = _require_hub()
    text = message.text or ""
    m = re.search(r"/scan\s+(solana|tron)\s+([A-Za-z0-9]+)", text, re.IGNORECASE)
    if not m:
        await message.answer("Pick chain first, then paste address.", reply_markup=scan_quick_menu())
        return

    limiter = await hub.rate_limiter.check(
        key=f"rl:scan:{message.chat.id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H')}",
        limit=_settings.wallet_scan_limit_per_hour,
        window_seconds=3600,
    )
    if not limiter.allowed:
        await message.answer("Wallet scan limit reached for this hour.")
        return

    chain, address = m.group(1).lower(), m.group(2)
    result = await hub.wallet_service.scan(chain, address, chat_id=message.chat.id)
    await message.answer(wallet_scan_template(result), reply_markup=wallet_actions(chain, address))


@router.message(Command("alert"))
async def alert_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()

    if text.startswith("/alert list"):
        alerts = await hub.alerts_service.list_alerts(message.chat.id)
        if not alerts:
            await message.answer("No active alerts.")
            return
        rows = ["<b>Active Alerts</b>", ""]
        for a in alerts:
            rows.append(f"<code>#{a.id}</code>  <b>{a.symbol}</b>  {a.condition}  {a.target_price}  <i>[{a.status}]</i>")
        first = alerts[0]
        await _remember_source_context(
            message.chat.id,
            exchange=first.source_exchange,
            market_kind=first.market_kind,
            instrument_id=first.instrument_id,
            symbol=first.symbol,
            context="alerts list",
        )
        await message.answer("\n".join(rows))
        return

    if text.startswith("/alert clear"):
        count = await hub.alerts_service.clear_user_alerts(message.chat.id)
        await message.answer(f"Cleared {count} alerts.")
        return

    if text.startswith("/alert pause"):
        count = await hub.alerts_service.pause_user_alerts(message.chat.id)
        await message.answer(f"Paused {count} alerts.")
        return

    if text.startswith("/alert resume"):
        count = await hub.alerts_service.resume_user_alerts(message.chat.id)
        await message.answer(f"Resumed {count} alerts.")
        return

    d_match = re.search(r"/alert\s+delete\s+(\d+)", text)
    if d_match:
        ok = await hub.alerts_service.delete_alert(message.chat.id, int(d_match.group(1)))
        await message.answer("Deleted." if ok else "Alert not found.")
        return

    a_match = re.search(r"/alert\s+add\s+([A-Za-z0-9]+)\s+(above|below|cross)\s+([0-9.]+)", text, re.IGNORECASE)
    if a_match:
        symbol, condition, price = a_match.group(1).upper(), a_match.group(2).lower(), float(a_match.group(3))
        alert = await hub.alerts_service.create_alert(message.chat.id, symbol, condition, price, source="command")
        await _remember_source_context(
            message.chat.id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        await message.answer(
            f"alert set for <b>{symbol}</b> at <b>{price}</b>. "
            "i'll ping you when we hit it. don't get liquidated"
        )
        return

    simple_match = re.search(
        r"^/alert\s+([A-Za-z0-9$]{2,20})\s+([0-9]+(?:\.[0-9]+)?)(?:\s+(above|below|cross))?\s*$",
        text,
        re.IGNORECASE,
    )
    if simple_match:
        symbol = simple_match.group(1).upper().lstrip("$")
        price = float(simple_match.group(2))
        condition = (simple_match.group(3) or "cross").lower()
        alert = await hub.alerts_service.create_alert(message.chat.id, symbol, condition, price, source="command")
        await _remember_source_context(
            message.chat.id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        await message.answer(
            f"alert set for <b>{symbol}</b> at <b>{price}</b>. "
            "i'll ping you when we hit it. don't get liquidated"
        )
        return

    await message.answer("Pick an alert action.", reply_markup=alert_quick_menu())


@router.message(Command("alerts"))
async def alerts_cmd(message: Message) -> None:
    hub = _require_hub()
    try:
        alerts = await hub.alerts_service.list_alerts(message.chat.id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("alerts_list_failed", extra={"event": "alerts_list_failed", "error": str(exc), "chat_id": message.chat.id})
        await message.answer("Alerts are temporarily unavailable. Try again in a few seconds.")
        return
    if not alerts:
        await message.answer("No active alerts.")
        return
    rows = ["<b>Active Alerts</b>", ""]
    for a in alerts:
        rows.append(f"<code>#{a.id}</code>  <b>{a.symbol}</b>  {a.condition}  {a.target_price}  <i>[{a.status}]</i>")
    first = alerts[0]
    await _remember_source_context(
        message.chat.id,
        exchange=first.source_exchange,
        market_kind=first.market_kind,
        instrument_id=first.instrument_id,
        symbol=first.symbol,
        context="alerts list",
    )
    await message.answer("\n".join(rows))


@router.message(Command("alertdel"))
async def alertdel_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()
    m = re.search(r"^/alertdel\s+(\d+)\s*$", text, re.IGNORECASE)
    if not m:
        try:
            alerts = await hub.alerts_service.list_alerts(message.chat.id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("alertdel_list_failed", extra={"event": "alertdel_list_failed", "error": str(exc), "chat_id": message.chat.id})
            await message.answer("Alerts are temporarily unavailable. Try again in a few seconds.")
            return
        if not alerts:
            await message.answer("No active alerts.", reply_markup=alert_quick_menu())
            return
        options = [(f"Delete #{a.id}", f"cmd:alertdel:{a.id}") for a in alerts[:8]]
        await message.answer("Tap an alert to delete.", reply_markup=simple_followup(options))
        return
    try:
        ok = await hub.alerts_service.delete_alert(message.chat.id, int(m.group(1)))
    except Exception as exc:  # noqa: BLE001
        logger.exception("alertdel_failed", extra={"event": "alertdel_failed", "error": str(exc), "chat_id": message.chat.id})
        await message.answer("Delete failed on my side. Try again in a few seconds.")
        return
    await message.answer("Deleted." if ok else "Alert not found.")


@router.message(Command("alertclear"))
async def alertclear_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()
    m = re.search(r"^/alertclear\s+([A-Za-z0-9$]{2,20})\s*$", text, re.IGNORECASE)
    if m:
        symbol = m.group(1).upper().lstrip("$")
        count = await hub.alerts_service.delete_alerts_by_symbol(message.chat.id, symbol)
        await message.answer(f"Cleared {count} alerts for {symbol}.")
        return
    await message.answer("Pick clear action.", reply_markup=simple_followup([("Clear all alerts", "cmd:alert:clear"), ("Clear by symbol", "cmd:alert:clear_symbol")]))


@router.message(Command("tradecheck"))
async def tradecheck_cmd(message: Message) -> None:
    await _wizard_set(message.chat.id, {"step": "symbol", "data": {}})
    await message.answer("Tradecheck wizard: send symbol (e.g., ETH).")


@router.message(Command("findpair"))
async def findpair_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    query = args.strip()
    if not query:
        await message.answer("Pick find mode.", reply_markup=findpair_quick_menu())
        return
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", query):
        await _dispatch_command_text(message, f"coin around {query}")
        return
    await _dispatch_command_text(message, f"find pair {query}")


@router.message(Command("setup"))
async def setup_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    if not args.strip():
        await message.answer("Choose setup input mode.", reply_markup=setup_quick_menu())
        return
    await _dispatch_command_text(message, args.strip())


@router.message(Command("margin"))
async def margin_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Choose setup input mode.", reply_markup=setup_quick_menu())
        return
    await _dispatch_command_text(message, text)


@router.message(Command("pnl"))
async def pnl_cmd(message: Message) -> None:
    raw = (message.text or "").strip()
    args = raw.split(maxsplit=1)[1] if len(raw.split(maxsplit=1)) > 1 else ""
    text = args.strip()
    if not text:
        await message.answer("Choose setup input mode.", reply_markup=setup_quick_menu())
        return
    await _dispatch_command_text(message, text)


@router.message(Command("join"))
async def join_cmd(message: Message) -> None:
    hub = _require_hub()
    if not message.from_user:
        await message.answer("Could not identify user for join.")
        return
    try:
        payload = await hub.giveaway_service.join_active(message.chat.id, message.from_user.id)
    except Exception as exc:  # noqa: BLE001
        await message.answer(f"couldn't join: {_safe_exc(exc)}")
        return
    await message.answer(f"you're in 🎉 giveaway <b>#{payload['giveaway_id']}</b> — participants: <b>{payload['participants']}</b>")


@router.message(Command("giveaway"))
async def giveaway_cmd(message: Message) -> None:
    hub = _require_hub()
    text = (message.text or "").strip()
    if not message.from_user:
        await message.answer("Could not identify sender.")
        return

    if re.search(r"^/giveaway\s+status\b", text, flags=re.IGNORECASE):
        payload = await hub.giveaway_service.status(message.chat.id)
        await message.answer(giveaway_status_template(payload))
        return

    if re.search(r"^/giveaway\s+join\b", text, flags=re.IGNORECASE):
        try:
            payload = await hub.giveaway_service.join_active(message.chat.id, message.from_user.id)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"couldn't join: {_safe_exc(exc)}")
            return
        await message.answer(f"you're in 🎉 giveaway <b>#{payload['giveaway_id']}</b> — participants: <b>{payload['participants']}</b>")
        return

    if re.search(r"^/giveaway\s+end\b", text, flags=re.IGNORECASE):
        try:
            payload = await hub.giveaway_service.end_giveaway(message.chat.id, message.from_user.id)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"couldn't end giveaway: {_safe_exc(exc)}")
            return
        if payload.get("winner_user_id"):
            await message.answer(
                f"🏆 giveaway <b>#{payload.get('giveaway_id')}</b> closed.\n"
                f"winner: <code>{payload.get('winner_user_id')}</code>\n"
                f"prize: <b>{payload.get('prize', '—')}</b>"
            )
        else:
            await message.answer(f"giveaway ended with no winner. {payload.get('note')}")
        return

    if re.search(r"^/giveaway\s+reroll\b", text, flags=re.IGNORECASE):
        try:
            payload = await hub.giveaway_service.reroll(message.chat.id, message.from_user.id)
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"reroll failed: {_safe_exc(exc)}")
            return
        await message.answer(
            f"🔄 reroll done for giveaway <b>#{payload.get('giveaway_id')}</b>\n"
            f"new winner: <code>{payload.get('winner_user_id')}</code>\n"
            f"prev: <code>{payload.get('previous_winner_user_id', '—')}</code>"
        )
        return

    start_match = re.search(r"^/giveaway\s+start\s+(\S+)(?:\s+(.+))?$", text, flags=re.IGNORECASE)
    if start_match:
        duration_raw = start_match.group(1)
        duration_seconds = _parse_duration_to_seconds(duration_raw)
        if duration_seconds is None:
            await message.answer("Invalid duration. Example: /giveaway start 10m prize \"50 USDT\"")
            return
        tail = (start_match.group(2) or "").strip()
        winners_match = re.search(r"\bwinners?\s*=?\s*(\d+)\b", tail, flags=re.IGNORECASE)
        winners_requested = int(winners_match.group(1)) if winners_match else 1
        tail = re.sub(r"\bwinners?\s*=?\s*\d+\b", "", tail, flags=re.IGNORECASE).strip()
        tail = re.sub(r"^\s*prize\s+", "", tail, flags=re.IGNORECASE).strip()
        prize = (tail or "Prize").strip("'\"")
        try:
            payload = await hub.giveaway_service.start_giveaway(
                group_chat_id=message.chat.id,
                admin_chat_id=message.from_user.id,
                duration_seconds=duration_seconds,
                prize=prize,
            )
        except Exception as exc:  # noqa: BLE001
            await message.answer(f"couldn't start giveaway: {_safe_exc(exc)}")
            return
        note = ""
        if winners_requested > 1:
            note = "\nNote: multi-winner draw will run as sequential rerolls after the first winner."
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\n"
            f"Ends at: {payload['end_time']}\nUsers enter with /join or /giveaway join{note}"
        )
        return

    await message.answer("Pick giveaway action.", reply_markup=giveaway_menu(is_admin=hub.giveaway_service.is_admin(message.from_user.id)))


@router.callback_query(F.data.startswith("cmd:"))
async def command_menu_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    hub = _require_hub()
    chat_id = callback.message.chat.id
    data = callback.data or ""
    parts = data.split(":")
    if len(parts) < 2:
        await callback.answer()
        return

    def _menu_for(name: str):
        mapping = {
            "alpha": ("Pick analysis shortcut.", alpha_quick_menu()),
            "watch": ("Pick watch shortcut.", watch_quick_menu()),
            "chart": ("Pick chart shortcut.", chart_quick_menu()),
            "heatmap": ("Pick heatmap symbol.", heatmap_quick_menu()),
            "rsi": ("Pick RSI scanner preset.", rsi_quick_menu()),
            "ema": ("Pick EMA scanner preset.", ema_quick_menu()),
            "news": ("Pick news mode.", news_quick_menu()),
            "alert": ("Pick alert action.", alert_quick_menu()),
            "findpair": ("Pick find mode.", findpair_quick_menu()),
            "setup": ("Choose setup input mode.", setup_quick_menu()),
            "scan": ("Pick chain first.", scan_quick_menu()),
            "giveaway": ("Pick giveaway action.", giveaway_menu(is_admin=hub.giveaway_service.is_admin(callback.from_user.id))),
        }
        return mapping.get(name)

    if parts[1] == "menu":
        menu = _menu_for(parts[2] if len(parts) > 2 else "")
        if menu:
            await callback.message.answer(menu[0], reply_markup=menu[1])
        await callback.answer()
        return

    async def _dispatch_with_typing(synthetic_text: str) -> None:
        async def _run() -> None:
            await _dispatch_command_text(callback.message, synthetic_text)
            await callback.answer()

        await _run_with_typing_lock(callback.bot, chat_id, _run)

    action = parts[1]
    if action == "alpha":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": ""})
            await callback.message.answer("Send symbol and optional tf, e.g. `SOL 4h`.")
            await callback.answer()
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(f"{parts[2]} {parts[3]}")
            return
    if action == "watch":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "watch "})
            await callback.message.answer("Send symbol and optional tf, e.g. `BTC 1h`.")
            await callback.answer()
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(f"watch {parts[2]} {parts[3]}")
            return
    if action == "chart":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "chart "})
            await callback.message.answer("Send symbol and optional tf, e.g. `ETH 4h`.")
            await callback.answer()
            return
        if len(parts) >= 4:
            await _dispatch_with_typing(f"chart {parts[2]} {parts[3]}")
            return
    if action == "heatmap":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "heatmap "})
            await callback.message.answer("Send symbol, e.g. `BTC`.")
            await callback.answer()
            return
        if len(parts) >= 3:
            await _dispatch_with_typing(f"heatmap {parts[2]}")
            return
    if action == "rsi":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "rsi "})
            await callback.message.answer("Send format: `1h oversold top 10 rsi14`.")
            await callback.answer()
            return
        if len(parts) >= 6:
            await _dispatch_with_typing(f"rsi top {parts[4]} {parts[2]} {parts[3]} rsi{parts[5]}")
            return
    if action == "ema":
        if len(parts) >= 3 and parts[2] == "custom":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "ema "})
            await callback.message.answer("Send format: `200 4h top 10`.")
            await callback.answer()
            return
        if len(parts) >= 5:
            await _dispatch_with_typing(f"ema {parts[2]} {parts[3]} top {parts[4]}")
            return
    if action == "news" and len(parts) >= 4:
        await _dispatch_with_typing(f"news {parts[2]} {parts[3]}")
        return
    if action == "alert":
        if len(parts) >= 3 and parts[2] == "create":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "alert "})
            await callback.message.answer(
                "send me the alert details:\n\n"
                "<code>SOL 100 above</code>\n"
                "<code>BTC 66000 below</code>\n"
                "<code>ETH 3200</code>  ← defaults to cross\n\n"
                "<i>format: symbol  price  [above | below | cross]</i>"
            )
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "list":
            await _dispatch_with_typing("list my alerts")
            return
        if len(parts) >= 3 and parts[2] == "clear":
            await _dispatch_with_typing("clear my alerts")
            return
        if len(parts) >= 3 and parts[2] == "pause":
            await _dispatch_with_typing("pause alerts")
            return
        if len(parts) >= 3 and parts[2] == "resume":
            await _dispatch_with_typing("resume alerts")
            return
        if len(parts) >= 3 and parts[2] == "delete":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "delete alert "})
            await callback.message.answer("Send alert id, e.g. `12`.")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "clear_symbol":
            await _cmd_wizard_set(chat_id, {"step": "alert_clear_symbol"})
            await callback.message.answer("Send symbol to clear, e.g. `SOL`.")
            await callback.answer()
            return
    if action == "findpair":
        if len(parts) >= 3 and parts[2] == "price":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "coin around "})
            await callback.message.answer("Send target price, e.g. `0.155`.")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "query":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": "find pair "})
            await callback.message.answer("Send name/ticker/context, e.g. `xion`.")
            await callback.answer()
            return
    if action == "setup":
        if len(parts) >= 3 and parts[2] == "wizard":
            await _wizard_set(chat_id, {"step": "symbol", "data": {}})
            await callback.message.answer("Tradecheck wizard: send symbol (e.g., ETH).")
            await callback.answer()
            return
        if len(parts) >= 3 and parts[2] == "freeform":
            await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": ""})
            await callback.message.answer("Paste setup text, e.g. `entry 2100 stop 2165 targets 2043 2027 1991`.")
            await callback.answer()
            return
    if action == "scan" and len(parts) >= 3:
        chain = "solana" if parts[2] == "solana" else "tron"
        await _cmd_wizard_set(chat_id, {"step": "dispatch_text", "prefix": f"scan {chain} "})
        await callback.message.answer(f"Paste {chain} address.")
        await callback.answer()
        return
    if action == "alertdel" and len(parts) >= 3:
        await _dispatch_with_typing(f"delete alert {parts[2]}")
        return

    await callback.answer()


@router.callback_query(F.data.startswith("gw:"))
async def giveaway_menu_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return
    hub = _require_hub()
    chat_id = callback.message.chat.id
    data = callback.data or ""
    parts = data.split(":")
    action = parts[1] if len(parts) > 1 else ""
    user_id = callback.from_user.id if callback.from_user else None

    async def _run_and_answer(runner) -> None:
        async def _run() -> None:
            await runner()
            await callback.answer()

        await _run_with_typing_lock(callback.bot, chat_id, _run)

    if action == "start":
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return
        await callback.message.answer("Pick duration.", reply_markup=giveaway_duration_menu())
        await callback.answer()
        return
    if action == "dur" and len(parts) >= 3:
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return
        duration_seconds = max(30, _as_int(parts[2], 600))
        await callback.message.answer("Pick number of winners.", reply_markup=giveaway_winners_menu(duration_seconds))
        await callback.answer()
        return
    if action == "win" and len(parts) >= 4:
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return
        duration_seconds = max(30, _as_int(parts[2], 600))
        winners = max(1, min(_as_int(parts[3], 1), 5))
        await _cmd_wizard_set(chat_id, {"step": "giveaway_prize", "duration_seconds": duration_seconds, "winners": winners})
        await callback.message.answer("Send giveaway prize text, e.g. `50 USDT`.")
        await callback.answer()
        return
    if action == "join":
        if not user_id:
            await callback.answer("No user", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await hub.giveaway_service.join_active(chat_id, user_id)
            except Exception as exc:  # noqa: BLE001
                await callback.message.answer(f"couldn't join: {exc}")
                return
            gw_id = payload.get("giveaway_id", "?")
            participants = payload.get("participants", "?")
            await callback.message.answer(
                f"you're in 🎉\n\n"
                f"giveaway <b>#{gw_id}</b>\n"
                f"participants so far: <b>{participants}</b>\n\n"
                f"<i>good luck, fren.</i>"
            )

        await _run_and_answer(_runner)
        return
    if action == "status":
        async def _runner() -> None:
            payload = await hub.giveaway_service.status(chat_id)
            await callback.message.answer(giveaway_status_template(payload))

        await _run_and_answer(_runner)
        return
    if action == "end":
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await hub.giveaway_service.end_giveaway(chat_id, user_id)
            except Exception as exc:  # noqa: BLE001
                await callback.message.answer(f"couldn't end giveaway: {exc}")
                return
            if payload.get("winner_user_id"):
                await callback.message.answer(
                    f"🏆 giveaway <b>#{payload.get('giveaway_id')}</b> closed.\n\n"
                    f"winner: <code>{payload.get('winner_user_id')}</code>\n"
                    f"prize: <b>{payload.get('prize', '—')}</b>"
                )
            else:
                note = payload.get("note") or "no participants"
                await callback.message.answer(f"giveaway ended with no winner. {note}")

        await _run_and_answer(_runner)
        return
    if action == "reroll":
        if not hub.giveaway_service.is_admin(user_id or 0):
            await callback.answer("Admin only", show_alert=True)
            return

        async def _runner() -> None:
            try:
                payload = await hub.giveaway_service.reroll(chat_id, user_id)
            except Exception as exc:  # noqa: BLE001
                await callback.message.answer(f"reroll failed: {exc}")
                return
            await callback.message.answer(
                f"🔄 reroll done for giveaway <b>#{payload.get('giveaway_id')}</b>\n\n"
                f"new winner: <code>{payload.get('winner_user_id')}</code>\n"
                f"prev winner: <code>{payload.get('previous_winner_user_id', '—')}</code>"
            )

        await _run_and_answer(_runner)
        return

    await callback.answer()


@router.callback_query(F.data.startswith("settings:"))
async def settings_callbacks(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    hub = _require_hub()
    data = callback.data or ""
    chat_id = callback.message.chat.id

    if data == "settings:toggle:anon_mode":
        cur = await hub.user_service.get_settings(chat_id)
        new_settings = await hub.user_service.update_settings(chat_id, {"anon_mode": not cur.get("anon_mode", True)})
    elif data == "settings:toggle:formal_mode":
        cur = await hub.user_service.get_settings(chat_id)
        new_settings = await hub.user_service.update_settings(chat_id, {"formal_mode": not cur.get("formal_mode", False)})
    elif data.startswith("settings:set:"):
        _, _, key, value = data.split(":", 3)
        new_settings = await hub.user_service.update_settings(chat_id, {key: value})
    else:
        await callback.answer("Unknown settings action", show_alert=True)
        return

    await callback.message.edit_text(settings_text(new_settings), reply_markup=settings_menu(new_settings))
    await callback.answer("Updated")


@router.callback_query(F.data.startswith("set_alert:"))
async def set_alert_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    symbol = (callback.data or "").split(":", 1)[1]
    await _set_pending_alert(callback.message.chat.id, symbol)
    await callback.message.answer(
        f"send me the target price for <b>{symbol}</b>.\n"
        f"e.g. <code>{symbol} 100</code> or <code>alert {symbol} 100 above</code>"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("show_levels:"))
async def show_levels_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    hub = _require_hub()
    symbol = (callback.data or "").split(":", 1)[1]
    payload = await hub.cache.get_json(f"last_analysis:{callback.message.chat.id}:{symbol}")
    if not payload:
        await callback.answer("No cached levels — run a fresh analysis first.", show_alert=True)
        return
    entry = payload.get("entry", "—")
    tp1 = payload.get("tp1", "—")
    tp2 = payload.get("tp2", "—")
    sl = payload.get("sl", "—")
    await callback.message.answer(
        f"<b>{symbol}</b> key levels\n\n"
        f"entry    <code>{entry}</code>\n"
        f"target 1  <code>{tp1}</code>\n"
        f"target 2  <code>{tp2}</code>\n"
        f"stop      <code>{sl}</code>"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("why:"))
async def why_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    hub = _require_hub()
    symbol = (callback.data or "").split(":", 1)[1]
    payload = await hub.cache.get_json(f"last_analysis:{callback.message.chat.id}:{symbol}")
    if not payload:
        await callback.answer("No context saved — run a fresh analysis first.", show_alert=True)
        return
    bullets = payload.get("why", [])
    if bullets:
        bullet_lines = "\n".join(f"· {w}" for w in bullets)
        text = f"<b>why {symbol}</b>\n\n{bullet_lines}"
    else:
        summary = payload.get("summary", "")
        text = f"<b>why {symbol}</b>\n\n{summary or 'no reasoning available for this setup.'}"
    await callback.message.answer(text)
    await callback.answer()


@router.callback_query(F.data.startswith("refresh:"))
async def refresh_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":", 1)[1]
        settings = await hub.user_service.get_settings(chat_id)
        payload = await hub.analysis_service.analyze(
            symbol,
            timeframes=_analysis_timeframes_from_settings(settings),
            ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=False,
            include_news=False,
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(callback.message, symbol, analysis_text)
        await callback.answer("Refreshed")

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("details:"))
async def details_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":", 1)[1]
        settings = await hub.user_service.get_settings(chat_id)
        payload = await hub.analysis_service.analyze(
            symbol,
            timeframes=_analysis_timeframes_from_settings(settings),
            ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=True,
            include_news=True,
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
            detailed=True,
        )
        await _send_ghost_analysis(callback.message, symbol, analysis_text)
        await callback.answer("Detailed mode")

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("derivatives:"))
async def derivatives_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":", 1)[1]
        payload = await hub.analysis_service.deriv_adapter.get_funding_and_oi(symbol)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("source_line") or ""),
            exchange=str(payload.get("source") or ""),
            market_kind="perp",
            symbol=symbol,
            context="derivatives",
        )
        funding = payload.get("funding_rate")
        oi = payload.get("open_interest")
        source = payload.get("source") or payload.get("source_line") or "live"
        funding_str = f"{float(funding)*100:.4f}%" if funding is not None else "n/a"
        oi_str = f"${float(oi)/1_000_000:.2f}B" if oi is not None else "n/a"
        await callback.message.answer(
            f"<b>{symbol}</b> derivatives\n\n"
            f"funding rate  <code>{funding_str}</code>\n"
            f"open interest <code>{oi_str}</code>\n\n"
            f"<i>source: {source}</i>"
        )
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("catalysts:"))
async def catalysts_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":", 1)[1]
        headlines = await hub.news_service.get_asset_headlines(symbol, limit=3)
        if not headlines:
            await callback.message.answer(f"no fresh catalysts for <b>{symbol}</b> right now. check back later.")
            await callback.answer()
            return
        lines = [f"<b>{symbol} catalysts</b>\n"]
        for item in headlines[:3]:
            title = item.get("title", "")
            url = item.get("url", "")
            source = item.get("source", "")
            if url:
                lines.append(f'· <a href="{url}">{title}</a>')
            else:
                lines.append(f"· {title}")
            if source:
                lines.append(f"  <i>{source}</i>")
            lines.append("")
        await callback.message.answer("\n".join(lines).strip(), disable_web_page_preview=True)
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("backtest:"))
async def backtest_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    symbol = (callback.data or "").split(":", 1)[1]
    await callback.message.answer(
        f"drop your <b>{symbol}</b> trade details and i'll check it.\n\n"
        f"format: <code>{symbol} entry 2100 stop 2060 targets 2140 2180 2220 timeframe 1h</code>\n\n"
        "<i>or just paste it in natural language — i'll figure it out.</i>"
    )
    await callback.answer()


@router.callback_query(F.data.startswith("save_wallet:"))
async def save_wallet_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, chain, address = (callback.data or "").split(":", 2)
        await hub.wallet_service.scan(chain, address, chat_id=chat_id, save=True)
        await callback.answer("Wallet saved", show_alert=True)

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:analysis:"))
async def quick_analysis_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        symbol = (callback.data or "").split(":")[-1]
        settings = await hub.user_service.get_settings(chat_id)
        payload = await hub.analysis_service.analyze(
            symbol,
            timeframes=_analysis_timeframes_from_settings(settings),
            ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=False,
            include_news=False,
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol,
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(callback.message, symbol, analysis_text)
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:analysis_tf:"))
async def quick_analysis_tf_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, symbol, timeframe = (callback.data or "").split(":", 3)
        settings = await hub.user_service.get_settings(chat_id)
        payload = await hub.analysis_service.analyze(
            symbol.upper(),
            timeframe=timeframe,
            timeframes=[timeframe],
            ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
            rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
            include_derivatives=False,
            include_news=False,
        )
        await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol.upper()}", payload, ttl=1800)
        await _remember_analysis_context(chat_id, symbol.upper(), payload.get("side"), payload)
        analysis_text = await _render_analysis_text(
            payload=payload,
            symbol=symbol.upper(),
            direction=payload.get("side"),
            settings=settings,
            chat_id=chat_id,
        )
        await _send_ghost_analysis(callback.message, symbol.upper(), analysis_text)
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:chart:"))
async def quick_chart_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, symbol, timeframe = (callback.data or "").split(":", 3)
        img, _ = await hub.chart_service.render_chart(symbol=symbol.upper(), timeframe=timeframe)
        await callback.message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol.upper()}-{timeframe}.png"),
            caption=f"{symbol.upper()} {timeframe} chart.",
        )
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:heatmap:"))
async def quick_heatmap_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, symbol = (callback.data or "").split(":", 2)
        img, _ = await hub.heatmap_service.render(symbol=symbol.upper())
        await callback.message.answer_photo(
            BufferedInputFile(img, filename=f"{symbol.upper()}-heatmap.png"),
            caption=f"{symbol.upper()} order-book heatmap.",
        )
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:rsi:"))
async def quick_rsi_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, mode, timeframe, limit_raw = (callback.data or "").split(":", 4)
        limit = max(1, min(_as_int(limit_raw, 5), 20))
        payload = await hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode="overbought" if mode == "overbought" else "oversold",
            limit=limit,
            rsi_length=14,
            symbol=None,
        )
        await callback.message.answer(rsi_scan_template(payload))
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("quick:news:"))
async def quick_news_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        _, _, mode = (callback.data or "").split(":", 2)
        mode_norm = "openai" if mode == "openai" else "crypto"
        topic = "openai" if mode_norm == "openai" else "crypto"
        payload = await hub.news_service.get_digest(topic=topic, mode=mode_norm, limit=6)
        await callback.message.answer(news_template(payload))
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("define:"))
async def define_easter_egg_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        settings = await hub.user_service.get_settings(chat_id)
        parts = (callback.data or "").split(":")
        action = parts[1] if len(parts) > 1 else ""

        if action == "analyze":
            timeframe = parts[2] if len(parts) > 2 else "1h"
            symbol = "DEFINE"
            payload = await hub.analysis_service.analyze(
                symbol,
                timeframe=timeframe,
                timeframes=[timeframe],
                ema_periods=_parse_int_list(settings.get("preferred_ema_periods", [20, 50, 200]), [20, 50, 200]),
                rsi_periods=_parse_int_list(settings.get("preferred_rsi_periods", [14]), [14]),
                include_derivatives=True,
                include_news=True,
            )
            await hub.cache.set_json(f"last_analysis:{chat_id}:{symbol}", payload, ttl=1800)
            await _remember_analysis_context(chat_id, symbol, payload.get("side"), payload)
            analysis_text = await _render_analysis_text(
                payload=payload,
                symbol=symbol,
                direction=payload.get("side"),
                settings=settings,
                chat_id=chat_id,
            )
            await _send_ghost_analysis(callback.message, symbol, analysis_text)
            await callback.answer()
            return

        if action == "chart":
            timeframe = parts[2] if len(parts) > 2 else "1h"
            try:
                img, _ = await hub.chart_service.render_chart(symbol="DEFINE", timeframe=timeframe)
            except Exception:  # noqa: BLE001
                await callback.message.answer("Drop a real ticker for chart, e.g. `chart SOL 1h`.")
                await callback.answer()
                return
            await callback.message.answer_photo(
                BufferedInputFile(img, filename=f"DEFINE-{timeframe}.png"),
                caption=f"DEFINE {timeframe} chart.",
            )
            await callback.answer()
            return

        if action == "heatmap":
            symbol = "DEFINE"
            try:
                img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
            except Exception:  # noqa: BLE001
                symbol = "BTC"
                img, meta = await hub.orderbook_heatmap_service.render_heatmap(symbol=symbol)
            await callback.message.answer_photo(
                BufferedInputFile(img, filename=f"{symbol}_heatmap.png"),
                caption=(
                    f"{meta['pair']} orderbook heatmap\n"
                    f"Best bid: {meta['best_bid']:.6f} | Best ask: {meta['best_ask']:.6f}\n"
                    f"Bid wall: {meta['bid_wall']:.6f} | Ask wall: {meta['ask_wall']:.6f}"
                ),
            )
            await callback.answer()
            return

        if action == "alert":
            await _set_pending_alert(chat_id, "DEFINE")
            await callback.message.answer("Send alert level for DEFINE, e.g. DEFINE 0.50")
            await callback.answer()
            return

        if action == "news":
            payload = await hub.news_service.get_digest(topic="DEFINE", mode="crypto", limit=6)
            await callback.message.answer(news_template(payload))
            await callback.answer()
            return

        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.callback_query(F.data.startswith("top:"))
async def top_rsi_callback(callback: CallbackQuery) -> None:
    if not await _acquire_callback_once(callback):
        with suppress(Exception):
            await callback.answer()
        return

    chat_id = callback.message.chat.id

    async def _run() -> None:
        hub = _require_hub()
        parts = (callback.data or "").split(":")
        mode = parts[1] if len(parts) > 1 else "oversold"
        timeframe = parts[2] if len(parts) > 2 else "1h"
        mode = "overbought" if mode == "overbought" else "oversold"
        payload = await hub.rsi_scanner_service.scan(
            timeframe=timeframe,
            mode=mode,
            limit=10,
            rsi_length=14,
            symbol=None,
        )
        await callback.message.answer(rsi_scan_template(payload))
        await callback.answer()

    await _run_with_typing_lock(callback.bot, chat_id, _run)


@router.message(F.text)
async def route_text(message: Message) -> None:
    hub = _require_hub()
    text = message.text or ""
    chat_id = message.chat.id
    raw_text = text.strip()

    if raw_text.startswith("/"):
        return

    if message.chat.type in ("group", "supergroup"):
        ft_match = re.search(r"\bfree\s*talk\s*mode\s*(on|off)\b", raw_text, flags=re.IGNORECASE)
        if ft_match:
            if not await _is_group_admin(message):
                await message.answer("Only group admins can toggle free talk mode.")
                return
            enabled = ft_match.group(1).lower() == "on"
            await _set_group_free_talk(message.chat.id, enabled)
            await message.answer(f"Group free talk mode {'ON' if enabled else 'OFF'}.")
            return

    if message.chat.type in ("group", "supergroup"):
        free_talk_enabled = await _group_free_talk_enabled(message.chat.id)
        mentioned = _mentions_bot(text, hub.bot_username)
        reply_to_bot = _is_reply_to_bot(message, hub)
        clear_intent = _looks_like_clear_intent(text)
        if not (free_talk_enabled or mentioned or reply_to_bot or clear_intent):
            return
        text = _strip_bot_mention(text, hub.bot_username)
        if not text:
            await message.answer("Send a request in plain text, e.g. `SOL long`, `cpi news`, `chart btc 1h`, or `alert me when SOL hits 50`.")
            return

    if re.search(r"\b(my|show)\s+(user\s*)?id\b|\bwhat('?s| is)\s+my\s+id\b", text, flags=re.IGNORECASE):
        if not message.from_user:
            await message.answer("Could not read your user id from this update.")
            return
        await message.answer(
            f"Your user id: {message.from_user.id}\n"
            f"Current chat id: {message.chat.id}"
        )
        return

    if _is_source_query(text):
        await message.answer(await _source_reply_for_chat(chat_id, text))
        return

    if not await _acquire_message_once(message):
        logger.info(
            "duplicate_message_ignored",
            extra={
                "event": "duplicate_message_ignored",
                "chat_id": chat_id,
                "message_id": message.message_id,
            },
        )
        return

    if not await _check_req_limit(chat_id):
        await message.answer("Rate limit hit. Try again in a minute.")
        return

    # Fast-path: pure greetings — respond immediately, no LLM or market data needed.
    # This avoids cold-start timeouts and lock contention for short casual messages.
    _GREETING_RE = re.compile(
        r"^(gm|gn|gg|gm fren|gn fren|good\s*morning|good\s*night|"
        r"hi|hey|hello|sup|yo|wassup|wagmi|lgtm|lfg|ngmi|ser|fren|anon|"
        r"wen\s*moon|wen\s*lambo|wen\s*pump|wen\s*bull|wen\s*dump|"
        r"still\s*alive|you\s*there|you\s*alive|are\s*you\s*there)[\s!?.]*$",
        re.IGNORECASE,
    )
    if _GREETING_RE.match(raw_text.strip()):
        import random
        _gm_replies = [
            "gm fren 👋 charts are open, tape is moving. what are we hunting today?",
            "gm anon ☀️ market's breathing. drop a ticker or ask anything.",
            "gm 👋 still alive, still watching. what do you need?",
            "gm fren — locked in. throw me a coin or question.",
            "gm anon. BTC still the anchor, alts still lagging dominance. what's the play?",
            "gm ☕ fresh session. give me a ticker, a question, or ask what's moving.",
            "gm — charts loaded, alerts armed. what are we doing today?",
        ]
        _gn_replies = [
            "gn fren 🌙 set your alerts before you sleep.",
            "gn anon. the market doesn't sleep but you should.",
            "gn — if you haven't set alerts, do it now.",
        ]
        low = raw_text.strip().lower()
        if low.startswith("gn") or "night" in low:
            await message.answer(random.choice(_gn_replies))
        else:
            await message.answer(random.choice(_gm_replies))
        return

    lock = _chat_lock(chat_id)
    if lock.locked():
        # Avoid flooding busy notices if user sends many messages quickly.
        if await hub.cache.set_if_absent(f"busy_notice:{chat_id}", ttl=5):
            await message.answer("Still processing your previous request. Give me a few seconds.")
        return

    start_ts = datetime.now(timezone.utc)
    stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(message.bot, chat_id, stop))
    try:
        async with lock:
            cmd_wizard = await _cmd_wizard_get(chat_id)
            if cmd_wizard:
                step = str(cmd_wizard.get("step") or "").strip().lower()
                if step == "dispatch_text":
                    prefix = str(cmd_wizard.get("prefix") or "")
                    await _cmd_wizard_clear(chat_id)
                    typed = text.strip()
                    if not typed:
                        await message.answer("Send the requested details to continue.")
                        return
                    await _dispatch_command_text(message, f"{prefix}{typed}".strip())
                    return
                if step == "giveaway_prize":
                    await _cmd_wizard_clear(chat_id)
                    if not message.from_user:
                        await message.answer("Could not identify sender.")
                        return
                    prize = text.strip().strip("'\"") or "Prize"
                    duration_seconds = max(30, _as_int(cmd_wizard.get("duration_seconds"), 600))
                    winners_requested = max(1, min(_as_int(cmd_wizard.get("winners"), 1), 5))
                    try:
                        payload = await hub.giveaway_service.start_giveaway(
                            group_chat_id=chat_id,
                            admin_chat_id=message.from_user.id,
                            duration_seconds=duration_seconds,
                            prize=prize,
                        )
                    except Exception as exc:  # noqa: BLE001
                        await message.answer(str(exc))
                        return
                    note = ""
                    if winners_requested > 1:
                        note = "\nNote: multi-winner draw runs as sequential rerolls after first winner."
                    await message.answer(
                        f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\n"
                        f"Ends at: {payload['end_time']}\nUsers enter with /join or /giveaway join{note}",
                        reply_markup=giveaway_menu(is_admin=True),
                    )
                    return
                if step == "alert_clear_symbol":
                    await _cmd_wizard_clear(chat_id)
                    symbol = text.strip().upper().lstrip("$")
                    if not re.fullmatch(r"[A-Z0-9]{2,20}", symbol):
                        await message.answer("Invalid symbol. Send a ticker like SOL.")
                        return
                    count = await hub.alerts_service.delete_alerts_by_symbol(chat_id, symbol)
                    await message.answer(f"Cleared {count} alerts for {symbol}.")
                    return

            wizard = await _wizard_get(chat_id)
            if wizard:
                step = wizard.get("step")
                data = wizard.get("data", {})
                if step == "symbol":
                    data["symbol"] = text.strip().upper()
                    await _wizard_set(chat_id, {"step": "timeframe", "data": data})
                    await message.answer("Timeframe? (15m / 1h / 4h)")
                    return
                if step == "timeframe":
                    data["timeframe"] = text.strip().lower()
                    await _wizard_set(chat_id, {"step": "timestamp", "data": data})
                    await message.answer("Timestamp? (ISO, yesterday, or 2 hours ago)")
                    return
                if step == "timestamp":
                    ts = parse_timestamp(text)
                    if not ts:
                        await message.answer("Could not parse timestamp. Try 'yesterday' or ISO datetime.")
                        return
                    data["timestamp"] = ts.isoformat()
                    await _wizard_set(chat_id, {"step": "levels", "data": data})
                    await message.answer("Send levels: entry <x> stop <y> targets <a> <b> ...")
                    return
                if step == "levels":
                    entry_m = re.search(r"entry\s*([0-9.]+)", text, re.IGNORECASE)
                    stop_m = re.search(r"stop\s*([0-9.]+)", text, re.IGNORECASE)
                    targets_m = re.search(r"targets?\s*([0-9.\s]+)", text, re.IGNORECASE)
                    if not entry_m or not stop_m or not targets_m:
                        await message.answer("Format: entry <x> stop <y> targets <a> <b>")
                        return
                    targets = [float(x) for x in re.findall(r"[0-9.]+", targets_m.group(1))]
                    data.update(
                        {
                            "entry": float(entry_m.group(1)),
                            "stop": float(stop_m.group(1)),
                            "targets": targets,
                            "timestamp": datetime.fromisoformat(data["timestamp"]),
                            "mode": "ambiguous",
                        }
                    )
                    result = await hub.trade_verify_service.verify(**data)
                    await _save_trade_check(chat_id, data, result)
                    await _remember_source_context(
                        chat_id,
                        source_line=str(result.get("source_line") or ""),
                        symbol=data["symbol"],
                        context="trade check",
                    )
                    await message.answer(trade_verification_template(result))
                    await _wizard_clear(chat_id)
                    return

            pending_alert_symbol = await _get_pending_alert(chat_id)
            if pending_alert_symbol:
                price_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
                if price_match:
                    target = float(price_match.group(1))
                    alert = await hub.alerts_service.create_alert(chat_id, pending_alert_symbol, "cross", target, source="button")
                    await _clear_pending_alert(chat_id)
                    await _remember_source_context(
                        chat_id,
                        exchange=alert.source_exchange,
                        market_kind=alert.market_kind,
                        instrument_id=alert.instrument_id,
                        symbol=pending_alert_symbol,
                        context="alert",
                    )
                    await message.answer(
                        f"alert set for <b>{pending_alert_symbol}</b> at <b>{target}</b>. "
                        "i'll ping you when we hit it. don't get liquidated"
                    )
                    return

            settings = await hub.user_service.get_settings(chat_id)
            text_lower = text.lower().strip()

            # Special Ghost Easter eggs / overrides
            if "define trading" in text_lower or ("define" in text_lower and len(text_lower.split()) <= 3):
                await message.bot.send_chat_action(message.chat.id, ChatAction.TYPING)
                await asyncio.sleep(0.8)

                funny_line = (
                    "the art of losing money faster than a casino while staring at candles "
                    "until your eyes bleed. buy low, sell high, don't get rekt"
                )
                await message.answer(funny_line, reply_markup=_define_keyboard())
                return

            # Bonus: catch casual scanner phrases and route directly.
            if "rsi top" in text_lower or "overbought list" in text_lower or "strong coins" in text_lower:
                if "oversold" in text_lower:
                    await _dispatch_command_text(message, "rsi top 10 1h oversold")
                elif "overbought" in text_lower:
                    await _dispatch_command_text(message, "rsi top 10 1h overbought")
                elif "rsi top" in text_lower:
                    await _dispatch_command_text(message, text_lower)
                else:
                    await _dispatch_command_text(message, "coins to watch 5")
                return

            followup_context = await _recent_analysis_context(chat_id)
            if _looks_like_analysis_followup(text, followup_context):
                followup_reply = await _llm_followup_reply(
                    text,
                    followup_context or {},
                    chat_id=chat_id,
                )
                if followup_reply:
                    await message.answer(followup_reply)
                    return

            parsed = parse_message(text)
            chat_mode = _openai_chat_mode()

            if hub.llm_client and chat_mode == "chat_only":
                llm_reply = await _llm_market_chat_reply(text, settings, chat_id=chat_id)
                if llm_reply:
                    await _send_llm_reply(message, llm_reply)
                    return
                await message.answer("signal unclear right now. try again in a sec.")
                return

            if hub.llm_client and chat_mode == "llm_first":
                routed = await _llm_route_message(text)
                if routed:
                    try:
                        if await _handle_routed_intent(message, settings, routed):
                            return
                    except Exception:  # noqa: BLE001
                        pass
                llm_reply = await _llm_market_chat_reply(text, settings, chat_id=chat_id)
                if llm_reply:
                    await _send_llm_reply(message, llm_reply)
                    return

            if chat_mode in {"hybrid", "tool_first"} and (parsed.intent == Intent.UNKNOWN or (parsed.requires_followup and parsed.intent == Intent.UNKNOWN)):
                routed = await _llm_route_message(text)
                if routed:
                    try:
                        if await _handle_routed_intent(message, settings, routed):
                            return
                    except Exception:  # noqa: BLE001
                        pass

            if parsed.requires_followup:
                if parsed.intent == Intent.UNKNOWN:
                    llm_reply = await _llm_market_chat_reply(text, settings, chat_id=chat_id)
                    if llm_reply:
                        await _send_llm_reply(message, llm_reply)
                        return
                    english_phrase = is_likely_english_phrase(text)
                    symbol_hint = None if english_phrase else _extract_action_symbol_hint(text)
                    if symbol_hint:
                        await message.answer(
                            f"pick an action for <b>{symbol_hint}</b>:",
                            reply_markup=smart_action_menu(symbol_hint),
                        )
                    else:
                        await message.answer(unknown_prompt(), reply_markup=smart_action_menu(None))
                    return
                if parsed.intent == Intent.ANALYSIS and not parsed.entities.get("symbol"):
                    kb = simple_followup(
                        [
                            ("BTC", "quick:analysis:BTC"),
                            ("ETH", "quick:analysis:ETH"),
                            ("SOL", "quick:analysis:SOL"),
                        ]
                    )
                    await message.answer(parsed.followup_question or "Need one detail.", reply_markup=kb)
                    return
                await message.answer(parsed.followup_question or unknown_prompt())
                return

            try:
                if await _handle_parsed_intent(message, parsed, settings):
                    return
                llm_reply = await _llm_market_chat_reply(text, settings, chat_id=chat_id)
                if llm_reply:
                    await _send_llm_reply(message, llm_reply)
                    return
                symbol_hint = _extract_action_symbol_hint(text)
                await message.answer(
                    parsed.followup_question or unknown_prompt(),
                    reply_markup=smart_action_menu(symbol_hint) if symbol_hint else None,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.exception("handle_parsed_intent_error", extra={"event": "handle_parsed_intent_error", "chat_id": chat_id})
                await message.answer(
                    f"couldn't complete that — <i>{_safe_exc(exc)}</i>\n"
                    "try again with a bit more detail."
                )
                return
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "route_text_unhandled_error",
            extra={"event": "route_text_unhandled_error", "chat_id": chat_id},
        )
        with suppress(Exception):
            await message.answer(f"something broke on my end. try again in a sec. (<i>{_safe_exc(exc)}</i>)")
    finally:
        stop.set()
        typing_task.cancel()
        with suppress(Exception):
            await typing_task
        latency_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        logger.info("message_processed", extra={"event": "message_processed", "chat_id": chat_id, "latency_ms": latency_ms})
