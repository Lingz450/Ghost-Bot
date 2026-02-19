from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from datetime import datetime, timezone
import logging

from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from app.bot.keyboards import analysis_actions, settings_menu, simple_followup, wallet_actions
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
from app.core.nlu import Intent, parse_message, parse_timestamp
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


def _chat_lock(chat_id: int) -> asyncio.Lock:
    lock = _CHAT_LOCKS.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        _CHAT_LOCKS[chat_id] = lock
    return lock


async def _acquire_message_once(message: Message, ttl: int = 60 * 60 * 6) -> bool:
    hub = _require_hub()
    key = f"seen:message:{message.chat.id}:{message.message_id}"
    return await hub.cache.set_if_absent(key, ttl=ttl)


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
    result = await hub.rate_limiter.check(
        key=f"rl:req:{chat_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
        limit=_settings.request_rate_limit_per_minute,
        window_seconds=60,
    )
    return result.allowed


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
        f"User style preference: {style}\n"
        f"User message: {cleaned}\n"
        "Keep answer concise. If this is a crypto setup request that lacks details, ask one short follow-up question."
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

    if intent == "smalltalk":
        llm_reply = await _llm_fallback_reply(raw_text, settings, chat_id=chat_id)
        await message.answer(llm_reply or smalltalk_reply(settings))
        return True

    if intent == "general_chat":
        llm_reply = await _llm_fallback_reply(raw_text, settings, chat_id=chat_id)
        if llm_reply:
            await message.answer(llm_reply)
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
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        await message.answer(trade_plan_template(payload, settings), reply_markup=analysis_actions(symbol))
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
        lines.append("")
        lines.append("Not financial advice.")
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
        if not symbol or price is None:
            await message.answer("Give me symbol and price, e.g. `alert SOL above 100`.")
            return True
        op = str(params.get("operator") or params.get("condition") or "cross").strip().lower()
        if op in {">", ">=", "above", "gt", "gte"}:
            condition = "above"
        elif op in {"<", "<=", "below", "lt", "lte"}:
            condition = "below"
        else:
            condition = "cross"
        alert = await hub.alerts_service.create_alert(chat_id, symbol, condition, float(price))
        await _remember_source_context(
            chat_id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=symbol,
            context="alert",
        )
        await message.answer(
            f"Alert created: #{alert.id}\n"
            f"Condition: {alert.symbol} {alert.condition} {alert.target_price}\n"
            "I trigger once and avoid spam."
        )
        return True

    if intent == "alert_list":
        alerts = await hub.alerts_service.list_alerts(chat_id)
        if not alerts:
            await message.answer("No active alerts.")
        else:
            lines = [
                (f"#{a.id} {a.symbol} {a.condition} {a.target_price} [{a.status}]").strip()
                for a in alerts
            ]
            first = alerts[0]
            await _remember_source_context(
                chat_id,
                exchange=first.source_exchange,
                market_kind=first.market_kind,
                instrument_id=first.instrument_id,
                symbol=first.symbol,
                context="alerts list",
            )
            await message.answer("Active alerts:\n" + "\n".join(lines))
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
        if not symbol or entry is None or stop is None or not targets:
            await message.answer("Need symbol, entry, stop, and at least one target.")
            return True
        direction = str(params.get("side") or params.get("direction") or "").strip().lower() or None
        timeframe = str(params.get("timeframe", "1h")).strip() or "1h"
        amount_usd = _as_float(params.get("amount") or params.get("amount_usd") or params.get("margin"))
        leverage = _as_float(params.get("leverage"))
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
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        await message.answer(trade_plan_template(payload, settings), reply_markup=analysis_actions(symbol))
        return True

    if parsed.intent == Intent.SETUP_REVIEW:
        timeframe = parsed.entities.get("timeframe", "1h")
        tfs = parsed.entities.get("timeframes") or []
        if tfs:
            timeframe = tfs[0]
        payload = await hub.setup_review_service.review(
            symbol=parsed.entities["symbol"],
            timeframe=timeframe,
            entry=float(parsed.entities["entry"]),
            stop=float(parsed.entities["stop"]),
            targets=[float(x) for x in parsed.entities["targets"]],
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
        lines.append("")
        lines.append("Not financial advice.")
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
        alert = await hub.alerts_service.create_alert(
            chat_id,
            parsed.entities["symbol"],
            parsed.entities.get("condition", "cross"),
            float(parsed.entities["target_price"]),
        )
        await _remember_source_context(
            chat_id,
            exchange=alert.source_exchange,
            market_kind=alert.market_kind,
            instrument_id=alert.instrument_id,
            symbol=alert.symbol,
            context="alert",
        )
        await message.answer(
            f"Alert created: #{alert.id}\n"
            f"Condition: {alert.symbol} {alert.condition} {alert.target_price}\n"
            "I trigger once and avoid spam."
        )
        return True

    if parsed.intent == Intent.ALERT_LIST:
        alerts = await hub.alerts_service.list_alerts(chat_id)
        if not alerts:
            await message.answer("No active alerts.")
        else:
            lines = [
                (f"#{a.id} {a.symbol} {a.condition} {a.target_price} [{a.status}]").strip()
                for a in alerts
            ]
            first = alerts[0]
            await _remember_source_context(
                chat_id,
                exchange=first.source_exchange,
                market_kind=first.market_kind,
                instrument_id=first.instrument_id,
                symbol=first.symbol,
                context="alerts list",
            )
            await message.answer("Active alerts:\n" + "\n".join(lines))
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
    await message.answer(
        "Market bot online. Send plain text like 'SOL long', 'Coins to watch 5', or 'ping me when BTC hits 70000'.\n"
        "Not financial advice."
    )


@router.message(Command("help"))
async def help_cmd(message: Message) -> None:
    await message.answer(help_text())


@router.message(Command("id"))
async def id_cmd(message: Message) -> None:
    if not message.from_user:
        await message.answer("Could not read your user id from this update.")
        return
    await message.answer(
        f"Your user id: {message.from_user.id}\n"
        f"Current chat id: {message.chat.id}"
    )


@router.message(Command("settings"))
async def settings_cmd(message: Message) -> None:
    hub = _require_hub()
    settings = await hub.user_service.get_settings(message.chat.id)
    await message.answer(settings_text(settings), reply_markup=settings_menu(settings))


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
    topic = None
    mode = "crypto"
    if " " in text:
        topic = text.split(" ", 1)[1].strip()
    if topic:
        lowered = topic.lower()
        if re.search(r"\b(openai|chatgpt|gpt|codex)\b", lowered):
            mode = "openai"
            topic = "openai"
        elif re.search(r"\b(cpi|inflation)\b", lowered):
            mode = "macro"
            topic = "cpi"
        elif re.search(r"\b(fomc|fed|powell|macro|rates?)\b", lowered):
            mode = "macro"
            topic = "macro"
    payload = await hub.news_service.get_digest(topic=topic, mode=mode, limit=6)
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
        await message.answer("Usage: /scan <solana|tron> <address>")
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
        rows = [
            (f"#{a.id} {a.symbol} {a.condition} {a.target_price} [{a.status}]").strip()
            for a in alerts
        ]
        first = alerts[0]
        await _remember_source_context(
            message.chat.id,
            exchange=first.source_exchange,
            market_kind=first.market_kind,
            instrument_id=first.instrument_id,
            symbol=first.symbol,
            context="alerts list",
        )
        await message.answer("Active alerts:\n" + "\n".join(rows))
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
            f"Alert created: #{alert.id}\nCondition: {symbol} {condition} {price}\n"
            "I will trigger once on crossing and avoid spam."
        )
        return

    await message.answer(
        "Use /alert add <symbol> <above|below|cross> <price> | /alert list | /alert delete <id> | /alert clear | /alert pause | /alert resume"
    )


@router.message(Command("tradecheck"))
async def tradecheck_cmd(message: Message) -> None:
    await _wizard_set(message.chat.id, {"step": "symbol", "data": {}})
    await message.answer("Tradecheck wizard: send symbol (e.g., ETH).")


@router.message(Command("join"))
async def join_cmd(message: Message) -> None:
    hub = _require_hub()
    if not message.from_user:
        await message.answer("Could not identify user for join.")
        return
    try:
        payload = await hub.giveaway_service.join_active(message.chat.id, message.from_user.id)
    except Exception as exc:  # noqa: BLE001
        await message.answer(str(exc))
        return
    await message.answer(f"Joined giveaway #{payload['giveaway_id']}. Participants: {payload['participants']}")


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

    if re.search(r"^/giveaway\s+end\b", text, flags=re.IGNORECASE):
        try:
            payload = await hub.giveaway_service.end_giveaway(message.chat.id, message.from_user.id)
        except Exception as exc:  # noqa: BLE001
            await message.answer(str(exc))
            return
        if payload.get("winner_user_id"):
            await message.answer(
                f"Giveaway #{payload['giveaway_id']} ended.\nWinner: {payload['winner_user_id']}\nPrize: {payload['prize']}"
            )
        else:
            await message.answer(f"Giveaway ended with no winner. {payload.get('note')}")
        return

    if re.search(r"^/giveaway\s+reroll\b", text, flags=re.IGNORECASE):
        try:
            payload = await hub.giveaway_service.reroll(message.chat.id, message.from_user.id)
        except Exception as exc:  # noqa: BLE001
            await message.answer(str(exc))
            return
        await message.answer(
            f"Reroll complete for giveaway #{payload['giveaway_id']}.\n"
            f"New winner: {payload['winner_user_id']} (prev: {payload.get('previous_winner_user_id')})"
        )
        return

    start_match = re.search(r"^/giveaway\s+start\s+(\S+)(?:\s+prize\s+(.+))?$", text, flags=re.IGNORECASE)
    if start_match:
        duration_raw = start_match.group(1)
        duration_seconds = _parse_duration_to_seconds(duration_raw)
        if duration_seconds is None:
            await message.answer("Invalid duration. Example: /giveaway start 10m prize \"50 USDT\"")
            return
        prize_raw = (start_match.group(2) or "Prize").strip()
        prize = prize_raw.strip("'\"")
        try:
            payload = await hub.giveaway_service.start_giveaway(
                group_chat_id=message.chat.id,
                admin_chat_id=message.from_user.id,
                duration_seconds=duration_seconds,
                prize=prize,
            )
        except Exception as exc:  # noqa: BLE001
            await message.answer(str(exc))
            return
        await message.answer(
            f"Giveaway #{payload['id']} started.\nPrize: {payload['prize']}\n"
            f"Ends at: {payload['end_time']}\nUsers enter with /join"
        )
        return

    await message.answer("Usage: /giveaway start 10m prize \"50 USDT\" | /giveaway end | /giveaway reroll | /giveaway status")


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
    await callback.message.answer(f"Send alert level for {symbol}, e.g. {symbol} 100")
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
        await callback.answer("No cached levels. Refresh first.", show_alert=True)
        return
    await callback.message.answer(
        f"{symbol} levels\nEntry: {payload['entry']}\nTP1: {payload['tp1']}\nTP2: {payload['tp2']}\nSL: {payload['sl']}"
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
        await callback.answer("No context saved.", show_alert=True)
        return
    lines = [f"Why {symbol}:"] + [f"- {w}" for w in payload.get("why", [])]
    await callback.message.answer("\n".join(lines))
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
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        await callback.message.answer(trade_plan_template(payload, settings), reply_markup=analysis_actions(symbol))
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
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        await callback.message.answer(trade_plan_template(payload, settings), reply_markup=analysis_actions(symbol))
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
        await callback.message.answer(
            f"{symbol} derivatives\nFunding: {payload.get('funding_rate')}\nOpen interest: {payload.get('open_interest')}"
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
            await callback.message.answer(f"No fresh catalysts found for {symbol} right now.")
            await callback.answer()
            return
        lines = [f"{symbol} catalysts:"]
        for item in headlines[:3]:
            lines.append(f"- {item['title']}")
            lines.append(f"  {item['url']}")
        await callback.message.answer("\n".join(lines))
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
        f"Send trade details like: check this trade from yesterday: {symbol} entry 2100 stop 2060 targets 2140 2180 timeframe 1h"
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
        await _remember_source_context(
            chat_id,
            source_line=str(payload.get("data_source_line") or ""),
            symbol=symbol,
            context="analysis",
        )
        await callback.message.answer(trade_plan_template(payload, settings), reply_markup=analysis_actions(symbol))
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

    start_ts = datetime.now(timezone.utc)
    stop = asyncio.Event()
    typing_task = asyncio.create_task(_typing_loop(message.bot, chat_id, stop))
    lock = _chat_lock(chat_id)
    try:
        async with lock:
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
                        f"Alert created: #{alert.id} {pending_alert_symbol} cross {target}"
                    )
                    return

            parsed = parse_message(text)
            settings = await hub.user_service.get_settings(chat_id)
            chat_mode = _openai_chat_mode()

            if hub.llm_client and chat_mode == "chat_only":
                llm_reply = await _llm_fallback_reply(text, settings, chat_id=chat_id)
                if llm_reply:
                    await message.answer(llm_reply)
                    return
                await message.answer("OpenAI is temporarily unavailable. Try again in a few seconds.")
                return

            if hub.llm_client and chat_mode == "llm_first":
                routed = await _llm_route_message(text)
                if routed:
                    try:
                        if await _handle_routed_intent(message, settings, routed):
                            return
                    except Exception:  # noqa: BLE001
                        pass
                llm_reply = await _llm_fallback_reply(text, settings, chat_id=chat_id)
                if llm_reply:
                    await message.answer(llm_reply)
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
                    llm_reply = await _llm_fallback_reply(text, settings, chat_id=chat_id)
                    if llm_reply:
                        await message.answer(llm_reply)
                        return
                    await message.answer(unknown_prompt())
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
                llm_reply = await _llm_fallback_reply(text, settings, chat_id=chat_id)
                await message.answer(llm_reply or parsed.followup_question or unknown_prompt())
                return
            except Exception as exc:  # noqa: BLE001
                await message.answer(
                    f"Could not complete that request cleanly: {exc}\n"
                    "Try again with a bit more detail."
                )
                return
    finally:
        stop.set()
        typing_task.cancel()
        with suppress(Exception):
            await typing_task
        latency_ms = int((datetime.now(timezone.utc) - start_ts).total_seconds() * 1000)
        logger.info("message_processed", extra={"event": "message_processed", "chat_id": chat_id, "latency_ms": latency_ms})
