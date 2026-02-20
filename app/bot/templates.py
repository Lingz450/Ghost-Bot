from __future__ import annotations

import random
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from app.core.fmt import fmt_price, safe_html

# ---------------------------------------------------------------------------
# Personality pools
# ---------------------------------------------------------------------------

SMALLTALK_REPLIES = [
    "Ghost Alpha online. Scanning the tape and filtering noise.\nDrop a ticker, ask for news, or run an RSI scan.",
    "All systems green. Market is choppy but clean levels still print.\nSend a coin and direction.",
    "Quiet session. I'm watching liquidity and breakout zones.\nWhat are we hunting?",
    "Operational. Alerts armed, charts loaded.\nSend me a coin and direction.",
    "Lurking where clean entries live.\nWant analysis, watchlist, or news?",
    "Stable and tracking. If it moves, I'll see it.\nScalp or swing today?",
    "Locked in. Market is noisy, signal is not.\nTry <code>SOL long</code> or <code>ETH short</code>.",
    "Signal check complete. Give me a ticker and I'll map entry/targets fast.",
    "Ghost Alpha active. Watching BTC lead and alt reactions.\nWhat do you want first?",
    "Scanner is hot. Trade plans, alerts, wallet scans — your call.",
]

WILD_CLOSERS = [
    "Drop the next ticker and I'll map it fast.",
    "Want me to set the alert level now?",
    "Keep it clean. One setup at a time.",
]

STANDARD_CLOSERS = [
    "Send the next chart if you want a second read.",
]

UNKNOWN_FOLLOWUPS = [
    "Give me a clear target: <code>SOL long</code>, <code>cpi news</code>, <code>chart BTC 1h</code>, or <code>alert me when BTC hits 70000</code>.",
    "I can route free text. Try <code>ETH short 4h</code>, <code>rsi top 10 1h oversold</code>, or <code>list my alerts</code>.",
    "Send the intent directly: <code>coins to watch 5</code>, <code>openai updates</code>, or <code>scan solana &lt;address&gt;</code>.",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tone_mode(settings: dict) -> str:
    if settings.get("formal_mode"):
        return "formal"
    tone = str(settings.get("tone_mode", "wild")).lower()
    return "wild" if tone not in {"formal", "standard", "wild"} else tone


def _pick(options: list[str]) -> str:
    return random.choice(options)


def _render_summary(summary: str, settings: dict) -> str:
    text = safe_html(summary.strip())
    if not settings.get("formal_mode"):
        prof = settings.get("profanity_level", "light")
        if prof == "medium":
            text = text.replace("volatile", "wild")
    return text


def relative_updated(ts_iso: str | None) -> str:
    if not ts_iso:
        return ""
    try:
        then = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - then.astimezone(timezone.utc)
        minutes = int(delta.total_seconds() // 60)
        return f"updated {minutes}m ago"
    except Exception:  # noqa: BLE001
        return ts_iso


def _clean_url(raw: str) -> str:
    if not raw:
        return "n/a"
    try:
        parts = urlsplit(raw)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:  # noqa: BLE001
        return raw


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def trade_plan_template(plan: dict, settings: dict, detailed: bool = False) -> str:
    symbol = safe_html(str(plan.get("symbol") or "asset").upper())
    side = str(plan.get("side") or "").strip().lower()
    price = plan.get("price")
    summary = _render_summary(plan.get("summary", "").strip(), settings)
    context = safe_html(str(plan.get("market_context_text") or "").strip())

    side_label = f" · {side}" if side in {"long", "short"} else ""
    header = f"<b>{symbol}{side_label}</b>"

    if isinstance(price, (int, float)):
        lead = f"{safe_html(fmt_price(float(price)))} — {summary}"
    else:
        lead = summary or f"{symbol} setup mapped."

    lines = [header, "", lead]
    if context:
        lines.append(f"<i>backdrop: {context}</i>")

    lines += [
        "",
        f"• <b>entry</b>   {safe_html(plan['entry'])}",
        f"• <b>target</b>  {safe_html(plan['tp1'])} → {safe_html(plan['tp2'])}",
        f"• <b>stop</b>    {safe_html(plan['sl'])}",
    ]

    why_items = plan.get("why", [])
    if why_items:
        lines.append("")
        for bullet in (why_items if detailed else why_items[:2]):
            lines.append(f"  — {safe_html(bullet)}")

    if detailed and plan.get("mtf_snapshot"):
        lines += ["", "<b>mtf:</b>"]
        for row in plan["mtf_snapshot"][:4]:
            lines.append(f"  {safe_html(row)}")

    if detailed and plan.get("input_notes"):
        lines.append("")
        for note in plan["input_notes"][:3]:
            lines.append(f"<i>note: {safe_html(note)}</i>")

    if detailed:
        updated = plan.get("updated_at")
        if updated:
            lines += ["", f"<i>{relative_updated(updated) or updated}</i>"]

    lines += ["", f"<i>{safe_html(plan.get('risk', 'Stay nimble and cut it if structure breaks.'))}</i>"]
    return "\n".join(lines)


def watchlist_template(payload: dict) -> str:
    summary = safe_html(payload.get("summary", ""))
    lines = [
        "<b>Watchlist</b>",
        "",
        summary,
        "",
    ]
    for idx, item in enumerate(payload.get("items", []), start=1):
        lines.append(f"<b>{idx}.</b> {safe_html(item)}")
    return "\n".join(lines)


def news_template(payload: dict) -> str:
    summary = safe_html(payload.get("summary", ""))
    vibe = safe_html(payload.get("vibe", ""))
    headlines = payload.get("headlines", [])[:5]

    lines = [
        "<b>News Digest</b>",
        "",
        summary,
    ]

    if not headlines:
        lines += ["", "No fresh items from configured feeds.", "", f"<i>{vibe}</i>"]
        return "\n".join(lines)

    lines.append("")
    for idx, item in enumerate(headlines, start=1):
        title = safe_html(item.get("title", ""))
        source = safe_html(item.get("source", ""))
        url = _clean_url(item.get("url", ""))
        lines.append(f"<b>{idx}. {title}</b>")
        if url and url != "n/a":
            lines.append(f'<i>{source}</i> · <a href="{url}">read →</a>')
        else:
            lines.append(f"<i>{source}</i>")
        lines.append("")

    updated = relative_updated(payload.get("updated_at")) or payload.get("updated_at", "")
    if updated:
        lines.append(f"<i>{safe_html(updated)}</i>")
    if vibe:
        lines.append(f"<i>{vibe}</i>")
    return "\n".join(lines)


def wallet_scan_template(payload: dict) -> str:
    chain = safe_html(str(payload.get("chain", "")).upper())
    address = safe_html(str(payload.get("address", "")))
    balance_line = f"{payload['native_balance']:.6f} {safe_html(payload['native_symbol'])}"
    if payload.get("native_usd") is not None:
        balance_line += f" (~${payload['native_usd']:.2f})"

    lines = [
        f"<b>{chain} Wallet Scan</b>",
        "",
        f"<code>{address}</code>",
        "",
        f"<b>Native balance</b>  {balance_line}",
    ]

    tokens = payload.get("tokens", [])[:8]
    if tokens:
        lines.append("")
        lines.append("<b>Top tokens:</b>")
        for token in tokens:
            sym = safe_html(token.get("symbol", "UNK"))
            amt = token.get("amount", 0)
            lines.append(f"  {sym}  {amt}")
    else:
        lines += ["", "No non-native holdings detected."]

    if payload.get("resources"):
        lines += ["", "<b>Resources:</b>"]
        for k, v in payload["resources"].items():
            lines.append(f"  {safe_html(k)}: {safe_html(str(v))}")

    tx_count = len(payload.get("recent_transactions", []))
    lines += ["", f"<b>Recent tx count</b>  {tx_count}"]

    warnings = payload.get("warnings", [])[:4]
    if warnings:
        lines += ["", "<b>Warnings:</b>"]
        for warn in warnings:
            lines.append(f"  ⚠ {safe_html(warn)}")
    return "\n".join(lines)


def cycle_template(payload: dict) -> str:
    summary = safe_html(payload.get("summary", ""))
    confidence = payload.get("confidence", 0)
    lines = [
        "<b>Cycle Check</b>",
        "",
        summary,
        f"<i>Confidence: {confidence:.0%}</i>",
        "",
    ]
    for b in payload.get("bullets", []):
        lines.append(f"— {safe_html(b)}")
    return "\n".join(lines)


def trade_verification_template(payload: dict) -> str:
    symbol = safe_html(str(payload.get("symbol", "")))
    if payload.get("result") == "not_filled":
        return (
            f"<b>Trade Check · {symbol}</b>\n\n"
            f"Result: <b>not filled</b>\n"
            f"{safe_html(payload.get('note', ''))}"
        )

    direction = safe_html(str(payload.get("direction", "")))
    result = safe_html(str(payload.get("result", "")))
    mode = safe_html(str(payload.get("mode", "")))

    lines = [
        f"<b>Trade Check · {symbol} {direction}</b>",
        f"Result: <b>{result}</b> · mode: {mode}",
        "",
        f"Filled at:   {safe_html(str(payload.get('filled_at', 'n/a')))}",
        f"First hit:   {safe_html(str(payload.get('first_hit', 'n/a')))}",
        f"MFE:         {safe_html(str(payload.get('mfe', 'n/a')))}",
        f"MAE:         {safe_html(str(payload.get('mae', 'n/a')))}",
        f"R multiple:  {safe_html(str(payload.get('r_multiple', 'n/a')))}",
    ]
    return "\n".join(lines)


def correlation_template(payload: dict) -> str:
    summary = safe_html(payload.get("summary", ""))
    lines = ["<b>Correlation</b>", "", summary, ""]
    for b in payload.get("bullets", []):
        lines.append(f"— {safe_html(b)}")
    return "\n".join(lines)


def rsi_scan_template(payload: dict) -> str:
    tf = safe_html(str(payload.get("timeframe", "")))
    rsi_len = safe_html(str(payload.get("rsi_length", 14)))
    summary = safe_html(payload.get("summary", ""))

    items = payload.get("items", [])
    if not items:
        return (
            f"<b>RSI Scan · {tf} · RSI({rsi_len})</b>\n\n"
            f"{summary}\n\n"
            "No results for that request. Try another timeframe or symbol."
        )

    # Derive direction label from first item's note
    first_note = str(items[0].get("note", "")).lower()
    direction = "Overbought" if "overbought" in first_note else "Oversold" if "oversold" in first_note else "Scan"

    lines = [
        f"<b>RSI Scan · {tf} · RSI({rsi_len}) · {direction}</b>",
        "",
        summary,
        "",
    ]
    for idx, row in enumerate(items, start=1):
        sym = safe_html(str(row.get("symbol", "")))
        rsi_val = row.get("rsi", "—")
        note = safe_html(str(row.get("note", "")))
        try:
            rsi_fmt = f"{float(rsi_val):.1f}"
        except (TypeError, ValueError):
            rsi_fmt = safe_html(str(rsi_val))
        lines.append(f"<b>{idx}. {sym}</b>   RSI {rsi_fmt} — <i>{note}</i>")
    return "\n".join(lines)


def pair_find_template(payload: dict) -> str:
    query = safe_html(str(payload.get("query", "")))
    summary = safe_html(payload.get("summary", ""))
    matches = payload.get("matches", [])

    lines = [
        "<b>Pair Finder</b>",
        f"<i>query: {query}</i>",
        "",
        summary,
        "",
    ]
    if not matches:
        lines.append("No direct match. Try ticker, full name, or contract address.")
    else:
        for idx, row in enumerate(matches, start=1):
            sym = safe_html(str(row.get("symbol", "")))
            name = safe_html(str(row.get("name", "")))
            pair = safe_html(str(row.get("pair") or "n/a"))
            tradable = "✓" if row.get("tradable_binance") else "✗"
            price = row.get("price")
            price_txt = f"${price}" if price is not None else "n/a"
            lines.append(f"<b>{idx}. {sym}</b> <i>({name})</i>")
            lines.append(f"   Pair: <code>{pair}</code> · Tradable: {tradable} · ${safe_html(str(price_txt))}")
            lines.append("")
    lines.append("<i>Missing candles? I can still give narrative + execution rules.</i>")
    return "\n".join(lines)


def price_guess_template(payload: dict) -> str:
    summary = safe_html(payload.get("summary", ""))
    matches = payload.get("matches", [])

    lines = ["<b>Price Search</b>", "", summary, ""]
    if not matches:
        lines.append("No close candidates found. Try a wider hint or exact range.")
        return "\n".join(lines)

    for idx, row in enumerate(matches[:10], start=1):
        sym = safe_html(str(row.get("symbol", "")))
        name = safe_html(str(row.get("name", "")))
        tradable = "✓" if row.get("tradable_binance") else "✗"
        price = safe_html(str(row.get("price", "")))
        lines.append(f"<b>{idx}. {sym}</b> <i>({name})</i>  ${price} · Tradable: {tradable}")
    return "\n".join(lines)


def _fmt_price(v: object) -> str:
    """Format a price value cleanly (strip trailing zeros)."""
    try:
        f = float(str(v))
    except (TypeError, ValueError):
        return str(v)
    if abs(f) >= 1:
        return f"{f:.4f}".rstrip("0").rstrip(".")
    return f"{f:.8f}".rstrip("0").rstrip(".")


def setup_review_template(payload: dict, settings: dict) -> str:
    tone = _tone_mode(settings)
    symbol    = safe_html(str(payload.get("symbol", "")))
    direction = safe_html(str(payload.get("direction", "")))
    verdict   = safe_html(str(payload.get("verdict", "")).upper())
    tf        = safe_html(str(payload.get("timeframe", "1h")))

    if tone == "wild":
        opener = "i see the level. here's where it breaks."
    elif tone == "standard":
        opener = "Setup review complete."
    else:
        opener = "Setup review completed."

    entry_val  = _fmt_price(payload.get("entry", ""))
    stop_val   = _fmt_price(payload.get("stop", ""))
    targets    = payload.get("targets", [])
    rr_first   = safe_html(str(payload.get("rr_first", "")))
    rr_best    = safe_html(str(payload.get("rr_best", "")))
    stop_atr   = safe_html(str(payload.get("stop_atr", "")))
    entry_ctx  = safe_html(str(payload.get("entry_context", "")))
    stop_note  = safe_html(str(payload.get("stop_note", "")))
    risk_line  = safe_html(str(payload.get("risk_line", "")))

    # Build targets line with TP labels
    target_lines: list[str] = []
    for i, t in enumerate(targets, 1):
        target_lines.append(f"tp{i}: <b>${safe_html(_fmt_price(t))}</b>")

    sug = payload.get("suggested", {})

    def _sug_line(key: str, label: str) -> str:
        val  = _fmt_price(sug.get(key, ""))
        why  = safe_html(str(sug.get(f"{key}_why", "")))
        if why:
            return f"  {label}:  <b>${val}</b>  <i>({why})</i>"
        return f"  {label}:  <b>${val}</b>"

    lines = [
        f"<b>{symbol} {direction} — {verdict}</b>",
        f"<i>{opener}</i>",
        "",
        f"entry:  <b>${safe_html(entry_val)}</b>",
        f"stop:   <b>${safe_html(stop_val)}</b>",
    ]
    for tl in target_lines:
        lines.append(tl)

    lines += [
        "",
        f"R/R:  first <b>{rr_first}R</b>  ·  best <b>{rr_best}R</b>",
        f"ATR ({tf}):  {stop_atr}",
        f"<i>{entry_ctx}</i>",
        f"<i>{stop_note}</i>",
        "",
        "<b>fred's levels:</b>",
        _sug_line("entry", "entry"),
        _sug_line("stop",  "sl"),
        _sug_line("tp1",   "tp1"),
        _sug_line("tp2",   "tp2"),
    ]

    position = payload.get("position")
    if position:
        lines += [
            "",
            "<b>position sizing:</b>",
            f"  margin:    <b>${safe_html(str(position.get('margin_usd', '')))}",
            f"  leverage:  <b>{safe_html(str(position.get('leverage', '')))}x</b>",
            f"  notional:  <b>${safe_html(str(position.get('notional_usd', '')))}</b>",
            f"  qty:       {safe_html(str(position.get('qty', '')))}",
            f"  stop PnL:  <b>${safe_html(str(position.get('stop_pnl_usd', '')))}</b>",
        ]
        for row in position.get("tp_pnls", [])[:3]:
            lines.append(
                f"  tp {safe_html(_fmt_price(row.get('tp', '')))}: "
                f"<b>${safe_html(str(row.get('pnl_usd', '')))}</b>"
            )

    size_note = safe_html(str(payload.get("size_note", "")).strip())
    if size_note:
        lines += ["", f"<i>{size_note}</i>"]

    lines += ["", f"<i>{risk_line}</i>"]

    if tone == "wild":
        lines.append(_pick(WILD_CLOSERS))
    elif tone == "standard":
        lines.append(_pick(STANDARD_CLOSERS))
    return "\n".join(lines)


def trade_math_template(payload: dict, settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "wild":
        opener = "Trade math locked. Here's the risk map."
    elif tone == "standard":
        opener = "Trade math summary."
    else:
        opener = "Trade risk/reward summary."

    direction = safe_html(str(payload.get("direction", "")))
    entry = safe_html(str(payload.get("entry", "")))
    stop = safe_html(str(payload.get("stop", "")))
    targets = ", ".join(safe_html(str(x)) for x in payload.get("targets", []))
    risk_pu = safe_html(str(payload.get("risk_per_unit", "")))
    best_r = safe_html(str(payload.get("best_r", "")))

    lines = [
        "<b>Trade Math</b>",
        f"<i>{opener}</i>",
        "",
        f"Direction: {direction}",
        f"Entry:     {entry}",
        f"Stop:      {stop}",
        f"Targets:   {targets}",
        "",
        f"Risk/unit: {risk_pu}",
        f"Best R:    {best_r}",
    ]
    for row in payload.get("rows", [])[:4]:
        lines.append(f"  TP {safe_html(str(row.get('tp', '')))}: {safe_html(str(row.get('r_multiple', '')))}R")

    position = payload.get("position")
    if position:
        lines += [
            "",
            "<b>Position sizing:</b>",
            f"  Margin:    ${safe_html(str(position.get('margin_usd', '')))}",
            f"  Leverage:  {safe_html(str(position.get('leverage', '')))}x",
            f"  Notional:  ${safe_html(str(position.get('notional_usd', '')))}",
            f"  Qty:       {safe_html(str(position.get('qty', '')))}",
            f"  Stop PnL:  ${safe_html(str(position.get('stop_pnl_usd', '')))}",
        ]
        for row in position.get("tp_pnls", [])[:4]:
            lines.append(f"  TP {safe_html(str(row.get('tp', '')))}: ${safe_html(str(row.get('pnl_usd', '')))}")
    return "\n".join(lines)


def asset_unsupported_template(payload: dict, settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "wild":
        opener = "Can't chart this one clean right now."
        closer = "Send a chart link or contract address and I'll work with that."
    elif tone == "standard":
        opener = "Can't fetch full technical data for this asset right now."
        closer = "Try BTC, ETH, SOL, or share chart/contract details."
    else:
        opener = "Data is currently unavailable for this asset."
        closer = "Please share additional context or another symbol."

    sym = safe_html(str(payload.get("symbol", "")))
    reason = safe_html(str(payload.get("reason", "")))
    narrative = safe_html(str(payload.get("narrative", "")))
    safe_action = safe_html(str(payload.get("safe_action", "")))

    lines = [
        f"<b>{sym}</b>",
        f"<i>{opener}</i>",
        "",
        f"Reason: {reason}",
        "",
        narrative,
        f"<i>{safe_action}</i>",
    ]
    if payload.get("headlines"):
        lines += ["", "<b>Recent context:</b>"]
        for item in payload["headlines"][:2]:
            lines.append(f"  — {safe_html(item.get('title', ''))}")
    if payload.get("alternatives"):
        alts = ", ".join(safe_html(a) for a in payload["alternatives"])
        lines += ["", f"Alternatives I can map now: {alts}"]
    lines += ["", closer]
    return "\n".join(lines)


def giveaway_status_template(payload: dict) -> str:
    if payload.get("active"):
        return (
            f"<b>Giveaway #{payload['id']} — ACTIVE</b>\n\n"
            f"Prize: {safe_html(str(payload['prize']))}\n"
            f"Participants: {payload['participants']}\n"
            f"Ends in: {payload['seconds_left']}s\n\n"
            "Use /join to enter."
        )
    if payload.get("message"):
        return safe_html(str(payload["message"]))
    winner = payload.get("winner_user_id")
    winner_txt = str(winner) if winner else "none"
    return (
        f"<b>Giveaway #{payload.get('id')} — {safe_html(str(payload.get('status', '')))}</b>\n\n"
        f"Prize: {safe_html(str(payload.get('prize', '')))}\n"
        f"Winner: {winner_txt}"
    )


def help_text() -> str:
    lines = [
        "<b>Ghost Alpha — Quick Reference</b>",
        "",
        "<b>Free-talk examples</b>",
        "<code>SOL long</code>",
        "<code>SOL 4h</code>",
        "<code>chart BTC 1h</code>",
        "<code>rsi top 10 1h oversold</code>",
        "<code>ema 200 4h top 10</code>",
        "<code>latest crypto news</code>",
        "<code>cpi news</code>",
        "<code>alert me when SOL hits 100</code>",
        "<code>list my alerts</code>",
        "<code>clear my alerts</code>",
        "<code>find pair xion</code>",
        "<code>coin around 0.155</code>",
        "<code>scan solana &lt;address&gt;</code>",
        "<code>check this trade from yesterday: ETH entry 2100 stop 2165 targets 2043 2027 1991 timeframe 1h</code>",
        "",
        "<b>Slash commands</b>",
        "<code>/alpha &lt;symbol&gt; [tf] [ema=..] [rsi=..]</code>",
        "<code>/watch &lt;symbol&gt; [tf]</code>",
        "<code>/chart &lt;symbol&gt; [tf]</code>",
        "<code>/heatmap &lt;symbol&gt;</code>",
        "<code>/rsi &lt;tf&gt; &lt;overbought|oversold&gt; [topN] [len]</code>",
        "<code>/ema &lt;ema_len&gt; &lt;tf&gt; [topN]</code>",
        "<code>/news [crypto|openai|cpi|fomc] [limit]</code>",
        "<code>/alert &lt;symbol&gt; &lt;price&gt; [above|below|cross]</code>",
        "<code>/alerts  /alertdel &lt;id&gt;  /alertclear [symbol]</code>",
        "<code>/findpair &lt;price_or_query&gt;</code>",
        "<code>/setup &lt;freeform setup text&gt;</code>",
        "<code>/scan &lt;chain&gt; &lt;address&gt;</code>",
        "<code>/tradecheck  /cycle</code>",
        "<code>/giveaway  /join</code>",
        "",
        "<i>Tip: free-talk works — commands are optional.</i>",
    ]
    return "\n".join(lines)


def settings_text(settings: dict) -> str:
    lines = [
        "<b>Settings</b>",
        "",
        f"risk_profile:          {safe_html(str(settings.get('risk_profile', '')))}",
        f"preferred_timeframe:   {safe_html(str(settings.get('preferred_timeframe', '')))}",
        f"preferred_timeframes:  {safe_html(str(settings.get('preferred_timeframes', '')))}",
        f"preferred_ema_periods: {safe_html(str(settings.get('preferred_ema_periods', '')))}",
        f"preferred_rsi_periods: {safe_html(str(settings.get('preferred_rsi_periods', '')))}",
        f"preferred_exchange:    {safe_html(str(settings.get('preferred_exchange', '')))}",
        f"tone_mode:             {safe_html(str(settings.get('tone_mode', '')))}",
        f"anon_mode:             {safe_html(str(settings.get('anon_mode', '')))}",
        f"profanity_level:       {safe_html(str(settings.get('profanity_level', '')))}",
        f"formal_mode:           {safe_html(str(settings.get('formal_mode', '')))}",
    ]
    return "\n".join(lines)


def smalltalk_reply(settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "formal":
        return "Ghost Alpha is running normally.\nWhat would you like: analysis, alerts, wallet scan, or market news?"
    if tone == "standard":
        return "Ghost Alpha online.\nWant analysis, alerts, wallet scan, or market news?"
    return random.choice(SMALLTALK_REPLIES)


def unknown_prompt() -> str:
    return random.choice(UNKNOWN_FOLLOWUPS)
