from __future__ import annotations

import random
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

SMALLTALK_REPLIES = [
    "Ghost Alpha online. Scanning the tape and filtering noise. You want a setup, an alert, or a wallet scan?",
    "All systems green. Market is choppy but clean levels still print. Drop a ticker.",
    "Quiet session. I am watching liquidity and breakout zones. What are we hunting?",
    "Operational. Alerts armed, charts loaded. Send a coin and direction.",
    "I am good. Lurking where clean entries live. Want analysis, watchlist, or news?",
    "Stable and tracking. If it moves, I will see it. Scalp or swing today?",
    "Locked in. Market is noisy, signal is not. Say `SOL long` or `ETH short`.",
    "Signal check complete. Give me a ticker and I will map entry/targets fast.",
    "Ghost Alpha active. Watching BTC lead and alt reactions. What do you want first?",
    "Scanner is hot. I can run trade plans, alerts, and wallet scans. Your call.",
]

WILD_ACKS = [
    "Ghost Alpha in position. I see the level.",
    "Scanner locked. Signal is cleaner than the noise.",
    "I tracked your setup. Here is the real risk.",
    "Tape is talking. Let us keep this surgical.",
]

WILD_WARNINGS = [
    "You are one bad wick away from pain if sizing is loose.",
    "If liquidity hunts your stop, do not revenge the move.",
    "Respect invalidation or the market will do it for you.",
]

WILD_CLOSERS = [
    "Drop the next ticker and I will map it fast.",
    "If you want, I can set the alert level now.",
    "Keep it clean. One setup at a time.",
]

STANDARD_ACKS = [
    "Setup processed.",
    "Analysis ready.",
    "Plan generated.",
]

STANDARD_WARNINGS = [
    "Keep risk controlled and respect your stop.",
]

STANDARD_CLOSERS = [
    "Send another ticker if you want a follow-up.",
]

UNKNOWN_FOLLOWUPS = [
    "Give me a clear target: `SOL long`, `cpi news`, `chart BTC 1h`, or `alert me when BTC hits 70000`.",
    "I can route free text. Try `ETH short 4h`, `rsi top 10 1h oversold`, or `list my alerts`.",
    "Send the intent directly: `coins to watch 5`, `openai updates`, `scan solana <address>`.",
]


def _style_prefix(settings: dict) -> str:
    if settings.get("formal_mode"):
        return ""
    if settings.get("anon_mode", True):
        return "anon: "
    return ""


def _tone_mode(settings: dict) -> str:
    if settings.get("formal_mode"):
        return "formal"
    tone = str(settings.get("tone_mode", "wild")).lower()
    return "wild" if tone not in {"formal", "standard", "wild"} else tone


def _pick(options: list[str]) -> str:
    return random.choice(options)


def _render_summary(summary: str, settings: dict) -> str:
    text = summary.strip()
    if settings.get("formal_mode"):
        return text
    prof = settings.get("profanity_level", "light")
    if prof == "medium":
        text = text.replace("volatile", "wild")
    return f"{_style_prefix(settings)}{text}"


def trade_plan_template(plan: dict, settings: dict, detailed: bool = False) -> str:
    tone = _tone_mode(settings)
    if tone == "wild":
        ack = _pick(WILD_ACKS)
        warning = _pick(WILD_WARNINGS)
        closer = _pick(WILD_CLOSERS)
    elif tone == "standard":
        ack = _pick(STANDARD_ACKS)
        warning = _pick(STANDARD_WARNINGS)
        closer = _pick(STANDARD_CLOSERS)
    else:
        ack = ""
        warning = "Manage risk and respect invalidation."
        closer = ""

    lines = [*( [ack] if ack else [] ), _render_summary(plan["summary"], settings), ""]
    lines.extend(
        [
            "Entry: " + plan["entry"],
            "TP1: " + plan["tp1"],
            "TP2: " + plan["tp2"],
            "SL: " + plan["sl"],
            "",
        ]
    )

    why_items = plan.get("why", [])
    for bullet in (why_items if detailed else why_items[:2]):
        lines.append(f"- {bullet}")

    if detailed and plan.get("mtf_snapshot"):
        lines.append("")
        lines.append("MTF snapshot:")
        for row in plan["mtf_snapshot"][:4]:
            lines.append(f"- {row}")

    if detailed and plan.get("input_notes"):
        lines.append("")
        for note in plan["input_notes"][:3]:
            lines.append(f"Note: {note}")

    if detailed:
        updated = plan.get("updated_at")
        if updated:
            lines.append("")
            lines.append(f"Updated: {relative_updated(updated) or updated}")

    lines.append(warning)
    if closer:
        lines.append(closer)
    lines.append(plan.get("risk", "Not financial advice."))
    return "\n".join(lines)


def watchlist_template(payload: dict) -> str:
    lines = [
        payload["summary"],
        "Scanning strongest rotation + narrative names.",
        "",
    ]
    for idx, item in enumerate(payload["items"], start=1):
        lines.append(f"{idx}. {item}")

    lines.append("")
    lines.append("Not financial advice. Use risk controls.")
    return "\n".join(lines)


def news_template(payload: dict) -> str:
    lines = [payload["summary"], "", "Top headlines to track:"]
    headlines = payload.get("headlines", [])[:5]
    if not headlines:
        lines.append("- No fresh items from configured feeds.")
        lines.append("")
        lines.append(payload["vibe"])
        lines.append("Not financial advice.")
        return "\n".join(lines)

    for idx, item in enumerate(headlines, start=1):
        clean_url = _clean_url(item["url"])
        lines.append("")
        lines.append(f"{idx}) {item['title']}")
        lines.append(f"Link: {clean_url}")
        lines.append("")

    lines.append(f"Updated: {relative_updated(payload.get('updated_at')) or payload.get('updated_at', 'n/a')}")
    lines.append(payload["vibe"])
    lines.append("Not financial advice.")
    return "\n".join(lines)


def wallet_scan_template(payload: dict) -> str:
    balance_line = f"{payload['native_balance']:.6f} {payload['native_symbol']}"
    if payload.get("native_usd") is not None:
        balance_line += f" (~${payload['native_usd']:.2f})"

    lines = [
        f"{payload['chain'].upper()} wallet snapshot ready.",
        f"Address scanned: {payload['address']}",
        "",
        f"Native balance: {balance_line}",
        "Top tokens:",
    ]

    tokens = payload.get("tokens", [])[:8]
    if not tokens:
        lines.append("- No non-native holdings detected.")
    else:
        for token in tokens:
            lines.append(f"- {token.get('symbol', 'UNK')}: {token.get('amount', 0)}")

    if payload.get("resources"):
        lines.append("Resources:")
        for k, v in payload["resources"].items():
            lines.append(f"- {k}: {v}")

    lines.append("Recent tx count: " + str(len(payload.get("recent_transactions", []))))
    lines.append("Warnings:")
    for warn in payload.get("warnings", [])[:4]:
        lines.append(f"- {warn}")

    lines.append("Not financial advice.")
    return "\n".join(lines)


def cycle_template(payload: dict) -> str:
    lines = [payload["summary"], "", f"Confidence: {payload['confidence']:.0%}"]
    for b in payload["bullets"]:
        lines.append(f"- {b}")
    lines.append("Not financial advice.")
    return "\n".join(lines)


def trade_verification_template(payload: dict) -> str:
    if payload.get("result") == "not_filled":
        text = (
            f"Trade check for {payload['symbol']}\n"
            f"Result: not filled\n"
            f"{payload['note']}\n"
            "Not financial advice."
        )
        return text

    lines = [
        f"Trade check completed for {payload['symbol']} ({payload['direction']}).",
        f"Result: {payload['result']} | Mode: {payload['mode']}",
        "",
        f"Filled at: {payload.get('filled_at')}",
        f"First hit: {payload.get('first_hit')}",
        f"MFE: {payload.get('mfe')}",
        f"MAE: {payload.get('mae')}",
        f"R multiple: {payload.get('r_multiple')}",
        "",
        "Not financial advice.",
    ]
    return "\n".join(lines)


def correlation_template(payload: dict) -> str:
    lines = [payload["summary"], ""]
    for b in payload["bullets"]:
        lines.append(f"- {b}")
    lines.append("Not financial advice.")
    return "\n".join(lines)


def help_text() -> str:
    lines = [
        "Ghost Alpha Help",
        "",
        "Free-talk examples:",
        "- SOL long",
        "- SOL 4h",
        "- chart BTC 1h",
        "- rsi top 10 1h oversold",
        "- ema 200 4h top 10",
        "- latest crypto news",
        "- cpi news",
        "- alert me when SOL hits 100",
        "- list my alerts",
        "- clear my alerts",
        "- find pair xion",
        "- coin around 0.155",
        "- scan solana <address>",
        "- check this trade from yesterday: ETH entry 2100 stop 2165 targets 2043 2027 1991 timeframe 1h",
        "",
        "Slash commands (optional):",
        "- /alpha <symbol> [tf] [ema=..] [rsi=..]",
        "- /watch <symbol> [tf]",
        "- /chart <symbol> [tf]",
        "- /heatmap <symbol>",
        "- /rsi <tf> <overbought|oversold> [topN] [len]",
        "- /ema <ema_len> <tf> [topN]",
        "- /news [crypto|openai|cpi|fomc] [limit]",
        "- /alert <symbol> <price> [above|below|cross]",
        "- /alerts | /alertdel <id> | /alertclear [symbol]",
        "- /findpair <price_or_query>",
        "- /setup <freeform setup text>",
        "- /scan <chain> <address>",
        "- /tradecheck | /cycle",
        "- /giveaway (opens buttons)",
        "- /join",
        "",
        "Tip: You can just type naturally. Commands are optional.",
    ]
    return "\n".join(lines)


def settings_text(settings: dict) -> str:
    return (
        "Settings:\n"
        f"- risk_profile: {settings.get('risk_profile')}\n"
        f"- preferred_timeframe: {settings.get('preferred_timeframe')}\n"
        f"- preferred_timeframes: {settings.get('preferred_timeframes')}\n"
        f"- preferred_ema_periods: {settings.get('preferred_ema_periods')}\n"
        f"- preferred_rsi_periods: {settings.get('preferred_rsi_periods')}\n"
        f"- preferred_exchange: {settings.get('preferred_exchange')}\n"
        f"- tone_mode: {settings.get('tone_mode')}\n"
        f"- anon_mode: {settings.get('anon_mode')}\n"
        f"- profanity_level: {settings.get('profanity_level')}\n"
        f"- formal_mode: {settings.get('formal_mode')}"
    )


def smalltalk_reply(settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "formal":
        return "Ghost Alpha is running normally. What would you like: analysis, alerts, wallet scan, or market news?"
    if tone == "standard":
        return "Ghost Alpha online. Want analysis, alerts, wallet scan, or market news?"
    return random.choice(SMALLTALK_REPLIES)


def setup_review_template(payload: dict, settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "wild":
        opener = "I see the level. Here is where it breaks."
        closer = _pick(WILD_CLOSERS)
    elif tone == "standard":
        opener = "Setup review complete."
        closer = _pick(STANDARD_CLOSERS)
    else:
        opener = "Setup review completed."
        closer = ""

    lines = [
        opener,
        f"{payload['symbol']} {payload['direction']} setup verdict: {payload['verdict'].upper()}",
        "",
        f"Entry: {payload['entry']}",
        f"SL: {payload['stop']}",
        "Targets: " + ", ".join(str(x) for x in payload["targets"]),
        "",
        f"R/R: first {payload['rr_first']}R | best {payload['rr_best']}R",
        f"Stop tightness: {payload['stop_atr']} ATR",
        f"Context: {payload['entry_context']}",
        f"Risk note: {payload['stop_note']}",
        "",
        "Improved levels:",
        f"- Entry: {payload['suggested']['entry']}",
        f"- SL: {payload['suggested']['stop']}",
        f"- TP1: {payload['suggested']['tp1']}",
        f"- TP2: {payload['suggested']['tp2']}",
        "",
        payload.get("size_note", ""),
        "",
        payload["risk_line"],
    ]
    position = payload.get("position")
    if position:
        lines.insert(-2, "Position sizing:")
        lines.insert(-2, f"- Margin: ${position['margin_usd']}")
        lines.insert(-2, f"- Leverage: {position['leverage']}x")
        lines.insert(-2, f"- Notional: ${position['notional_usd']}")
        lines.insert(-2, f"- Qty: {position['qty']}")
        lines.insert(-2, f"- Stop PnL: ${position['stop_pnl_usd']}")
        for row in position.get("tp_pnls", [])[:3]:
            lines.insert(-2, f"- TP {row['tp']}: ${row['pnl_usd']}")
        lines.insert(-2, "")
    if closer:
        lines.append(closer)
    return "\n".join(lines)


def trade_math_template(payload: dict, settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "wild":
        opener = "Trade math locked. Here is the risk map."
    elif tone == "standard":
        opener = "Trade math summary."
    else:
        opener = "Trade risk/reward summary."

    lines = [
        opener,
        f"Direction: {payload['direction']}",
        f"Entry: {payload['entry']}",
        f"Stop: {payload['stop']}",
        "Targets: " + ", ".join(str(x) for x in payload["targets"]),
        "",
        f"Risk per unit: {payload['risk_per_unit']}",
        f"Best R: {payload['best_r']}",
    ]
    for row in payload.get("rows", [])[:4]:
        lines.append(f"- TP {row['tp']}: {row['r_multiple']}R")

    position = payload.get("position")
    if position:
        lines.extend(
            [
                "",
                "Position sizing:",
                f"- Margin: ${position['margin_usd']}",
                f"- Leverage: {position['leverage']}x",
                f"- Notional: ${position['notional_usd']}",
                f"- Qty: {position['qty']}",
                f"- Stop PnL: ${position['stop_pnl_usd']}",
            ]
        )
        for row in position.get("tp_pnls", [])[:4]:
            lines.append(f"- TP {row['tp']}: ${row['pnl_usd']}")
    lines.extend(["", "Not financial advice."])
    return "\n".join(lines)


def asset_unsupported_template(payload: dict, settings: dict) -> str:
    tone = _tone_mode(settings)
    if tone == "wild":
        opener = "I can't chart this one clean right now."
        closer = "Send a chart link or contract address and I'll work with that."
    elif tone == "standard":
        opener = "I can't fetch full technical data for this asset right now."
        closer = "Try BTC, ETH, SOL, or share chart/contract details."
    else:
        opener = "Data is currently unavailable for this asset."
        closer = "Please share additional context or another symbol."

    lines = [
        opener,
        f"Symbol: {payload['symbol']}",
        f"Reason: {payload['reason']}",
        "",
        f"Narrative: {payload['narrative']}",
        f"Safe action: {payload['safe_action']}",
    ]
    if payload.get("headlines"):
        lines.append("")
        lines.append("Recent context:")
        for item in payload["headlines"][:2]:
            lines.append(f"- {item['title']}")
    if payload.get("alternatives"):
        lines.append("")
        lines.append("Alternatives I can map now: " + ", ".join(payload["alternatives"]))
    lines.append(closer)
    lines.append("Not financial advice.")
    return "\n".join(lines)


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


def rsi_scan_template(payload: dict) -> str:
    lines = [payload["summary"], ""]
    if not payload.get("items"):
        lines.append("No RSI rows available right now for that request.")
        lines.append("Try another timeframe or symbol.")
        return "\n".join(lines)
    for idx, row in enumerate(payload["items"], start=1):
        lines.append(f"{idx}. {row['symbol']} RSI({payload['rsi_length']}) {payload['timeframe']} = {row['rsi']} ({row['note']})")
    lines.append("")
    lines.append("Not financial advice.")
    return "\n".join(lines)


def pair_find_template(payload: dict) -> str:
    lines = [payload["summary"], f"Query: {payload['query']}", ""]
    matches = payload.get("matches", [])
    if not matches:
        lines.append("No direct match. Try ticker, full name, or contract address.")
    else:
        for idx, row in enumerate(matches, start=1):
            pair = row.get("pair") or "n/a"
            tradable = "YES" if row.get("tradable_binance") else "NO"
            price = row.get("price")
            price_txt = f"${price}" if price is not None else "n/a"
            lines.append(f"{idx}. {row['symbol']} ({row.get('name')})")
            lines.append(f"   Pair: {pair} | Tradable: {tradable} | Price: {price_txt}")
    lines.append("")
    lines.append("If candles are missing, I can still provide narrative + safer execution rules.")
    return "\n".join(lines)


def price_guess_template(payload: dict) -> str:
    lines = [payload["summary"], ""]
    matches = payload.get("matches", [])
    if not matches:
        lines.append("No close candidates found. Try a wider hint or exact range.")
        return "\n".join(lines)
    for idx, row in enumerate(matches[:10], start=1):
        tradable = "YES" if row.get("tradable_binance") else "NO"
        lines.append(f"{idx}. {row['symbol']} ({row.get('name')}) - ${row['price']} | Tradable: {tradable}")
    lines.append("")
    lines.append("Not financial advice.")
    return "\n".join(lines)


def giveaway_status_template(payload: dict) -> str:
    if payload.get("active"):
        return (
            f"Giveaway #{payload['id']} is ACTIVE\n"
            f"Prize: {payload['prize']}\n"
            f"Participants: {payload['participants']}\n"
            f"Ends in: {payload['seconds_left']}s\n"
            "Users join with /join"
        )
    if payload.get("message"):
        return payload["message"]
    winner = payload.get("winner_user_id")
    winner_txt = str(winner) if winner else "none"
    return (
        f"Latest giveaway #{payload.get('id')} ({payload.get('status')})\n"
        f"Prize: {payload.get('prize')}\n"
        f"Winner: {winner_txt}"
    )


def unknown_prompt() -> str:
    return random.choice(UNKNOWN_FOLLOWUPS)
