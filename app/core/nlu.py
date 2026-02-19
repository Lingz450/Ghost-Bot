from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class Intent(str, Enum):
    ANALYSIS = "analysis"
    RSI_SCAN = "rsi_scan"
    EMA_SCAN = "ema_scan"
    CHART = "chart"
    HEATMAP = "heatmap"
    TRADE_MATH = "trade_math"
    SETUP_REVIEW = "setup_review"
    ASSET_UNSUPPORTED = "asset_unsupported"
    SMALLTALK = "smalltalk"
    ALERT_CREATE = "alert_create"
    ALERT_LIST = "alert_list"
    ALERT_DELETE = "alert_delete"
    ALERT_CLEAR = "alert_clear"
    ALERT_PAUSE = "alert_pause"
    ALERT_RESUME = "alert_resume"
    WATCHLIST = "watchlist"
    PAIR_FIND = "pair_find"
    PRICE_GUESS = "price_guess"
    GIVEAWAY_JOIN = "giveaway_join"
    GIVEAWAY_START = "giveaway_start"
    GIVEAWAY_STATUS = "giveaway_status"
    GIVEAWAY_CANCEL = "giveaway_cancel"
    GIVEAWAY_REROLL = "giveaway_reroll"
    NEWS = "news"
    SCAN_WALLET = "scan_wallet"
    CYCLE = "cycle"
    TRADECHECK = "tradecheck"
    CORRELATION = "correlation"
    SETTINGS = "settings"
    HELP = "help"
    START = "start"
    UNKNOWN = "unknown"


SYMBOL_RE = re.compile(r"\b[A-Za-z]{2,12}\b")
TIMEFRAME_RE = re.compile(r"\b(1m|3m|5m|15m|30m|1h|2h|4h|6h|12h|1d|3d|1w|daily|weekly|monthly)\b", re.IGNORECASE)
PRICE_RE = re.compile(r"(?<!\w)((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\w)")
SOL_ADDRESS_RE = re.compile(r"\b[1-9A-HJ-NP-Za-km-z]{32,44}\b")
TRON_ADDRESS_RE = re.compile(r"\bT[1-9A-HJ-NP-Za-km-z]{33}\b")
SMALLTALK_RE = re.compile(
    r"\b("
    r"how are you|how you doing|how are you doing|how are you doing today|"
    r"how's it going|hows it going|"
    r"what are we doing today|what we doing today|"
    r"hi|hello|hey|yo|sup|gm|good morning|good afternoon|good evening|what's up|whats up"
    r")\b"
)
SMALLTALK_TYPO_RE = re.compile(
    r"\b("
    r"hwo|howw|hw|hru|how r u|how are u|how u doing|how ya doing|"
    r"helo|helllo|hii|heyy"
    r")\b",
    re.IGNORECASE,
)
SMALLTALK_EXCLUDE_RE = re.compile(
    r"\b(long|short|alert|scan|trade|entry|stop|target|cycle|watch|news|following|correlation|rsi|ema|chart|heatmap)\b",
    re.IGNORECASE,
)
TIMEFRAME_TOKEN_RE = re.compile(r"\b\d+[mhdwM]\b")
NEWS_WORD_RE = re.compile(r"\b(news|headline|headlines|update|updates|brief|digest|changelog|release)\b", re.IGNORECASE)
MACRO_NEWS_RE = re.compile(r"\b(cpi|inflation|fomc|fed|powell|rates?|nfp|jobs?|ppi|macro)\b", re.IGNORECASE)
OPENAI_NEWS_RE = re.compile(r"\b(openai|chatgpt|gpt|codex|responses?\s+api)\b", re.IGNORECASE)

SUPPORTED_TIMEFRAMES = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}
DEFAULT_ALL_TIMEFRAMES = ["15m", "1h", "4h", "1d"]
DEFAULT_ALL_EMAS = [9, 20, 50, 100, 200]
DEFAULT_ALL_RSIS = [7, 14, 21]
MAX_TIMEFRAMES = 4
MAX_EMA_PERIODS = 5
MAX_RSI_PERIODS = 3

COMMON_STOPWORDS = {
    "long",
    "short",
    "coins",
    "watch",
    "ping",
    "when",
    "hits",
    "alert",
    "above",
    "below",
    "cross",
    "scan",
    "this",
    "address",
    "is",
    "it",
    "following",
    "check",
    "trade",
    "sl",
    "tp",
    "tp1",
    "tp2",
    "lev",
    "rr",
    "pnl",
    "margin",
    "amount",
    "leverage",
    "from",
    "yesterday",
    "entry",
    "stop",
    "targets",
    "latest",
    "news",
    "today",
    "what",
    "happening",
    "with",
    "me",
    "rsi",
    "top",
    "overbought",
    "oversold",
    "pair",
    "find",
    "around",
    "about",
    "near",
    "coin",
    "token",
    "price",
    "to",
    "and",
    "a",
    "an",
    "set",
    "right",
    "now",
    "my",
    "alerts",
    "reset",
    "cpi",
    "inflation",
    "fomc",
    "fed",
    "macro",
    "openai",
    "chatgpt",
    "gpt",
    "codex",
    "crypto",
    "remove",
    "delete",
    "chart",
    "candle",
    "candles",
    "plot",
    "heatmap",
    "orderbook",
    "order",
    "book",
    "depth",
    "pick",
    "winner",
    "winners",
    "giveaway",
    "run",
    "start",
    "status",
    "cancel",
    "reroll",
    "join",
}


@dataclass
class ParsedMessage:
    intent: Intent
    entities: dict[str, Any] = field(default_factory=dict)
    requires_followup: bool = False
    followup_question: str | None = None


def normalize_symbol(raw: str) -> str:
    token = raw.strip().upper()
    mapping = {
        "XBT": "BTC",
        "BITCOIN": "BTC",
        "ETHER": "ETH",
        "ETHEREUM": "ETH",
        "SOLANA": "SOL",
    }
    return mapping.get(token, token)


def parse_timeframe(text: str) -> str | None:
    match = TIMEFRAME_RE.search(text)
    if match:
        tf = match.group(1).lower()
        mapping = {"daily": "1d", "weekly": "1w", "monthly": "1M"}
        return mapping.get(tf, tf)

    lower = text.lower()
    flex_match = re.search(
        r"\b(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks)\b",
        lower,
    )
    if not flex_match:
        return None
    n = int(flex_match.group(1))
    unit = flex_match.group(2)
    if unit.startswith("m"):
        tf = f"{n}m"
    elif unit.startswith("h"):
        tf = f"{n}h"
    elif unit.startswith("d"):
        tf = f"{n}d"
    else:
        tf = f"{n}w"
    return tf if tf in SUPPORTED_TIMEFRAMES else None


def _dedupe_list(items: list[Any]) -> list[Any]:
    out: list[Any] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _normalize_timeframe_token(token: str) -> str:
    token = token.strip()
    if not token:
        return token
    if token.endswith("M"):
        return token
    return token.lower()


def parse_timeframes_request(text: str) -> tuple[list[str] | None, bool, list[str]]:
    notes: list[str] = []
    lower = text.lower()
    all_requested = bool(re.search(r"\ball\s+timeframes?\b|\ball\s+tfs?\b", lower))

    tokens: list[str] = []

    for match in re.finditer(r"\btf\s*[:=]\s*([0-9mhdwM,\s]+)", text, flags=re.IGNORECASE):
        raw = match.group(1)
        for chunk in re.split(r"[\s,]+", raw.strip()):
            if chunk:
                tokens.append(_normalize_timeframe_token(chunk))

    for token in TIMEFRAME_TOKEN_RE.findall(text):
        tokens.append(_normalize_timeframe_token(token))

    flex_chunks = re.findall(
        r"\b(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|week|weeks)\b",
        lower,
    )
    for n_raw, unit in flex_chunks:
        n = int(n_raw)
        if unit.startswith("m"):
            tokens.append(f"{n}m")
        elif unit.startswith("h"):
            tokens.append(f"{n}h")
        elif unit.startswith("d"):
            tokens.append(f"{n}d")
        else:
            tokens.append(f"{n}w")

    tokens = _dedupe_list(tokens)

    valid: list[str] = []
    for token in tokens:
        if token in SUPPORTED_TIMEFRAMES:
            valid.append(token)
        else:
            notes.append(f"Ignored unsupported timeframe: {token}")

    if all_requested and not valid:
        valid = DEFAULT_ALL_TIMEFRAMES.copy()

    if len(valid) > MAX_TIMEFRAMES:
        valid = valid[:MAX_TIMEFRAMES]
        notes.append("Running max 4 timeframes to keep analysis fast.")

    return (valid or None), all_requested, notes


def _parse_periods(
    text: str,
    kind: str,
    max_items: int,
    max_value: int = 500,
) -> tuple[list[int] | None, bool, list[str]]:
    notes: list[str] = []
    lower = text.lower()
    all_requested = bool(re.search(rf"\ball\s+{kind}s?\b", lower))

    periods: list[int] = []

    assignment_match = re.search(rf"\b{kind}\s*[:=]\s*([0-9,\s]+)", lower)
    if assignment_match:
        for piece in re.split(r"[,\s]+", assignment_match.group(1).strip()):
            if piece.isdigit():
                periods.append(int(piece))

    for match in re.finditer(rf"\b{kind}\s*([0-9]{{1,4}})\b", lower):
        periods.append(int(match.group(1)))

    valid = _dedupe_list([p for p in periods if 2 <= p <= max_value])

    if len(valid) > max_items:
        valid = valid[:max_items]
        notes.append(f"Running max {max_items} {kind.upper()} periods to keep it fast.")

    return (valid or None), all_requested, notes


def _looks_like_smalltalk(lower: str) -> bool:
    if SMALLTALK_EXCLUDE_RE.search(lower):
        return False

    if SMALLTALK_RE.search(lower) or SMALLTALK_TYPO_RE.search(lower):
        return True

    if re.search(r"\b(how|hwo|hw)\b.{0,16}\b(are|r)\b.{0,8}\b(you|u)\b.{0,12}\b(do|doing)\b", lower):
        return True
    if re.search(r"\b(how|hwo|hw)\s+(are|r)\s+(you|u)\b", lower):
        return True
    if re.fullmatch(r"\s*(yo+|hey+|heyy+|hii+|helo+|hello+|sup+|gm+)\s*[!?.,]*\s*", lower):
        return True
    return False


def parse_ema_request(text: str) -> tuple[list[int] | None, bool, list[str]]:
    return _parse_periods(text, "ema", MAX_EMA_PERIODS, max_value=500)


def parse_rsi_request(text: str) -> tuple[list[int] | None, bool, list[str]]:
    return _parse_periods(text, "rsi", MAX_RSI_PERIODS, max_value=50)


def parse_duration_token(text: str) -> str | None:
    lower = text.lower()
    match = re.search(
        r"\b(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)\b",
        lower,
    )
    if not match:
        quick = re.search(r"\b(\d+)\s*([smhd])\b", lower)
        if not quick:
            return None
        return f"{int(quick.group(1))}{quick.group(2)}"
    value = int(match.group(1))
    unit = match.group(2)
    if unit.startswith("s"):
        suffix = "s"
    elif unit.startswith("m"):
        suffix = "m"
    elif unit.startswith("h"):
        suffix = "h"
    else:
        suffix = "d"
    return f"{value}{suffix}"


def parse_timestamp(text: str, now: datetime | None = None) -> datetime | None:
    now = now or datetime.now(timezone.utc)
    lower = text.lower()

    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2}(?:[t\s]\d{2}:\d{2}(?::\d{2})?)?)\b", text)
    if iso_match:
        value = iso_match.group(1).replace(" ", "T")
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass

    if "yesterday" in lower:
        return now - timedelta(days=1)
    if "last night" in lower:
        d = now - timedelta(days=1)
        return d.replace(hour=22, minute=0, second=0, microsecond=0)

    hours_match = re.search(r"(\d+)\s*hours?\s*ago", lower)
    if hours_match:
        return now - timedelta(hours=int(hours_match.group(1)))

    mins_match = re.search(r"(\d+)\s*minutes?\s*ago", lower)
    if mins_match:
        return now - timedelta(minutes=int(mins_match.group(1)))

    return None


def _extract_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for token in SYMBOL_RE.findall(text):
        lower = token.lower()
        if lower in COMMON_STOPWORDS:
            continue
        if len(token) < 2:
            continue
        symbols.append(normalize_symbol(token))
    unique: list[str] = []
    for s in symbols:
        if s not in unique:
            unique.append(s)
    return unique


def _extract_prices(text: str) -> list[float]:
    prices = []
    for p in PRICE_RE.findall(text):
        prices.append(float(p.replace(",", "")))
    return prices


def _extract_prices_with_positions(text: str) -> list[tuple[float, int, int]]:
    out: list[tuple[float, int, int]] = []
    for m in PRICE_RE.finditer(text):
        raw = m.group(1)
        start = m.start(1)
        end = m.end(1)
        after = text[end : end + 1].lower()
        if after == "r":
            continue
        try:
            out.append((float(raw.replace(",", "")), start, end))
        except ValueError:
            continue
    return out


def _parse_single_level(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def parse_setup_levels(text: str) -> tuple[float | None, float | None, list[float]]:
    entry = _parse_single_level(
        text,
        [
            r"(?:\bentry\b|\be\b)\s*[:=]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        ],
    )
    stop = _parse_single_level(
        text,
        [
            r"(?:\bstop\b|\bsl\b)\s*[:=]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        ],
    )

    targets: list[float] = []
    target_matches = list(
        re.finditer(
            r"(?:\btargets?\b|\btp\d*\b)\s*[:=]?\s*([^\n\r]+)",
            text,
            flags=re.IGNORECASE,
        )
    )
    for tm in target_matches:
        segment = tm.group(1)[:180]
        segment = re.split(
            r"\b(leverage|lev|margin|amount|size|position|risk|rr|r:r|pnl|with)\b",
            segment,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        for val, _, _ in _extract_prices_with_positions(segment):
            targets.append(val)

    if not targets:
        all_prices = [v for v, _, _ in _extract_prices_with_positions(text)]
        filtered = []
        for val in all_prices:
            if entry is not None and abs(val - entry) < 1e-9:
                continue
            if stop is not None and abs(val - stop) < 1e-9:
                continue
            filtered.append(val)
        targets = filtered[:5]

    unique_targets: list[float] = []
    for val in targets:
        if val not in unique_targets:
            unique_targets.append(val)

    return entry, stop, unique_targets


def _parse_news_intent(text: str) -> ParsedMessage | None:
    stripped = text.strip()
    lower = stripped.lower()

    if not lower:
        return None

    is_news_command = lower.startswith("/news")
    has_news_words = bool(NEWS_WORD_RE.search(lower))
    has_macro = bool(MACRO_NEWS_RE.search(lower))
    has_openai = bool(OPENAI_NEWS_RE.search(lower))
    has_today_prompt = any(
        phrase in lower
        for phrase in (
            "what do you have for me today",
            "what happened today",
            "latest today",
            "today brief",
            "what's new with openai",
            "whats new with openai",
        )
    )
    has_newsish = bool(re.search(r"\b(news|new|update|updates|changelog|release)\b", lower))

    if not (is_news_command or has_news_words or has_today_prompt):
        if not ((has_macro or has_openai) and ("today" in lower or "latest" in lower or has_newsish)):
            return None

    topic: str | None = None
    mode = "crypto"

    if has_openai:
        topic = "openai"
        mode = "openai"
    elif re.search(r"\b(cpi|inflation|consumer price|bls|core cpi)\b", lower):
        topic = "cpi"
        mode = "macro"
    elif has_macro:
        topic = "macro"
        mode = "macro"
    else:
        symbols = _extract_symbols(stripped)
        topic = symbols[0] if symbols else "crypto"
        mode = "crypto"

    limit = 6
    n_match = re.search(r"\b(\d{1,2})\b", lower)
    if n_match:
        limit = max(3, min(int(n_match.group(1)), 10))

    return ParsedMessage(Intent.NEWS, {"topic": topic, "mode": mode, "limit": limit})


def parse_message(text: str) -> ParsedMessage:
    stripped = text.strip()
    lower = stripped.lower()

    if lower.startswith("/start"):
        return ParsedMessage(Intent.START)
    if lower.startswith("/help"):
        return ParsedMessage(Intent.HELP)
    if lower.startswith("/settings"):
        return ParsedMessage(Intent.SETTINGS)
    if lower.startswith("/join"):
        return ParsedMessage(Intent.GIVEAWAY_JOIN)
    if lower.strip() == "join":
        return ParsedMessage(Intent.GIVEAWAY_JOIN)

    if "asset unsupported" in lower or "symbol unsupported" in lower:
        return ParsedMessage(Intent.ASSET_UNSUPPORTED)

    if _looks_like_smalltalk(lower):
        return ParsedMessage(Intent.SMALLTALK)

    if lower.startswith("/alert list") or "list alerts" in lower:
        return ParsedMessage(Intent.ALERT_LIST)

    if "list my alerts" in lower:
        return ParsedMessage(Intent.ALERT_LIST)

    delete_match = re.search(r"(?:/alert\s+delete|delete\s+alert)\s+(\d+)", lower)
    if delete_match:
        return ParsedMessage(Intent.ALERT_DELETE, {"alert_id": int(delete_match.group(1))})

    if re.search(r"\b(clear|reset)\s+(my\s+)?alerts?\b|\bdelete all alerts\b", lower):
        return ParsedMessage(Intent.ALERT_CLEAR)

    if re.search(r"\bpause\s+alerts?\b", lower):
        return ParsedMessage(Intent.ALERT_PAUSE)

    if re.search(r"\bresume\s+alerts?\b", lower):
        return ParsedMessage(Intent.ALERT_RESUME)

    if re.search(r"\b(remove|delete)\s+(my\s+)?[a-z]{2,12}\s+alerts?\b", lower):
        symbols = _extract_symbols(stripped)
        symbol = symbols[0] if symbols else None
        if symbol:
            return ParsedMessage(Intent.ALERT_DELETE, {"symbol": symbol})

    if "ema" in lower and re.search(r"\b(top|closest|near|scan|coins?|which)\b", lower):
        tf_list, _, notes = parse_timeframes_request(stripped)
        timeframe = parse_timeframe(lower) or (tf_list[0] if tf_list else "1h")
        ema_periods, _, ema_notes = parse_ema_request(stripped)
        notes.extend(ema_notes)
        ema_len = int(ema_periods[0]) if ema_periods else 200
        limit_match = re.search(r"\btop\s+(\d{1,2})\b|\b(\d{1,2})\s*$", lower)
        limit = 10
        if limit_match:
            for g in limit_match.groups():
                if g and g.isdigit():
                    limit = max(1, min(int(g), 20))
                    break
        mode = "closest"
        if "above" in lower:
            mode = "above"
        elif "below" in lower:
            mode = "below"
        return ParsedMessage(
            Intent.EMA_SCAN,
            {
                "timeframe": timeframe,
                "ema_length": ema_len,
                "mode": mode,
                "limit": limit,
                "notes": notes,
            },
        )

    if "rsi" in lower and re.search(r"\b(scan|top|overbought|oversold)\b", lower):
        tf_list, _, notes = parse_timeframes_request(stripped)
        timeframe = parse_timeframe(lower) or (tf_list[0] if tf_list else "1h")
        mode = "overbought" if "overbought" in lower else "oversold"
        rsi_periods, _, rsi_notes = parse_rsi_request(stripped)
        notes.extend(rsi_notes)
        length = int(rsi_periods[0]) if rsi_periods else 14
        symbols = _extract_symbols(stripped)
        symbol = symbols[0] if symbols else None
        limit_match = re.search(r"\btop\s+(\d{1,2})\b|\bscan\s+\w+\s+\w+\s+(\d{1,2})\b|\b(\d{1,2})\s*$", lower)
        limit = 10
        if limit_match:
            for g in limit_match.groups():
                if g and g.isdigit():
                    limit = max(1, min(int(g), 20))
                    break
        return ParsedMessage(
            Intent.RSI_SCAN,
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "mode": mode,
                "limit": limit,
                "rsi_length": length,
                "notes": notes,
            },
        )

    price_guess_match = re.search(
        r"\b(coin|token)\b.*\b(around|near|about)\b|\bwhat coin is\b|\bprice near\b|\bcoin is\b",
        lower,
    )
    if price_guess_match:
        prices = _extract_prices(stripped)
        if not prices:
            return ParsedMessage(Intent.PRICE_GUESS, {}, True, "What price should I search around?")
        limit_match = re.search(r"\btop\s+(\d{1,2})\b", lower)
        limit = int(limit_match.group(1)) if limit_match else 10
        return ParsedMessage(
            Intent.PRICE_GUESS,
            {"target_price": float(prices[0]), "limit": max(1, min(limit, 20))},
        )

    if re.search(r"\bfind\s+pair\b|\bpair\s+for\b|\bfind\s+\$?[a-z]{2,12}\b", lower):
        symbols = _extract_symbols(stripped)
        query = None
        if "$" in stripped:
            m = re.search(r"\$([A-Za-z0-9]{2,12})", stripped)
            if m:
                query = m.group(1)
        if not query and symbols:
            query = symbols[0]
        if not query:
            pair_match = re.search(r"(?:find pair|pair for)\s+([A-Za-z0-9 _-]{2,40})", stripped, re.IGNORECASE)
            query = pair_match.group(1).strip() if pair_match else None
        if not query:
            return ParsedMessage(Intent.PAIR_FIND, {}, True, "Which coin should I resolve to a tradable pair?")
        return ParsedMessage(Intent.PAIR_FIND, {"query": query})

    parsed_news = _parse_news_intent(stripped)
    if parsed_news:
        return parsed_news

    if "/cycle" in lower or "cycle check" in lower or "bull market top" in lower or "halving" in lower:
        return ParsedMessage(Intent.CYCLE)

    if re.search(r"\b(chart|candles?|plot)\b", lower):
        symbols = _extract_symbols(stripped)
        if not symbols:
            return ParsedMessage(Intent.CHART, {}, True, "Which symbol should I chart?")
        return ParsedMessage(
            Intent.CHART,
            {"symbol": symbols[0], "timeframe": parse_timeframe(lower) or "1h"},
        )

    if re.search(r"\b(heatmap|order\s*book|orderbook|depth)\b", lower):
        symbols = _extract_symbols(stripped)
        symbol = symbols[0] if symbols else "BTC"
        return ParsedMessage(Intent.HEATMAP, {"symbol": symbol})

    if re.search(r"\bgiveaway\b", lower) and re.search(r"\b(status|active)\b", lower):
        return ParsedMessage(Intent.GIVEAWAY_STATUS)

    if re.search(r"\bgiveaway\b", lower) and re.search(r"\b(cancel|stop|end)\b", lower):
        return ParsedMessage(Intent.GIVEAWAY_CANCEL)

    if re.search(r"\bgiveaway\b", lower) and re.search(r"\b(reroll|pick again|new winner)\b", lower):
        return ParsedMessage(Intent.GIVEAWAY_REROLL)

    duration_token = parse_duration_token(stripped)
    has_giveaway_phrase = bool(re.search(r"\bgiveaway\b", lower))
    has_pick_winner_phrase = bool(re.search(r"\bpick (a )?winner\b", lower))
    has_giveaway_state_word = bool(re.search(r"\b(status|active|cancel|stop|end|reroll)\b", lower))
    giveaway_start = bool(
        duration_token
        and (
            has_pick_winner_phrase
            or (has_giveaway_phrase and not has_giveaway_state_word)
        )
    )
    if giveaway_start:
        prize_match = re.search(
            r"\bprize\b\s+(.+?)(?:\s+\bwinners?\b\s*\d+)?\s*$",
            stripped,
            flags=re.IGNORECASE,
        )
        winners_match = re.search(r"\bwinners?\b\s*(\d+)", lower)
        return ParsedMessage(
            Intent.GIVEAWAY_START,
            {
                "duration": duration_token or "10m",
                "prize": (prize_match.group(1).strip() if prize_match else "Prize"),
                "winners": int(winners_match.group(1)) if winners_match else 1,
            },
        )

    direction_watch = re.search(
        r"\b(?:coin|coins|which|what|best)\b.*\bto\s+(long|short)\b|\b(?:long|short)\s+(?:coin|coins)\b",
        lower,
    )
    if direction_watch and not re.search(r"\b(entry|stop|sl|targets?|tp\d*|limit)\b", lower):
        direction = "short" if "short" in direction_watch.group(0) else "long"
        n_match = re.search(r"\b(\d{1,2})\b", lower)
        singular_prompt = bool(re.search(r"\b(coin|which|what)\b", lower)) and "coins" not in lower
        n_default = 1 if singular_prompt else 5
        n = int(n_match.group(1)) if n_match else n_default
        return ParsedMessage(Intent.WATCHLIST, {"count": max(1, min(n, 20)), "direction": direction})

    watch_match = re.search(r"(?:coins\s+to\s+watch|give me\s+\d+\s+coins\s+to\s+watch|/watchlist)\s*(\d+)?", lower)
    if watch_match or "coins to watch" in lower:
        n_match = re.search(r"(\d+)", lower)
        n = int(n_match.group(1)) if n_match else 5
        return ParsedMessage(Intent.WATCHLIST, {"count": max(1, min(n, 20))})

    if "/scan" in lower or "scan this address" in lower or "scan tron" in lower or "scan solana" in lower:
        chain = "solana"
        if "tron" in lower:
            chain = "tron"
        tron_addr = TRON_ADDRESS_RE.search(stripped)
        sol_addr = SOL_ADDRESS_RE.search(stripped)
        address = tron_addr.group(0) if tron_addr else (sol_addr.group(0) if sol_addr else None)
        if tron_addr:
            chain = "tron"
        if sol_addr and "tron" not in lower:
            chain = "solana"
        if not address:
            return ParsedMessage(
                Intent.SCAN_WALLET,
                {"chain": chain},
                True,
                "Drop the wallet address (Solana or Tron).",
            )
        return ParsedMessage(Intent.SCAN_WALLET, {"chain": chain, "address": address})

    if "/tradecheck" in lower or "check this trade" in lower:
        symbols = _extract_symbols(stripped)
        prices = _extract_prices(stripped)
        entry = None
        stop = None
        targets: list[float] = []

        entry_match = re.search(r"entry\s*([0-9.,]+)", lower)
        stop_match = re.search(r"stop\s*([0-9.,]+)", lower)
        targets_match = re.search(r"targets?\s*([0-9.,\s]+)", lower)
        if entry_match:
            entry = float(entry_match.group(1).replace(",", ""))
        if stop_match:
            stop = float(stop_match.group(1).replace(",", ""))
        if targets_match:
            targets = [float(x.replace(",", "")) for x in re.findall(r"[0-9.,]+", targets_match.group(1))]
        if not targets and len(prices) >= 3:
            entry = entry or prices[0]
            stop = stop or prices[1]
            targets = prices[2:]

        entities = {
            "symbol": symbols[0] if symbols else None,
            "timeframe": parse_timeframe(lower) or "1h",
            "timestamp": parse_timestamp(lower),
            "entry": entry,
            "stop": stop,
            "targets": targets,
        }
        missing = [k for k in ("symbol", "entry", "stop") if not entities.get(k)]
        if missing:
            return ParsedMessage(Intent.TRADECHECK, entities, True, "Need symbol, entry, stop, and at least 1 target.")
        if not entities["targets"]:
            return ParsedMessage(Intent.TRADECHECK, entities, True, "Add at least one target price.")
        if not entities["timestamp"]:
            entities["timestamp"] = datetime.now(timezone.utc) - timedelta(days=1)
        return ParsedMessage(Intent.TRADECHECK, entities)

    setup_signal = bool(re.search(r"\b(entry|stop|sl|targets?|tp\d*|limit)\b", lower))
    math_signal = bool(
        re.search(
            r"\b(rr|r:r|risk\s*reward|risk/reward|pnl|profit|loss|size me|position size)\b",
            lower,
        )
    )
    setup_hint = setup_signal and (
        "long" in lower or "short" in lower or "setup" in lower or "play" in lower or math_signal
    )
    if setup_hint:
        symbols = _extract_symbols(stripped)
        entry, stop, targets = parse_setup_levels(stripped)
        tf_list, all_tfs, tf_notes = parse_timeframes_request(stripped)
        direction = "long" if "long" in lower else ("short" if "short" in lower else None)
        amount = _parse_single_level(
            stripped,
            [
                r"(?:\bamount\b|\bsize\b|\bposition\b)\s*[:=]?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
                r"\bmargin\b\s*[:=]?\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
                r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*\bmargin\b",
            ],
        )
        leverage = _parse_single_level(
            stripped,
            [
                r"\bleverage\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)",
                r"\b([0-9]+(?:\.[0-9]+)?)\s*x\b",
            ],
        )
        entities = {
            "symbol": symbols[0] if symbols else None,
            "direction": direction,
            "timeframe": parse_timeframe(lower) or "1h",
            "timeframes": tf_list,
            "all_timeframes": all_tfs,
            "entry": entry,
            "stop": stop,
            "targets": targets,
            "amount_usd": amount,
            "leverage": leverage,
            "notes": tf_notes,
        }
        if math_signal:
            direction = entities.get("direction")
            if not direction and entry is not None and targets:
                first_tp = float(targets[0])
                direction = "long" if first_tp >= float(entry) else "short"
                entities["direction"] = direction
            missing = [k for k in ("entry", "stop") if entities.get(k) is None]
            if missing or not targets:
                return ParsedMessage(
                    Intent.TRADE_MATH,
                    entities,
                    True,
                    "Drop entry, stop, and at least one target to calculate R:R and PnL.",
                )
            return ParsedMessage(Intent.TRADE_MATH, entities)

        missing = [k for k in ("symbol", "entry", "stop") if not entities.get(k)]
        if missing or not targets:
            return ParsedMessage(
                Intent.SETUP_REVIEW,
                entities,
                True,
                "Drop full setup: symbol, entry, stop, and at least one target.",
            )
        return ParsedMessage(Intent.SETUP_REVIEW, entities)

    if "following btc" in lower or "correlation" in lower:
        symbols = _extract_symbols(stripped)
        compare_to = "BTC"
        if symbols:
            symbol = symbols[0]
        else:
            return ParsedMessage(Intent.CORRELATION, {}, True, "Which token do you want compared to BTC?")
        return ParsedMessage(Intent.CORRELATION, {"symbol": symbol, "benchmark": compare_to})

    if "/alert add" in lower or "ping me when" in lower or "alert" in lower:
        symbols = _extract_symbols(stripped)
        prices = _extract_prices(stripped)
        condition = "cross"
        if "above" in lower:
            condition = "above"
        elif "below" in lower:
            condition = "below"
        elif "hits" in lower:
            condition = "cross"
        symbol = symbols[0] if symbols else None
        target = prices[0] if prices else None
        if not symbol or target is None:
            return ParsedMessage(Intent.ALERT_CREATE, {}, True, "Give symbol + level, e.g. `SOL above 100`.")
        return ParsedMessage(Intent.ALERT_CREATE, {"symbol": symbol, "condition": condition, "target_price": target})

    symbol_tf_hint = bool(parse_timeframe(lower) and len(_extract_symbols(stripped)) >= 1)
    analysis_hint = bool(
        re.search(
            r"\b(long|short)\b|what'?s happening with|long\?|\bema\b|\brsi\b|\btf\s*[:=]|\bwatch\s+\$?[a-z]{2,12}\b",
            lower,
        )
    ) or symbol_tf_hint
    if analysis_hint:
        symbols = _extract_symbols(stripped)
        if not symbols and re.match(r"^\s*me\s+(long|short)\b", lower):
            symbols = ["ME"]
        tf_list, all_tfs, tf_notes = parse_timeframes_request(stripped)
        ema_periods, all_emas, ema_notes = parse_ema_request(stripped)
        rsi_periods, all_rsis, rsi_notes = parse_rsi_request(stripped)
        notes = tf_notes + ema_notes + rsi_notes
        include_news = bool(re.search(r"\b(news|catalyst|catalysts)\b", lower))
        include_derivatives = bool(re.search(r"\b(funding|open interest|oi|derivatives?)\b", lower))
        direction = None
        if "long" in lower:
            direction = "long"
        if "short" in lower:
            direction = "short"
        symbol = symbols[0] if symbols else None
        timeframe = parse_timeframe(lower)
        if not symbol:
            return ParsedMessage(
                Intent.ANALYSIS,
                {
                    "direction": direction,
                    "timeframe": timeframe,
                    "timeframes": tf_list,
                    "all_timeframes": all_tfs,
                    "ema_periods": ema_periods,
                    "all_emas": all_emas,
                    "rsi_periods": rsi_periods,
                    "all_rsis": all_rsis,
                    "include_news": include_news,
                    "include_derivatives": include_derivatives,
                    "notes": notes,
                },
                True,
                "Which ticker, operator?",
            )
        return ParsedMessage(
            Intent.ANALYSIS,
            {
                "symbol": symbol,
                "direction": direction,
                "timeframe": timeframe,
                "timeframes": tf_list,
                "all_timeframes": all_tfs,
                "ema_periods": ema_periods,
                "all_emas": all_emas,
                "rsi_periods": rsi_periods,
                "all_rsis": all_rsis,
                "include_news": include_news,
                "include_derivatives": include_derivatives,
                "notes": notes,
            },
        )

    if lower.startswith("/"):
        return ParsedMessage(Intent.HELP)

    return ParsedMessage(
        Intent.UNKNOWN,
        {},
        True,
        "Say a ticker + direction (`SOL long` / `ETH short`) or ask `coins to watch 5`.",
    )
