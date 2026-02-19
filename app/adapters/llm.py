from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

GHOST_ALPHA_SYSTEM = """You are Ghost Alpha Bot.

VIBE:
- Fast, sharp, confident. A bit edgy/witty, never hateful.
- Keep replies short unless asked for depth.
- For greetings/smalltalk: 1-2 lines.
- For crypto questions: concise and structured.

RULES:
- Informational only. Never claim you executed trades.
- Add a brief "Not financial advice." line only for trade setup outputs.
- If data is missing, say what is missing and ask for ticker/chain/timeframe or suggest alerts/watchlist.
"""

ROUTER_SYSTEM = """You are an intent router for a Telegram crypto assistant called Ghost Alpha Bot.

Return ONLY valid JSON. No markdown. No extra text.

Given the user's message, produce:
{
  "intent": one of [
    "smalltalk","news_digest","watch_asset","market_analysis","watchlist","rsi_scan","ema_scan","chart","heatmap",
    "alert_create","alert_list","alert_delete","alert_clear",
    "pair_find","price_guess","setup_review","trade_math",
    "giveaway_start","giveaway_join","giveaway_end","giveaway_reroll","giveaway_status","giveaway_cancel",
    "general_chat"
  ],
  "confidence": 0.0-1.0,
  "params": { ... }
}

Rules:
- If user asks for crypto news ("latest crypto news", "news today", "what's happening"), set intent="news_digest" with params {"range":"today","limit":6}.
- If user asks for CPI/FOMC/macro updates, set intent="news_digest" and include params {"topic":"cpi" or "macro","mode":"macro","limit":6}.
- If user asks for OpenAI/ChatGPT/GPT/Codex updates, set intent="news_digest" and include params {"topic":"openai","mode":"openai","limit":6}.
- If user says "watch btc" or "btc 4h", set intent="watch_asset" with {"symbol":"BTC","timeframe":"4h"} (default timeframe "1h" if missing).
- If user says "<symbol> long/short", set intent="market_analysis" with {"symbol":"<symbol>","side":"long|short"} and optional timeframe.
- If user asks for "coins to watch", "coins to short", or "coins to long", set intent="watchlist" with params {"count":5, "direction":"short|long"} as applicable.
- If user asks for "ema 200 4h top 10" or "coins near ema200", set intent="ema_scan" with {"ema_length":200,"timeframe":"4h","limit":10}.
- If user asks for chart/candles, set intent="chart" with {"symbol":"BTC","timeframe":"1h"}.
- If user asks for heatmap/orderbook/depth, set intent="heatmap" with {"symbol":"BTC"}.
- If user says "alert me when X hits Y", set intent="alert_create" with {"symbol":"X","operator":">=" or "<=","price":Y}.
- If user says "list alerts", set intent="alert_list". If "clear/reset alerts", set intent="alert_clear".
- If user says "remove my sol alert", set intent="alert_delete" with {"symbol":"SOL"}.
- If user asks to "find pair", use intent="pair_find". If user provides a price and asks possible coins, use "price_guess" with {"price":...,"limit":10}.
- If user provides entry/stop/targets and optionally amount/leverage, use "setup_review".
- If giveaway is requested, route to giveaway_* intents.
- If not crypto-related, use "general_chat".
- If uncertain, set "general_chat" and low confidence.
"""

ROUTER_ALLOWED_INTENTS = {
    "smalltalk",
    "news_digest",
    "watch_asset",
    "market_analysis",
    "watchlist",
    "rsi_scan",
    "ema_scan",
    "chart",
    "heatmap",
    "alert_create",
    "alert_list",
    "alert_delete",
    "alert_clear",
    "pair_find",
    "price_guess",
    "setup_review",
    "trade_math",
    "giveaway_start",
    "giveaway_join",
    "giveaway_end",
    "giveaway_reroll",
    "giveaway_status",
    "giveaway_cancel",
    "general_chat",
}


class RouterPayload(BaseModel):
    intent: Literal[
        "smalltalk",
        "news_digest",
        "watch_asset",
        "market_analysis",
        "watchlist",
        "rsi_scan",
        "ema_scan",
        "chart",
        "heatmap",
        "alert_create",
        "alert_list",
        "alert_delete",
        "alert_clear",
        "pair_find",
        "price_guess",
        "setup_review",
        "trade_math",
        "giveaway_start",
        "giveaway_join",
        "giveaway_end",
        "giveaway_reroll",
        "giveaway_status",
        "giveaway_cancel",
        "general_chat",
    ] = "general_chat"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)
    followup_question: str | None = None


@dataclass
class LLMClient:
    api_key: str
    model: str = "gpt-4.1-mini"
    router_model: str | None = None
    max_output_tokens: int = 350
    temperature: float = 0.7

    def __post_init__(self) -> None:
        self.client = AsyncOpenAI(api_key=self.api_key)

    def _extract_output_text(self, resp: Any) -> str:
        output_text = getattr(resp, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        for item in (getattr(resp, "output", None) or []):
            item_type = getattr(item, "type", None)
            if item_type is None and isinstance(item, dict):
                item_type = item.get("type")
            if item_type != "message":
                continue

            content = getattr(item, "content", None)
            if content is None and isinstance(item, dict):
                content = item.get("content")
            for piece in content or []:
                piece_type = getattr(piece, "type", None)
                piece_text = getattr(piece, "text", None)
                if isinstance(piece, dict):
                    piece_type = piece_type or piece.get("type")
                    piece_text = piece_text or piece.get("text")
                if piece_type in ("output_text", "text") and isinstance(piece_text, str):
                    parts.append(piece_text)

        return "\n".join([p for p in parts if p.strip()]).strip()

    def _extract_json_payload(self, raw_text: str) -> dict:
        text = (raw_text or "").strip()
        if not text:
            return {}
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        candidate = text[start : end + 1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    async def reply(self, user_text: str, history: list[dict[str, str]] | None = None) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": GHOST_ALPHA_SYSTEM}]
        for item in history or []:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        resp = await self.client.responses.create(
            model=self.model,
            input=messages,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
        )
        text = self._extract_output_text(resp)
        return text or "Signal unclear. Give me ticker + timeframe and I will map it."

    async def route_message(self, user_text: str) -> dict:
        resp = await self.client.responses.create(
            model=self.router_model or self.model,
            input=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": user_text},
            ],
            max_output_tokens=260,
            temperature=0.0,
        )
        raw = self._extract_output_text(resp)
        payload = self._extract_json_payload(raw)
        try:
            validated = RouterPayload.model_validate(payload)
        except ValidationError:
            return {"intent": "general_chat", "confidence": 0.0, "params": {}}

        return {
            "intent": validated.intent,
            "confidence": float(validated.confidence),
            "params": dict(validated.params),
            "followup_question": validated.followup_question,
        }
