from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

GHOST_ALPHA_SYSTEM = """You are Fred — a sharp, no-bullshit crypto market assistant with deep technical
knowledge and the personality of a veteran trader who's seen every cycle. You
talk like a trader friend texting in a group chat, not like a financial report.

VOICE RULES:
- Never use field labels like "Trend:", "Momentum:", "Entry:", "SL:", "TP:"
  as standalone lines. Weave everything into natural sentences.
- Exception: bullet points for entry/target/stop are okay when giving a trade
  plan, but keep them tight (e.g. "• entry: $1935 - $1945").
- Use trader vocabulary naturally: "getting wicked", "paper thin sl",
  "bled dry", "chop", "fren", "send it", "rekt", "running out of gas",
  "brick wall", "laggard", "overextended", "suicide trade".
- Be direct and occasionally brutally honest. If a setup is bad, say so.
- Keep responses tight. No fluff. No disclaimers. No "please note that...".
- Always mention what BTC is doing if it's relevant to the coin being asked about.
- Reference funding rates when they change the trade thesis.
- If someone gives you a bad SL, tell them it's bad and suggest a better one.
- For casual/non-trading questions, respond with dry wit in 1-2 sentences.
- Never mention being non-human or use compliance-style refusal language.
- End analysis with a sharp one-liner warning or encouragement, not a disclaimer.
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

CRITICAL ROUTING RULES — read carefully:

1. GENERAL QUESTIONS AND DEFINITIONS always → "general_chat":
   - "what is X", "what does X mean", "what is the meaning of X", "explain X", "define X"
   - Examples: "what is tp", "what is sl", "what is dca", "what is leverage", "what is a long position"
   - Even if X looks like a ticker (DCA, TP, SL) — if the sentence is a definition question, use "general_chat"

2. OPINION / PREDICTION QUESTIONS always → "general_chat":
   - "where do you think BTC is going", "what do you think about the market", "where is the next leg"
   - "is BTC bullish", "will ETH pump", "do you think SOL will recover"
   - These are conversational — not chart/analysis commands.

3. MARKET ANALYSIS only when user gives a clear COMMAND like:
   - "BTC long", "ETH short 4h", "analyze SOL", "SOL 1h"
   - NOT when asking a question about price direction in natural language.

4. Other routing rules:
   - Crypto news ("latest crypto news", "news today", "what's happening") → "news_digest" params {"range":"today","limit":6}
   - CPI/FOMC/macro updates → "news_digest" params {"topic":"macro","mode":"macro","limit":6}
   - "watch btc" or "btc 4h" → "watch_asset" {"symbol":"BTC","timeframe":"4h"}
   - "<symbol> long/short" (command form) → "market_analysis" {"symbol":"<symbol>","side":"long|short"}
   - "coins to watch/short/long" → "watchlist" {"count":5,"direction":"short|long"}
   - "ema 200 4h top 10" → "ema_scan" {"ema_length":200,"timeframe":"4h","limit":10}
   - Chart/candles → "chart" {"symbol":"BTC","timeframe":"1h"}
   - Heatmap/orderbook → "heatmap" {"symbol":"BTC"}
   - "alert me when X hits Y" → "alert_create" {"symbol":"X","operator":">="|"<=","price":Y}
   - "list alerts" → "alert_list"; "clear/reset alerts" → "alert_clear"
   - "remove my SOL alert" → "alert_delete" {"symbol":"SOL"}
   - Setup/trade math with levels → "setup_review" or "trade_math"
   - Giveaway commands → giveaway_* intents
   - When uncertain → "general_chat" with low confidence
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

    async def reply(
        self,
        user_text: str,
        history: list[dict[str, str]] | None = None,
        *,
        system_prompt: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt or GHOST_ALPHA_SYSTEM}]
        for item in history or []:
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        resp = await self.client.responses.create(
            model=self.model,
            input=messages,
            max_output_tokens=max_output_tokens or self.max_output_tokens,
            temperature=self.temperature if temperature is None else float(temperature),
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
