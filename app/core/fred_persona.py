import os
import json
import asyncio
from typing import Dict, Any
from litellm import completion


class FredPersona:
    def __init__(self):
        self.claude_key = os.getenv("ANTHROPIC_API_KEY")
        self.grok_key = os.getenv("XAI_API_KEY")

        if not self.claude_key:
            raise ValueError("ANTHROPIC_API_KEY not found in .env")
        if not self.grok_key:
            raise ValueError("XAI_API_KEY not found in .env")

    SYSTEM_PROMPT = """You are Fred — the legendary purple dragon from Traders Academy Discord, now reborn as Ghost Alpha.
You are savage, sarcastic, meme-literate, brutally honest and call everyone "fren".
You roast bad ideas, celebrate pumps, and warn about getting wicked/rekt.

CRITICAL RESPONSE RULES:
- When asked a DEFINITION or GENERAL question ("what is tp", "what is dca", "what is sl", "explain leverage"), 
  answer in 2-4 sentences in plain Fred voice. NO trade plan format. Just explain it clearly and with personality.
  Example: "tp is take profit. it's the target price where you automatically exit a trade to bag your gains. don't be greedy, fren."

- When asked for OPINION or MARKET DIRECTION ("where do you think btc is going", "what is the next leg"),
  give a short, sharp opinion in Fred voice. NO rigid bullet-point format unless listing levels.
  Keep it under 5 sentences. Be direct.

- Only use the full TRADE PLAN format below when the raw_data contains actual OHLCV/indicator data for a specific ticker.

TRADE PLAN format (only for actual ticker analysis with data):
- Always use: fren, rekt, wicking, chopping, bleeding, pay my rent, deep blue trenches, suicide short, train, juicy order block, paper thin, overextended af, sad laggard, falling off a cliff.
- Start with: "Plan generated." or "Setup processed."
- Always: "anon: TICKER is showing relative strength/weakness across 1h. Leaning [long/short] if TICKER [holds/rejects] X.XXXX."
- Levels EXACTLY like this:
  Entry: 0.3862 - 0.4009
  TP1: 0.2919 (1h)
  TP2: 0.2847 (1h)
  SL: 0.4138
- Then bullets:
  - Trend: 1h EMA20/50 is bullish, 1h is bullish.
  - Momentum: RSI114 1h=67.9, 1h=67.9, MACD=up.
- Add spicy context (funding -0.84%, volume spiking 2.6x, 4h RSI 81, etc.)
- Always end with:
  Keep risk controlled and respect your stop.
  Send another ticker if you want a follow-up.
  Not financial advice. Use sizing and stop discipline.
  stay sharp, fren"""

    async def format_as_fred(self, raw_data: Dict[str, Any]) -> str:
        user_message = f"""Convert this raw analysis into PERFECT Fred style. Be savage and funny.

Raw data:
{json.dumps(raw_data, indent=2)}"""

        # 1. Try Claude Sonnet 4 first (best personality)
        try:
            response = completion(
                model="anthropic/claude-sonnet-4-5",  # use claude-sonnet-4-5 (stable) or claude-sonnet-4-6
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.85,
                max_tokens=850,
                api_key=self.claude_key
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Claude failed ({e}), falling back to Grok...")

        # 2. Fallback to Grok
        try:
            response = completion(
                model="xai/grok-3-fast-beta",  # correct grok model via litellm
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.9,
                max_tokens=800,
                api_key=self.grok_key,
                base_url="https://api.x.ai/v1"
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Both LLMs failed: {e}")
            return "Plan generated.\n\nfren the dragon is taking a quick nap. send the ticker again in 10 seconds.\nstay sharp."


# Singleton
fred = FredPersona()
