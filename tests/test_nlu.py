from __future__ import annotations

import pytest

from app.core.nlu import Intent, parse_message


@pytest.mark.parametrize(
    "text,expected_intent,expected_keys",
    [
        ("SOL long", Intent.ANALYSIS, ["symbol", "direction"]),
        ("ETH short", Intent.ANALYSIS, ["symbol", "direction"]),
        ("ME long", Intent.ANALYSIS, ["symbol"]),
        ("Long?", Intent.ANALYSIS, []),
        ("What's happening with BTC?", Intent.ANALYSIS, ["symbol"]),
        ("Coins to watch 5", Intent.WATCHLIST, ["count"]),
        ("Give me 10 coins to watch", Intent.WATCHLIST, ["count"]),
        ("rsi top 10 1h oversold", Intent.RSI_SCAN, ["timeframe", "mode", "limit"]),
        ("top rsi 4h overbought", Intent.RSI_SCAN, ["timeframe", "mode"]),
        ("scan rsi sol 4h", Intent.RSI_SCAN, ["symbol", "timeframe"]),
        ("ping me when SOL hits 100", Intent.ALERT_CREATE, ["symbol", "target_price"]),
        ("set an alert btc 65000", Intent.ALERT_CREATE, ["symbol", "target_price"]),
        ("alert BTC above 70000", Intent.ALERT_CREATE, ["symbol", "target_price", "condition"]),
        ("remove my sol alert", Intent.ALERT_DELETE, ["symbol"]),
        ("clear my alerts", Intent.ALERT_CLEAR, []),
        ("reset alerts", Intent.ALERT_CLEAR, []),
        ("scan this address 9xQeWvG816bUx9EPfXfVn8A2fB7a4ri3W2h7sG2Tttz", Intent.SCAN_WALLET, ["address", "chain"]),
        ("scan tron TQn9Y2khEsLJW1ChVWFMSMeRDow5KcbLSE", Intent.SCAN_WALLET, ["address", "chain"]),
        ("find pair xion", Intent.PAIR_FIND, ["query"]),
        ("find $PHB", Intent.PAIR_FIND, ["query"]),
        ("coin around 0.155", Intent.PRICE_GUESS, ["target_price"]),
        ("are we at the bull market top?", Intent.CYCLE, []),
        ("cycle check", Intent.CYCLE, []),
        (
            "check this trade from yesterday: ETH entry 2100 stop 2165 targets 2043 2027 1991 timeframe 1h",
            Intent.TRADECHECK,
            ["symbol", "entry", "stop", "targets", "timeframe"],
        ),
        ("is BIRB following BTC movement?", Intent.CORRELATION, ["symbol", "benchmark"]),
        ("/start", Intent.START, []),
        ("/help", Intent.HELP, []),
        ("/settings", Intent.SETTINGS, []),
        ("/watchlist 8", Intent.WATCHLIST, ["count"]),
        ("/alert add sol above 120", Intent.ALERT_CREATE, ["symbol", "target_price"]),
        ("/alert list", Intent.ALERT_LIST, []),
        ("/alert delete 11", Intent.ALERT_DELETE, ["alert_id"]),
        ("/scan solana 9xQeWvG816bUx9EPfXfVn8A2fB7a4ri3W2h7sG2Tttz", Intent.SCAN_WALLET, ["chain", "address"]),
        ("/tradecheck", Intent.TRADECHECK, []),
        ("/news", Intent.NEWS, []),
        ("cpi news", Intent.NEWS, ["topic", "mode"]),
        ("macro update", Intent.NEWS, ["topic", "mode"]),
        ("inflation headlines today", Intent.NEWS, ["topic", "mode"]),
        ("openai updates", Intent.NEWS, ["topic", "mode"]),
        ("latest crypto news", Intent.NEWS, ["topic", "mode"]),
        ("btc news today", Intent.NEWS, ["topic", "mode"]),
        ("what happened today", Intent.NEWS, ["topic", "mode"]),
        ("/cycle", Intent.CYCLE, []),
        ("latest news today", Intent.NEWS, []),
        ("what do you have for me today", Intent.NEWS, []),
        ("how are you doing today", Intent.SMALLTALK, []),
        ("gm", Intent.SMALLTALK, []),
        ("yo", Intent.SMALLTALK, []),
        ("/join", Intent.GIVEAWAY_JOIN, []),
        ("join", Intent.GIVEAWAY_JOIN, []),
        ("SOL 4h", Intent.ANALYSIS, ["symbol", "timeframe"]),
        ("chart btc 1h", Intent.CHART, ["symbol", "timeframe"]),
        ("heatmap sol", Intent.HEATMAP, ["symbol"]),
        ("ema 200 4h top 10", Intent.EMA_SCAN, ["timeframe", "ema_length", "limit"]),
        ("SOL long 15m ema9 ema21 rsi14", Intent.ANALYSIS, ["symbol", "timeframes", "ema_periods", "rsi_periods"]),
        ("ETH short tf=1h,4h ema=20,50,200 rsi=14,21", Intent.ANALYSIS, ["timeframes", "ema_periods", "rsi_periods"]),
        ("ETH long all timeframes all emas all rsis", Intent.ANALYSIS, ["all_timeframes", "all_emas", "all_rsis"]),
        ("BTC limit long entry 66300 sl 64990 tp 69200 72000", Intent.SETUP_REVIEW, ["symbol", "entry", "stop", "targets"]),
        (
            "long BTC entry 66300 sl 64990 tp 69200 72000 amount 100 leverage 10",
            Intent.SETUP_REVIEW,
            ["symbol", "entry", "stop", "targets", "amount_usd", "leverage"],
        ),
        (
            "entry 100 sl 95 tp 110 lev 10x with 200 margin what is rr and pnl",
            Intent.TRADE_MATH,
            ["entry", "stop", "targets", "amount_usd", "leverage"],
        ),
        ("SOL short e 158 sl 166 targets 149 144", Intent.SETUP_REVIEW, ["symbol", "entry", "stop", "targets"]),
        ("btc long 4h", Intent.ANALYSIS, ["symbol", "timeframe"]),
        ("eth short 15m", Intent.ANALYSIS, ["symbol", "timeframe"]),
        ("watch btc", Intent.ANALYSIS, ["symbol"]),
        ("watch BTC", Intent.ANALYSIS, ["symbol"]),
        ("Coin to short right now", Intent.WATCHLIST, ["count", "direction"]),
        ("coin to long now", Intent.WATCHLIST, ["count", "direction"]),
        ("what are we doing today", Intent.SMALLTALK, []),
        ("alert sol below 80", Intent.ALERT_CREATE, ["condition", "target_price"]),
        ("scan solana", Intent.SCAN_WALLET, ["chain"]),
        ("check this trade: btc entry 50000 stop 49000 targets 51000", Intent.TRADECHECK, ["symbol", "entry", "stop", "targets"]),
        ("giveaway 30m prize 25usdt winners 2", Intent.GIVEAWAY_START, ["duration", "prize", "winners"]),
        ("pick giveaway winner in 30 mins", Intent.GIVEAWAY_START, ["duration"]),
        ("pick a winner in 10 minutes", Intent.GIVEAWAY_START, ["duration"]),
    ],
)
def test_parse_message_cases(text: str, expected_intent: Intent, expected_keys: list[str]) -> None:
    parsed = parse_message(text)
    assert parsed.intent == expected_intent
    for key in expected_keys:
        assert key in parsed.entities


def test_analysis_followup_when_missing_symbol() -> None:
    parsed = parse_message("long?")
    assert parsed.intent == Intent.ANALYSIS
    assert parsed.requires_followup


def test_scan_followup_when_missing_address() -> None:
    parsed = parse_message("scan tron")
    assert parsed.intent == Intent.SCAN_WALLET
    assert parsed.requires_followup


def test_unknown_fallback_prompt_is_actionable() -> None:
    parsed = parse_message("explain your modules")
    assert parsed.intent == Intent.UNKNOWN
    assert parsed.requires_followup
    assert "SOL long" in (parsed.followup_question or "")


def test_analysis_caps_timeframes_and_periods() -> None:
    parsed = parse_message("SOL long tf=1m,3m,5m,15m,30m ema=5,9,20,50,100,200 rsi=3,7,14,21")
    assert parsed.intent == Intent.ANALYSIS
    assert len(parsed.entities.get("timeframes", [])) <= 4
    assert len(parsed.entities.get("ema_periods", [])) <= 5
    assert len(parsed.entities.get("rsi_periods", [])) <= 3
    notes = parsed.entities.get("notes", [])
    assert notes


def test_setup_review_amount_leverage_values() -> None:
    parsed = parse_message("long BTC entry 66300 sl 64990 tp 69200 72000 amount 100 leverage 10")
    assert parsed.intent == Intent.SETUP_REVIEW
    assert parsed.entities["amount_usd"] == 100.0
    assert parsed.entities["leverage"] == 10.0


def test_price_guess_limit_parse() -> None:
    parsed = parse_message("top 7 coin around 2.08")
    assert parsed.intent == Intent.PRICE_GUESS
    assert parsed.entities["limit"] == 7
