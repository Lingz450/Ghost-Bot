from __future__ import annotations

from app.bot.templates import (
    correlation_template,
    news_template,
    rsi_scan_template,
    trade_plan_template,
    trade_verification_template,
    watchlist_template,
)


def test_trade_plan_template_hides_source_lines() -> None:
    payload = {
        "summary": "SOL looks constructive.",
        "condition": "Leaning long if SOL holds 90.0",
        "entry": "90.0 - 91.0",
        "tp1": "94.0 (1h)",
        "tp2": "98.0 (4h)",
        "sl": "87.5",
        "why": ["Trend up", "Momentum recovering"],
        "data_source_line": "Data source: Bybit Spot (SOLUSDT) | Updated: 30s ago",
        "updated_at": "2026-02-19T00:00:00+00:00",
        "risk": "Not financial advice.",
    }
    text = trade_plan_template(payload, {"formal_mode": False, "tone_mode": "standard", "anon_mode": False})
    assert "Data source:" not in text
    assert "Source:" not in text


def test_watchlist_template_hides_source_line() -> None:
    payload = {
        "summary": "Short-side watchlist.",
        "items": ["SOL - watch rejection", "ETH - watch rejection"],
        "source_line": "Data source: router",
    }
    text = watchlist_template(payload)
    assert "Data source:" not in text
    assert "Source:" not in text


def test_news_template_hides_source_label() -> None:
    payload = {
        "summary": "Macro risk rising.",
        "headlines": [
            {"title": "CPI print due next week", "source": "Example", "url": "https://example.com/a"},
        ],
        "updated_at": "2026-02-19T00:00:00+00:00",
        "vibe": "Risk-on fading.",
    }
    text = news_template(payload)
    assert "Source:" not in text
    assert "Link:" in text


def test_scan_and_correlation_templates_hide_source_line() -> None:
    rsi_payload = {
        "summary": "RSI scan",
        "items": [{"symbol": "SOL", "rsi": 22.4, "note": "oversold"}],
        "rsi_length": 14,
        "timeframe": "1h",
        "source_line": "Data source: router",
    }
    corr_payload = {
        "summary": "Tracks BTC",
        "bullets": ["corr 0.7", "beta 1.2"],
        "source_line": "Data source: router",
    }
    rsi_text = rsi_scan_template(rsi_payload)
    corr_text = correlation_template(corr_payload)
    assert "Data source:" not in rsi_text
    assert "Data source:" not in corr_text


def test_trade_verification_template_hides_source_line() -> None:
    payload = {
        "symbol": "ETH",
        "direction": "short",
        "result": "win",
        "mode": "ambiguous",
        "filled_at": 1,
        "first_hit": 2,
        "mfe": 1.1,
        "mae": 0.4,
        "r_multiple": 2.5,
        "source_line": "Data source: router",
    }
    text = trade_verification_template(payload)
    assert "Data source:" not in text
