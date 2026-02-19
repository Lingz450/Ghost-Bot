from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.adapters.llm import RouterPayload


def test_router_payload_valid() -> None:
    payload = RouterPayload.model_validate(
        {
            "intent": "news_digest",
            "confidence": 0.82,
            "params": {"topic": "cpi", "mode": "macro", "limit": 6},
        }
    )
    assert payload.intent == "news_digest"
    assert payload.confidence == 0.82
    assert payload.params["topic"] == "cpi"


def test_router_payload_invalid_intent() -> None:
    with pytest.raises(ValidationError):
        RouterPayload.model_validate(
            {
                "intent": "TA_SINGLE",
                "confidence": 0.9,
                "params": {},
            }
        )


def test_router_payload_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        RouterPayload.model_validate(
            {
                "intent": "general_chat",
                "confidence": 1.3,
                "params": {},
            }
        )
