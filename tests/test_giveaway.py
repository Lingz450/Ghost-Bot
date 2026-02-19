from __future__ import annotations

import pytest

from app.services.giveaway import GiveawayService


class _DummyRng:
    def choice(self, seq):
        return seq[0]


@pytest.mark.asyncio
async def test_choose_winner_avoids_last_winner_when_possible() -> None:
    service = GiveawayService(db_factory=None, admin_chat_ids=[1], min_participants=2)
    service.rng = _DummyRng()

    async def _participants(_session, _giveaway_id):
        return [111, 222, 333]

    async def _last_winner(_session, _group_chat_id):
        return 111

    service._participant_ids = _participants  # type: ignore[method-assign]
    service._last_winner = _last_winner  # type: ignore[method-assign]

    winner, note = await service._choose_winner(None, 99, 10)  # type: ignore[arg-type]
    assert note == "ok"
    assert winner == 222


@pytest.mark.asyncio
async def test_choose_winner_falls_back_when_no_alternate() -> None:
    service = GiveawayService(db_factory=None, admin_chat_ids=[1], min_participants=1)
    service.rng = _DummyRng()

    async def _participants(_session, _giveaway_id):
        return [777]

    async def _last_winner(_session, _group_chat_id):
        return 777

    service._participant_ids = _participants  # type: ignore[method-assign]
    service._last_winner = _last_winner  # type: ignore[method-assign]

    winner, note = await service._choose_winner(None, 99, 10)  # type: ignore[arg-type]
    assert note == "ok"
    assert winner == 777
