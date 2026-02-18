from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.core.config import get_settings
from app.core.container import ServiceHub

logger = logging.getLogger(__name__)


class WorkerScheduler:
    def __init__(self, hub: ServiceHub) -> None:
        self.hub = hub
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler(timezone="UTC")

    async def _notify(self, chat_id: int, text: str) -> None:
        await self.hub.bot.send_message(chat_id=chat_id, text=text)

    async def _process_alerts(self) -> None:
        count = await self.hub.alerts_service.process_alerts(self._notify)
        logger.info("alerts_processed", extra={"event": "alerts_processed", "count": count})

    async def _refresh_news_cache(self) -> None:
        await self.hub.news_service.get_daily_brief(limit=10)

    async def _process_giveaways(self) -> None:
        count = await self.hub.giveaway_service.process_due_giveaways(self._notify)
        logger.info("giveaways_processed", extra={"event": "giveaways_processed", "count": count})

    async def _refresh_scan_universe(self) -> None:
        payload = await self.hub.rsi_scanner_service.refresh_universe(self.settings.rsi_scan_universe_size)
        logger.info("scan_universe_refreshed", extra={"event": "scan_universe_refreshed", **payload})

    async def _refresh_scan_indicators(self) -> None:
        payload = await self.hub.rsi_scanner_service.refresh_indicators(force=False)
        logger.info("scan_indicators_refreshed", extra={"event": "scan_indicators_refreshed", **payload})

    def start(self) -> None:
        self.scheduler.add_job(self._process_alerts, "interval", seconds=self.settings.alert_check_interval_sec, max_instances=1)
        self.scheduler.add_job(self._refresh_news_cache, "interval", minutes=15, max_instances=1)
        self.scheduler.add_job(self._process_giveaways, "interval", seconds=20, max_instances=1)
        self.scheduler.add_job(self._refresh_scan_universe, "interval", minutes=30, max_instances=1)
        self.scheduler.add_job(self._refresh_scan_indicators, "interval", minutes=5, max_instances=1)
        self.scheduler.start()

    def stop(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
