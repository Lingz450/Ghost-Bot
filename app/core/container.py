from __future__ import annotations

from dataclasses import dataclass

from aiogram import Bot

from app.adapters.llm import LLMClient
from app.adapters.market_router import MarketDataRouter
from app.core.cache import RedisCache
from app.core.rate_limit import RateLimiter
from app.services.alerts import AlertsService
from app.services.audit import AuditService
from app.services.correlation import CorrelationService
from app.services.cycles import CyclesService
from app.services.discovery import DiscoveryService
from app.services.ema_scanner import EMAScannerService
from app.services.giveaway import GiveawayService
from app.services.market_analysis import MarketAnalysisService
from app.services.news import NewsService
from app.services.orderbook_heatmap import OrderbookHeatmapService
from app.services.rsi_scanner import RSIScannerService
from app.services.setup_review import SetupReviewService
from app.services.charting import ChartService
from app.services.trade_verify import TradeVerifyService
from app.services.users import UserService
from app.services.wallet_scan import WalletScanService
from app.services.watchlist import WatchlistService


@dataclass
class ServiceHub:
    bot: Bot
    bot_username: str | None
    llm_client: LLMClient | None
    market_router: MarketDataRouter
    cache: RedisCache
    rate_limiter: RateLimiter
    user_service: UserService
    audit_service: AuditService
    analysis_service: MarketAnalysisService
    alerts_service: AlertsService
    wallet_service: WalletScanService
    trade_verify_service: TradeVerifyService
    setup_review_service: SetupReviewService
    watchlist_service: WatchlistService
    news_service: NewsService
    cycles_service: CyclesService
    correlation_service: CorrelationService
    rsi_scanner_service: RSIScannerService
    ema_scanner_service: EMAScannerService
    chart_service: ChartService
    orderbook_heatmap_service: OrderbookHeatmapService
    discovery_service: DiscoveryService
    giveaway_service: GiveawayService
