from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, Update
from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import text

from app.adapters.derivatives import DerivativesAdapter
from app.adapters.llm import LLMClient
from app.adapters.market_router import MarketDataRouter
from app.adapters.news_sources import NewsSourcesAdapter
from app.adapters.ohlcv import OHLCVAdapter
from app.adapters.prices import PriceAdapter
from app.adapters.solana import SolanaAdapter
from app.adapters.tron import TronAdapter
from app.bot.handlers import init_handlers, router
from app.core.cache import RedisCache
from app.core.config import Settings, get_settings
from app.core.container import ServiceHub
from app.core.http import ResilientHTTPClient
from app.core.logging import setup_logging
from app.core.rate_limit import RateLimiter
from app.db.session import AsyncSessionLocal
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
from app.workers.scheduler import WorkerScheduler

logger = logging.getLogger(__name__)


async def _sync_bot_commands(bot: Bot) -> None:
    command_specs = [
        ("admins", "Show bot admins"),
        ("alert", "Create a price alert"),
        ("alertclear", "Clear alerts (all or by symbol)"),
        ("alertdel", "Delete one alert by ID"),
        ("alerts", "List your active alerts"),
        ("alpha", "Full market analysis (multi-timeframe)"),
        ("chart", "Send candlestick chart image"),
        ("cycle", "Cycle check"),
        ("ema", "EMA scan (near EMA levels)"),
        ("findpair", "Find coin by price / partial name"),
        ("giveaway", "Admin: start/end/reroll giveaways"),
        ("heatmap", "Orderbook heatmap snapshot"),
        ("help", "Show examples + what I can do"),
        ("id", "Show your user/chat id"),
        ("join", "Join active giveaway"),
        ("margin", "Position size + margin calculator"),
        ("news", "Crypto + macro + OpenAI news digest"),
        ("pnl", "PnL calculator (entry/exit/size/lev)"),
        ("price", "Latest price + 24h stats"),
        ("rsi", "RSI scan (overbought/oversold)"),
        ("scan", "Wallet scan (solana/tron address)"),
        ("settings", "Preferences (default TF, risk, etc.)"),
        ("setup", "RR + PnL + margin from entry/SL/TP"),
        ("start", "Start the bot / show quick intro"),
        ("tradecheck", "Verify trade outcome from timestamp"),
        ("watch", "Quick levels + bias for a coin"),
        ("watchlist", "Coins to watch list"),
    ]
    commands = [
        BotCommand(command=command, description=description)
        for command, description in sorted(command_specs, key=lambda x: x[0])
    ]
    await bot.set_my_commands(commands)


def build_hub(settings: Settings, bot: Bot, cache: RedisCache, http: ResilientHTTPClient) -> ServiceHub:
    rate_limiter = RateLimiter(cache)
    llm_client = None
    if settings.openai_api_key:
        llm_client = LLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            router_model=settings.openai_router_model or None,
            max_output_tokens=settings.openai_max_output_tokens,
            temperature=settings.openai_temperature,
        )

    market_router = MarketDataRouter(
        http=http,
        cache=cache,
        binance_base_url=settings.binance_base_url,
        binance_futures_base_url=settings.binance_futures_base_url,
        bybit_base_url=settings.bybit_base_url,
        okx_base_url=settings.okx_base_url,
        mexc_base_url=settings.mexc_base_url,
        blofin_base_url=settings.blofin_base_url,
        enable_binance=settings.enable_binance,
        enable_bybit=settings.enable_bybit,
        enable_okx=settings.enable_okx,
        enable_mexc=settings.enable_mexc,
        enable_blofin=settings.enable_blofin,
        exchange_priority=settings.exchange_priority,
        market_prefer_spot=settings.market_prefer_spot,
        best_source_ttl_hours=settings.best_source_ttl_hours,
        instruments_ttl_min=settings.instruments_ttl_min,
    )

    price_adapter = PriceAdapter(
        http=http,
        cache=cache,
        binance_base=settings.binance_base_url,
        coingecko_base=settings.coingecko_base_url,
        test_mode=settings.test_mode,
        mock_prices=settings.mock_prices,
        market_router=market_router,
    )
    ohlcv_adapter = OHLCVAdapter(
        http=http,
        cache=cache,
        binance_base=settings.binance_base_url,
        coingecko_base=settings.coingecko_base_url,
        market_router=market_router,
    )
    deriv_adapter = DerivativesAdapter(
        http=http,
        cache=cache,
        futures_base=settings.binance_futures_base_url,
        market_router=market_router,
    )
    news_adapter = NewsSourcesAdapter(
        http=http,
        cache=cache,
        rss_feeds=settings.rss_feed_list(),
        cryptopanic_key=settings.cryptopanic_api_key,
        openai_rss_feeds=settings.openai_rss_feed_list(),
    )
    solana_adapter = SolanaAdapter(http=http, rpc_url=settings.solana_rpc_url)
    tron_adapter = TronAdapter(http=http, api_url=settings.tron_api_url, api_key=settings.trongrid_api_key)

    news_service = NewsService(news_adapter, llm_client=llm_client)
    rsi_scanner_service = RSIScannerService(
        http=http,
        cache=cache,
        ohlcv_adapter=ohlcv_adapter,
        market_router=market_router,
        coingecko_base=settings.coingecko_base_url,
        binance_base=settings.binance_base_url,
        db_factory=AsyncSessionLocal,
        universe_size=settings.rsi_scan_universe_size,
        scan_timeframes=settings.rsi_scan_timeframes_list(),
        concurrency=settings.rsi_scan_concurrency,
        freshness_minutes=settings.rsi_scan_freshness_minutes,
        live_fallback_universe=settings.rsi_scan_live_fallback_universe,
    )
    discovery_service = DiscoveryService(
        http=http,
        cache=cache,
        market_router=market_router,
        price_adapter=price_adapter,
        binance_base=settings.binance_base_url,
        coingecko_base=settings.coingecko_base_url,
    )
    giveaway_service = GiveawayService(
        db_factory=AsyncSessionLocal,
        admin_chat_ids=settings.admin_ids_list(),
        min_participants=settings.giveaway_min_participants,
    )
    ema_scanner_service = EMAScannerService(
        http=http,
        cache=cache,
        ohlcv_adapter=ohlcv_adapter,
        market_router=market_router,
        binance_base=settings.binance_base_url,
        db_factory=AsyncSessionLocal,
        freshness_minutes=settings.rsi_scan_freshness_minutes,
        live_fallback_universe=settings.rsi_scan_live_fallback_universe,
        concurrency=settings.rsi_scan_concurrency,
    )
    chart_service = ChartService(ohlcv_adapter=ohlcv_adapter)
    orderbook_heatmap_service = OrderbookHeatmapService(market_router=market_router)

    return ServiceHub(
        bot=bot,
        bot_username=None,
        llm_client=llm_client,
        market_router=market_router,
        cache=cache,
        rate_limiter=rate_limiter,
        user_service=UserService(AsyncSessionLocal),
        audit_service=AuditService(AsyncSessionLocal),
        # analysis service defaults tuned for low latency; deep data can still be requested on demand
        analysis_service=MarketAnalysisService(
            price_adapter,
            ohlcv_adapter,
            deriv_adapter,
            news_service,
            fast_mode=settings.analysis_fast_mode,
            default_timeframes=settings.analysis_default_timeframes_list(),
            include_derivatives_default=settings.analysis_include_derivatives_default,
            include_news_default=settings.analysis_include_news_default,
            request_timeout_sec=settings.analysis_request_timeout_sec,
        ),
        alerts_service=AlertsService(
            db_factory=AsyncSessionLocal,
            cache=cache,
            price_adapter=price_adapter,
            market_router=market_router,
            alerts_limit_per_day=settings.alerts_create_limit_per_day,
            cooldown_minutes=settings.alert_cooldown_min,
            max_deviation_pct=settings.alert_max_deviation_pct,
        ),
        wallet_service=WalletScanService(db_factory=AsyncSessionLocal, solana=solana_adapter, tron=tron_adapter, price=price_adapter),
        trade_verify_service=TradeVerifyService(ohlcv_adapter),
        setup_review_service=SetupReviewService(ohlcv_adapter),
        watchlist_service=WatchlistService(
            http=http,
            news_adapter=news_adapter,
            market_router=market_router,
            coingecko_base=settings.coingecko_base_url,
            include_btc_eth=settings.include_btc_eth_watchlist,
        ),
        news_service=news_service,
        cycles_service=CyclesService(ohlcv_adapter),
        correlation_service=CorrelationService(ohlcv_adapter, price_adapter),
        rsi_scanner_service=rsi_scanner_service,
        ema_scanner_service=ema_scanner_service,
        chart_service=chart_service,
        orderbook_heatmap_service=orderbook_heatmap_service,
        discovery_service=discovery_service,
        giveaway_service=giveaway_service,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)

    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

    cache = RedisCache(settings.redis_url)
    http = ResilientHTTPClient()

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()

    hub = build_hub(settings, bot, cache, http)
    try:
        me = await bot.get_me()
        hub.bot_username = me.username.lower() if me.username else None
    except Exception:  # noqa: BLE001
        hub.bot_username = None

    try:
        await _sync_bot_commands(bot)
    except Exception as exc:  # noqa: BLE001
        logger.warning("set_bot_commands_failed", extra={"event": "set_bot_commands_failed", "error": str(exc)})
    init_handlers(hub)
    dp.include_router(router)

    scheduler = None
    if not settings.serverless_mode:
        scheduler = WorkerScheduler(hub)
        scheduler.start()

    polling_task = None
    if settings.serverless_mode and not settings.telegram_use_webhook:
        logger.warning("serverless_mode_enabled_without_webhook", extra={"event": "serverless_warning"})

    if settings.telegram_use_webhook and settings.telegram_auto_set_webhook:
        webhook_url = settings.telegram_webhook_url.rstrip("/") + settings.telegram_webhook_path
        try:
            await bot.set_webhook(webhook_url, secret_token=settings.telegram_webhook_secret or None)
            logger.info("webhook_configured", extra={"event": "webhook", "url": webhook_url})
        except Exception as exc:  # noqa: BLE001
            # Do not crash the whole API on webhook registration failures.
            logger.exception(
                "webhook_configure_failed",
                extra={"event": "webhook_error", "url": webhook_url, "error": str(exc)},
            )
    elif not settings.telegram_use_webhook and not settings.serverless_mode:
        polling_task = asyncio.create_task(dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types()))

    app.state.settings = settings
    app.state.hub = hub
    app.state.dp = dp
    app.state.bot = bot
    app.state.http = http
    app.state.cache = cache
    app.state.scheduler = scheduler
    app.state.polling_task = polling_task

    try:
        yield
    finally:
        if scheduler:
            scheduler.stop()
        if polling_task:
            polling_task.cancel()
            with contextlib.suppress(Exception):
                await polling_task
        await bot.session.close()
        await http.close()
        await cache.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Ghost Alpha Bot", version="1.0.0", lifespan=lifespan)

    def _cron_authorized(req: Request) -> bool:
        # Native Vercel cron invocations include this header.
        if req.headers.get("x-vercel-cron"):
            return True
        if not settings.cron_secret:
            return True
        auth = req.headers.get("authorization", "")
        if auth == f"Bearer {settings.cron_secret}":
            return True
        if req.headers.get("x-cron-secret", "") == settings.cron_secret:
            return True
        return False

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict:
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            pong = await app.state.cache.redis.ping()
            if not pong:
                raise RuntimeError("Redis ping failed")
            return {"status": "ready"}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(settings.telegram_webhook_path)
    async def telegram_webhook(req: Request) -> dict:
        app_settings = app.state.settings
        if not app_settings.telegram_use_webhook:
            raise HTTPException(status_code=400, detail="Webhook mode disabled")

        if app_settings.telegram_webhook_secret:
            secret = req.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if secret != app_settings.telegram_webhook_secret:
                raise HTTPException(status_code=403, detail="Invalid secret")

        payload = await req.json()
        update = Update.model_validate(payload)
        await app.state.dp.feed_update(app.state.bot, update)
        return {"ok": True}

    @app.post("/test/mock-price")
    async def mock_price(payload: dict) -> dict:
        settings = app.state.settings
        if not settings.test_mode:
            raise HTTPException(status_code=403, detail="TEST_MODE disabled")

        symbol = payload.get("symbol")
        price = payload.get("price")
        if not symbol or price is None:
            raise HTTPException(status_code=400, detail="symbol and price required")

        await app.state.hub.analysis_service.price_adapter.set_mock_price(symbol, float(price))
        return {"ok": True, "symbol": symbol.upper(), "price": float(price)}

    @app.api_route("/tasks/alerts/run", methods=["GET", "POST"])
    async def task_alerts(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        async def _notify(chat_id: int, text: str) -> None:
            try:
                await app.state.bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("task_alert_notify_failed", extra={"event": "task_alert_notify_failed", "chat_id": chat_id, "error": str(exc)})

        try:
            count = await app.state.hub.alerts_service.process_alerts(_notify)
            return {"ok": True, "processed": count, "task": "alerts", "ts": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("task_alerts_failed", extra={"event": "task_alerts_failed", "error": str(exc)})
            return {"ok": False, "processed": 0, "task": "alerts", "error": str(exc), "ts": datetime.now(timezone.utc).isoformat()}

    @app.api_route("/tasks/giveaways/run", methods=["GET", "POST"])
    async def task_giveaways(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        async def _notify(chat_id: int, text: str) -> None:
            try:
                await app.state.bot.send_message(chat_id=chat_id, text=text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("task_giveaway_notify_failed", extra={"event": "task_giveaway_notify_failed", "chat_id": chat_id, "error": str(exc)})

        try:
            count = await app.state.hub.giveaway_service.process_due_giveaways(_notify)
            return {"ok": True, "processed": count, "task": "giveaways", "ts": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("task_giveaways_failed", extra={"event": "task_giveaways_failed", "error": str(exc)})
            return {"ok": False, "processed": 0, "task": "giveaways", "error": str(exc), "ts": datetime.now(timezone.utc).isoformat()}

    @app.api_route("/tasks/news/warm", methods=["GET", "POST"])
    async def task_news(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        try:
            await app.state.hub.news_service.get_daily_brief(limit=10)
            return {"ok": True, "task": "news", "ts": datetime.now(timezone.utc).isoformat()}
        except Exception as exc:  # noqa: BLE001
            logger.exception("task_news_failed", extra={"event": "task_news_failed", "error": str(exc)})
            return {"ok": False, "task": "news", "error": str(exc), "ts": datetime.now(timezone.utc).isoformat()}

    @app.api_route("/tasks/rsi/refresh", methods=["GET", "POST"])
    async def task_rsi_refresh(req: Request) -> dict:
        if not _cron_authorized(req):
            raise HTTPException(status_code=401, detail="Unauthorized")

        force = str(req.query_params.get("force", "")).lower() in {"1", "true", "yes"}
        try:
            universe = await app.state.hub.rsi_scanner_service.refresh_universe(app.state.settings.rsi_scan_universe_size)
            payload = await app.state.hub.rsi_scanner_service.refresh_indicators(force=force)
            return {"ok": True, "task": "rsi_refresh", "force": force, "universe": universe, **payload}
        except Exception as exc:  # noqa: BLE001
            logger.exception("task_rsi_refresh_failed", extra={"event": "task_rsi_refresh_failed", "error": str(exc)})
            return {
                "ok": False,
                "task": "rsi_refresh",
                "force": force,
                "updated": 0,
                "timeframes": [],
                "symbols": 0,
                "error": str(exc),
                "ts": datetime.now(timezone.utc).isoformat(),
            }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
