from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ghost-alpha-bot"
    env: str = "dev"
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8000

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_use_webhook: bool = False
    telegram_webhook_url: str = ""
    telegram_webhook_path: str = "/telegram/webhook"
    telegram_webhook_secret: str = ""
    telegram_auto_set_webhook: bool = Field(default=True, alias="TELEGRAM_AUTO_SET_WEBHOOK")
    serverless_mode: bool = Field(default=False, alias="SERVERLESS_MODE")
    cron_secret: str = Field(default="", alias="CRON_SECRET")
    # LLM providers — Claude is primary, Grok is fallback, OpenAI is last resort
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    xai_api_key: str = Field(default="", alias="XAI_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_router_model: str = Field(default="", alias="OPENAI_ROUTER_MODEL")
    openai_max_output_tokens: int = Field(default=350, alias="OPENAI_MAX_OUTPUT_TOKENS")
    openai_temperature: float = Field(default=0.7, alias="OPENAI_TEMPERATURE")
    openai_router_min_confidence: float = Field(default=0.6, alias="OPENAI_ROUTER_MIN_CONFIDENCE")
    openai_chat_mode: str = Field(default="hybrid", alias="OPENAI_CHAT_MODE")
    openai_chat_history_turns: int = Field(default=12, alias="OPENAI_CHAT_HISTORY_TURNS")

    database_url: str = "postgresql+asyncpg://ghost:ghost@postgres:5432/ghost_bot"
    redis_url: str = "redis://redis:6379/0"

    binance_base_url: str = "https://api.binance.com"
    binance_futures_base_url: str = "https://fapi.binance.com"
    bybit_base_url: str = Field(default="https://api.bybit.com", alias="BYBIT_BASE_URL")
    okx_base_url: str = Field(default="https://www.okx.com", alias="OKX_BASE_URL")
    mexc_base_url: str = Field(default="https://api.mexc.com", alias="MEXC_BASE_URL")
    blofin_base_url: str = Field(default="https://openapi.blofin.com", alias="BLOFIN_BASE_URL")
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    enable_binance: bool = Field(default=True, alias="ENABLE_BINANCE")
    enable_bybit: bool = Field(default=True, alias="ENABLE_BYBIT")
    enable_okx: bool = Field(default=True, alias="ENABLE_OKX")
    enable_mexc: bool = Field(default=False, alias="ENABLE_MEXC")
    enable_blofin: bool = Field(default=False, alias="ENABLE_BLOFIN")
    exchange_priority: str = Field(default="binance,bybit,okx,mexc,blofin", alias="EXCHANGE_PRIORITY")
    market_prefer_spot: bool = Field(default=True, alias="MARKET_PREFER_SPOT")
    best_source_ttl_hours: int = Field(default=12, alias="BEST_SOURCE_TTL_HOURS")
    instruments_ttl_min: int = Field(default=45, alias="INSTRUMENTS_TTL_MIN")

    cryptopanic_api_key: str = ""
    news_rss_feeds: str = (
        "https://www.coindesk.com/arc/outboundfeeds/rss/;"
        "https://cointelegraph.com/rss;"
        "https://www.theblock.co/rss.xml"
    )
    openai_rss_feeds: str = Field(
        default=(
        "https://developers.openai.com/changelog/rss.xml;"
        "https://openai.com/news/rss.xml"
        ),
        alias="OPENAI_RSS_FEEDS",
    )

    solana_rpc_url: str = "https://api.mainnet-beta.solana.com"
    tron_api_url: str = "https://api.trongrid.io"
    trongrid_api_key: str = ""

    request_rate_limit_per_minute: int = 20
    wallet_scan_limit_per_hour: int = 10
    alerts_create_limit_per_day: int = 10
    alert_max_deviation_pct: float = Field(default=30.0, alias="ALERT_MAX_DEVIATION_PCT")

    alert_check_interval_sec: int = 30
    alert_cooldown_min: int = 30
    admin_chat_ids: str = ""
    broadcast_enabled: bool = Field(default=False, alias="BROADCAST_ENABLED")
    broadcast_channel_ids: str = Field(default="", alias="BROADCAST_CHANNEL_IDS")
    broadcast_interval_minutes: int = Field(default=15, alias="BROADCAST_INTERVAL_MINUTES")
    broadcast_rate_limit_minutes: int = Field(default=60, alias="BROADCAST_RATE_LIMIT_MINUTES")
    giveaway_min_participants: int = 2
    analysis_fast_mode: bool = Field(default=True, alias="ANALYSIS_FAST_MODE")
    analysis_default_timeframes: str = Field(default="1h", alias="ANALYSIS_DEFAULT_TIMEFRAMES")
    analysis_include_derivatives_default: bool = Field(default=False, alias="ANALYSIS_INCLUDE_DERIVATIVES_DEFAULT")
    analysis_include_news_default: bool = Field(default=False, alias="ANALYSIS_INCLUDE_NEWS_DEFAULT")
    analysis_request_timeout_sec: float = Field(default=8.0, alias="ANALYSIS_REQUEST_TIMEOUT_SEC")
    rsi_scan_universe_size: int = Field(default=500, alias="RSI_SCAN_UNIVERSE_SIZE")
    rsi_scan_scan_timeframes: str = Field(default="15m,1h,4h,1d", alias="RSI_SCAN_SCAN_TIMEFRAMES")
    rsi_scan_concurrency: int = Field(default=12, alias="RSI_SCAN_CONCURRENCY")
    rsi_scan_freshness_minutes: int = Field(default=45, alias="RSI_SCAN_FRESHNESS_MINUTES")
    rsi_scan_live_fallback_universe: int = Field(default=120, alias="RSI_SCAN_LIVE_FALLBACK_UNIVERSE")

    default_timeframe: str = "1h"
    include_btc_eth_watchlist: bool = True

    test_mode: bool = False
    mock_prices: str = ""

    def rss_feed_list(self) -> List[str]:
        return [x.strip() for x in self.news_rss_feeds.split(";") if x.strip()]

    def openai_rss_feed_list(self) -> List[str]:
        return [x.strip() for x in self.openai_rss_feeds.split(";") if x.strip()]

    def admin_ids_list(self) -> List[int]:
        out: List[int] = []
        for item in self.admin_chat_ids.split(","):
            raw = item.strip()
            if not raw:
                continue
            try:
                out.append(int(raw))
            except ValueError:
                continue
        return out

    def analysis_default_timeframes_list(self) -> List[str]:
        out: List[str] = []
        for item in self.analysis_default_timeframes.split(","):
            tf = item.strip()
            if tf:
                out.append(tf)
        return out or ["1h"]

    def broadcast_channel_ids_list(self) -> List[int]:
        out: List[int] = []
        for item in self.broadcast_channel_ids.split(","):
            raw = item.strip()
            if not raw:
                continue
            try:
                out.append(int(raw))
            except ValueError:
                continue
        return out

    def rsi_scan_timeframes_list(self) -> List[str]:
        out: List[str] = []
        for item in self.rsi_scan_scan_timeframes.split(","):
            tf = item.strip()
            if tf:
                out.append(tf)
        return out or ["15m", "1h", "4h", "1d"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
