# Ghost Alpha Bot

Production-ready Telegram bot for crypto analysis, alerts, wallet scans, cycle checks, trade verification, watchlists, news briefs, and BTC-correlation checks.

## What it does

- Trade plans from natural language (`SOL long`, `ETH short`, `Long?`) using TA + market narrative
- Price alerts (`ping me when SOL hits 100`) with one-shot trigger + anti-spam dedupe/cooldown
- RSI scanner (`rsi top 10 1h oversold`, `top rsi 4h overbought`)
- EMA scanner (`ema 200 4h top 10`, `which coins are near ema50 1h`)
- Chart images (`chart btc 1h`, `show me sol chart 4h`)
- Orderbook heatmap (`heatmap btc`, `orderbook depth sol`)
- Wallet scans for Solana/Tron (`scan solana <addr>`, `scan tron <addr>`)
- Cycle checks (`cycle check`, `are we near top?`)
- Trade verification (`check this trade from yesterday: ...`) with same-candle ambiguity modes
- Pair/price discovery (`find pair xion`, `coin around 0.155`)
- Giveaway admin flow (`/giveaway ...`, `/join`) with anti-back-to-back winner rule
- Watchlists (`Coins to watch 5`)
- Daily news briefs (`latest news today`) with source links
- Correlation (`is BIRB following BTC?`) with corr + beta context
- User settings (`/settings`): anon/formal/profanity/risk/timeframe/exchange

This bot is analysis only. It does **not** place trades.

## Stack

- Python 3.11
- aiogram 3 (Telegram)
- FastAPI (health, readiness, webhook, test-mode endpoints)
- Postgres + SQLAlchemy + Alembic
- Redis (cache + rate limiting + dedupe)
- APScheduler (background jobs)
- httpx (resilient adapters with retry/backoff + circuit breaker)
- pandas/numpy (TA + analytics)
- pytest (unit tests)

## Repository layout

- `app/main.py` - bootstrap, FastAPI app, webhook/polling runtime
- `app/bot/handlers.py` - Telegram handlers (commands + NLU routing + callbacks)
- `app/core/nlu.py` - deterministic regex/heuristic intent/entity parser
- `app/services/` - analysis/alerts/wallet/news/watchlist/cycle/correlation/trade-verify
- `app/adapters/` - market router + multi-exchange adapters + RSS/Solana/Tron adapters
- `app/adapters/exchanges/` - Binance, Bybit, OKX (and optional MEXC/BloFin) market clients
- `app/db/models.py` - DB schema models
- `app/db/migrations/versions/0001_initial.py` - initial Alembic migration
- `app/workers/scheduler.py` - periodic alert monitor + news refresh
- `api/index.py` - Vercel ASGI entrypoint
- `vercel.json` - Vercel rewrites + cron schedule
- `tests/` - parser + indicator + tradecheck tests

## Quick start (Docker Compose)

1. Copy env:

```bash
cp .env.example .env
```

2. Set at minimum:

- `TELEGRAM_BOT_TOKEN`

3. Start:

```bash
docker compose up --build
```

4. Health checks:

- `GET http://localhost:8000/health`
- `GET http://localhost:8000/ready`

The app runs migrations on start (`alembic upgrade head`) then launches the bot + API.

## Long polling vs webhook

Default is long polling:

- `TELEGRAM_USE_WEBHOOK=false`

Webhook mode:

- Set `TELEGRAM_USE_WEBHOOK=true`
- Set `TELEGRAM_WEBHOOK_URL=https://your-domain.com`
- Optional: set `TELEGRAM_WEBHOOK_SECRET`
- Ensure your reverse proxy routes `POST /telegram/webhook` to port `8000`

## Deploy on Vercel

1. Push this repo to GitHub and import it into Vercel.
2. Set build/runtime to Python (Vercel auto-detects from `api/index.py`).
3. Set required env vars in Vercel project:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_USE_WEBHOOK=true`
- `TELEGRAM_WEBHOOK_URL=https://<your-vercel-domain>`
- `TELEGRAM_WEBHOOK_PATH=/telegram/webhook`
- `SERVERLESS_MODE=true`
- `TELEGRAM_AUTO_SET_WEBHOOK=true`
- `DATABASE_URL` (use hosted Postgres, e.g. Neon/Supabase)
- `REDIS_URL` (use hosted Redis, e.g. Upstash/Redis Cloud)
- `CRON_SECRET` (used to secure `/tasks/*` cron endpoints)
4. Run DB migrations against your hosted Postgres before first traffic:
```bash
alembic upgrade head
```
5. Deploy. For Hobby plan (no sub-daily Vercel cron), use the included GitHub Actions scheduler:
- Add GitHub repo secret `VERCEL_BASE_URL=https://<your-vercel-domain>`
- Add GitHub repo secret `CRON_SECRET=<same value as Vercel env>`
- Workflow `.github/workflows/serverless-tasks.yml` runs every 5 minutes and calls:
  - `/tasks/alerts/run`
  - `/tasks/giveaways/run`
  - `/tasks/rsi/refresh`
  - `/tasks/news/warm` (every 15 minutes)

Notes:
- Docker Compose is for local/VPS only; Vercel needs external DB/Redis.
- In `SERVERLESS_MODE`, polling and APScheduler are disabled automatically.

### Neon + Upstash quick setup

1. Create a Neon project and copy the **pooled** Postgres connection string.
2. Use async driver format in env:
   - `postgresql://...` or `postgres://...` also works now (auto-normalized), but recommended:
   - `DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@HOST/DB?ssl=require`
3. Create an Upstash Redis database and copy the Redis TCP URL (not REST URL):
   - `REDIS_URL=rediss://default:PASSWORD@HOST:6379`
4. Put both into Vercel Project Settings -> Environment Variables.
5. Run migrations once against Neon before production traffic:
```bash
alembic upgrade head
```

## Environment variables

Required:

- `TELEGRAM_BOT_TOKEN`

Primary runtime:

- `DATABASE_URL` (default: `postgresql+asyncpg://ghost:ghost@postgres:5432/ghost_bot`)
- `REDIS_URL` (default: `redis://redis:6379/0`)
- `TELEGRAM_USE_WEBHOOK`
- `TELEGRAM_WEBHOOK_URL`
- `TELEGRAM_WEBHOOK_PATH`
- `TELEGRAM_WEBHOOK_SECRET`
- `TELEGRAM_AUTO_SET_WEBHOOK` (default `true`)
- `SERVERLESS_MODE` (default `false`; set `true` on Vercel)
- `CRON_SECRET` (optional but recommended for `/tasks/*`)
  - Note: Vercel native cron calls are accepted via `x-vercel-cron` header automatically.
- `OPENAI_API_KEY` (optional, enables freeform Q&A fallback)
- `OPENAI_MODEL` (default `gpt-4.1-mini`)
- `OPENAI_ROUTER_MODEL` (optional; use a smaller model just for JSON intent routing)
- `OPENAI_MAX_OUTPUT_TOKENS` (default `350`)
- `OPENAI_TEMPERATURE` (default `0.7`)
- `OPENAI_ROUTER_MIN_CONFIDENCE` (default `0.6`, LLM intent router execution threshold)
- `OPENAI_CHAT_MODE` (default `hybrid`; options: `tool_first`, `hybrid`, `llm_first`, `chat_only`)
- `OPENAI_CHAT_HISTORY_TURNS` (default `12`, short memory window for chat continuity)
- `RSI_SCAN_UNIVERSE_SIZE` (default `500`)
- `RSI_SCAN_SCAN_TIMEFRAMES` (default `15m,1h,4h,1d`)
- `RSI_SCAN_CONCURRENCY` (default `12`)
- `RSI_SCAN_FRESHNESS_MINUTES` (default `45`)
- `RSI_SCAN_LIVE_FALLBACK_UNIVERSE` (default `120`)

Data source/adapters:

- `BINANCE_BASE_URL`
- `BINANCE_FUTURES_BASE_URL`
- `BYBIT_BASE_URL`
- `OKX_BASE_URL`
- `MEXC_BASE_URL`
- `BLOFIN_BASE_URL`
- `ENABLE_BINANCE`
- `ENABLE_BYBIT`
- `ENABLE_OKX`
- `ENABLE_MEXC`
- `ENABLE_BLOFIN`
- `EXCHANGE_PRIORITY` (default `binance,bybit,okx,mexc,blofin`)
- `MARKET_PREFER_SPOT` (default `true`; TA/price/ohlcv route as spot-first then perp)
- `BEST_SOURCE_TTL_HOURS` (default `12`)
- `INSTRUMENTS_TTL_MIN` (default `45`)
- `COINGECKO_BASE_URL`
- `NEWS_RSS_FEEDS`
- `OPENAI_RSS_FEEDS` (optional, official OpenAI update/news feeds)
- `CRYPTOPANIC_API_KEY` (optional)
- `SOLANA_RPC_URL`
- `TRON_API_URL`
- `TRONGRID_API_KEY` (optional)

Limits/reliability:

- `REQUEST_RATE_LIMIT_PER_MINUTE` (default 20)
- `WALLET_SCAN_LIMIT_PER_HOUR` (default 10)
- `ALERTS_CREATE_LIMIT_PER_DAY` (default 10)
- `ALERT_CHECK_INTERVAL_SEC` (default 30)
- `ALERT_COOLDOWN_MIN` (default 30)
- `ALERT_MAX_DEVIATION_PCT` (default 30; rejects unrealistic alert levels too far from current price)
- `ADMIN_CHAT_IDS` (comma-separated Telegram user IDs allowed to run giveaway admin commands)
- `GIVEAWAY_MIN_PARTICIPANTS` (default 2)
- `ANALYSIS_FAST_MODE` (default `true`)
- `ANALYSIS_DEFAULT_TIMEFRAMES` (default `1h`)
- `ANALYSIS_INCLUDE_DERIVATIVES_DEFAULT` (default `false`)
- `ANALYSIS_INCLUDE_NEWS_DEFAULT` (default `false`)
- `ANALYSIS_REQUEST_TIMEOUT_SEC` (default `8`)

Behavior/test mode:

- `DEFAULT_TIMEFRAME`
- `INCLUDE_BTC_ETH_WATCHLIST`
- `TEST_MODE` (default true in `.env.example`)
- `MOCK_PRICES` (e.g. `SOL:100,BTC:70000`)

## Reliability and safety

- Retries + exponential backoff for HTTP adapters
- Circuit breaker per upstream host
- Redis cache keys:
  - `price:<symbol>`
  - `ohlcv:<symbol>:<tf>:<limit>`
  - `news:crypto`
  - `news:openai`
  - `funding:<symbol>`
  - `rsi:universe:<N>`
- Alert anti-spam:
  - one-shot status transition to `triggered`
  - minute-bucket dedupe key
  - cooldown timestamp
- Per-user rate limits for requests/scans/alerts
- Minimal PII storage: Telegram chat ID and optional saved wallets
- Wallet scans are public-chain-only and include a no-attribution warning

## Telegram commands

- `/start`
- `/help`
- `/settings`
- `/alpha <symbol> [tf] [ema=..] [rsi=..]`
- `/watch <symbol> [tf]`
- `/chart <symbol> [tf]`
- `/heatmap <symbol>`
- `/rsi <tf> <overbought|oversold> [topN] [len]`
- `/ema <ema_len> <tf> [topN]`
- `/news [crypto|openai|cpi|fomc] [limit]`
- `/alert <symbol> <price> [above|below|cross]`
- `/alerts`
- `/alertdel <id>`
- `/alertclear [symbol]`
- `/findpair <price_or_query>`
- `/setup <freeform setup text>`
- `/watchlist [N]`
- `/alert add <symbol> <above|below|cross> <price>`
- `/alert list`
- `/alert delete <id>`
- `/alert clear`
- `/alert pause`
- `/alert resume`
- `/scan <chain> <address>`
- `/tradecheck` (interactive wizard)
- `/news`
- `/news cpi`
- `/news openai`
- `/cycle`
- `/giveaway start <10m|1h|1d> <prize> [winners=N]`
- `/giveaway join`
- `/giveaway end`
- `/giveaway reroll`
- `/giveaway status`
- `/join`

Natural language is fully supported, no strict command requirement.
If you send a slash command without enough arguments, the bot now opens interactive button pickers (duration/timeframe/mode/etc.) instead of only showing usage text.
When configured with `OPENAI_API_KEY`, an LLM JSON intent router handles free-form phrasing and maps to deterministic tools (analysis/alerts/news/scans/etc.).

OpenAI response strategy is configurable:

- `OPENAI_CHAT_MODE=tool_first`: deterministic parser first, LLM mainly fallback
- `OPENAI_CHAT_MODE=hybrid`: deterministic + LLM router for unknowns (default)
- `OPENAI_CHAT_MODE=llm_first`: LLM router first, then LLM chat fallback
- `OPENAI_CHAT_MODE=chat_only`: every message goes straight to OpenAI chat (ChatGPT-like behavior)
Analysis uses a fast default path (price + TA) and exposes on-demand detail buttons: `More detail`, `Derivatives`, `News`.

### Multi-exchange fallback (automatic)

- Users never choose exchange.
- Market data routing priority is configurable by `EXCHANGE_PRIORITY`.
- Router behavior:
  1. Try spot first in priority order (Binance -> Bybit -> OKX -> optional MEXC/BloFin).
  2. If no spot market works, fallback to perp in the same order.
  3. Cache best source per symbol (`best_source:<symbol>:spot|perp`) and reuse it.
- Source metadata is kept internally for continuity/debugging and alert failover.
- User-facing replies stay clean by default; ask `source?` or `which exchange?` to reveal the last result source on demand.

Advanced analysis syntax is supported:

- `SOL long 15m ema9 ema21 rsi14`
- `ETH short tf=1h,4h ema=20,50,200 rsi=14,21`
- `BTC long all timeframes all emas all rsis`
- `BTC limit long entry 66300 sl 64990 tp 69200 72000` (setup review mode)
- `watch btc` / `watch BTC` (quick analysis intent)
- `rsi top 10 1h oversold` / `top rsi 4h overbought`
- `find pair xion` / `find $PHB`
- `coin around 0.155`

Slash equivalents (optional for power users):

- `/alpha SOL 4h ema=20,50,200 rsi=14`
- `/chart BTC 1h`
- `/rsi 1h oversold 10 14`
- `/ema 200 4h 10`
- `/alert SOL 100`
- `/alerts`
- `/alertdel 12`
- `/alertclear SOL`
- `/findpair 0.155`
- `/setup long BTC entry 66300 sl 64990 tp 69200 72000 amount 100 leverage 10`

Group behavior:

- Default: bot responds in groups when mentioned (`@GhotalphaBot`), on `/commands`, when replying to the bot, or for clear trading intents (`alert/chart/rsi/ema/news/btc/eth/sol`).
- Admin toggle in group: `free talk mode on` / `free talk mode off`.
- In private chats: bot responds to all messages.

## Operator guide

1. ChatGPT-like default replies:
- Set `OPENAI_API_KEY`.
- Set `OPENAI_CHAT_MODE=llm_first` (recommended).
- If you want every single message to be pure OpenAI chat, set `OPENAI_CHAT_MODE=chat_only`.

2. Group free-talk controls:
- In a group, admins can toggle: `free talk mode on` / `free talk mode off`.
- OFF: bot responds on mention/reply/clear intent.
- ON: bot can respond without mention (still rate-limited).

3. Scanner universe + speed tuning:
- `RSI_SCAN_UNIVERSE_SIZE` -> set 500 to 1000.
- `RSI_SCAN_SCAN_TIMEFRAMES` -> e.g. `15m,1h,4h,1d`.
- `RSI_SCAN_CONCURRENCY` -> increase cautiously if your host is strong.
- `RSI_SCAN_FRESHNESS_MINUTES` -> lower for fresher scanner data.
- `RSI_SCAN_LIVE_FALLBACK_UNIVERSE` -> fallback scan size when snapshots are stale.

4. Alert realism guard:
- `ALERT_MAX_DEVIATION_PCT` rejects unrealistic far targets by default.

5. Vercel serverless tasks:
- Keep GitHub Actions scheduler enabled (`.github/workflows/serverless-tasks.yml`) to run alerts/giveaways/scanner/news tasks every few minutes on Hobby.

## Manual QA (acceptance scenarios)

1. Send `SOL long`
- Expected: 1-2 line summary + Entry/TP1/TP2/SL block + signal/narrative bullets + risk line + inline buttons

2. Send `ETH short`
- Expected: short-bias trade plan in same structured format

3. Send `Coins to watch 5`
- Expected: day theme + 5-symbol watchlist with one-line catalysts

4. Send `what are the latest news for today`
- Expected: compact brief + headlines + links + vibe line

4b. Send `cpi news` or `openai updates`
- Expected: topic-filtered digest (macro/OpenAI) with source links; if no strict match, bot falls back to latest flow and says so

5. Send `ping me when SOL hits 100`
- Expected: alert created with ID and normalized condition

6. Trigger alert in test mode
- Set `TEST_MODE=true`
- Call:

```bash
curl -X POST http://localhost:8000/test/mock-price \
  -H "Content-Type: application/json" \
  -d '{"symbol":"SOL","price":100}'
```

- Expected: worker triggers alert once and marks it `triggered`

7. Send `scan solana <address>`
- Expected: native balance, token breakdown, tx context, warnings, `Save wallet` button

8. Send `check this trade from yesterday: ETH entry 2100 stop 2165 targets 2043 2027 1991 timeframe 1h`
- Expected: win/loss/ambiguous + first-hit time + MFE/MAE + R multiple

9. Send `is BIRB following BTC?`
- Expected: verdict + corr/beta + relative performance bullets

10. Send `ema 200 4h top 10` or `chart btc 1h`
- Expected: EMA proximity scan list and chart image response

## Test mode (local-safe)

- Works without premium API keys (RSS + public endpoints + fallbacks)
- Mockable prices via:
  - `MOCK_PRICES` env at startup
  - `POST /test/mock-price` at runtime

## Local dev (without Docker)

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows
pip install -r requirements.txt
alembic upgrade head
python -m app.main
```

## Tests

```bash
python -m pytest -q
```

Includes:

- 30+ NLU parse cases
- TA deterministic checks
- Trade verification ambiguity mode checks

## VPS deployment notes

1. Provision VPS with Docker + Docker Compose
2. Clone repo and configure `.env`
3. Run `docker compose up -d --build`
4. Put Nginx/Caddy in front of `:8000` if using webhook
5. Monitor `/health` and `/ready`
6. Persist Docker volumes (`pgdata`, `redisdata`)

## Privacy note

- No exchange private keys stored
- No user API keys stored
- Wallet scans only process public-chain data
- Do not use outputs for doxxing, harassment, or attribution

## Assumptions

- Binance public endpoints are primary market/ohlcv source
- CoinGecko fallback handles symbols unavailable on Binance
- Derivatives sentiment is best-effort and omitted gracefully when unavailable
- Trade verification defaults to `ambiguous` same-candle mode
- Not financial advice
