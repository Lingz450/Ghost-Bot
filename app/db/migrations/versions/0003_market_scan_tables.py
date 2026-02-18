# ruff: noqa
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_market_scan_tables"
down_revision = "0002_giveaways"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "universe_symbols",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("exchange", sa.String(length=20), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("quote_volume_24h", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("symbol", "exchange", name="uq_universe_symbol_exchange"),
    )
    op.create_index("ix_universe_symbols_symbol", "universe_symbols", ["symbol"], unique=False)
    op.create_index("ix_universe_symbols_exchange", "universe_symbols", ["exchange"], unique=False)
    op.create_index("ix_universe_symbols_rank", "universe_symbols", ["rank"], unique=False)
    op.create_index("ix_universe_symbols_updated_at", "universe_symbols", ["updated_at"], unique=False)

    op.create_table(
        "indicator_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("timeframe", sa.String(length=10), nullable=False),
        sa.Column("exchange", sa.String(length=20), nullable=False),
        sa.Column("close_price", sa.Float(), nullable=True),
        sa.Column("rsi14", sa.Float(), nullable=True),
        sa.Column("ema20", sa.Float(), nullable=True),
        sa.Column("ema50", sa.Float(), nullable=True),
        sa.Column("ema100", sa.Float(), nullable=True),
        sa.Column("ema200", sa.Float(), nullable=True),
        sa.Column("computed_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("symbol", "timeframe", "exchange", name="uq_indicator_symbol_tf_exchange"),
    )
    op.create_index("ix_indicator_snapshots_symbol", "indicator_snapshots", ["symbol"], unique=False)
    op.create_index("ix_indicator_snapshots_timeframe", "indicator_snapshots", ["timeframe"], unique=False)
    op.create_index("ix_indicator_snapshots_exchange", "indicator_snapshots", ["exchange"], unique=False)
    op.create_index("ix_indicator_snapshots_rsi14", "indicator_snapshots", ["rsi14"], unique=False)
    op.create_index("ix_indicator_snapshots_computed_at", "indicator_snapshots", ["computed_at"], unique=False)


def downgrade() -> None:
    op.drop_table("indicator_snapshots")
    op.drop_table("universe_symbols")

