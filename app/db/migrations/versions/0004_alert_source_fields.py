# ruff: noqa
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0004_alert_source_fields"
down_revision = "0003_market_scan_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("alerts", sa.Column("source_exchange", sa.String(length=20), nullable=True))
    op.add_column("alerts", sa.Column("instrument_id", sa.String(length=40), nullable=True))
    op.add_column("alerts", sa.Column("market_kind", sa.String(length=10), nullable=True))
    op.create_index("ix_alerts_source_exchange", "alerts", ["source_exchange"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_alerts_source_exchange", table_name="alerts")
    op.drop_column("alerts", "market_kind")
    op.drop_column("alerts", "instrument_id")
    op.drop_column("alerts", "source_exchange")
