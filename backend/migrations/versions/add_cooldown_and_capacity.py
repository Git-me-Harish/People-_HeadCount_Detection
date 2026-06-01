"""add cooldown_minutes to alerts and max_capacity to cameras

Revision ID: 001_cooldown_capacity
Revises: 
Create Date: 2025-01-01 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "001_cooldown_capacity"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Alert cooldown — default 10 min, industry standard
    with op.batch_alter_table("alerts") as batch_op:
        batch_op.add_column(
            sa.Column("cooldown_minutes", sa.Integer(), nullable=False, server_default="10")
        )

    # Camera capacity planning
    with op.batch_alter_table("cameras") as batch_op:
        batch_op.add_column(
            sa.Column("max_capacity", sa.Integer(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("alerts") as batch_op:
        batch_op.drop_column("cooldown_minutes")
    with op.batch_alter_table("cameras") as batch_op:
        batch_op.drop_column("max_capacity")