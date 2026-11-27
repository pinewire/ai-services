"""baseline existing schema

Revision ID: 25782eb03114
Revises:
Create Date: 2026-09-12 12:47:00.024497

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '25782eb03114'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Autogenerate produced an empty body here, because it was run against a
    # database whose tables the application had already created at startup.
    # That made `alembic upgrade head` a no-op on an empty database, and the
    # next revision then failed on a table that had never been created —
    # which only showed up the first time this ran against a genuinely fresh
    # Postgres rather than a developer's existing one.
    #
    # Building the baseline from the models is a deliberate trade. A baseline
    # revision is allowed to couple to the models, because its whole job is to
    # bring an empty database up to the state the next revision expects.
    # Every revision after this one uses explicit DDL.
    #
    # Imported inside the function rather than at module scope so that
    # `alembic history` and friends don't need the application importable.
    import app.models  # noqa: F401  registers the tables on Base.metadata
    from app.db import Base

    bind = op.get_bind()

    # pgvector must exist before any table declaring a vector column is
    # created. The later resize revision does this too, deliberately: each
    # revision has to stand on its own.
    bind.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    Base.metadata.create_all(bind=bind, checkfirst=True)


def downgrade() -> None:
    import app.models  # noqa: F401
    from app.db import Base

    Base.metadata.drop_all(bind=op.get_bind())
