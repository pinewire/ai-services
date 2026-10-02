"""resize embedding vectors for text-embedding-3-small native dimensions

Revision ID: 8d1c4d2a9f31
Revises: 25782eb03114
"""

from typing import Sequence, Union

from alembic import op

revision: str = "8d1c4d2a9f31"
down_revision: Union[str, None] = "25782eb03114"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_embedding")
    # Existing vectors cannot be meaningfully resized; re-ingest the KB after this migration.
    op.execute(
        "ALTER TABLE kb_chunks ALTER COLUMN embedding TYPE vector(1536) "
        "USING NULL::vector(1536)"
    )
    op.create_index(
        "ix_kb_chunks_embedding",
        "kb_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_kb_chunks_embedding")
    op.execute(
        "ALTER TABLE kb_chunks ALTER COLUMN embedding TYPE vector(1024) "
        "USING NULL::vector(1024)"
    )
    op.create_index(
        "ix_kb_chunks_embedding",
        "kb_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
