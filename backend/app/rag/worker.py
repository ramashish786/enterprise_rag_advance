"""Arq worker: runs ingestion in a separate process.

In production this is invoked as:
    arq app.rag.worker.WorkerSettings

The API enqueues via `queue.enqueue_job('ingest_document', document_id, path)`.
"""
from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.session import get_session
from app.rag.ingestion import process_document

logger = get_logger(__name__)


async def ingest_document(ctx, document_id: str, file_path: str) -> None:
    from uuid import UUID
    async with get_session() as session:
        await process_document(session, document_id=UUID(document_id), file_path=file_path)


async def startup(ctx) -> None:
    configure_logging()
    logger.info("ingestion_worker_started")


class WorkerSettings:
    functions = [ingest_document]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 4
    job_timeout = 600
