"""Document routes (admin): upload, list, delete."""
from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _db_session, require_permission
from app.audit.service import AuditLogService
from app.auth.service import SecurityPrincipal
from app.db.models import Document, DocumentStatus, DocumentVisibility
from app.rag.ingestion import process_document, register_document
from app.schemas.api import DocumentOut

router = APIRouter(prefix="/api/documents", tags=["documents"])

UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    visibility: DocumentVisibility = Form(DocumentVisibility.INTERNAL),
    department_id: UUID | None = Form(None),
    principal: SecurityPrincipal = Depends(require_permission("document", "upload")),
    session: AsyncSession = Depends(_db_session),
):
    # Save to disk
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".pdf", ".docx", ".pptx", ".txt", ".md"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unsupported extension: {ext}")

    dest = UPLOAD_DIR / f"{uuid4().hex}{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = await register_document(
        session,
        filename=file.filename or dest.name,
        file_path=dest,
        mime_type=file.content_type or "application/octet-stream",
        uploaded_by=principal.user_id,
        department_id=department_id,
        visibility=visibility,
    )

    # Enqueue ingestion. When Arq queue isn't available we run inline as a
    # fallback (dev convenience). In production this always uses the queue.
    try:
        from arq import create_pool
        from arq.connections import RedisSettings
        from app.core.config import settings
        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        await pool.enqueue_job("ingest_document", str(doc.id), str(dest))
    except Exception:
        # Fallback: inline processing (blocks the request; fine for tests/dev)
        await process_document(session, document_id=doc.id, file_path=dest)

    await AuditLogService(session).log(
        event_type="document.uploaded", action="upload", outcome="success",
        user_id=principal.user_id, resource_type="document", resource_id=str(doc.id),
        payload={"filename": doc.filename, "visibility": visibility.value},
    )

    return DocumentOut(
        id=doc.id, filename=doc.filename, status=doc.status.value,
        visibility=doc.visibility.value, version=doc.version, created_at=doc.created_at,
    )


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    principal: SecurityPrincipal = Depends(require_permission("document", "upload")),
    session: AsyncSession = Depends(_db_session),
):
    rows = (await session.execute(select(Document).order_by(Document.created_at.desc()))).scalars().all()
    return [
        DocumentOut(id=d.id, filename=d.filename, status=d.status.value,
                    visibility=d.visibility.value, version=d.version, created_at=d.created_at)
        for d in rows
    ]


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    principal: SecurityPrincipal = Depends(require_permission("document", "delete")),
    session: AsyncSession = Depends(_db_session),
):
    doc = (await session.execute(select(Document).where(Document.id == document_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    await session.delete(doc)
    await session.commit()
    await AuditLogService(session).log(
        event_type="document.deleted", action="delete", outcome="success",
        user_id=principal.user_id, resource_type="document", resource_id=str(document_id),
    )
    return {"ok": True}
