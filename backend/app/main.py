"""FastAPI application entrypoint.

Wires all routers, configures middleware (CORS, request-id logging), maps
domain exceptions to HTTP responses.
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.api import (
    admin_routes,
    auth_routes,
    chat_routes,
    document_routes,
    policy_routes,
)
from app.core.config import settings
from app.core.exceptions import InsuranceAgentError
from app.core.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    configure_logging()
    logger = get_logger(__name__)

    app = FastAPI(
        title="Enterprise Insurance AI Chat Agent",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # every log line during a request carries this id
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        structlog.contextvars.bind_contextvars(request_id=req_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["x-request-id"] = req_id
        return response

    # map our own exceptions to HTTP responses
    @app.exception_handler(InsuranceAgentError)
    async def domain_error_handler(request: Request, exc: InsuranceAgentError):
        return JSONResponse(
            status_code=exc.http_status,
            content={"error": str(exc), "context": exc.context},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": "validation", "detail": exc.errors()})

    # Routers
    app.include_router(auth_routes.router)
    app.include_router(chat_routes.router)
    app.include_router(document_routes.router)
    app.include_router(admin_routes.router)
    app.include_router(policy_routes.router)

    @app.get("/health")
    async def health():
        return {"ok": True}

    logger.info("app_ready", env=settings.app_env)
    return app


app = create_app()
