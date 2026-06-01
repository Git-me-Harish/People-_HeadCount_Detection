"""PeopleSense API entrypoint — Phase 2."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import get_settings
from .db import init_db
from .routers import alerts, analytics, auth, cameras, detect, jobs, stream
from .routers import (
    api_tokens,
    audit,
    heatmaps,
    notifications,
    plan,
    public,
    reports,
    templates,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    logger.info("PeopleSense API %s started (env=%s)", __version__, settings.environment)
    yield
    logger.info("PeopleSense API shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "PeopleSense Cloud — multi-tenant crowd intelligence SaaS. "
            "Authenticate via /api/v1/auth/login then call detection endpoints."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_timing_middleware(request: Request, call_next):
        """Log request duration; attach X-Request-ID for tracing."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
        response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
        if elapsed_ms > 2000:
            logger.warning("Slow request: %s %s → %dms", request.method, request.url.path, elapsed_ms)
        return response

    # --------------- meta routes ---------------

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "version": __version__, "environment": settings.environment}

    @app.get("/healthz", tags=["meta"], include_in_schema=False)
    def healthz() -> dict:
        """Kubernetes / Fly.io liveness probe."""
        return {"status": "ok"}

    @app.get("/", tags=["meta"])
    def root() -> JSONResponse:
        return JSONResponse(
            {
                "name": settings.app_name,
                "version": __version__,
                "docs": "/docs",
                "api": settings.api_v1_prefix,
            }
        )

    # --------------- routers ---------------
    prefix = settings.api_v1_prefix

    # Phase 1
    app.include_router(auth.router, prefix=prefix)
    app.include_router(cameras.router, prefix=prefix)
    app.include_router(detect.router, prefix=prefix)
    app.include_router(jobs.router, prefix=prefix)
    app.include_router(analytics.router, prefix=prefix)
    app.include_router(alerts.router, prefix=prefix)
    app.include_router(stream.router, prefix=prefix)

    # Phase 2
    app.include_router(templates.router, prefix=prefix)
    app.include_router(notifications.router, prefix=prefix)
    app.include_router(api_tokens.router, prefix=prefix)
    app.include_router(audit.router, prefix=prefix)
    app.include_router(reports.router, prefix=prefix)
    app.include_router(public.router, prefix=prefix)
    app.include_router(heatmaps.router, prefix=prefix)
    app.include_router(plan.router, prefix=prefix)

    return app


app = create_app()
