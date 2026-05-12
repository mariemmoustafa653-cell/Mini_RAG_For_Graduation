"""
Mini RAG E-Learning Platform — FastAPI Application Entry Point.

An AI-powered multi-tenant educational assistant that uses
Retrieval-Augmented Generation (RAG) to answer questions,
summarize content, generate quizzes, explain topics, create
flashcards, and translate materials from uploaded PDF documents.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger

from app.config import settings
from app.api.routes import router
from app.utils.database import init_database
from app.utils.logger import setup_logger


# ── Initialize Logging ──────────────────────────────────────
setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # ── Startup ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Mini RAG E-Learning Platform Starting...")
    logger.info("=" * 60)

    # Initialize database
    init_database()

    # Ensure required directories exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.VECTOR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Upload directory: {settings.UPLOAD_DIR}")
    logger.info(f"Vector data directory: {settings.VECTOR_DATA_DIR}")
    logger.info(f"Database: {settings.DATABASE_PATH}")
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    logger.info(f"LLM model: {settings.LLM_MODEL}")
    logger.info(f"Chunk size: {settings.CHUNK_SIZE}, overlap: {settings.CHUNK_OVERLAP}")
    logger.info(f"Top-K retrieval: {settings.TOP_K}")
    logger.info(f"Similarity threshold: {settings.SIMILARITY_THRESHOLD}")
    logger.info(f"Temperature: {settings.TEMPERATURE}")
    logger.info(f"Max retries: {settings.MAX_RETRIES}")
    logger.info("Platform ready! ✨")
    logger.info("=" * 60)

    yield

    # ── Shutdown ────────────────────────────────────────
    logger.info("Platform shutting down...")


# ── Create FastAPI App ──────────────────────────────────────
app = FastAPI(
    title="Mini RAG E-Learning Platform",
    description=(
        "AI-powered multi-tenant educational assistant using RAG. "
        "Upload course PDFs and interact with an AI tutor that answers questions, "
        "summarizes, generates quizzes, explains, creates flashcards, and translates."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routes ──────────────────────────────────────────────
app.include_router(router)

# ── Static Files (Frontend) ────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── Root Endpoints ──────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the frontend dashboard."""
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "name": "Mini RAG E-Learning Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "api": "/api",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "mini-rag"}
