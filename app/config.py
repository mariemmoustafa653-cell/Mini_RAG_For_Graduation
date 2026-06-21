"""
Application configuration loaded from environment variables.
Uses pydantic-settings for type-safe config management.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


# Resolve base directory at module level
_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for the Mini RAG platform."""

    # ── Gemini ──────────────────────────────────────────────
    GEMINI_API_KEY: str = ""

    # ── Model Configuration ─────────────────────────────────
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    FALLBACK_EMBEDDING_MODEL: str = "models/gemini-embedding-2-preview"
    LLM_MODEL: str = "gemini-2.5-flash"
    FALLBACK_LLM_MODEL: str = "gemini-flash-latest"

    # ── Chunking ────────────────────────────────────────────
    CHUNK_SIZE: int = 800  # Reduced from 1000 to save tokens
    CHUNK_OVERLAP: int = 150

    # ── Retrieval ───────────────────────────────────────────
    TOP_K: int = 3  # Reduced from 5 to save context tokens
    SIMILARITY_THRESHOLD: float = 0.25  # Minimum cosine similarity to accept a chunk

    # ── LLM Generation ─────────────────────────────────────
    TEMPERATURE: float = 0.3  # Low for factual accuracy
    MAX_TOKENS: int = 1000  # Reduced from 2000 to prevent excessive generation

    # ── Resilience ─────────────────────────────────────────
    MAX_RETRIES: int = 5  # Increased retries
    RETRY_MIN_WAIT: float = 2.0  # Increased initial wait
    RETRY_MAX_WAIT: float = 60.0  # Increased max wait for backoff

    # ── Optimization ────────────────────────────────────────
    CACHE_TTL: int = 3600  # seconds

    # ── Upload ──────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 20

    # ── Server ──────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Paths (computed from BASE_DIR) ──────────────────────
    BASE_DIR: Path = _BASE_DIR
    DATA_DIR: Path = _BASE_DIR

    @property
    def UPLOAD_DIR(self) -> Path:
        return self.DATA_DIR / "uploads"

    @property
    def VECTOR_DATA_DIR(self) -> Path:
        return self.DATA_DIR / "vector_data"

    @property
    def DATABASE_PATH(self) -> Path:
        return self.DATA_DIR / "mini_rag.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def model_post_init(self, __context):
        """Ensure required directories exist after initialization."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


# Singleton settings instance
settings = Settings()
