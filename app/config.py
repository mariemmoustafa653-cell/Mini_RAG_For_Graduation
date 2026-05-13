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
    EMBEDDING_MODEL: str = "models/gemini-embedding-2"
    LLM_MODEL: str = "gemini-2.0-flash"
    FALLBACK_LLM_MODEL: str = "gemini-flash-latest"

    # ── Chunking ────────────────────────────────────────────
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # ── Retrieval ───────────────────────────────────────────
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.25  # Minimum cosine similarity to accept a chunk

    # ── LLM Generation ─────────────────────────────────────
    TEMPERATURE: float = 0.3  # Low for factual accuracy
    MAX_TOKENS: int = 2000

    # ── Resilience ─────────────────────────────────────────
    MAX_RETRIES: int = 3
    RETRY_MIN_WAIT: float = 1.0  # seconds
    RETRY_MAX_WAIT: float = 10.0  # seconds

    # ── Upload ──────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 20

    # ── Server ──────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Paths (computed from BASE_DIR) ──────────────────────
    BASE_DIR: Path = _BASE_DIR
    UPLOAD_DIR: Path = _BASE_DIR / "uploads"
    VECTOR_DATA_DIR: Path = _BASE_DIR / "vector_data"
    DATABASE_PATH: Path = _BASE_DIR / "mini_rag.db"

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
