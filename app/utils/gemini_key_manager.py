"""
Gemini API key rotation manager.

Provides automatic key fallback when quota/rate-limit errors (429) are hit.
Maintains a shared pool of API keys that both the LLM and embedding services use.
"""

import threading
from google import genai
from loguru import logger


class GeminiKeyManager:
    """
    Manages a pool of Gemini API keys with round-robin fallback on 429 errors.

    Thread-safe: uses a lock to protect key rotation.
    """

    def __init__(self, keys: list[str]):
        if not keys or not any(k.strip() for k in keys):
            logger.error("No Gemini API keys configured!")
            self._keys = [""]
        else:
            self._keys = [k.strip() for k in keys if k.strip()]

        self._index = 0
        self._lock = threading.Lock()
        # Cache clients per key to avoid re-creating them
        self._clients: dict[str, dict[str, genai.Client]] = {}

        logger.info(f"GeminiKeyManager initialized with {len(self._keys)} API key(s)")

    @property
    def current_key(self) -> str:
        """Return the currently active API key."""
        with self._lock:
            return self._keys[self._index]

    def get_client(self, api_version: str = "v1") -> genai.Client:
        """
        Get a Gemini client for the current key + api_version.
        Clients are cached per (key, api_version) pair.
        """
        key = self.current_key
        cache_key = f"{id(key)}:{api_version}"

        if cache_key not in self._clients:
            self._clients[cache_key] = genai.Client(
                api_key=key,
                http_options={"api_version": api_version},
            )
        return self._clients[cache_key]

    def rotate_key(self) -> bool:
        """
        Rotate to the next API key.

        Returns True if a new key is available, False if all keys have been
        exhausted (wrapped back to the starting key).
        """
        with self._lock:
            if len(self._keys) <= 1:
                return False

            prev = self._index
            self._index = (self._index + 1) % len(self._keys)
            wrapped = self._index == 0

            logger.warning(
                f"Rotating Gemini API key: slot {prev} -> {self._index} "
                f"({'wrapped — all keys tried' if wrapped else 'next key'})"
            )
            return not wrapped  # False when we've tried all keys

    @staticmethod
    def is_quota_error(exception: Exception) -> bool:
        """Check if an exception is a 429 / quota / rate-limit error."""
        err = str(exception).lower()
        return any(kw in err for kw in ("429", "quota", "exhausted", "rate"))


# ── Singleton ───────────────────────────────────────────────

_manager: GeminiKeyManager | None = None


def init_key_manager(keys: list[str]) -> GeminiKeyManager:
    """Initialize the global key manager (called once at startup)."""
    global _manager
    _manager = GeminiKeyManager(keys)
    return _manager


def get_key_manager() -> GeminiKeyManager:
    """Return the global key manager, initializing from settings if needed."""
    global _manager
    if _manager is None:
        from app.config import settings
        keys = settings.gemini_api_keys_list
        _manager = GeminiKeyManager(keys)
    return _manager
