"""
Embedding service using Google Gemini's multilingual embedding model.
Generates vector representations for document chunks and user queries.
Includes retry logic for API resilience.
"""

from functools import lru_cache
from google import genai
from google.genai import types
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.utils.logger import tenacity_before_sleep_log
from app.utils.gemini_key_manager import get_key_manager


def _get_client():
    """Get the Gemini API client via key manager."""
    return get_key_manager().get_client(api_version="v1beta")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using Gemini.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    if not texts:
        return []

    # Client is initialized lazily


    # Gemini batch limit for embeddings is typically 100 texts
    batch_size = 50  # Reduced batch size for better reliability
    all_embeddings = []

    logger.info(f"Generating embeddings for {len(texts)} chunks...")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Clean batch: replace empty strings with a space
        batch = [t if t.strip() else " " for t in batch]
        logger.debug(f"Processing batch {i // batch_size + 1}: {len(batch)} texts")

        batch_embeddings = _embed_batch(batch)
        
        # Validation: ensure batch call returned correct number of embeddings
        if len(batch_embeddings) != len(batch):
            logger.warning(
                f"Batch embedding count mismatch: expected {len(batch)}, got {len(batch_embeddings)}. "
                "Falling back to individual embedding calls for this batch."
            )
            # Fallback to single calls for this specific batch
            batch_embeddings = []
            for text in batch:
                batch_embeddings.append(_embed_single(text, task_type="RETRIEVAL_DOCUMENT"))

        all_embeddings.extend(batch_embeddings)

    # Final validation
    if len(all_embeddings) != len(texts):
        error_msg = f"FATAL: Final count mismatch! Chunks ({len(texts)}) != Embeddings ({len(all_embeddings)})"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Successfully generated {len(all_embeddings)} embeddings (dim={len(all_embeddings[0])})")
    return all_embeddings


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    before_sleep=tenacity_before_sleep_log,
    reraise=True,
)
def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed a single batch using Gemini API with fallback and key rotation."""
    km = get_key_manager()
    c = km.get_client(api_version="v1beta")
    try:
        result = c.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        
        if not result.embeddings:
            return []
            
        return [e.values for e in result.embeddings]
        
    except Exception as e:
        # ── Key rotation on 429 / quota errors ──────────────
        if km.is_quota_error(e):
            if km.rotate_key():
                logger.warning("Embedding quota/rate-limit hit — rotated to next API key, retrying...")
                new_client = km.get_client(api_version="v1beta")
                result = new_client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                return [e.values for e in result.embeddings] if result.embeddings else []

        error_str = str(e).lower()
        if "404" in error_str and "not found" in error_str:
            logger.warning(f"Primary embedding model {settings.EMBEDDING_MODEL} failed. Trying fallback.")
            try:
                result = c.models.embed_content(
                    model=settings.FALLBACK_EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                return [e.values for e in result.embeddings] if result.embeddings else []
            except Exception as fe:
                logger.error(f"Fallback embedding failed: {fe}")
        
        logger.error(f"Gemini batch embedding error: {str(e)}")
        return []


@lru_cache(maxsize=128)
def get_query_embedding(query: str) -> list[float]:
    """
    Generate an embedding for a single query string with caching.
    """
    # Client is initialized lazily

    query = query.strip() if query.strip() else " "

    try:
        return _embed_single(query)
    except Exception as e:
        logger.error(f"Query embedding error: {str(e)}")
        raise RuntimeError(f"Failed to embed query: {str(e)}") from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    before_sleep=tenacity_before_sleep_log,
    reraise=True,
)
def _embed_single(query: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    """Embed a single string using Gemini API with fallback and key rotation."""
    km = get_key_manager()
    c = km.get_client(api_version="v1beta")
    try:
        result = c.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        if not result.embeddings:
            raise ValueError("API returned no embeddings")
            
        embedding = result.embeddings[0].values
        if embedding and isinstance(embedding[0], list):
            embedding = embedding[0]
        return embedding
        
    except Exception as e:
        # ── Key rotation on 429 / quota errors ──────────────
        if km.is_quota_error(e):
            if km.rotate_key():
                logger.warning("Embedding single quota/rate-limit hit — rotated to next API key, retrying...")
                new_client = km.get_client(api_version="v1beta")
                result = new_client.models.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    contents=query,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                if result.embeddings:
                    embedding = result.embeddings[0].values
                    if embedding and isinstance(embedding[0], list):
                        embedding = embedding[0]
                    return embedding

        error_str = str(e).lower()
        if "404" in error_str and "not found" in error_str:
            logger.warning(f"Primary model {settings.EMBEDDING_MODEL} failed. Trying fallback.")
            try:
                result = c.models.embed_content(
                    model=settings.FALLBACK_EMBEDDING_MODEL,
                    contents=query,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                if result.embeddings:
                    embedding = result.embeddings[0].values
                    if embedding and isinstance(embedding[0], list):
                        embedding = embedding[0]
                    return embedding
            except Exception as fe:
                logger.error(f"Fallback single embedding failed: {fe}")

        logger.error(f"Gemini single embedding error: {str(e)}")
        raise RuntimeError(f"Gemini Embedding API error: {str(e)}") from e
