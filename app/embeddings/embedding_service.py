"""
Embedding service using OpenAI's multilingual embedding model.
Generates vector representations for document chunks and user queries.
Includes retry logic for API resilience.
"""

from typing import Optional

from openai import OpenAI
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.config import settings


# Module-level client (initialized lazily)
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info(f"OpenAI client initialized (embedding model: {settings.EMBEDDING_MODEL})")
    return _client


def _make_retry_decorator():
    """Create a retry decorator using current settings."""
    return retry(
        stop=stop_after_attempt(settings.MAX_RETRIES),
        wait=wait_exponential(
            min=settings.RETRY_MIN_WAIT,
            max=settings.RETRY_MAX_WAIT,
        ),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.
    
    Uses OpenAI's text-embedding-3-small which supports
    multilingual content (Arabic + English).
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    if not texts:
        return []

    client = _get_client()

    # Process in batches of 100 (OpenAI limit is 2048 but smaller is safer)
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Clean batch: replace empty strings with a space to avoid API errors
        batch = [t if t.strip() else " " for t in batch]
        logger.debug(f"Embedding batch {i // batch_size + 1}: {len(batch)} texts")

        batch_embeddings = _embed_batch(client, batch)
        all_embeddings.extend(batch_embeddings)

    logger.info(f"Generated {len(all_embeddings)} embeddings (dim={len(all_embeddings[0])})")
    return all_embeddings


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    before_sleep=before_sleep_log(logger, "WARNING"),
    reraise=True,
)
def _embed_batch(client: OpenAI, batch: list[str]) -> list[list[float]]:
    """Embed a single batch with retry logic."""
    try:
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=batch,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.warning(f"Embedding API call failed (will retry): {e}")
        raise


def get_query_embedding(query: str) -> list[float]:
    """
    Generate an embedding for a single query string.
    
    Args:
        query: User's question or search query
    
    Returns:
        Embedding vector as a list of floats
    """
    client = _get_client()
    query = query.strip() if query.strip() else " "

    try:
        return _embed_single(client, query)
    except Exception as e:
        logger.error(f"Query embedding error after retries: {e}")
        raise RuntimeError(f"Failed to embed query: {e}") from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
    before_sleep=before_sleep_log(logger, "WARNING"),
    reraise=True,
)
def _embed_single(client: OpenAI, query: str) -> list[float]:
    """Embed a single query with retry logic."""
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=query,
    )
    embedding = response.data[0].embedding
    logger.debug(f"Query embedded: '{query[:50]}...' → dim={len(embedding)}")
    return embedding
