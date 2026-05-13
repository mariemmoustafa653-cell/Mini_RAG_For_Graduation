"""
Embedding service using Google Gemini's multilingual embedding model.
Generates vector representations for document chunks and user queries.
Includes retry logic for API resilience.
"""

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


# Global client instance
client = None


def _get_client():
    """Get or initialize the Gemini API client."""
    global client
    if client is None:
        if not settings.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not set in environment!")
        
        # Explicitly use v1 to avoid legacy v1beta issues
        client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options={"api_version": "v1"}
        )
        logger.info(f"Gemini API configured (embedding model: {settings.EMBEDDING_MODEL})")
    return client


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
    """Embed a single batch using Gemini API."""
    c = _get_client()
    try:
        # In the new google-genai SDK, passing a list to contents should work
        # but we must be careful with how the response is parsed.
        result = c.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        
        if not result.embeddings:
            return []
            
        # Extract values from each embedding object
        embeddings = [e.values for e in result.embeddings]
        
        logger.debug(f"API batch response: received {len(embeddings)} embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"Gemini batch embedding error: {str(e)}")
        # We don't raise here, we let get_embeddings handle the fallback
        return []


def get_query_embedding(query: str) -> list[float]:
    """
    Generate an embedding for a single query string.
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
    """Embed a single string using Gemini API."""
    c = _get_client()
    try:
        result = c.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        # result.embeddings is a list even for single content
        if not result.embeddings:
            raise ValueError("API returned no embeddings")
            
        embedding = result.embeddings[0].values
        
        # Gemini sometimes returns a list of lists if content was interpreted as multiple parts
        if embedding and isinstance(embedding[0], list):
            logger.warning("Single embedding returned nested list, flattening...")
            embedding = embedding[0]
            
        return embedding
    except Exception as e:
        logger.error(f"Gemini single embedding error: {str(e)}")
        raise RuntimeError(f"Gemini Embedding API error: {str(e)}") from e
