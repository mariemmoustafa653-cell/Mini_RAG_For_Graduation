"""
Semantic retrieval layer with hallucination control.
Orchestrates query embedding, vector search, metadata filtering,
relevance threshold enforcement, and confidence scoring.
"""

from typing import Optional

from loguru import logger

from app.config import settings
from app.embeddings.embedding_service import get_query_embedding
from app.vector_store.faiss_store import search


# ── Hallucination Control Constants ─────────────────────────
NO_CONTEXT_MESSAGE = "No relevant content found in the uploaded materials."
LOW_CONFIDENCE_WARNING = (
    "⚠️ Note: The retrieved content has low relevance to your question. "
    "The answer may not be fully accurate based on the uploaded materials."
)


def retrieve(
    teacher_id: str,
    query: str,
    top_k: Optional[int] = None,
    from_page: Optional[int] = None,
    to_page: Optional[int] = None,
    doc_id: Optional[int] = None,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a query.
    
    Pipeline:
    1. Embed the user query
    2. Search the teacher's FAISS index
    3. Apply metadata filters (teacher_id, doc_id, page range)
    4. Enforce relevance threshold (hallucination guard)
    5. Return top-k ranked results with confidence metadata
    
    Args:
        teacher_id: Teacher's unique identifier for data isolation
        query: User's question or search query
        top_k: Number of results to return (defaults to settings.TOP_K)
        from_page: Optional minimum page number
        to_page: Optional maximum page number
        doc_id: Optional document ID filter
    
    Returns:
        List of relevant chunk dicts with scores, ordered by relevance.
        Each chunk includes a 'confidence' field: 'high', 'medium', or 'low'.
    """
    if top_k is None:
        top_k = settings.TOP_K

    logger.info(
        f"Retrieving for teacher={teacher_id}, query='{query[:80]}...', "
        f"top_k={top_k}, doc_id={doc_id}, pages={from_page}-{to_page}"
    )

    # Step 1: Embed the query
    query_embedding = get_query_embedding(query)

    # Step 2 & 3: Search with metadata filtering
    results = search(
        teacher_id=teacher_id,
        query_embedding=query_embedding,
        top_k=top_k,
        from_page=from_page,
        to_page=to_page,
        doc_id=doc_id,
    )

    # Step 4: Hallucination guard — filter by relevance threshold
    threshold = settings.SIMILARITY_THRESHOLD
    filtered_results = []

    for chunk in results:
        score = chunk.get("score", 0)

        # Assign confidence level
        if score >= threshold * 2:
            chunk["confidence"] = "high"
        elif score >= threshold:
            chunk["confidence"] = "medium"
        else:
            chunk["confidence"] = "low"

        # Only include chunks above the minimum threshold
        if score >= threshold:
            filtered_results.append(chunk)

    # Log filtering stats
    dropped = len(results) - len(filtered_results)
    if dropped > 0:
        logger.info(
            f"Hallucination guard: dropped {dropped}/{len(results)} chunks "
            f"below threshold {threshold}"
        )

    logger.info(
        f"Retrieved {len(filtered_results)} chunks for query "
        f"(threshold={threshold}, dropped={dropped})"
    )
    return filtered_results


def build_context(chunks: list[dict]) -> str:
    """
    Build a formatted context string from retrieved chunks.
    
    Each chunk is labeled with its page number and confidence level
    for transparency in the LLM prompt.
    
    Args:
        chunks: List of retrieved chunk dicts
    
    Returns:
        Formatted context string for the LLM prompt
    """
    if not chunks:
        return NO_CONTEXT_MESSAGE

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        score = chunk.get("score", 0)
        confidence = chunk.get("confidence", "unknown")
        context_parts.append(
            f"[Source {i} — Page {page} | relevance: {score:.2f} | confidence: {confidence}]\n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Add low-confidence warning if average score is below threshold
    avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
    if avg_score < settings.SIMILARITY_THRESHOLD * 1.5:
        context = f"{LOW_CONFIDENCE_WARNING}\n\n{context}"

    return context


def get_retrieval_confidence(chunks: list[dict]) -> dict:
    """
    Calculate overall retrieval confidence metrics.
    
    Useful for the API to inform the frontend about
    response reliability.
    
    Args:
        chunks: List of retrieved chunk dicts
    
    Returns:
        Dict with confidence metrics
    """
    if not chunks:
        return {
            "overall": "none",
            "avg_score": 0.0,
            "max_score": 0.0,
            "num_high": 0,
            "num_medium": 0,
            "num_low": 0,
        }

    scores = [c.get("score", 0) for c in chunks]
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)

    num_high = sum(1 for c in chunks if c.get("confidence") == "high")
    num_medium = sum(1 for c in chunks if c.get("confidence") == "medium")
    num_low = sum(1 for c in chunks if c.get("confidence") == "low")

    # Determine overall confidence
    if num_high >= len(chunks) // 2:
        overall = "high"
    elif num_high + num_medium >= len(chunks) // 2:
        overall = "medium"
    else:
        overall = "low"

    return {
        "overall": overall,
        "avg_score": round(avg_score, 3),
        "max_score": round(max_score, 3),
        "num_high": num_high,
        "num_medium": num_medium,
        "num_low": num_low,
    }
