"""
FAISS vector store with per-teacher index isolation.
Manages vector storage, similarity search, and metadata filtering.
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from loguru import logger

from app.config import settings


# In-memory cache of loaded indexes
_indexes: dict[str, faiss.Index] = {}
_metadata: dict[str, list[dict]] = {}


def _get_teacher_dir(teacher_id: str) -> Path:
    """Get the storage directory for a teacher's vector data."""
    teacher_dir = settings.VECTOR_DATA_DIR / teacher_id
    teacher_dir.mkdir(parents=True, exist_ok=True)
    return teacher_dir


def _load_index(teacher_id: str) -> tuple[Optional[faiss.Index], list[dict]]:
    """Load a teacher's FAISS index and metadata from disk."""
    teacher_dir = _get_teacher_dir(teacher_id)
    index_path = teacher_dir / "index.faiss"
    meta_path = teacher_dir / "metadata.json"

    index = None
    meta = []

    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            logger.debug(f"Loaded FAISS index for teacher {teacher_id}: {index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to load FAISS index for {teacher_id}: {e}")

    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata for {teacher_id}: {e}")

    return index, meta


def _save_index(teacher_id: str, index: faiss.Index, meta: list[dict]):
    """Persist a teacher's FAISS index and metadata to disk."""
    teacher_dir = _get_teacher_dir(teacher_id)

    faiss.write_index(index, str(teacher_dir / "index.faiss"))

    with open(teacher_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.debug(f"Saved FAISS index for teacher {teacher_id}: {index.ntotal} vectors")


def _ensure_loaded(teacher_id: str):
    """Ensure a teacher's index is loaded into memory."""
    if teacher_id not in _indexes:
        index, meta = _load_index(teacher_id)
        _indexes[teacher_id] = index
        _metadata[teacher_id] = meta


def add_chunks(
    teacher_id: str,
    chunks: list[dict],
    embeddings: list[list[float]],
):
    """
    Add document chunks and their embeddings to the teacher's vector store.
    
    Args:
        teacher_id: Teacher's unique identifier
        chunks: List of chunk metadata dicts
        embeddings: Corresponding embedding vectors
    """
    if not chunks or not embeddings:
        logger.warning("No chunks or embeddings to add")
        return

    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) count mismatch")

    _ensure_loaded(teacher_id)

    # Convert embeddings to numpy array
    vectors = np.array(embeddings, dtype=np.float32)
    dimension = vectors.shape[1]

    # Create or get existing index
    if _indexes[teacher_id] is None:
        # Create new index with L2 distance (works well for cosine with normalized vectors)
        _indexes[teacher_id] = faiss.IndexFlatIP(dimension)
        _metadata[teacher_id] = []
        logger.info(f"Created new FAISS index for teacher {teacher_id} (dim={dimension})")

    # Normalize vectors for inner product (equivalent to cosine similarity)
    faiss.normalize_L2(vectors)

    # Add to index
    _indexes[teacher_id].add(vectors)

    # Store metadata (without the text embedding, just the chunk info)
    for chunk in chunks:
        _metadata[teacher_id].append({
            "teacher_id": chunk["teacher_id"],
            "doc_id": chunk["doc_id"],
            "page": chunk["page"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "language": chunk["language"],
        })

    # Persist to disk
    _save_index(teacher_id, _indexes[teacher_id], _metadata[teacher_id])

    logger.info(
        f"Added {len(chunks)} chunks to teacher {teacher_id}'s index "
        f"(total: {_indexes[teacher_id].ntotal})"
    )


def search(
    teacher_id: str,
    query_embedding: list[float],
    top_k: int = 5,
    from_page: Optional[int] = None,
    to_page: Optional[int] = None,
    doc_id: Optional[int] = None,
) -> list[dict]:
    """
    Search for similar chunks in a teacher's vector store.
    
    Args:
        teacher_id: Teacher's unique identifier
        query_embedding: Query vector
        top_k: Number of results to return
        from_page: Optional minimum page number filter
        to_page: Optional maximum page number filter
        doc_id: Optional document ID filter
    
    Returns:
        List of chunk dicts with similarity scores, ordered by relevance
    """
    _ensure_loaded(teacher_id)

    index = _indexes.get(teacher_id)
    meta = _metadata.get(teacher_id, [])

    if index is None or index.ntotal == 0:
        logger.warning(f"No vectors found for teacher {teacher_id}")
        return []

    # Prepare query vector
    query_vec = np.array([query_embedding], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    # Search with extra results to account for metadata filtering
    has_filters = from_page is not None or to_page is not None or doc_id is not None
    search_k = min(top_k * 4, index.ntotal) if has_filters else min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, search_k)

    # Build results with metadata and filtering
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue

        chunk_meta = meta[idx]

        # Apply doc_id filtering
        if doc_id is not None and chunk_meta.get("doc_id") != doc_id:
            continue

        # Apply page filtering
        if from_page is not None and chunk_meta["page"] < from_page:
            continue
        if to_page is not None and chunk_meta["page"] > to_page:
            continue

        results.append({
            **chunk_meta,
            "score": float(score),
        })

        if len(results) >= top_k:
            break

    logger.debug(
        f"Search for teacher {teacher_id}: {len(results)} results "
        f"(doc_id={doc_id}, pages {from_page}-{to_page}, top_k={top_k})"
    )
    return results


def get_index_stats(teacher_id: str) -> dict:
    """Get statistics about a teacher's vector index."""
    _ensure_loaded(teacher_id)

    index = _indexes.get(teacher_id)
    meta = _metadata.get(teacher_id, [])

    if index is None:
        return {"total_vectors": 0, "total_documents": 0, "total_pages": 0}

    doc_ids = set(m.get("doc_id") for m in meta)
    pages = set(m.get("page") for m in meta)

    return {
        "total_vectors": index.ntotal,
        "total_documents": len(doc_ids),
        "total_pages": len(pages),
    }


def delete_document_vectors(teacher_id: str, doc_id: int):
    """
    Remove all vectors for a specific document from the teacher's index.
    
    Note: FAISS doesn't support efficient deletion, so we rebuild the index
    without the deleted document's vectors.
    """
    _ensure_loaded(teacher_id)

    meta = _metadata.get(teacher_id, [])
    if not meta:
        return

    # Find indices to keep
    keep_indices = [i for i, m in enumerate(meta) if m.get("doc_id") != doc_id]

    if len(keep_indices) == len(meta):
        logger.debug(f"No vectors found for doc_id={doc_id} in teacher {teacher_id}")
        return

    if not keep_indices:
        # All vectors belong to this document — clear everything
        _indexes[teacher_id] = None
        _metadata[teacher_id] = []
        _save_index_empty(teacher_id)
        logger.info(f"Cleared all vectors for teacher {teacher_id} (doc_id={doc_id})")
        return

    # Rebuild index without deleted vectors
    index = _indexes[teacher_id]
    all_vectors = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d)
    all_vectors = all_vectors.reshape(index.ntotal, index.d)

    kept_vectors = np.array([all_vectors[i] for i in keep_indices], dtype=np.float32)
    kept_meta = [meta[i] for i in keep_indices]

    new_index = faiss.IndexFlatIP(index.d)
    new_index.add(kept_vectors)

    _indexes[teacher_id] = new_index
    _metadata[teacher_id] = kept_meta
    _save_index(teacher_id, new_index, kept_meta)

    logger.info(
        f"Deleted doc_id={doc_id} vectors from teacher {teacher_id} "
        f"({len(meta) - len(keep_indices)} removed, {len(keep_indices)} remaining)"
    )


def _save_index_empty(teacher_id: str):
    """Remove index files when clearing all vectors."""
    teacher_dir = _get_teacher_dir(teacher_id)
    index_path = teacher_dir / "index.faiss"
    meta_path = teacher_dir / "metadata.json"

    if index_path.exists():
        index_path.unlink()
    if meta_path.exists():
        meta_path.unlink()
