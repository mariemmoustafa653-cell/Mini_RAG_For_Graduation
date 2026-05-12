"""
Text chunking module.
Splits document pages into semantic chunks with metadata preservation.
"""

from loguru import logger

from app.config import settings
from app.utils.arabic_utils import detect_language


def chunk_document(
    pages: list[dict],
    teacher_id: str,
    doc_id: int,
) -> list[dict]:
    """
    Split extracted pages into overlapping chunks with metadata.
    
    Uses recursive character-based splitting with configurable
    chunk size and overlap to maintain semantic coherence.
    
    Args:
        pages: List of {"page": int, "text": str, "language": str}
        teacher_id: Teacher's unique identifier
        doc_id: Document database ID
    
    Returns:
        List of chunk dicts with full metadata
    """
    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP
    chunks = []
    chunk_index = 0

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        language = page_data.get("language", detect_language(text))

        # Split page text into chunks
        page_chunks = _split_text(text, chunk_size, chunk_overlap)

        for chunk_text in page_chunks:
            if not chunk_text.strip():
                continue

            chunks.append({
                "teacher_id": teacher_id,
                "doc_id": doc_id,
                "page": page_num,
                "chunk_index": chunk_index,
                "text": chunk_text.strip(),
                "language": language,
            })
            chunk_index += 1

    logger.info(
        f"Chunked document {doc_id}: {len(chunks)} chunks "
        f"from {len(pages)} pages (size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks using separator hierarchy.
    
    Tries to split on paragraph breaks first, then sentences,
    then words, falling back to character-level splitting.
    """
    if len(text) <= chunk_size:
        return [text]

    # Separator hierarchy: prefer natural boundaries
    separators = ["\n\n", "\n", ". ", "، ", " "]

    return _recursive_split(text, separators, chunk_size, overlap)


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if len(text) <= chunk_size:
        return [text]

    # Find the best separator that exists in the text
    best_sep = None
    for sep in separators:
        if sep in text:
            best_sep = sep
            break

    if best_sep is None:
        # Fallback: hard split at chunk_size
        return _hard_split(text, chunk_size, overlap)

    # Split on the chosen separator
    parts = text.split(best_sep)
    chunks = []
    current_chunk = ""

    for part in parts:
        # If adding this part exceeds chunk size
        candidate = current_chunk + best_sep + part if current_chunk else part

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)

            # If the part itself is too long, recursively split it
            if len(part) > chunk_size:
                remaining_seps = separators[separators.index(best_sep) + 1:]
                sub_chunks = _recursive_split(part, remaining_seps, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = part

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Apply overlap between chunks
    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap)

    return chunks


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text at fixed character boundaries when no separator works."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if overlap > 0 else end

        # Prevent infinite loop
        if start >= len(text) - overlap:
            break

    return chunks


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap from the end of each chunk to the start of the next."""
    if len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
        overlapped.append(overlap_text + " " + chunks[i])

    return overlapped
