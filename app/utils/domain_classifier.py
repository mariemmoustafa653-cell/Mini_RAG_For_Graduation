"""
Domain classifier for the low-confidence fallback branch.

Determines whether a user question belongs to the same academic domain
as the uploaded course material, so the system can either:
  - answer from general domain knowledge (in-domain), or
  - reject the question (out-of-domain), or
  - fall back to the existing behaviour (uncertain / error).

This module is intentionally self-contained.  It reuses the existing
``generate()`` LLM call and ``get_documents_by_teacher()`` DB lookup
but does NOT touch retrieval, FAISS, embeddings, or prompt templates.
"""

from loguru import logger


# ── General-Knowledge Fallback Prompts ─────────────────────
# These are deliberately separate from the RAG prompts in
# app/prompts/templates.py, which explicitly forbid outside knowledge.

GENERAL_KNOWLEDGE_SYSTEM_PROMPT = (
    "You are an educational AI assistant. "
    "The student's uploaded course materials did not contain a direct answer, "
    "but the question falls within the same academic domain. "
    "You may answer using your general knowledge of this domain. "
    "Keep your response clear, accurate, and educational. "
    "Respond in the same language as the student's question."
)

GENERAL_KNOWLEDGE_USER_PROMPT = (
    "STUDENT'S QUESTION:\n{question}\n\n"
    "INSTRUCTIONS:\n"
    "- Answer this question using your general knowledge of the subject.\n"
    "- Be accurate and educational.\n"
    "- If you are not confident in the answer, say so.\n"
    "- Do NOT fabricate citations or page numbers."
)


# ── Domain Classification ──────────────────────────────────

_CLASSIFICATION_SYSTEM_PROMPT = (
    "You are a strict academic-domain classifier. "
    "You will be given information about uploaded course documents and a student question. "
    "Determine whether the question belongs to the SAME academic domain as the documents. "
    "Reply with EXACTLY one word: YES, NO, or UNCERTAIN. "
    "Be conservative: if you are not sure, reply UNCERTAIN."
)

_CLASSIFICATION_USER_TEMPLATE = (
    "UPLOADED DOCUMENTS:\n{doc_info}\n\n"
    "SAMPLE CONTENT FROM DOCUMENTS:\n{chunk_sample}\n\n"
    "STUDENT QUESTION:\n{question}\n\n"
    "Is this question within the same academic domain as the uploaded documents? "
    "Reply with EXACTLY one word: YES, NO, or UNCERTAIN."
)


def classify_domain(
    teacher_id: str,
    question: str,
    chunks: list[dict],
) -> str:
    """
    Classify whether *question* is in the same domain as the teacher's
    uploaded materials.

    Args:
        teacher_id: The teacher whose documents define the domain.
        question:   The student's question.
        chunks:     Retrieved chunks (may be empty or low-scoring).

    Returns:
        ``"in_domain"``, ``"out_of_domain"``, or ``"uncertain"``.
        On ANY error the function returns ``"uncertain"`` so that the
        caller can safely fall back to existing behaviour.
    """
    try:
        # ---- gather domain signals --------------------------------
        from app.utils.database import get_documents_by_teacher

        docs = get_documents_by_teacher(teacher_id)

        if not docs:
            # No documents at all → nothing to classify against
            logger.debug("Domain classifier: no documents found, returning uncertain.")
            return "uncertain"

        # Build a concise description of the uploaded material
        doc_lines = []
        for d in docs:
            name = d.get("original_filename", d.get("filename", "unknown"))
            pages = d.get("total_pages", "?")
            doc_lines.append(f"- {name} ({pages} pages)")
        doc_info = "\n".join(doc_lines)

        # Grab a small sample of chunk text for additional signal
        chunk_sample = ""
        if chunks:
            sample_texts = [
                c.get("text", "")[:200] for c in chunks[:3] if c.get("text")
            ]
            chunk_sample = "\n---\n".join(sample_texts) if sample_texts else "(no text available)"
        else:
            chunk_sample = "(no retrieved content)"

        # ---- ask the LLM -----------------------------------------
        from app.llm.llm_service import generate

        user_prompt = _CLASSIFICATION_USER_TEMPLATE.format(
            doc_info=doc_info,
            chunk_sample=chunk_sample,
            question=question,
        )

        raw = generate(
            system_prompt=_CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
            max_tokens=50,
        )

        # ---- parse the response -----------------------------------
        answer = raw.strip().upper().rstrip(".!,")

        if answer == "YES":
            return "in_domain"
        elif answer == "NO":
            return "out_of_domain"
        else:
            logger.info(f"Domain classifier returned ambiguous answer: '{raw.strip()}'")
            return "uncertain"

    except Exception as exc:
        logger.warning(f"Domain classification failed (preserving fallback): {exc}")
        return "uncertain"
