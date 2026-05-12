"""
API routes for the Mini RAG e-learning platform.
All endpoints for document upload, AI educational actions,
document management, and system monitoring.
"""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from loguru import logger

from app.ingestion.pdf_processor import (
    validate_file,
    extract_text_from_pdf,
    save_upload,
    PDFProcessingError,
)
from app.processing.chunker import chunk_document
from app.embeddings.embedding_service import get_embeddings
from app.vector_store.faiss_store import add_chunks, get_index_stats, delete_document_vectors
from app.retrieval.retriever import retrieve, build_context, get_retrieval_confidence
from app.router.prompt_router import route, get_supported_actions, UnsupportedActionError
from app.llm.llm_service import generate
from app.utils.database import (
    insert_document,
    update_document_chunks,
    get_documents_by_teacher,
    get_document_by_id,
    delete_document,
    get_all_stats,
)


router = APIRouter(prefix="/api", tags=["Mini RAG"])


# ── Request/Response Models ─────────────────────────────────

class AIRequest(BaseModel):
    """Common request model for all AI actions."""
    teacher_id: str
    message: str
    doc_id: Optional[int] = None  # Filter by specific document
    from_page: Optional[int] = None
    to_page: Optional[int] = None


class AIResponse(BaseModel):
    """Common response model for all AI actions."""
    action: str
    response: str
    sources: list[dict]
    teacher_id: str
    confidence: dict  # Retrieval confidence metrics


class UploadResponse(BaseModel):
    """Response model for document upload."""
    message: str
    document_id: int
    filename: str
    total_pages: int
    total_chunks: int
    teacher_id: str


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    teacher_id: str
    documents: list[dict]
    index_stats: dict


# ── Document Upload ─────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    teacher_id: str = Form(...),
):
    """
    Upload a PDF document for processing.
    
    Pipeline:
    1. Validate file (type, size)
    2. Save to disk
    3. Extract text from PDF
    4. Chunk text with metadata
    5. Generate embeddings
    6. Store in FAISS vector index
    7. Save document metadata to SQLite
    """
    logger.info(f"Upload request: teacher={teacher_id}, file={file.filename}")

    # Step 1: Validate
    # Read content to check size
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset for save_upload

    error = validate_file(file.filename, file_size)
    if error:
        raise HTTPException(status_code=400, detail=error)

    try:
        # Step 2: Save file
        saved_path, stored_filename, file_size = await save_upload(file, teacher_id)

        # Step 3: Extract text
        pages = extract_text_from_pdf(saved_path)
        total_pages = len(pages)

        # Step 4: Save metadata to DB (to get doc_id)
        doc_id = insert_document(
            teacher_id=teacher_id,
            filename=stored_filename,
            original_filename=file.filename,
            total_pages=total_pages,
            total_chunks=0,  # Updated after chunking
            file_size_bytes=file_size,
        )

        # Step 5: Chunk document
        chunks = chunk_document(pages, teacher_id, doc_id)
        total_chunks = len(chunks)

        # Step 5b: Update chunk count in database (fix for the original bug)
        update_document_chunks(doc_id, total_chunks)

        # Step 6: Generate embeddings
        chunk_texts = [c["text"] for c in chunks]
        embeddings = get_embeddings(chunk_texts)

        # Step 7: Store in FAISS
        add_chunks(teacher_id, chunks, embeddings)

        logger.info(
            f"Upload complete: doc_id={doc_id}, pages={total_pages}, chunks={total_chunks}"
        )

        return UploadResponse(
            message="Document uploaded and processed successfully!",
            document_id=doc_id,
            filename=file.filename,
            total_pages=total_pages,
            total_chunks=total_chunks,
            teacher_id=teacher_id,
        )

    except PDFProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


# ── AI Action Handler ───────────────────────────────────────

async def _handle_ai_action(action: str, request: AIRequest) -> AIResponse:
    """
    Common handler for all AI actions.
    
    Pipeline:
    1. Retrieve relevant chunks from vector store (with hallucination guard)
    2. Build context from chunks
    3. Route to appropriate prompt template
    4. Generate AI response
    5. Attach confidence metrics
    """
    try:
        # Step 1: Retrieve relevant content (with threshold filtering)
        chunks = retrieve(
            teacher_id=request.teacher_id,
            query=request.message,
            from_page=request.from_page,
            to_page=request.to_page,
            doc_id=request.doc_id,
        )

        # Step 2: Calculate retrieval confidence
        confidence = get_retrieval_confidence(chunks)

        # Step 3: Build context
        context = build_context(chunks)

        # Step 4: Route to prompt
        system_prompt, user_prompt = route(action, context, request.message)

        # Step 5: Generate response
        ai_response = generate(system_prompt, user_prompt)

        # Build source references
        sources = [
            {
                "page": c.get("page"),
                "doc_id": c.get("doc_id"),
                "score": round(c.get("score", 0), 3),
                "confidence": c.get("confidence", "unknown"),
                "preview": c.get("text", "")[:150] + "...",
            }
            for c in chunks
        ]

        return AIResponse(
            action=action,
            response=ai_response,
            sources=sources,
            teacher_id=request.teacher_id,
            confidence=confidence,
        )

    except UnsupportedActionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"AI action '{action}' failed: {e}")
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")


# ── AI Endpoints ────────────────────────────────────────────

@router.post("/chat", response_model=AIResponse)
async def chat(request: AIRequest):
    """Ask a question about the uploaded course materials."""
    return await _handle_ai_action("chat", request)


@router.post("/summarize", response_model=AIResponse)
async def summarize(request: AIRequest):
    """Summarize selected pages or topics from the course materials."""
    return await _handle_ai_action("summarize", request)


@router.post("/quiz", response_model=AIResponse)
async def quiz(request: AIRequest):
    """Generate a quiz based on the course materials."""
    return await _handle_ai_action("quiz", request)


@router.post("/explain", response_model=AIResponse)
async def explain(request: AIRequest):
    """Get a simple explanation of a topic from the course materials."""
    return await _handle_ai_action("explain", request)


@router.post("/flashcards", response_model=AIResponse)
async def flashcards(request: AIRequest):
    """Generate study flashcards from the course materials."""
    return await _handle_ai_action("flashcards", request)


@router.post("/translate", response_model=AIResponse)
async def translate(request: AIRequest):
    """Translate content between Arabic and English."""
    return await _handle_ai_action("translate", request)


# ── Document Management ─────────────────────────────────────

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(teacher_id: str):
    """List all uploaded documents for a teacher."""
    documents = get_documents_by_teacher(teacher_id)
    stats = get_index_stats(teacher_id)

    return DocumentListResponse(
        teacher_id=teacher_id,
        documents=documents,
        index_stats=stats,
    )


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: int, teacher_id: str):
    """Delete a document and its vectors."""
    deleted = delete_document(doc_id, teacher_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove vectors
    delete_document_vectors(teacher_id, doc_id)

    return {"message": "Document deleted successfully", "document_id": doc_id}


# ── Re-indexing ─────────────────────────────────────────────

@router.post("/reindex/{doc_id}")
async def reindex_document(doc_id: int, teacher_id: str):
    """
    Re-process and re-index a previously uploaded document.
    
    Useful when chunking/embedding settings have changed.
    """
    doc = get_document_by_id(doc_id)
    if not doc or doc["teacher_id"] != teacher_id:
        raise HTTPException(status_code=404, detail="Document not found")

    from app.config import settings
    from pathlib import Path

    # Find the file on disk
    file_path = settings.UPLOAD_DIR / teacher_id / doc["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Source PDF file not found on disk")

    try:
        # Remove old vectors
        delete_document_vectors(teacher_id, doc_id)

        # Re-extract text
        pages = extract_text_from_pdf(str(file_path))

        # Re-chunk
        chunks = chunk_document(pages, teacher_id, doc_id)
        total_chunks = len(chunks)

        # Update DB
        update_document_chunks(doc_id, total_chunks)

        # Re-embed
        chunk_texts = [c["text"] for c in chunks]
        embeddings = get_embeddings(chunk_texts)

        # Re-store
        add_chunks(teacher_id, chunks, embeddings)

        logger.info(f"Re-indexed doc_id={doc_id}: {total_chunks} chunks")

        return {
            "message": "Document re-indexed successfully",
            "document_id": doc_id,
            "total_chunks": total_chunks,
        }

    except Exception as e:
        logger.error(f"Re-indexing failed for doc_id={doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Re-indexing failed: {str(e)}")


# ── Info & Stats Endpoints ──────────────────────────────────

@router.get("/actions")
async def list_actions():
    """List all supported AI actions."""
    return {
        "actions": get_supported_actions(),
        "description": {
            "chat": "Ask questions about your course materials",
            "summarize": "Get summaries of selected content",
            "quiz": "Generate quizzes for self-assessment",
            "explain": "Get simple explanations of complex topics",
            "flashcards": "Create study flashcards",
            "translate": "Translate between Arabic and English",
        },
    }


@router.get("/stats")
async def platform_stats():
    """
    Get platform-wide statistics.
    
    Returns teacher count, document count, total pages/chunks processed.
    """
    db_stats = get_all_stats()
    return {
        "platform": "Mini RAG E-Learning",
        "stats": db_stats,
    }
