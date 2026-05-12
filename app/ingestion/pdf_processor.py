"""
PDF document processing module.
Handles file validation, text extraction, and error handling.
"""

import os
from pathlib import Path
from typing import Optional

import pdfplumber
from loguru import logger

from app.config import settings
from app.utils.arabic_utils import clean_text, normalize_arabic, detect_language


# ── Validation Error Messages ───────────────────────────────
UNSUPPORTED_FILE_TYPE = "Unsupported file type. Please upload a PDF document only."
FILE_TOO_LARGE = "File size exceeds the allowed limit (20MB)."
INVALID_PDF = "Unable to process this PDF. Please upload a valid document."


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


def validate_file(filename: str, file_size: int) -> Optional[str]:
    """
    Validate an uploaded file before processing.
    
    Args:
        filename: Original filename from upload
        file_size: File size in bytes
    
    Returns:
        Error message string if validation fails, None if valid
    """
    # Check file extension
    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        logger.warning(f"Rejected file with extension: {ext}")
        return UNSUPPORTED_FILE_TYPE

    # Check file size
    if file_size > settings.max_file_size_bytes:
        logger.warning(f"Rejected file: {file_size} bytes exceeds {settings.MAX_FILE_SIZE_MB}MB limit")
        return FILE_TOO_LARGE

    return None


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of dicts: [{"page": 1, "text": "...", "language": "en"}, ...]
    
    Raises:
        PDFProcessingError: If PDF cannot be read or is corrupted
    """
    pages = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing PDF: {pdf_path} ({total_pages} pages)")

            for i, page in enumerate(pdf.pages):
                try:
                    raw_text = page.extract_text() or ""
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i + 1}: {e}")
                    raw_text = ""

                # Clean and normalize text
                cleaned = clean_text(raw_text)

                # Detect language and apply Arabic normalization if needed
                lang = detect_language(cleaned)
                if lang == "ar":
                    cleaned = normalize_arabic(cleaned)

                if cleaned:  # Only include pages with actual content
                    pages.append({
                        "page": i + 1,
                        "text": cleaned,
                        "language": lang,
                    })

            if not pages:
                logger.warning(f"No extractable text found in: {pdf_path}")
                raise PDFProcessingError(INVALID_PDF)

            logger.info(f"Extracted text from {len(pages)}/{total_pages} pages")
            return pages

    except PDFProcessingError:
        raise
    except Exception as e:
        logger.error(f"Failed to process PDF {pdf_path}: {e}")
        raise PDFProcessingError(INVALID_PDF) from e


async def save_upload(file, teacher_id: str) -> tuple[str, str, int]:
    """
    Save an uploaded file to disk.
    
    Args:
        file: FastAPI UploadFile object
        teacher_id: Teacher's unique identifier
    
    Returns:
        Tuple of (saved_path, stored_filename, file_size)
    """
    import uuid

    # Create teacher-specific upload directory
    teacher_dir = settings.UPLOAD_DIR / teacher_id
    teacher_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename to avoid collisions
    unique_id = uuid.uuid4().hex[:8]
    original_name = Path(file.filename).stem
    stored_filename = f"{original_name}_{unique_id}.pdf"
    save_path = teacher_dir / stored_filename

    # Read and save file content
    content = await file.read()
    file_size = len(content)

    with open(save_path, "wb") as f:
        f.write(content)

    logger.info(f"Saved upload: {save_path} ({file_size} bytes)")
    return str(save_path), stored_filename, file_size
