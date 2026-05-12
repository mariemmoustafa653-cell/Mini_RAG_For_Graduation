"""
SQLite database for document metadata.
Tracks uploaded documents per teacher for management and listing.
"""

import sqlite3
from datetime import datetime
from typing import Optional
from loguru import logger
from app.config import settings


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory."""
    conn = sqlite3.connect(str(settings.DATABASE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Create the documents table if it doesn't exist."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teacher_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                total_pages INTEGER NOT NULL DEFAULT 0,
                total_chunks INTEGER NOT NULL DEFAULT 0,
                file_size_bytes INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_teacher_id 
            ON documents(teacher_id)
        """)
        conn.commit()
        logger.info("Database initialized successfully")
    finally:
        conn.close()


def insert_document(
    teacher_id: str,
    filename: str,
    original_filename: str,
    total_pages: int,
    total_chunks: int,
    file_size_bytes: int,
) -> int:
    """Insert a new document record and return its ID."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO documents 
                (teacher_id, filename, original_filename, total_pages, total_chunks, file_size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                teacher_id,
                filename,
                original_filename,
                total_pages,
                total_chunks,
                file_size_bytes,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        doc_id = cursor.lastrowid
        logger.info(f"Document saved: id={doc_id}, teacher={teacher_id}, file={original_filename}")
        return doc_id
    finally:
        conn.close()


def update_document_chunks(doc_id: int, total_chunks: int):
    """
    Update the chunk count for a document after processing.

    This fixes the issue where total_chunks was set to 0 at insert
    time (before chunking) and never updated.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE documents SET total_chunks = ? WHERE id = ?",
            (total_chunks, doc_id),
        )
        conn.commit()
        logger.debug(f"Updated doc_id={doc_id} total_chunks={total_chunks}")
    finally:
        conn.close()


def get_documents_by_teacher(teacher_id: str) -> list[dict]:
    """Retrieve all documents for a teacher."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM documents WHERE teacher_id = ? ORDER BY created_at DESC",
            (teacher_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_document_by_id(doc_id: int) -> Optional[dict]:
    """Retrieve a single document by ID."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_document(doc_id: int, teacher_id: str) -> bool:
    """Delete a document record. Returns True if deleted."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM documents WHERE id = ? AND teacher_id = ?",
            (doc_id, teacher_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Document deleted: id={doc_id}, teacher={teacher_id}")
        return deleted
    finally:
        conn.close()


def get_all_stats() -> dict:
    """
    Get global platform statistics.
    
    Returns:
        Dict with total_teachers, total_documents, total_pages, total_chunks
    """
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT 
                COUNT(DISTINCT teacher_id) as total_teachers,
                COUNT(*) as total_documents,
                COALESCE(SUM(total_pages), 0) as total_pages,
                COALESCE(SUM(total_chunks), 0) as total_chunks,
                COALESCE(SUM(file_size_bytes), 0) as total_size_bytes
            FROM documents
        """).fetchone()
        return dict(row) if row else {
            "total_teachers": 0,
            "total_documents": 0,
            "total_pages": 0,
            "total_chunks": 0,
            "total_size_bytes": 0,
        }
    finally:
        conn.close()
