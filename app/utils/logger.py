"""
Structured logging configuration using loguru.
Provides consistent, readable logs across all modules.
"""

import sys
from loguru import logger


def setup_logger():
    """Configure application-wide logging."""
    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )

    # File handler for persistent logs
    logger.add(
        "logs/mini_rag_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

    return logger


# Initialize logger on import
setup_logger()
