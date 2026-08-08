"""
Centralized logging setup.

Every module should call get_logger(__name__) instead of configuring
its own logging — this guarantees a consistent format across the
entire pipeline (ingestion, streaming, sentiment, analytics, api)
and writes to both console and a rotating log file.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _init_root_logger() -> None:
    global _initialized
    if _initialized:
        return

    log_dir: Path = settings.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rotates at 5MB, keeps 5 backups, so logs/ doesn't grow unbounded
    # during a 24-48 hour continuous run (Week 11 integration testing)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Usage in any module:
        from config.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("message")
    """
    _init_root_logger()
    return logging.getLogger(name)