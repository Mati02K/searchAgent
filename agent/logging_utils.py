from __future__ import annotations

import logging
import os
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Return a file-backed logger with stable formatting.

    Env vars:
    - SEARCH_AGENT_LOG_FILE (default: logs/search_agent.log)
    - SEARCH_AGENT_LOG_LEVEL (default: INFO)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = os.getenv("SEARCH_AGENT_LOG_LEVEL", "INFO").upper()
    log_file = os.getenv("SEARCH_AGENT_LOG_FILE", "logs/search_agent.log")
    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(getattr(logging, log_level, logging.INFO))
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
