import logging
import sys
from pathlib import Path
from core.config import settings

def setup_logger(name: str) -> logging.Logger:
    """Sets up a structured logger that writes to both terminal and a physical log file in storage/logs/"""
    logger = logging.getLogger(name)
    
    # If logger already has handlers, return it to avoid duplicate logs
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # File Handler
    log_file = settings.LOG_DIR / "app.log"
    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
