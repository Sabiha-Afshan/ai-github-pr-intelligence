"""Logging configuration for the application."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.settings import get_settings
from src.utils.paths import LOG_DIRECTORY

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_file_name: str = "application.log",
) -> None:
    """
    Configure console and rotating-file logging.

    Calling this function more than once will not add duplicate
    handlers.
    """

    settings = get_settings()

    LOG_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    root_logger = logging.getLogger()

    if getattr(
        root_logger,
        "_github_pr_logging_configured",
        False,
    ):
        return

    log_level = getattr(
        logging,
        settings.log_level.upper(),
        logging.INFO,
    )

    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    log_file_path = Path(LOG_DIRECTORY / log_file_name)

    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8",
    )

    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    root_logger._github_pr_logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured named logger."""

    configure_logging()

    return logging.getLogger(name)
