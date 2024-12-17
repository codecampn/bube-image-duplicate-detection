import logging

from ..config import config


def setup_logging() -> None:
    """Set up the logging for the application."""
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format="%(asctime)s - %(levelname)s - %(name)s  - %(message)s",
        filename="application.log",
    )
