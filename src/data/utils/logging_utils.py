import os
import logging
from typing import Dict, List, Optional, Union, Tuple

def configure_logger(log_path: str) -> logging.Logger:
    """
    Configures and returns a module-specific logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        # Console handler for level INFO and above
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # File handler for level DEBUG and above
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger