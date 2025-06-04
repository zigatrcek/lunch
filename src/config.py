"""
Configuration module for the Restaurant Menu Extractor.

This module loads environment variables from a .env file and provides
easy access to configuration settings throughout the application.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class that loads settings from environment variables."""

    # Facebook API Configuration
    FACEBOOK_PAGE_ID: str = os.getenv("FACEBOOK_PAGE_ID", "")
    FACEBOOK_ACCESS_TOKEN: str = os.getenv("FACEBOOK_ACCESS_TOKEN", "")

    # Gemini AI Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Tesseract Configuration
    TESSERACT_CMD: Optional[str] = os.getenv("TESSERACT_CMD")

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./menu.db")

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/menu_extractor.log")

    @classmethod
    def validate_required_config(cls) -> list[str]:
        """
        Validate that all required configuration variables are set.

        Returns:
            List of missing configuration variables. Empty list if all are set.
        """
        missing = []

        if not cls.FACEBOOK_PAGE_ID:
            missing.append("FACEBOOK_PAGE_ID")
        if not cls.FACEBOOK_ACCESS_TOKEN:
            missing.append("FACEBOOK_ACCESS_TOKEN")
        if not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")

        return missing

    @classmethod
    def setup_logging_directory(cls) -> None:
        """Create the logging directory if it doesn't exist."""
        log_path = Path(cls.LOG_FILE)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def setup_logging(cls) -> None:
        """
        Set up logging configuration for the application.

        Creates log directory if it doesn't exist and configures both
        file and console logging with appropriate formatters.
        """
        # Create logging directory
        cls.setup_logging_directory()

        # Configure logging level
        log_level = getattr(logging, cls.LOG_LEVEL.upper(), logging.INFO)

        # Create formatters
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_formatter = logging.Formatter(
            fmt="%(levelname)s - %(name)s - %(message)s"
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        # Clear any existing handlers
        root_logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(cls.LOG_FILE)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # Log that logging has been configured
        logging.info("Logging configuration completed")


# Create a global config instance
config = Config()

# Initialize logging when the module is imported
config.setup_logging()
