"""
Configuration module for the Restaurant Menu Extractor.

This module loads environment variables from a .env file and provides
easy access to configuration settings throughout the application.
"""

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


# Create a global config instance
config = Config()
