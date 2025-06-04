"""
OCR Service module for extracting text from menu images.

This module provides functionality to perform Optical Character Recognition (OCR)
on restaurant menu images using Tesseract with Slovenian language support.
"""

import logging

import pytesseract
from PIL import Image

from src.config import config

logger = logging.getLogger(__name__)


def extract_text_from_image(
    image: Image.Image,
    lang: str = "slv",
    psm: int = 3,
    custom_config: str = ""
) -> str:
    """
    Extract text from an image using Tesseract OCR.
    
    Args:
        image: PIL Image object
        lang: Language code for Tesseract (default: 'slv' for Slovenian)
        psm: Page Segmentation Mode for Tesseract (default: 3 for automatic page segmentation)
        custom_config: Additional Tesseract configuration options
        
    Returns:
        str: Extracted text from the image
        
    Raises:
        TypeError: If image is not a PIL Image object
        pytesseract.TesseractError: If Tesseract OCR fails
        Exception: For other unexpected errors during processing
    """
    try:
        # Validate input type
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image object, got {type(image)}")
        
        logger.info("Processing PIL Image object for OCR")
        
        # Configure Tesseract command path if specified in config
        if config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
            logger.debug(f"Using Tesseract command: {config.TESSERACT_CMD}")
        
        # Build Tesseract configuration
        tesseract_config = f"--psm {psm}"
        if custom_config:
            tesseract_config += f" {custom_config}"
        
        logger.info(f"Performing OCR with language '{lang}' and config '{tesseract_config}'")
        
        # Perform OCR
        extracted_text = pytesseract.image_to_string(
            image,
            lang=lang,
            config=tesseract_config
        )
        
        # Log basic statistics about the extracted text
        text_length = len(extracted_text.strip())
        line_count = len([line for line in extracted_text.split('\n') if line.strip()])
        
        logger.info(f"OCR completed successfully. Extracted {text_length} characters in {line_count} non-empty lines")
        
        if text_length == 0:
            logger.warning("No text was extracted from the image")
        
        return extracted_text
        
    except TypeError as e:
        logger.error(f"Invalid input type: {e}")
        raise TypeError(f"Invalid input: {e}")

    except pytesseract.TesseractError as e:
        logger.error(f"Tesseract OCR failed: {e}")
        raise pytesseract.TesseractError(f"OCR processing failed: {e}")
    
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during OCR processing: {e}")
        raise Exception(f"OCR processing failed with unexpected error: {e}")

