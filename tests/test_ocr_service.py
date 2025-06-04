"""Tests for the OCR service module.

Simple tests for extract_text_from_image in ocr_service.py.
"""

import pytest
from PIL import Image

from src import ocr_service


class TestOCRService:
    def test_extract_text_from_valid_image(self):
        # Create a simple white image with no text
        img = Image.new("RGB", (100, 50), color="white")
        text = ocr_service.extract_text_from_image(img)
        assert isinstance(text, str)
        # For a blank image, text should be empty or whitespace
        assert text.strip() == ""

    def test_extract_text_from_invalid_input(self):
        # Not a PIL Image
        with pytest.raises(TypeError):
            ocr_service.extract_text_from_image("not_an_image")

    def test_extract_text_with_custom_config(self):
        img = Image.new("L", (50, 50), color=255)  # White grayscale image
        text = ocr_service.extract_text_from_image(img, lang="slv", psm=6, custom_config="--oem 1")
        assert isinstance(text, str)

    def test_extract_text_from_empty_image(self):
        # Completely black image
        img = Image.new("RGB", (60, 30), color="black")
        text = ocr_service.extract_text_from_image(img)
        assert isinstance(text, str)
        assert text.strip() == ""
