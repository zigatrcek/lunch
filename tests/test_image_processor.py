"""Tests for the image processing module.

Tests the functionality defined in implementation-plan.md Slice 1 Task 1:
- Function to open an image using Pillow
- Function to convert to grayscale
- Function to apply binarization (Otsu's method)
- Function to save/display preprocessed image for debugging
"""

from pathlib import Path

import PIL.Image as PILImage
import pytest

from src import image_processing


class TestImageProcessor:
    """Test class for image preprocessing functionality."""

    @pytest.fixture
    def menu_image_path(self):
        """Path where test menu image should be uploaded."""
        test_image_dir = Path(__file__).parent / "test_files"
        test_image_dir.mkdir(exist_ok=True)
        return test_image_dir / "test_menu.jpg"

    def test_open_image(self, menu_image_path):
        """Test function to open an image using Pillow."""
        img = image_processing.open_image(menu_image_path)
        assert img is not None, "Failed to open image"
        assert isinstance(
            img, PILImage.Image
        ), "Image is not a PIL Image object"

        with pytest.raises(IOError):
            # Test with an invalid path
            image_processing.open_image("invalid_path.png")

    def test_complete_preprocessing_pipeline(self, menu_image_path):
        """
        Test complete image preprocessing workflow
        (black pixel extraction pipeline).
        """
        preprocessed_img = image_processing.preprocess_image(menu_image_path)
        assert preprocessed_img is not None, "Preprocessing failed"
        assert isinstance(
            preprocessed_img, PILImage.Image
        ), "Result is not a PIL Image object"
        assert preprocessed_img.mode == "L", (
            "Preprocessed image mode is not 'L' (grayscale)"
        )
        assert preprocessed_img.size == PILImage.open(menu_image_path).size, (
            "Preprocessed image dimensions do not match original"
        )

        with pytest.raises(IOError):
            # Test with an invalid path
            image_processing.preprocess_image("invalid_path.png")

    def test_extract_black_pixels(self):
        """Test extracting black or almost black pixels from an image."""
        img = PILImage.new("RGB", (10, 10), color=(0, 0, 0))
        # Draw a white square in the center
        for x in range(3, 7):
            for y in range(3, 7):
                img.putpixel((x, y), (255, 255, 255))
        # Should extract only the black border
        binary_img = image_processing.extract_black_pixels(img, threshold=40)
        assert binary_img is not None, "Failed to extract black pixels"
        assert isinstance(
            binary_img, PILImage.Image
        ), "Result is not a PIL Image object"
        assert binary_img.mode == "L", "Output image mode is not 'L'"
        # Center should be white (255), border should be black (0)
        for x in range(10):
            for y in range(10):
                px = binary_img.getpixel((x, y))
                if 3 <= x < 7 and 3 <= y < 7:
                    assert px == 255, (
                        f"Center pixel at ({x},{y}) should be white"
                    )
                else:
                    assert px == 0, (
                        f"Border pixel at ({x},{y}) should be black"
                    )
        # Test with invalid input
        with pytest.raises(TypeError):
            image_processing.extract_black_pixels("not_an_image")
