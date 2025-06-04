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
        test_image_dir = Path(__file__).parent / "test_images"
        test_image_dir.mkdir(exist_ok=True)
        return test_image_dir / "test_menu.jpg"

    def test_open_image(self, menu_image_path):
        """Test function to open an image using Pillow."""
        img = image_processing.open_image(menu_image_path)
        assert img is not None, "Failed to open image"
        assert isinstance(
            img, PILImage.Image), "Image is not a PIL Image object"

        with pytest.raises(IOError):
            # Test with an invalid path
            image_processing.open_image("invalid_path.png")

    def test_convert_to_grayscale(self):
        """Test function to convert image to grayscale."""
        img = PILImage.new("RGB", (100, 100),
                           color="blue")  # Create a dummy image
        grayscale_img = image_processing.convert_to_grayscale(img)
        assert grayscale_img is not None, "Failed to convert image to grayscale"
        assert isinstance(
            grayscale_img, PILImage.Image), "Result is not a PIL Image object"
        assert grayscale_img.mode == "L", "Grayscale image mode is not 'L'"
        assert grayscale_img.size == img.size, "Grayscale image dimensions do not match original"
        # Test with an invalid input (not a PIL Image)
        with pytest.raises(TypeError):
            image_processing.convert_to_grayscale("not_an_image")

    def test_apply_binarization(self):
        """Test function to apply binarization using Otsu's method."""
        # TODO: Test ImageProcessor.apply_binarization() function
        # Should accept grayscale PIL Image object as input
        # Should use Otsu's thresholding method
        # Should return binary image (black and white only)
        # Should handle edge cases (very dark/bright images)
        # Create a dummy grayscale image
        img = PILImage.new("L", (100, 100), color=128)
        binary_img = image_processing.apply_binarization(img)
        assert binary_img is not None, "Failed to apply binarization"
        assert isinstance(
            binary_img, PILImage.Image), "Result is not a PIL Image object"
        assert binary_img.mode == "L", "Binarized image mode is not 'L' (grayscale)"
        assert binary_img.size == img.size, "Binarized image dimensions do not match original"
        # Test with an invalid input (not a PIL Image)
        with pytest.raises(TypeError):
            image_processing.apply_binarization("not_an_image")

    def test_complete_preprocessing_pipeline(self, menu_image_path):
        """Test complete image preprocessing workflow."""
        # TODO: Test full pipeline from raw image to OCR-ready binary image
        # 1. Open image from path
        # 2. Convert to grayscale
        # 3. Apply Otsu's binarization
        # 4. Save result for debugging
        # Should work with actual menu image if available
        preprocessed_img = image_processing.preprocess_image(menu_image_path)
        assert preprocessed_img is not None, "Preprocessing failed"
        assert isinstance(preprocessed_img,
                          PILImage.Image), "Result is not a PIL Image object"
        assert preprocessed_img.mode == "L", "Preprocessed image mode is not 'L' (grayscale)"
        assert preprocessed_img.size == PILImage.open(
            menu_image_path).size, "Preprocessed image dimensions do not match original"
        # Test with an invalid path
        with pytest.raises(IOError):
            image_processing.preprocess_image("invalid_path.png")
