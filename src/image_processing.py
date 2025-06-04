import logging

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu

# Get logger for this module
logger = logging.getLogger(__name__)


def open_image(image_path: str) -> Image.Image:
    """
    Open an image using Pillow.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Image: PIL Image object.
    """
    logger.info(f"Opening image: {image_path}")
    try:
        img = Image.open(image_path)
        logger.info(
            f"Successfully opened image: {image_path} "
            f"(Size: {img.size}, Mode: {img.mode})"
        )
        return img
    except Exception as e:
        logger.error(f"Error opening image {image_path}: {e}")
        raise IOError(f"Error opening image: {e}")


def convert_to_grayscale(image: Image.Image) -> Image.Image:
    """
    Convert an image to grayscale.

    Args:
        image (Image): PIL Image object.

    Returns:
        Image: Grayscale PIL Image object.
    """
    logger.debug("Converting image to grayscale")
    if not isinstance(image, Image.Image):
        logger.error("Input is not a PIL Image object")
        raise TypeError("Input must be a PIL Image object")

    gray_img = image.convert("L")
    logger.info(
        f"Successfully converted image to grayscale (Size: {gray_img.size})"
    )
    return gray_img


def apply_binarization(image: Image.Image) -> Image.Image:
    """
    Apply binarization to an image using Otsu's method.

    Args:
        image (Image): Grayscale PIL Image object.

    Returns:
        Image: Binary (black and white) PIL Image object.
    """
    logger.debug("Applying binarization using Otsu's method")
    if not isinstance(image, Image.Image):
        logger.error("Input is not a PIL Image object")
        raise TypeError("Input must be a PIL Image object")

    # Ensure the image is in grayscale mode
    if image.mode != "L":
        logger.error(
            f"Image must be in grayscale mode, but got mode: {image.mode}"
        )
        raise ValueError("Image must be in grayscale mode for binarization")

    # Convert to numpy array for processing
    img_array = np.array(image)
    logger.debug(f"Image array shape: {img_array.shape}")

    # Apply Otsu's thresholding
    thresh = threshold_otsu(img_array)
    logger.debug(f"Otsu threshold value: {thresh}")
    binary_image = img_array > thresh

    binary_image = (binary_image * 255).astype(np.uint8)  # Convert to uint8
    result = Image.fromarray(binary_image, mode="L")
    logger.info(f"Successfully applied binarization (Threshold: {thresh:.2f})")
    return result


def preprocess_image(image_path: str) -> Image.Image:
    """
    Complete preprocessing pipeline for an image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Image: Preprocessed binary PIL Image object.
    """
    logger.info(f"Starting image preprocessing pipeline for: {image_path}")

    try:
        # Step 1: Open the image
        logger.debug("Step 1: Opening image")
        img = open_image(image_path)

        # Step 2: Convert to grayscale
        logger.debug("Step 2: Converting to grayscale")
        gray_img = convert_to_grayscale(img)

        # Step 3: Apply Otsu's binarization
        logger.debug("Step 3: Applying binarization")
        binary_img = apply_binarization(gray_img)

        logger.info(
            f"Successfully completed preprocessing pipeline for: {image_path}"
        )
        return binary_img

    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {e}")
        raise
