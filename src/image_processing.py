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


def extract_black_pixels(image: Image.Image, threshold: int = 40) -> Image.Image:
    """
    Extract pixels that are black or almost black from an image.
    All pixels where all RGB channels are below the threshold are set to black (0),
    others to white (255).

    Args:
        image (Image): PIL Image object (RGB or grayscale).
        threshold (int): Maximum value for a channel to be considered
            "almost black" (default: 40).

    Returns:
        Image: Binary (black and white) PIL Image object.
    """
    logger.debug(f"Extracting black pixels with threshold {threshold}")
    if not isinstance(image, Image.Image):
        logger.error("Input is not a PIL Image object")
        raise TypeError("Input must be a PIL Image object")

    # Convert to RGB if not already
    if image.mode != "RGB":
        img_rgb = image.convert("RGB")
    else:
        img_rgb = image

    arr = np.array(img_rgb)
    # Create mask: all channels below threshold
    mask = np.all(arr < threshold, axis=-1)
    # Create binary image: black where mask is True, white elsewhere
    binary_arr = np.where(mask, 0, 255).astype(np.uint8)
    # Convert to single channel (L mode)
    result = Image.fromarray(binary_arr, mode="L")
    logger.info(f"Extracted black pixels (threshold={threshold})")
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

        binary_img = extract_black_pixels(img, threshold=40)

        logger.info(
            f"Successfully completed preprocessing pipeline for: {image_path}"
        )
        return binary_img

    except Exception as e:
        logger.error(f"Failed to preprocess image {image_path}: {e}")
        raise


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    try:
        img_path = "tests/test_images/test_menu.jpg"
        logger.info(f"Running preprocessing on image: {img_path}")
        preprocessed_img = preprocess_image(img_path)
        plt.imshow(preprocessed_img, cmap='gray')
        plt.axis('off')
        plt.show()
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
