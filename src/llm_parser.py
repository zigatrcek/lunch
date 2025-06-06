"""
LLM Parser module for extracting structured menu data from OCR text.

This module uses Google's Gemini Flash API to convert raw OCR text from
restaurant menu images into structured JSON data.
"""

import logging
from typing import Any, Dict, List

import google.generativeai as genai
import instructor
from pydantic import BaseModel, Field

from src.config import config

logger = logging.getLogger(__name__)


class MenuItem(BaseModel):
    """
    Represents a single menu item with its details.
    """
    name: str = Field(..., description="Name of the menu item")
    date: str = Field(...,
                      description="Date of the menu item in YYYY-MM-DD format")
    price: str = Field(..., description="Price of the menu item in EUR")
    type: str = Field(...,
                      description="Type of the menu item (e.g., 'meat', 'fish', 'soup')")


class Menu(BaseModel):
    """
    Represents the entire menu with a list of menu items.
    """
    items: List[MenuItem] = Field(..., description="List of menu items")


def create_gemini_client(
    model: str = "gemini-2.5-flash-preview-05-20"
) -> instructor.AsyncInstructor:
    """
    Create and configure a Gemini client.

    Args:
        model: Model name to use for parsing (default: "gemini-2.5-flash-preview-05-20")

    Returns:
        genai.Client: Configured Gemini client

    Raises:
        ValueError: If GEMINI_API_KEY is not configured
        Exception: If client creation fails
    """
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    if not isinstance(model, str):
        raise ValueError("Model name must be a string")

    try:
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=model),
            mode=instructor.Mode.GEMINI_JSON,
        )
        logger.info("Gemini client created successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}")
        raise Exception(f"Failed to create Gemini client: {e}")


def create_menu_extraction_prompt(ocr_text: str) -> str:
    """
    Create a detailed prompt for menu extraction from OCR text.

    Args:
        ocr_text: Raw text extracted from menu image via OCR

    Returns:
        str: Formatted prompt for the LLM
    """
    prompt = f"""
You are an expert at extracting structured information from restaurant menu text. 
I will provide you with OCR text from a Slovenian restaurant's weekly lunch menu image.

Your task is to extract and structure the menu information into the specified JSON format.

IMPORTANT GUIDELINES:
1. Parse dates carefully - they may be in various Slovenian formats (e.g., "ponedeljek 2.12.", "torek, 3. december")
2. Convert all dates to YYYY-MM-DD format
3. Handle meal types marked with asterisks (*, **, ***) - One asterisk (*) indicates a meat dish, two (**) indicates a fish dish, and three (***) indicates a soup dish
4. Extract prices carefully - they may be formatted as "8,60 €", "8, 60 €", or "8.60€"
5. If you see "RENDELICE" or similar vertical text, this is usually decorative - ignore it
6. Group meals by day of the week
7. Separate regular menu items from daily specials
8. Handle partial or unclear text gracefully - make reasonable assumptions where necessary

MEAL CATEGORIES to look for:
- Daily lunch specials
- Soups (juhe)
- Main courses (glavne jedi)
- Regular menu items

OCR TEXT TO PARSE:
{ocr_text}

Please extract this information. Be thorough but handle unclear text gracefully.
"""
    return prompt


def parse_menu_with_gemini(
    ocr_text: str,
    client: instructor.client.Instructor,
) -> List[Dict[str, Any]]:
    """
    Parse menu data from OCR text using Gemini Flash API.

    Args:
        ocr_text: Raw OCR text from menu image
        client: instructor.AsyncInstructor client instance
    Returns:
        Dict[str, Any]: Structured menu data in JSON format
    Raises:
        ValueError: If input is invalid
        Exception: If API call fails or response is invalid
    """
    if not ocr_text:
        return []
    if not isinstance(ocr_text, str):
        raise ValueError("OCR text must be a string")
    if not isinstance(client, instructor.client.Instructor):
        raise ValueError(
            f"Client must be an instance of instructor.client.Instructor, but got {type(client)}")
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")

    prompt = create_menu_extraction_prompt(ocr_text)
    try:
        menu_response: Menu = client.chat.completions.create(
            response_model=Menu,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        logger.info("Successfully called Gemini API for menu extraction")

        if not menu_response or not isinstance(menu_response, Menu):
            logger.error("Invalid response format from Gemini API: ",
                         menu_response)
            raise ValueError("Invalid response from Gemini API")
        return menu_response

    except Exception as e:
        logger.error(f"Failed to call Gemini API: {e}")
        raise Exception(f"Failed to call Gemini API: {e}")


def extract_menu_from_ocr(
    ocr_text: str,
) -> Menu:
    """
    Complete pipeline to extract structured menu data from OCR text.

    This is the main function that combines prompt creation and API calling
    to extract structured menu data from raw OCR text.

    Args:
        ocr_text: Raw OCR text from menu image

    Returns:
        List[Dict[str, Any]]: Structured menu data

    Raises:
        ValueError: If input is invalid
        Exception: If extraction or validation fails
    """
    if not ocr_text:
        return []
    if not isinstance(ocr_text, str):
        raise ValueError("OCR text must be a string")

    try:
        client = create_gemini_client()
        menu_items_list = parse_menu_with_gemini(ocr_text, client)
        logger.info("Successfully extracted menu data from OCR text")
        return menu_items_list
    except Exception as e:
        logger.error(f"Failed to extract menu from OCR text: {e}")
        raise Exception(f"Failed to extract menu from OCR text: {e}")


if __name__ == "__main__":
    # Example usage
    with open("tests/test_files/test_ocr_text.txt", "r", encoding="utf-8") as f:
        ocr_text = f.read()
    try:
        menu_data = extract_menu_from_ocr(ocr_text)
        print(menu_data)
    except Exception as e:
        print(f"Error extracting menu data: {e}")
        exit(1)
