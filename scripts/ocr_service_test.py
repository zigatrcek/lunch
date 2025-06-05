#!/usr/bin/env python3
"""
Manual test script for OCR service functionality.

This script tests the OCR service with a sample image to verify
that text extraction is working correctly.
"""

import sys
from pathlib import Path

from src.image_processing import open_image
from src.ocr_service import extract_text_from_image

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))



def main():
    """Test OCR functionality with available test image."""
    print("=== OCR Service Test ===\n")


    # Test with sample image if available
    test_image_path = Path(__file__).parent.parent / \
        "tests" / "test_files" / "test_menu.jpg"
    
    if test_image_path.exists():
        print(f"\n3. Testing OCR with sample image: {test_image_path}")

        try:
            image = open_image(test_image_path)
            # Test basic OCR
            print("   Testing basic OCR extraction...")
            text = extract_text_from_image(image, lang="slv")
            
            if text.strip():
                print(f"   ✓ Extracted {len(text)} characters")
                print(f"   '{text}'")
            else:
                print("   ⚠ No text extracted")
        
        except Exception as e:
            print(f"   ✗ OCR test failed: {e}")
    else:
        print(f"\n3. No test image found at {test_image_path}")
        print("   To test with an actual image, place a menu image at:")
        print(f"   {test_image_path}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
