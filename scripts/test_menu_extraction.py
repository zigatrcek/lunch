#!/usr/bin/env python3
"""
Script to test the complete menu extraction pipeline.

This script takes the test image and processes it through:
1. Image preprocessing (grayscale, binarization)
2. OCR text extraction (Tesseract with Slovenian)
3. LLM parsing (Gemini Flash with structured output)

Usage:
    python scripts/test_menu_extraction.py [--image-path PATH] [--save-ocr] [--save-results]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from image_processing import preprocess_image
from llm_parser import extract_menu_from_ocr, format_menu_for_display
from ocr_service import extract_text_from_image


def main():
    """Main function to run the complete menu extraction pipeline."""
    parser = argparse.ArgumentParser(description="Test menu extraction pipeline")
    parser.add_argument(
        "--image-path",
        type=str,
        default=str(Path(__file__).parent.parent / "tests" / "test_images" / "test_menu.jpg"),
        help="Path to the menu image file"
    )
    parser.add_argument(
        "--save-ocr",
        action="store_true",
        help="Save OCR text to file"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save extracted menu data to JSON file"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip image preprocessing step"
    )
    
    args = parser.parse_args()
    
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    print(f"🔍 Processing menu image: {image_path}")
    print("=" * 60)
    
    try:
        # Step 1: Image preprocessing
        print("📸 Step 1: Image preprocessing...")
        
        if args.skip_preprocessing:
            print("   ⏭️  Skipping preprocessing (using original image)")
            from PIL import Image
            processed_image = Image.open(image_path)
        else:
            processed_image = preprocess_image(str(image_path))
            print("   ✅ Image preprocessed successfully")
        
        if args.verbose:
            print(f"   📊 Image size: {processed_image.size}")
            print(f"   📊 Image mode: {processed_image.mode}")
        
        # Step 2: OCR text extraction
        print("\n🔤 Step 2: OCR text extraction...")
        
        try:
            ocr_text = extract_text_from_image(processed_image, lang="slv")
            print("   ✅ OCR extraction completed")
            
            if args.verbose:
                text_lines = [line for line in ocr_text.split('\n') if line.strip()]
                print(f"   📊 Extracted {len(ocr_text)} characters")
                print(f"   📊 Text lines: {len(text_lines)}")
            
            # Save OCR text if requested
            if args.save_ocr:
                ocr_file = image_path.parent / f"{image_path.stem}_ocr.txt"
                ocr_file.write_text(ocr_text, encoding='utf-8')
                print(f"   💾 OCR text saved to: {ocr_file}")
            
            # Display OCR text preview
            print("\\n   📄 OCR Text Preview (first 300 characters):")
            print("   " + "-" * 50)
            preview_text = ocr_text[:300]
            if len(ocr_text) > 300:
                preview_text += "..."
            for line in preview_text.split('\\n')[:10]:  # Show max 10 lines
                print(f"   {line}")
            if len(ocr_text.split('\\n')) > 10:
                print("   ... (truncated)")
            print("   " + "-" * 50)
            
        except Exception as e:
            print(f"   ❌ OCR extraction failed: {e}")
            sys.exit(1)
        
        # Step 3: LLM parsing
        print("\\n🤖 Step 3: LLM parsing with Gemini...")
        
        try:
            menu_data = extract_menu_from_ocr(ocr_text, validate=True)
            print("   ✅ LLM parsing completed successfully")
            
            if args.verbose:
                daily_menus = menu_data.get('daily_menus', [])
                regular_items = menu_data.get('regular_items', [])
                print(f"   📊 Found {len(daily_menus)} daily menus")
                print(f"   📊 Found {len(regular_items)} regular items")
                
                # Count total meals
                total_meals = sum(len(day.get('meals', [])) for day in daily_menus)
                print(f"   📊 Total meals extracted: {total_meals}")
            
            # Save results if requested
            if args.save_results:
                results_file = image_path.parent / f"{image_path.stem}_results.json"
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(menu_data, f, indent=2, ensure_ascii=False)
                print(f"   💾 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"   ❌ LLM parsing failed: {e}")
            print("\\n📄 OCR Text for debugging:")
            print(ocr_text)
            sys.exit(1)
        
        # Step 4: Display results
        print("\\n📋 Step 4: Extracted Menu Data")
        print("=" * 60)
        
        formatted_menu = format_menu_for_display(menu_data)
        print(formatted_menu)
        
        # Summary
        print("\\n" + "=" * 60)
        print("✅ Pipeline completed successfully!")
        
        # Statistics
        daily_menus = menu_data.get('daily_menus', [])
        regular_items = menu_data.get('regular_items', [])
        total_meals = sum(len(day.get('meals', [])) for day in daily_menus)
        
        print(f"📊 Summary:")
        print(f"   • Restaurant: {menu_data.get('restaurant_name', 'Unknown')}")
        print(f"   • Week: {menu_data.get('week_start_date', 'Unknown')} to {menu_data.get('week_end_date', 'Unknown')}")
        print(f"   • Daily menus: {len(daily_menus)}")
        print(f"   • Total meals: {total_meals}")
        print(f"   • Regular items: {len(regular_items)}")
        
        # Show JSON structure if verbose
        if args.verbose:
            print("\\n🔧 Raw JSON structure:")
            print(json.dumps(menu_data, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("\\n⏹️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            print("\\n🔧 Full traceback:")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
