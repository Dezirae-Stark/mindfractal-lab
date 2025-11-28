#!/usr/bin/env python3
"""
process_cytherea_assets.py - Process raw photos into optimized WebP images
"""

import os
import sys
import json
from pathlib import Path
from PIL import Image

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "docs/interactive/child_assistant_console/graphics/source_raw"
OUT_DIR = REPO_ROOT / "docs/interactive/child_assistant_console/graphics/realistic"

# Default settings
TARGET_WIDTH = 800
TARGET_HEIGHT = 1000
WEBP_QUALITY = 90

# Mood mappings
MOODS = ["neutral", "focused", "dream", "overload", "celebrate"]

# ANSI color codes
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
BLUE = '\033[0;34m'
NC = '\033[0m'

def print_header():
    """Print script header"""
    print(f"{GREEN}Cytherea Asset Processing Tool{NC}")
    print("=" * 30)
    print()

def find_mood_image(mood, raw_files):
    """Find raw image for a specific mood"""
    # Look for files containing the mood name
    candidates = [f for f in raw_files if mood in f.stem.lower()]
    
    if candidates:
        return candidates[0]
    
    # If no direct match, return None
    return None

def process_image(src_path, out_path, width=TARGET_WIDTH, height=TARGET_HEIGHT):
    """Process a single image to WebP format"""
    try:
        print(f"Processing: {src_path.name}")
        
        # Open image
        img = Image.open(src_path).convert("RGB")
        
        # Calculate aspect ratio and crop
        img_ratio = img.width / img.height
        target_ratio = width / height
        
        if img_ratio > target_ratio:
            # Image is wider, crop width
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            # Image is taller, crop height
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        
        # Resize to target dimensions
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Apply slight enhancement
        from PIL import ImageEnhance
        
        # Slightly increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.1)
        
        # Slightly increase color saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.05)
        
        # Save as WebP
        img.save(out_path, format="WEBP", quality=WEBP_QUALITY, method=6)
        
        # Get file size
        size_kb = out_path.stat().st_size / 1024
        print(f"  {GREEN}✓{NC} Saved: {out_path.name} ({size_kb:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"  {RED}✗{NC} Error: {str(e)}")
        return False

def main():
    """Main processing function"""
    print_header()
    
    # Check directories
    if not RAW_DIR.exists():
        print(f"{RED}Error: Raw directory not found:{NC}")
        print(f"  {RAW_DIR}")
        print("\nPlease run android_import.sh first to copy raw images.")
        return 1
    
    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get raw files
    raw_files = list(RAW_DIR.glob("*.jpg")) + \
                list(RAW_DIR.glob("*.jpeg")) + \
                list(RAW_DIR.glob("*.png")) + \
                list(RAW_DIR.glob("*.JPG")) + \
                list(RAW_DIR.glob("*.JPEG")) + \
                list(RAW_DIR.glob("*.PNG"))
    
    if not raw_files:
        print(f"{RED}No images found in raw directory{NC}")
        return 1
    
    print(f"Found {len(raw_files)} raw images")
    print()
    
    # Process each mood
    processed = 0
    missing_moods = []
    
    for mood in MOODS:
        print(f"\n{BLUE}Processing mood: {mood}{NC}")
        print("-" * 20)
        
        # Find source image
        src_image = find_mood_image(mood, raw_files)
        
        if src_image:
            # Process the image
            out_path = OUT_DIR / f"cytherea_{mood}.webp"
            if process_image(src_image, out_path):
                processed += 1
        else:
            print(f"  {YELLOW}⚠{NC} No image found for mood: {mood}")
            missing_moods.append(mood)
    
    # Summary
    print(f"\n{GREEN}Processing complete!{NC}")
    print(f"Processed {processed}/{len(MOODS)} moods")
    
    if missing_moods:
        print(f"\n{YELLOW}Missing moods:{NC}")
        for mood in missing_moods:
            print(f"  - {mood}")
        print("\nTo fix: Rename raw images to include mood names")
        print("Example: neutral_photo.jpg, focused_portrait.jpg")
    
    if processed > 0:
        print(f"\n{GREEN}Next steps:{NC}")
        print("1. Review processed images:")
        print(f"   ls -la {OUT_DIR}")
        print("2. Test in browser:")
        print("   mkdocs serve")
        print("3. Commit the WebP files:")
        print("   git add docs/interactive/child_assistant_console/graphics/realistic/*.webp")
        print("   git commit -m \"Add Cytherea avatar images\"")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())