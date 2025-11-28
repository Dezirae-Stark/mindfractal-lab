#!/usr/bin/env python3
"""
process_two_images.py - Process two photos into 5 mood variations
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance

# Configuration
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "docs/interactive/child_assistant_console/graphics/source_raw"
OUT_DIR = REPO_ROOT / "docs/interactive/child_assistant_console/graphics/realistic"

# Image settings
TARGET_WIDTH = 800
TARGET_HEIGHT = 1000
WEBP_QUALITY = 90

# Mood mappings with enhancement settings
MOOD_SETTINGS = {
    "neutral": {
        "source": "image1",  # Use first image
        "contrast": 1.0,
        "color": 1.0,
        "brightness": 1.0
    },
    "focused": {
        "source": "image1",
        "contrast": 1.15,
        "color": 1.1,
        "brightness": 1.05
    },
    "dream": {
        "source": "image2",  # Use second image
        "contrast": 0.95,
        "color": 1.15,
        "brightness": 1.1
    },
    "overload": {
        "source": "image2",
        "contrast": 0.9,
        "color": 0.85,
        "brightness": 0.95
    },
    "celebrate": {
        "source": "image1",
        "contrast": 1.2,
        "color": 1.25,
        "brightness": 1.1
    }
}

def process_image(img, settings, out_path):
    """Process image with mood-specific enhancements"""
    # Apply enhancements
    if settings['contrast'] != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(settings['contrast'])
    
    if settings['color'] != 1.0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(settings['color'])
    
    if settings['brightness'] != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(settings['brightness'])
    
    # Save as WebP
    img.save(out_path, format="WEBP", quality=WEBP_QUALITY, method=6)
    size_kb = out_path.stat().st_size / 1024
    print(f"  ✓ Created: {out_path.name} ({size_kb:.1f} KB)")

def main():
    print("Cytherea Two-Image Processor")
    print("=" * 30)
    
    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find the two images
    images = list(RAW_DIR.glob("*.jpg")) + \
             list(RAW_DIR.glob("*.jpeg")) + \
             list(RAW_DIR.glob("*.JPG")) + \
             list(RAW_DIR.glob("*.JPEG"))
    
    if len(images) < 2:
        print(f"Error: Need at least 2 images, found {len(images)}")
        print(f"Looking in: {RAW_DIR}")
        return 1
    
    print(f"Found {len(images)} images:")
    for i, img in enumerate(images[:2]):
        print(f"  {i+1}. {img.name}")
    
    # Load and prepare base images
    print("\nLoading base images...")
    
    # Load first image
    img1 = Image.open(images[0]).convert("RGB")
    # Load second image (or reuse first if only one)
    img2 = Image.open(images[1] if len(images) > 1 else images[0]).convert("RGB")
    
    # Crop and resize both images
    base_images = {}
    for idx, (name, img) in enumerate([("image1", img1), ("image2", img2)]):
        # Calculate aspect ratio and crop
        img_ratio = img.width / img.height
        target_ratio = TARGET_WIDTH / TARGET_HEIGHT
        
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
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
        base_images[name] = img
        print(f"  ✓ Prepared base image {idx+1}")
    
    # Process each mood
    print("\nCreating mood variations...")
    for mood, settings in MOOD_SETTINGS.items():
        print(f"\n{mood.capitalize()} mood:")
        
        # Get base image
        base_img = base_images[settings['source']].copy()
        
        # Process and save
        out_path = OUT_DIR / f"cytherea_{mood}.webp"
        process_image(base_img, settings, out_path)
    
    print("\n✨ Processing complete!")
    print(f"\nAll mood variations saved to:")
    print(f"  {OUT_DIR}")
    
    print("\nNext steps:")
    print("1. Test locally: mkdocs serve")
    print("2. Commit the WebP files:")
    print("   git add docs/interactive/child_assistant_console/graphics/realistic/*.webp")
    print("   git commit -m \"Add Cytherea avatar images\"")
    print("   git push")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())