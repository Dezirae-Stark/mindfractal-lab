# Cytherea Asset Pipeline

This directory contains tools for processing raw photos into optimized WebP images for the Cytherea avatar system.

## Overview

The pipeline helps you:
1. Copy raw photos from your Android device
2. Process them into optimized WebP images
3. Place them in the correct directory for deployment

## Prerequisites

- Python 3.7+ with Pillow library
- Termux (for Android) or standard terminal
- Access to `/storage/emulated/0/Cytherea/` on Android

## Quick Start

1. **From Android (Termux)**:
   ```bash
   cd ~/mindfractal-lab
   bash tools/cytherea_asset_pipeline/android_import.sh
   ```

2. **Process images**:
   ```bash
   python tools/cytherea_asset_pipeline/process_cytherea_assets.py
   ```

3. **Verify results**:
   ```bash
   ls -la docs/interactive/child_assistant_console/graphics/realistic/
   ```

## File Naming

Raw files should be named to indicate their mood:
- `neutral_*.jpg` → `cytherea_neutral.webp`
- `focused_*.jpg` → `cytherea_focused.webp`
- `dream_*.jpg` → `cytherea_dream.webp`
- `overload_*.jpg` → `cytherea_overload.webp`
- `celebrate_*.jpg` → `cytherea_celebrate.webp`

## Configuration

Edit `sample_config.json` to customize:
- Input/output paths
- Image dimensions
- Quality settings
- Mood mappings

## Troubleshooting

- **Permission denied**: Run `termux-setup-storage` first
- **Module not found**: Install Pillow with `pip install Pillow`
- **No images found**: Check file naming patterns

## Privacy Note

Raw images are NOT committed to git. Only processed WebP files should be added to the repository.