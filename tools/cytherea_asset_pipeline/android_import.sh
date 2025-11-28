#!/data/data/com.termux/files/usr/bin/bash
# android_import.sh - helper script to copy raw Cytherea images into the repo

set -e

# Configuration
REPO_ROOT="${REPO_ROOT:-$HOME/mindfractal-lab}"
SRC_PATH="/storage/emulated/0/Cytherea"
DEST_PATH="$REPO_ROOT/docs/interactive/child_assistant_console/graphics/source_raw"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Cytherea Asset Import Tool${NC}"
echo "=============================="

# Check if source directory exists
if [ ! -d "$SRC_PATH" ]; then
    echo -e "${RED}Error: Source directory not found: $SRC_PATH${NC}"
    echo "Please ensure you have images in /storage/emulated/0/Cytherea/"
    exit 1
fi

# Create destination directory
mkdir -p "$DEST_PATH"

# Count available images
IMAGE_COUNT=$(ls -1 "$SRC_PATH"/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}No images found in $SRC_PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}Found $IMAGE_COUNT images to import${NC}"
echo

# Copy images with progress
echo "Copying raw Cytherea images..."
for img in "$SRC_PATH"/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        cp -v "$img" "$DEST_PATH/"
        echo -e "${GREEN}✓${NC} Copied: $filename"
    fi
done

echo
echo -e "${GREEN}Import complete!${NC}"
echo
echo "Next steps:"
echo "1. Review imported images in:"
echo "   $DEST_PATH"
echo
echo "2. Rename files to indicate mood:"
echo "   - neutral_*.jpg"
echo "   - focused_*.jpg"
echo "   - dream_*.jpg"
echo "   - overload_*.jpg"
echo "   - celebrate_*.jpg"
echo
echo "3. Run the processing script:"
echo "   cd \"$REPO_ROOT\""
echo "   python tools/cytherea_asset_pipeline/process_cytherea_assets.py"