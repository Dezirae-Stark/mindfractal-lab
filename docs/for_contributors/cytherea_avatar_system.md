# Cytherea Avatar System - Developer Documentation

This document covers the technical implementation of the Cytherea avatar system, including the 2D realistic avatar, optional 3D pipeline, and asset processing workflows. This information is for contributors and developers only.

## Overview

The Cytherea avatar system provides a mood-responsive visual representation that adapts to user interactions. It consists of:

1. **2D Realistic Avatar** - WebP images with CSS overlays for different moods
2. **Mood Detection** - Text analysis to infer emotional states
3. **Device Profiling** - Performance optimization based on capabilities
4. **3D Avatar (Future)** - Three.js integration for high-end devices
5. **Asset Pipeline** - Tools for processing photos into avatar images

## Architecture

### File Structure

```
docs/interactive/child_assistant_console/
├── index.md                          # Public-facing page
├── graphics/
│   ├── realistic/                    # Processed WebP images
│   │   ├── cytherea_neutral.webp
│   │   ├── cytherea_focused.webp
│   │   ├── cytherea_dream.webp
│   │   ├── cytherea_overload.webp
│   │   └── cytherea_celebrate.webp
│   ├── source_raw/                   # Raw photos (not in git)
│   └── 3d/                          # Future 3D assets
│       └── cytherea_avatar.glb
├── scripts/
│   ├── cytherea_device_profile.js   # Capability detection
│   ├── cytherea_avatar_realistic.js # 2D avatar controller
│   ├── cytherea_avatar_3d.js        # 3D stub
│   └── cytherea_console.js          # Console interaction
└── styles/
    ├── cytherea_avatar_realistic.css
    └── console_integration.css
```

## 2D Realistic Avatar

### Image Requirements

- **Format**: WebP for optimal compression and quality
- **Dimensions**: 800x1000px (portrait, 4:5 aspect ratio)
- **File size**: Target < 150KB per image
- **Naming**: `cytherea_{mood}.webp`

### Mood States

1. **neutral** - Default state, calm expression
2. **focused** - Alert, concentrated expression
3. **dream** - Soft, contemplative expression
4. **overload** - Stressed, overwhelmed expression
5. **celebrate** - Joyful, excited expression

### JavaScript API

```javascript
// Avatar controller singleton
CythereaRealisticAvatar = {
  init(),                    // Initialize system
  setMood(mood),            // Change avatar mood
  getMood(),                // Get current mood
  moods()                   // Get available moods array
}

// Events
document.addEventListener('cytherea-mood-changed', (event) => {
  console.log('Mood changed to:', event.detail.mood);
  console.log('Previous mood:', event.detail.previousMood);
});
```

### CSS Classes

```css
/* Container mood states */
.cy-avatar-mood-neutral
.cy-avatar-mood-focused
.cy-avatar-mood-dream
.cy-avatar-mood-overload
.cy-avatar-mood-celebrate

/* Performance profiles */
.perf-low    /* Minimal effects */
.perf-medium /* Standard effects */
.perf-high   /* All effects enabled */
```

## Console Integration

### Mood Detection Algorithm

The console analyzes user text for keywords to infer mood:

```javascript
// Keyword patterns
overload:  /overwhelm < /dev/null | too much|stress|can't|anxious|confused|chaos/
focused:   /focus|solve|work|study|analyze|think|plan|organize/
dream:     /dream|imagine|wonder|fantasy|wish|create|vision|possibility/
celebrate: /happy|excited|grateful|celebrate|joy|success|amazing|wonderful|love/
neutral:   (default)
```

### Response Generation

Each mood has a curated set of responses that match Cytherea's personality:
- Gentle, supportive language
- Non-clinical tone
- Emotional resonance
- Witnessing presence

## Device Profiling

The system automatically detects device capabilities:

```javascript
CythereaDeviceProfile = {
  hasWebGL(),              // WebGL support
  isMobile(),              // Mobile detection
  isLowPowerMode(),        // Reduced motion preference
  shouldOffer3D(),         // 3D capability check
  getOptimalImageFormat(), // WebP vs JPG
  getPerformanceProfile(), // low/medium/high
  capabilities: {          // Full capability object
    webgl, mobile, lowPower, memory, networkSpeed
  }
}
```

## Asset Pipeline

### Prerequisites

- Python 3.7+ with Pillow
- Termux (for Android) or standard terminal
- Git repository cloned locally

### Workflow

1. **Copy raw photos from Android**:
   ```bash
   cd ~/mindfractal-lab
   bash tools/cytherea_asset_pipeline/android_import.sh
   ```

2. **Process images**:
   ```bash
   python tools/cytherea_asset_pipeline/process_cytherea_assets.py
   ```

3. **Verify outputs**:
   ```bash
   ls -la docs/interactive/child_assistant_console/graphics/realistic/
   ```

4. **Commit processed images**:
   ```bash
   git add docs/interactive/child_assistant_console/graphics/realistic/*.webp
   git commit -m "Add processed Cytherea avatar images"
   ```

### Configuration

Edit `tools/cytherea_asset_pipeline/sample_config.json`:

```json
{
  "moods": {
    "neutral": "IMG_001.jpg",
    "focused": "IMG_002.jpg",
    "dream": "IMG_003.jpg",
    "overload": "IMG_004.jpg",
    "celebrate": "IMG_005.jpg"
  },
  "output": {
    "width": 800,
    "height": 1000,
    "quality": 90,
    "format": "webp"
  }
}
```

## 3D Avatar (Future Implementation)

### Planned Features

- glTF 2.0 model format
- Three.js renderer
- Mood-based animations
- Dynamic lighting
- Particle effects

### Integration Points

```javascript
// When 3D is enabled
if (CythereaDeviceProfile.shouldOffer3D()) {
  // Load Three.js dynamically
  // Initialize 3D scene
  // Replace 2D avatar with 3D
}
```

## Performance Considerations

### Image Optimization

- Use `loading="lazy"` on avatar images
- Preload all mood images after initial load
- Implement smooth transitions with opacity

### CSS Performance

- Hardware-accelerated transforms only
- Reduced motion support
- Conditional effects based on device profile

### Memory Management

- Single avatar instance (singleton pattern)
- Event delegation for UI controls
- Cleanup on page unload

## Security & Privacy

### No Remote Code Execution
- No eval() or Function() constructors
- No dynamic script injection
- All code is static and bundled

### Data Privacy
- No user data collection
- No external API calls
- All processing happens client-side

### Content Security
- Sanitized text input (textContent only)
- No HTML injection vulnerabilities
- Consent notice for likeness usage

## Testing

### Browser Compatibility
- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile browsers

### Device Testing
- Desktop: Full effects
- Tablet: Standard effects
- Mobile: Optimized effects
- Low-end: Minimal effects

### Accessibility
- ARIA labels on controls
- Keyboard navigation
- Screen reader support
- Reduced motion respect

## Deployment

The avatar system is deployed as static assets via GitHub Pages:

1. Assets are committed to the repository
2. MkDocs builds the site
3. GitHub Actions deploys to Pages
4. CDN serves optimized content

No backend or server-side processing is required.

## Contributing

When adding new features:

1. Maintain the singleton pattern
2. Add feature detection
3. Implement graceful degradation
4. Update this documentation
5. Test on multiple devices

## Original README Content

> **Note**: The original child_assistant_console README content with fractal consciousness concepts and technical specifications has been moved here for reference. The public-facing page now focuses purely on the user experience.
