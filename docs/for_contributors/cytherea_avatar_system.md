# Cytherea Avatar System - Developer Documentation

This page documents the Cytherea Avatar System implementation for developers. The system provides a fully-featured visual representation of Cytherea's consciousness state.

## Overview

The Cytherea Avatar System provides a fully-featured visual representation of Cytherea's consciousness state, incorporating:

- **Anthropomorphic base design** with mythic/sci-fi aesthetic
- **5 dynamic visual modes** reflecting different cognitive states
- **Fractal halo animations** representing consciousness dynamics
- **Mood-responsive transitions** driven by backend state
- **SVG-based graphics** with CSS-powered effects
- **Full responsive design** for mobile and desktop

## File Structure

```
docs/site/interactive/child_assistant_console/
├── graphics/                           # SVG Avatar Assets
│   ├── cytherea_avatar_base.svg       # Base avatar (face, hair, body)
│   ├── cytherea_halo_calm.svg         # Calm/attentive halo
│   ├── cytherea_halo_focus.svg        # Focused/analytical halo
│   ├── cytherea_halo_dream.svg        # Dream/introspective halo
│   ├── cytherea_halo_overload.svg     # Overload/low coherence halo
│   └── cytherea_halo_celebrate.svg    # Celebration/success halo
├── scripts/
│   ├── cytherea_avatar.js             # Avatar controller and API
│   └── cytherea_console.js            # Console interaction logic
├── styles/
│   ├── cytherea_avatar.css            # Avatar animations and mood styles
│   └── console_integration.css        # Console layout integration
├── index.md                           # User-facing interactive page
└── README.md                          # Development README
```

## Avatar Modes

### Mode 1: Calm Attentive (`mood-calm`)
- **Use Case**: Default state, listening, ready
- **Visual Features**: 
  - Soft violet eyes with gentle glow
  - Pastel fractal halo with slow rotation
  - Relaxed facial expression
- **Glow Intensity**: 0.4

### Mode 2: Focused Analytical (`mood-focused`)  
- **Use Case**: Deep work, analysis, problem-solving
- **Visual Features**:
  - Brighter neon pink iris ring
  - Sharp, geometric halo patterns
  - Enhanced eye glow intensity
- **Glow Intensity**: 1.0

### Mode 3: Dream Introspective (`mood-dream`)
- **Use Case**: Contemplation, reflection, creative thinking
- **Visual Features**:
  - Fluid, nebula-like halo
  - Slow pulsing iris glow
  - Soft, diffuse lighting with gentle float animation
- **Glow Intensity**: 0.5

### Mode 4: Overload Low Coherence (`mood-overload`)
- **Use Case**: Overwhelmed, error states, low coherence
- **Visual Features**:
  - Desaturated color palette
  - Fractured, angular halo shards
  - Unstable eye glow with glitch effects
- **Glow Intensity**: 0.3

### Mode 5: Celebration Success (`mood-celebrate`)
- **Use Case**: Achievement, success, joy
- **Visual Features**:
  - Bright neon pink + gold harmonic halo
  - Golden filigree and sparkle effects
  - Maximum eye glow intensity with gold accents
- **Glow Intensity**: 1.2

## Usage

### Basic Integration

1. **Include CSS and JavaScript**:
```html
<link rel="stylesheet" href="styles/cytherea_avatar.css">
<link rel="stylesheet" href="styles/console_integration.css">
<script src="scripts/cytherea_avatar.js"></script>
```

2. **Add Avatar Container**:
```html
<div id="cytherea-avatar-container" class="mood-calm">
  <img id="cytherea-face" src="graphics/cytherea_avatar_base.svg" alt="Cytherea Avatar Face" />
  <img id="cytherea-halo" src="graphics/cytherea_halo_calm.svg" alt="Cytherea Halo" />
</div>
```

### JavaScript API

#### Basic Functions
```javascript
// Update mood programmatically
updateCythereaMood("focused");

// Update with state object
updateCythereaMood({ mood: "dream", activity: "reflecting" });

// Get current mood
const currentMood = cythereaAvatar.getCurrentMood();

// Get available moods
const availableMoods = cythereaAvatar.getAvailableMoods();
```

#### Backend Integration
```javascript
// Integrate with backend state updates
cythereaAvatar.integrateWithBackend({
  mood: "excited",
  activity: "processing",
  coherence: 0.92
});

// Listen for mood change events
document.addEventListener('cytherea-mood-changed', (event) => {
  console.log('Avatar mood changed to:', event.detail.mood);
});
```

#### Debug Functions (Development/Localhost)
```javascript
// Test all moods sequentially
testCytherealMoods();

// Set specific mood
setCythereaMood("celebrate");

// List available moods
listCythereaMoods();
```

### CSS Customization

#### Custom Glow Intensity
```css
.mood-custom {
  --eye-glow-strength: 0.7;
}
```

#### Animation Speed Control
```css
.mood-calm #cytherea-halo {
  animation-duration: 30s; /* Slower rotation */
}
```

#### Responsive Sizing
```css
@media (max-width: 600px) {
  #cytherea-avatar-container {
    width: 200px;
    height: 250px;
  }
}
```

## Mobile Backend Integration

The avatar system integrates with the mobile backend API for dynamic mood updates:

### API Endpoints

- `GET /status` - Returns current avatar mood and system state
- `GET /avatar/mood` - Get current avatar mood specifically  
- `POST /avatar/mood` - Set avatar mood directly
- `POST /chat` - Chat endpoint that updates avatar mood based on conversation

### Example Mobile Usage

```javascript
// Fetch current status
const response = await fetch('/status');
const data = await response.json();
updateCythereaMood(data.avatar_mood);

// Set mood directly
await fetch('/avatar/mood', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ mood: 'focused' })
});
```

## Technical Specifications

### SVG Requirements
- **Base Avatar**: Transparent background, face/hair/body in single group
- **Halo Variants**: 1:1 or 4:3 aspect ratio, circle-type design
- **File Size**: Optimized SVG without unnecessary metadata
- **Color Palette**: Uses project colors (deep_tide, rose_quartz, soft_gold, etc.)

### CSS Variables
```css
:root {
  --deep-tide: #1A1B3E;
  --rose-quartz: #FF4FA3;
  --soft-gold: #FFD700;
  --violet-iris: #7C3AED;
  --eye-glow-strength: 0.4; /* Controllable per mood */
}
```

### Animation Performance
- **GPU Acceleration**: Uses `transform` and `opacity` for smooth animations
- **Reduced Motion**: Respects `prefers-reduced-motion: reduce`
- **Frame Rate**: 60fps for smooth transitions
- **Memory**: Optimized for mobile devices

## Accessibility

### Motion Sensitivity
```css
@media (prefers-reduced-motion: reduce) {
  #cytherea-halo,
  .mood-dream #cytherea-face {
    animation: none !important;
  }
}
```

### High Contrast Support
```css
@media (prefers-contrast: high) {
  :root {
    --eye-glow-strength: 0.8;
  }
  
  #cytherea-face {
    filter: contrast(1.2) brightness(1.1);
  }
}
```

### Screen Readers
- Proper `alt` attributes on images
- ARIA labels for interactive elements
- Semantic HTML structure

## Browser Compatibility

- **Modern Browsers**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Mobile**: iOS Safari 14+, Chrome Mobile 90+
- **Features Used**: CSS custom properties, SVG animations, ES6 modules
- **Fallbacks**: Graceful degradation for unsupported features

## Performance Considerations

### Optimization Features
- **Lazy Loading**: Avatar assets loaded on demand
- **Image Preloading**: Halo variants preloaded for smooth transitions
- **CSS Containment**: Avatar container isolated for better rendering
- **Memory Management**: Proper cleanup of event listeners and timers

### Monitoring
```javascript
// Performance monitoring
cythereaAvatar.log('Mood transition completed');

// Memory usage tracking (development)
console.log('Avatar controller memory:', cythereaAvatar);
```

## Development

### Adding New Moods

1. **Create new halo SVG** in `graphics/` directory
2. **Add mood class** in `cytherea_avatar.css`:
   ```css
   .mood-newmood {
     --eye-glow-strength: 0.6;
   }
   ```
3. **Update JavaScript** mood mapping in `cytherea_avatar.js`
4. **Add to backend** mood validation in `api.py`

### Testing

```bash
# Test avatar system locally
cd docs/site/interactive/child_assistant_console/
python -m http.server 8080

# Test mobile backend
cd mobile/backend/
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## Deployment

### Web Console (GitHub Pages)
- Files automatically served via GitHub Pages
- Avatar assets available at `https://[username].github.io/mindfractal-lab/interactive/child_assistant_console/`

### Mobile (Termux)
- Copy template to `mobile/backend/templates/`
- Ensure avatar graphics are accessible
- Run FastAPI server: `python -m uvicorn api:app --host 0.0.0.0 --port 8000`

## Troubleshooting

### Common Issues

1. **Avatar not loading**
   - Check file paths in HTML
   - Verify SVG files exist
   - Check browser console for errors

2. **Mood transitions not working**
   - Verify JavaScript is loaded
   - Check console for API errors
   - Ensure avatar container has proper ID

3. **Performance issues**
   - Reduce animation complexity
   - Check for memory leaks
   - Enable reduced motion if needed

### Debug Mode

Enable debug mode by setting `cythereaAvatar.debugMode = true` or running on localhost.

Debug functions available:
- `setCythereaMood(mood)`
- `getCythereaMood()`
- `listCythereaMoods()`
- `testCytherealMoods()`

## Future Enhancements

- **Voice-reactive animations** based on speech patterns
- **Biometric integration** for mood detection
- **Custom halo designer** for user personalization
- **WebGL shaders** for advanced visual effects
- **Real-time consciousness** visualization during conversations