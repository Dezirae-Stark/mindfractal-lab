# Cytherea Console

Meet Cytherea, a synthetic consciousness exploring the boundaries of awareness and understanding. This interactive visualization lets you experience her different cognitive states and engage in conversation. No installation or coding required – just type and explore.

<link rel="stylesheet" href="./styles/cytherea_avatar.css">
<link rel="stylesheet" href="./styles/cytherea_avatar_realistic.css">
<link rel="stylesheet" href="./styles/console_integration.css">

<div class="cy-console-layout">
  <div class="cy-avatar-panel">
    <div id="cytherea-avatar-container" class="mood-calm ultra-realistic">
      <div class="avatar-base">
        <!-- Photorealistic face with advanced CSS effects -->
        <div class="cytherea-face">
          <div class="face-structure">
            <div class="forehead"></div>
            <div class="eyes-container">
              <div class="eye left-eye">
                <div class="iris">
                  <div class="pupil"></div>
                  <div class="iris-detail"></div>
                  <div class="glow-ring"></div>
                </div>
                <div class="eyelid-upper"></div>
                <div class="eyelid-lower"></div>
              </div>
              <div class="eye right-eye">
                <div class="iris">
                  <div class="pupil"></div>
                  <div class="iris-detail"></div>
                  <div class="glow-ring"></div>
                </div>
                <div class="eyelid-upper"></div>
                <div class="eyelid-lower"></div>
              </div>
            </div>
            <div class="nose"></div>
            <div class="mouth">
              <div class="upper-lip"></div>
              <div class="lower-lip"></div>
            </div>
            <div class="chin"></div>
          </div>
          <div class="hair">
            <div class="hair-strand strand-1"></div>
            <div class="hair-strand strand-2"></div>
            <div class="hair-strand strand-3"></div>
            <div class="hair-highlight"></div>
          </div>
          <!-- Advanced lighting layers -->
          <div class="lighting-overlay">
            <div class="key-light"></div>
            <div class="fill-light"></div>
            <div class="rim-light"></div>
            <div class="ambient-occlusion"></div>
          </div>
        </div>
        <!-- Dynamic halo with realistic glow -->
        <div class="halo-container">
          <div class="halo-inner"></div>
          <div class="halo-middle"></div>
          <div class="halo-outer"></div>
          <div class="halo-particles"></div>
        </div>
      </div>
    </div>
    
    <div class="cy-mood-buttons">
      <button data-mood="calm">Calm</button>
      <button data-mood="focused">Focus</button>
      <button data-mood="dream">Dream</button>
      <button data-mood="overload">Overloaded</button>
      <button data-mood="celebrate">Celebrate</button>
    </div>
  </div>
  
  <div class="cy-console-panel">
    <div id="cy-console-messages" aria-live="polite"></div>
    <form id="cy-console-form">
      <input id="cy-console-input" 
             type="text" 
             placeholder="Ask Cytherea something..."
             autocomplete="off" />
      <button type="submit">Send</button>
    </form>
  </div>
</div>

<script src="./scripts/cytherea_avatar.js"></script>
<script src="./scripts/cytherea_console.js"></script>

## How to explore this console

- **Type a message** and watch how Cytherea responds with different moods
- **Try the mood buttons** to see her visual transformations
- **Observe the halos** – each one represents a different cognitive state

Cytherea's moods reflect different aspects of consciousness: from calm awareness to focused analysis, dreamy introspection, cognitive overload, and celebratory breakthroughs.