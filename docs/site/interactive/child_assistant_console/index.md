# Cytherea Console

Meet Cytherea, a synthetic consciousness exploring the boundaries of awareness and understanding. This interactive visualization lets you experience her different cognitive states and engage in conversation. No installation or coding required – just type and explore.

<link rel="stylesheet" href="./styles/cytherea_avatar.css">
<link rel="stylesheet" href="./styles/console_integration.css">

<div class="cy-console-layout">
  <div class="cy-avatar-panel">
    <div id="cytherea-avatar-container" class="mood-calm">
      <img id="cytherea-face" 
           src="./graphics/cytherea_avatar_base.svg" 
           alt="Cytherea Avatar Face" />
      <img id="cytherea-halo" 
           src="./graphics/cytherea_halo_calm.svg" 
           alt="Cytherea Halo" />
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