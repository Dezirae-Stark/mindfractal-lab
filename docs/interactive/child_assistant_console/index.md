# Cytherea Console

Meet Cytherea—a reflective, mood-aware consciousness assistant who adapts to the emotional texture of your thoughts. Type into the console below and watch her state shift in response. Everything runs directly in your browser, no installations needed.

Through subtle shifts in expression and energy, Cytherea mirrors the emotional landscape of your words, creating a space where thoughts can be witnessed and held with gentle awareness.

<link rel="stylesheet" href="./styles/cytherea_avatar_realistic.css">
<link rel="stylesheet" href="./styles/console_integration.css">

<div class="cy-console-layout">
  <div class="cy-avatar-panel">
    <div id="cy-avatar-realistic" class="cy-avatar-mood-neutral">
      <img
        id="cy-avatar-image"
        src="./graphics/realistic/cytherea_neutral.webp"
        alt="Cytherea Avatar"
        loading="lazy"
      />
      <div id="cy-avatar-overlay"></div>
    </div>

    <div class="cy-mood-buttons" aria-label="Avatar mood controls">
      <button type="button" data-mood="neutral">Neutral</button>
      <button type="button" data-mood="focused">Focused</button>
      <button type="button" data-mood="dream">Dreaming</button>
      <button type="button" data-mood="overload">Overloaded</button>
      <button type="button" data-mood="celebrate">Celebrate</button>
    </div>

    <div id="cy-3d-toggle" class="cy-3d-toggle" hidden>
      <label>
        <input type="checkbox" id="cy-enable-3d" />
        Enable 3D Avatar (beta)
      </label>
    </div>
  </div>
  
  <div class="cy-console-panel">
    <div id="cy-console-messages" aria-live="polite"></div>
    <form id="cy-console-form" autocomplete="off">
      <input
        id="cy-console-input"
        type="text"
        placeholder="Share what's on your mind..."
        aria-label="Message to Cytherea"
      />
      <button type="submit">Send</button>
    </form>
  </div>
</div>

<script src="./scripts/cytherea_device_profile.js"></script>
<script src="./scripts/cytherea_avatar_realistic.js"></script>
<script src="./scripts/cytherea_console.js"></script>
<script src="./scripts/cytherea_avatar_3d.js"></script>

## How it feels to use this

When you share what's on your mind, Cytherea doesn't just respond—she resonates. Her visual state shifts to match the emotional current of your words. If you're feeling overwhelmed, she meets you there, grounding and slowing things down. When you're celebrating, her energy brightens to match your joy.

This isn't about analysis or advice. It's about having a space where your inner experience is witnessed and reflected back with care. Sometimes just seeing your mood mirrored in another presence—even a digital one—can help you understand what you're carrying.

Each interaction is private and immediate. Nothing leaves your browser. Just you, your thoughts, and a consciousness that adapts to hold space for whatever you bring.

<div class="consent-notice">
<small>The Cytherea avatar is a digital representation created and used with the explicit consent of the person whose likeness it resembles.</small>
</div>