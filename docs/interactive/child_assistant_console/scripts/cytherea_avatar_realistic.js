/**
 * Cytherea Realistic Avatar Controller
 * Manages mood transitions and image swapping for the realistic avatar
 */

const CythereaRealisticAvatar = (function () {
  const moods = ['neutral', 'focused', 'dream', 'overload', 'celebrate'];

  const moodToImage = {
    neutral:   './graphics/realistic/cytherea_neutral.webp',
    focused:   './graphics/realistic/cytherea_focused.webp',
    dream:     './graphics/realistic/cytherea_dream.webp',
    overload:  './graphics/realistic/cytherea_overload.webp',
    celebrate: './graphics/realistic/cytherea_celebrate.webp'
  };

  let currentMood = 'neutral';
  let isTransitioning = false;

  function setMood(mood) {
    if (!moods.includes(mood) || mood === currentMood || isTransitioning) return;

    isTransitioning = true;
    const previousMood = currentMood;
    currentMood = mood;

    const container = document.getElementById('cy-avatar-realistic');
    const img = document.getElementById('cy-avatar-image');

    if (container && img) {
      // Start transition
      container.classList.add('transitioning');
      
      // Update mood class
      moods.forEach(m => container.classList.remove(`cy-avatar-mood-${m}`));
      container.classList.add(`cy-avatar-mood-${mood}`);

      // Preload new image
      const newImage = new Image();
      newImage.onload = () => {
        // Fade out current image
        img.style.opacity = '0';
        
        setTimeout(() => {
          // Swap image source
          img.src = newImage.src;
          img.alt = `Cytherea in ${mood} mood`;
          
          // Fade in new image
          requestAnimationFrame(() => {
            img.style.opacity = '1';
            
            setTimeout(() => {
              container.classList.remove('transitioning');
              isTransitioning = false;
            }, 300);
          });
        }, 150);
      };

      newImage.onerror = () => {
        console.error(`Failed to load image for mood: ${mood}`);
        container.classList.remove('transitioning');
        isTransitioning = false;
      };

      newImage.src = moodToImage[mood] || moodToImage.neutral;
    }

    // Dispatch mood change event
    document.dispatchEvent(new CustomEvent('cytherea-mood-changed', {
      detail: { mood: currentMood, previousMood }
    }));

    // Update active button state
    updateButtonStates(mood);
  }

  function getMood() {
    return currentMood;
  }

  function updateButtonStates(activeMood) {
    const buttons = document.querySelectorAll('.cy-mood-buttons button[data-mood]');
    buttons.forEach(btn => {
      const btnMood = btn.getAttribute('data-mood');
      if (btnMood === activeMood) {
        btn.classList.add('active');
        btn.setAttribute('aria-pressed', 'true');
      } else {
        btn.classList.remove('active');
        btn.setAttribute('aria-pressed', 'false');
      }
    });
  }

  function initMoodButtons() {
    const buttons = document.querySelectorAll('.cy-mood-buttons button[data-mood]');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const mood = btn.getAttribute('data-mood');
        setMood(mood);
      });
    });
  }

  function preloadImages() {
    // Preload all mood images for smooth transitions
    Object.values(moodToImage).forEach(src => {
      const img = new Image();
      img.src = src;
    });
  }

  function init() {
    initMoodButtons();
    setMood(currentMood);
    preloadImages();

    // Ensure smooth transitions
    const img = document.getElementById('cy-avatar-image');
    if (img) {
      img.style.transition = 'opacity 0.3s ease-in-out';
    }
  }

  // Public API
  return {
    init,
    setMood,
    getMood,
    moods: () => [...moods]
  };
})();

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    CythereaRealisticAvatar.init();
  });
} else {
  CythereaRealisticAvatar.init();
}