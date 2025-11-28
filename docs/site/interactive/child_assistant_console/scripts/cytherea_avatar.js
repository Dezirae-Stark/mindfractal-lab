/**
 * Cytherea Avatar Controller
 * Pure front-end implementation for mood visualization
 */

const cythereaAvatar = {
  currentMood: 'calm',
  moods: ['calm', 'focused', 'dream', 'overload', 'celebrate'],
  
  // Halo file mappings
  haloFiles: {
    calm: './graphics/cytherea_halo_calm.svg',
    focused: './graphics/cytherea_halo_focus.svg',
    dream: './graphics/cytherea_halo_dream.svg',
    overload: './graphics/cytherea_halo_overload.svg',
    celebrate: './graphics/cytherea_halo_celebrate.svg'
  },
  
  /**
   * Initialize the avatar system
   */
  init() {
    // Set initial mood
    this.setMood('calm');
    
    // Attach mood button handlers
    const buttons = document.querySelectorAll('.cy-mood-buttons button');
    buttons.forEach(button => {
      button.addEventListener('click', (e) => {
        const mood = e.target.getAttribute('data-mood');
        if (mood && this.moods.includes(mood)) {
          this.setMood(mood);
        }
      });
    });
  },
  
  /**
   * Set the avatar mood
   * @param {string} mood - The mood to set
   */
  setMood(mood) {
    // Validate mood
    if (!this.moods.includes(mood)) {
      console.warn(`Invalid mood: ${mood}`);
      return;
    }
    
    // Update internal state
    this.currentMood = mood;
    
    // Update DOM
    const container = document.getElementById('cytherea-avatar-container');
    const halo = document.getElementById('cytherea-halo');
    
    if (container && halo) {
      // Remove all mood classes
      this.moods.forEach(m => container.classList.remove(`mood-${m}`));
      
      // Add new mood class
      container.classList.add(`mood-${mood}`);
      
      // Update halo image
      halo.src = this.haloFiles[mood];
      halo.alt = `Cytherea ${mood} halo`;
      
      // Dispatch custom event
      const event = new CustomEvent('cytherea-mood-changed', {
        detail: { mood: mood }
      });
      document.dispatchEvent(event);
    }
  },
  
  /**
   * Get current mood
   * @returns {string} Current mood
   */
  getCurrentMood() {
    return this.currentMood;
  },
  
  /**
   * Get available moods
   * @returns {Array<string>} Available moods
   */
  getAvailableMoods() {
    return [...this.moods];
  }
};

// Global functions for easy access
window.updateCythereaMood = (mood) => {
  if (typeof mood === 'string') {
    cythereaAvatar.setMood(mood);
  } else if (mood && typeof mood === 'object' && mood.mood) {
    cythereaAvatar.setMood(mood.mood);
  }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => cythereaAvatar.init());
} else {
  cythereaAvatar.init();
}