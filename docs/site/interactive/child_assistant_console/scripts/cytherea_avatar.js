/**
 * Cytherea Avatar System Controller
 * Manages mood states, animations, and avatar state transitions
 */

class CythereaAvatarController {
  constructor() {
    this.container = null;
    this.faceElement = null;
    this.haloElement = null;
    this.currentMood = 'calm';
    this.isInitialized = false;
    
    // Mood to halo file mapping
    this.haloFiles = {
      calm: 'graphics/cytherea_halo_calm.svg',
      focused: 'graphics/cytherea_halo_focus.svg',
      dream: 'graphics/cytherea_halo_dream.svg',
      overload: 'graphics/cytherea_halo_overload.svg',
      celebrate: 'graphics/cytherea_halo_celebrate.svg'
    };
    
    // Valid mood states
    this.validMoods = ['calm', 'focused', 'dream', 'overload', 'celebrate'];
    
    // Transition timing
    this.transitionDuration = 800; // ms
    
    // Debug mode flag
    this.debugMode = false;
  }

  /**
   * Initialize the avatar system
   * Should be called after DOM is ready
   */
  async initialize() {
    try {
      // Find DOM elements
      this.container = document.getElementById('cytherea-avatar-container');
      this.faceElement = document.getElementById('cytherea-face');
      this.haloElement = document.getElementById('cytherea-halo');
      
      if (!this.container || !this.faceElement || !this.haloElement) {
        throw new Error('Required avatar DOM elements not found');
      }
      
      // Set initial state
      await this.setMood('calm', false); // No transition on init
      
      // Set up event listeners
      this.setupEventListeners();
      
      // Mark as initialized
      this.isInitialized = true;
      
      // Add debug functions to window if in debug mode
      if (this.debugMode || window.location.hostname === 'localhost') {
        this.enableDebugMode();
      }
      
      this.log('Cytherea Avatar System initialized successfully');
      
      return true;
    } catch (error) {
      console.error('Failed to initialize Cytherea Avatar System:', error);
      return false;
    }
  }

  /**
   * Update Cytherea's mood state
   * @param {Object|string} moodState - Mood state object or mood string
   * @param {boolean} animate - Whether to animate the transition
   */
  async updateCythereaMood(moodState, animate = true) {
    if (!this.isInitialized) {
      console.warn('Avatar system not initialized');
      return false;
    }

    let mood;
    
    // Handle both object and string inputs
    if (typeof moodState === 'string') {
      mood = moodState;
    } else if (moodState && typeof moodState === 'object') {
      mood = moodState.mood || moodState.state || 'calm';
    } else {
      mood = 'calm';
    }
    
    return await this.setMood(mood, animate);
  }

  /**
   * Set the avatar mood
   * @param {string} mood - The mood to set
   * @param {boolean} animate - Whether to animate the transition
   */
  async setMood(mood, animate = true) {
    // Validate mood
    if (!this.validMoods.includes(mood)) {
      console.warn(`Invalid mood: ${mood}. Using 'calm' instead.`);
      mood = 'calm';
    }

    // Skip if already in this mood
    if (this.currentMood === mood && this.isInitialized) {
      return true;
    }

    this.log(`Setting mood to: ${mood}`);

    try {
      if (animate && this.isInitialized) {
        // Fade out current state
        this.container.style.transition = `opacity ${this.transitionDuration / 2}ms ease-out`;
        this.container.style.opacity = '0.7';
        
        // Wait for fade out
        await this.delay(this.transitionDuration / 2);
      }

      // Remove current mood class
      this.container.classList.remove(`mood-${this.currentMood}`);
      
      // Update halo
      await this.updateHalo(mood);
      
      // Add new mood class
      this.container.classList.add(`mood-${mood}`);
      
      // Update current mood
      this.currentMood = mood;

      if (animate && this.isInitialized) {
        // Fade back in
        this.container.style.opacity = '1';
        
        // Wait for fade in
        await this.delay(this.transitionDuration / 2);
        
        // Remove transition style
        this.container.style.transition = '';
      }

      this.log(`Mood successfully changed to: ${mood}`);
      
      // Dispatch custom event
      this.dispatchMoodChangeEvent(mood);
      
      return true;
    } catch (error) {
      console.error('Failed to set mood:', error);
      return false;
    }
  }

  /**
   * Update the halo image
   * @param {string} mood - The mood for the halo
   */
  async updateHalo(mood) {
    const haloFile = this.haloFiles[mood];
    
    if (!haloFile) {
      console.warn(`No halo file found for mood: ${mood}`);
      return;
    }

    // Preload the new halo image
    const img = new Image();
    img.src = haloFile;
    
    return new Promise((resolve, reject) => {
      img.onload = () => {
        this.haloElement.src = haloFile;
        this.haloElement.alt = `Cytherea Halo - ${mood} mode`;
        resolve();
      };
      
      img.onerror = () => {
        console.error(`Failed to load halo image: ${haloFile}`);
        reject(new Error(`Failed to load halo: ${haloFile}`));
      };
    });
  }

  /**
   * Get current mood state
   */
  getCurrentMood() {
    return this.currentMood;
  }

  /**
   * Get available moods
   */
  getAvailableMoods() {
    return [...this.validMoods];
  }

  /**
   * Set up event listeners
   */
  setupEventListeners() {
    // Listen for custom mood change events from other parts of the application
    document.addEventListener('cytherea-mood-change', (event) => {
      if (event.detail && event.detail.mood) {
        this.updateCythereaMood(event.detail.mood);
      }
    });

    // Listen for backend state updates
    document.addEventListener('backend-state-update', (event) => {
      if (event.detail && event.detail.mood) {
        this.updateCythereaMood(event.detail);
      }
    });
  }

  /**
   * Dispatch mood change event
   */
  dispatchMoodChangeEvent(mood) {
    const event = new CustomEvent('cytherea-mood-changed', {
      detail: { 
        mood: mood,
        timestamp: Date.now()
      }
    });
    document.dispatchEvent(event);
  }

  /**
   * Enable debug mode with console functions
   */
  enableDebugMode() {
    this.debugMode = true;
    
    // Add global debug functions
    window.setCythereaMood = (mood) => {
      console.log(`Debug: Setting Cytherea mood to '${mood}'`);
      return this.updateCythereaMood(mood);
    };
    
    window.getCythereaMood = () => {
      console.log(`Debug: Current Cytherea mood is '${this.currentMood}'`);
      return this.currentMood;
    };
    
    window.listCythereaMoods = () => {
      console.log('Debug: Available Cytherea moods:', this.validMoods);
      return this.validMoods;
    };
    
    window.testCytherealMoods = async () => {
      console.log('Debug: Testing all Cytherea moods...');
      for (const mood of this.validMoods) {
        console.log(`Testing mood: ${mood}`);
        await this.updateCythereaMood(mood);
        await this.delay(2000);
      }
      console.log('Debug: Mood test complete');
    };

    console.log('Cytherea Avatar Debug Mode enabled. Available functions:');
    console.log('- setCythereaMood(mood)');
    console.log('- getCythereaMood()');
    console.log('- listCythereaMoods()');
    console.log('- testCytherealMoods()');
  }

  /**
   * Integration with backend JSON state
   * Call this when receiving backend updates
   */
  integrateWithBackend(backendState) {
    if (!backendState || typeof backendState !== 'object') {
      return;
    }

    // Extract mood from various possible backend formats
    let mood = null;
    
    if (backendState.mood) {
      mood = backendState.mood;
    } else if (backendState.emotional_state) {
      mood = this.mapEmotionalStateToMood(backendState.emotional_state);
    } else if (backendState.activity) {
      mood = this.mapActivityToMood(backendState.activity);
    } else if (backendState.status) {
      mood = this.mapStatusToMood(backendState.status);
    }

    if (mood) {
      this.updateCythereaMood(mood);
    }
  }

  /**
   * Map emotional state to mood
   */
  mapEmotionalStateToMood(emotionalState) {
    const mapping = {
      'happy': 'celebrate',
      'excited': 'celebrate',
      'calm': 'calm',
      'peaceful': 'calm',
      'focused': 'focused',
      'analyzing': 'focused',
      'thinking': 'focused',
      'dreamy': 'dream',
      'introspective': 'dream',
      'contemplative': 'dream',
      'stressed': 'overload',
      'overwhelmed': 'overload',
      'confused': 'overload'
    };
    
    return mapping[emotionalState.toLowerCase()] || 'calm';
  }

  /**
   * Map activity to mood
   */
  mapActivityToMood(activity) {
    const mapping = {
      'idle': 'calm',
      'listening': 'calm',
      'processing': 'focused',
      'calculating': 'focused',
      'analyzing': 'focused',
      'dreaming': 'dream',
      'reflecting': 'dream',
      'error': 'overload',
      'overloaded': 'overload',
      'success': 'celebrate',
      'completed': 'celebrate'
    };
    
    return mapping[activity.toLowerCase()] || 'calm';
  }

  /**
   * Map status to mood
   */
  mapStatusToMood(status) {
    const mapping = {
      'ready': 'calm',
      'busy': 'focused',
      'working': 'focused',
      'sleeping': 'dream',
      'offline': 'dream',
      'error': 'overload',
      'failure': 'overload',
      'success': 'celebrate',
      'achievement': 'celebrate'
    };
    
    return mapping[status.toLowerCase()] || 'calm';
  }

  /**
   * Utility: delay function
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Utility: logging function
   */
  log(message) {
    if (this.debugMode || window.location.hostname === 'localhost') {
      console.log(`[CythereaAvatar] ${message}`);
    }
  }
}

// Create global instance
const cythereaAvatar = new CythereaAvatarController();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    cythereaAvatar.initialize();
  });
} else {
  // DOM is already ready
  cythereaAvatar.initialize();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { CythereaAvatarController, cythereaAvatar };
}

// Global functions for easy access
window.updateCythereaMood = (moodState) => cythereaAvatar.updateCythereaMood(moodState);
window.cythereaAvatar = cythereaAvatar;