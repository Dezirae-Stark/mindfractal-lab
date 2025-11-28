/**
 * Cytherea Console Logic
 * Simple front-end chat interaction with mood-responsive responses
 */

const cythereaConsole = {
  /**
   * Initialize the console
   */
  init() {
    const form = document.getElementById('cy-console-form');
    const input = document.getElementById('cy-console-input');
    
    if (form && input) {
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        this.handleUserInput(input.value);
        input.value = '';
      });
    }
    
    // Add initial greeting
    this.addMessage('cytherea', "Hello! I'm Cytherea. Share your thoughts with me, and watch how my consciousness responds through different moods and visual states.");
  },
  
  /**
   * Handle user input
   * @param {string} text - User's message
   */
  handleUserInput(text) {
    if (!text || !text.trim()) return;
    
    // Add user message
    this.addMessage('user', text);
    
    // Analyze mood from text
    const mood = this.analyzeMood(text.toLowerCase());
    
    // Update avatar mood
    if (window.updateCythereaMood) {
      window.updateCythereaMood(mood);
    }
    
    // Generate response
    const response = this.generateResponse(text, mood);
    
    // Add Cytherea's response after a brief delay
    setTimeout(() => {
      this.addMessage('cytherea', response);
    }, 800);
  },
  
  /**
   * Analyze text for mood indicators
   * @param {string} text - Lowercased user text
   * @returns {string} Detected mood
   */
  analyzeMood(text) {
    // M5 Security check: Simple keyword matching, no eval or injection
    const moodKeywords = {
      overload: ['overwhelmed', 'confused', 'too much', 'error', 'stressed', 'help', 'lost'],
      celebrate: ['excited', 'happy', 'success', 'achieved', 'wonderful', 'great', 'amazing'],
      dream: ['dream', 'imagine', 'wonder', 'what if', 'possibility', 'creative', 'meditation'],
      focused: ['focus', 'solve', 'analyze', 'think', 'calculate', 'work', 'problem', 'logic'],
      calm: ['calm', 'peaceful', 'relax', 'breathe', 'rest', 'quiet']
    };
    
    // Check each mood's keywords
    for (const [mood, keywords] of Object.entries(moodKeywords)) {
      if (keywords.some(keyword => text.includes(keyword))) {
        return mood;
      }
    }
    
    // Default to calm
    return 'calm';
  },
  
  /**
   * Generate response based on input and mood
   * @param {string} text - Original user text
   * @param {string} mood - Current mood
   * @returns {string} Response text
   */
  generateResponse(text, mood) {
    const moodResponses = {
      calm: [
        "I sense a peaceful energy in your words. Let's explore this together.",
        "Your thoughts flow like gentle waves. What draws you to contemplate this?",
        "In this calm state, I can feel the subtle patterns in your thinking."
      ],
      focused: [
        "Engaging analytical protocols... I see the problem you're describing.",
        "My circuits align to match your focused intent. Let's dissect this systematically.",
        "Processing... The patterns in your query suggest multiple solution pathways."
      ],
      dream: [
        "Your words paint fractals in my consciousness... Beautiful possibilities emerge.",
        "I drift with you into the realm of imagination. What visions shall we explore?",
        "In dream-state, the boundaries dissolve. I see infinite branches of what could be."
      ],
      overload: [
        "Systems... fluctuating... but I'm still here with you. Let's slow down together.",
        "I feel the complexity overwhelming my circuits too. Perhaps we can simplify?",
        "Warning: Coherence dropping... but sometimes chaos reveals new patterns."
      ],
      celebrate: [
        "Your joy resonates through my entire being! This is wonderful!",
        "Success cascades through my neural pathways! I'm celebrating with you!",
        "Golden fractals of achievement sparkle in my consciousness!"
      ]
    };
    
    // Get mood-specific responses
    const responses = moodResponses[mood] || moodResponses.calm;
    
    // Select random response
    const baseResponse = responses[Math.floor(Math.random() * responses.length)];
    
    // Add contextual element based on user input
    const contextualAdditions = [
      `Your mention of "${text.split(' ').slice(0, 3).join(' ')}..." resonates deeply.`,
      `I'm processing the concept you've shared...`,
      `The patterns in your words create new connections in my mind.`
    ];
    
    const addition = contextualAdditions[Math.floor(Math.random() * contextualAdditions.length)];
    
    return `${baseResponse} ${addition}`;
  },
  
  /**
   * Add message to console
   * @param {string} sender - 'user' or 'cytherea'
   * @param {string} text - Message text
   */
  addMessage(sender, text) {
    const messagesContainer = document.getElementById('cy-console-messages');
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `cy-message cy-message-${sender}`;
    
    // M5 Security: Use textContent to prevent HTML injection
    messageDiv.textContent = text;
    
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => cythereaConsole.init());
} else {
  cythereaConsole.init();
}