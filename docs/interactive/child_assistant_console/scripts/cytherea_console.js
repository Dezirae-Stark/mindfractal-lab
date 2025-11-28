/**
 * Cytherea Console Integration
 * Handles chat interface and mood-aware responses
 */

(function () {
  const form = document.getElementById('cy-console-form');
  const input = document.getElementById('cy-console-input');
  const messages = document.getElementById('cy-console-messages');

  if (!form || !input || !messages) return;

  function appendMessage(text, role) {
    const div = document.createElement('div');
    div.className = role === 'user'
      ? 'cy-message cy-message-user'
      : 'cy-message cy-message-cytherea';
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }

  function inferMoodFromText(text) {
    const lower = text.toLowerCase();

    // Check for overload indicators
    if (lower.match(/overwhelm|too much|stress|can't|anxious|confused|chaos/)) {
      return 'overload';
    }
    
    // Check for focus indicators
    if (lower.match(/focus|solve|work|study|analyze|think|plan|organize/)) {
      return 'focused';
    }
    
    // Check for dream indicators
    if (lower.match(/dream|imagine|wonder|fantasy|wish|create|vision|possibility/)) {
      return 'dream';
    }
    
    // Check for celebration indicators
    if (lower.match(/happy|excited|grateful|celebrate|joy|success|amazing|wonderful|love/)) {
      return 'celebrate';
    }
    
    return 'neutral';
  }

  function generateCythereaReply(text, mood) {
    // Mood-specific response patterns
    const responses = {
      overload: [
        "You're carrying a lot right now. Let's slow things down and take this one piece at a time.",
        "I feel the weight of what you're holding. Remember, you don't have to process everything at once.",
        "That sounds overwhelming. Let's find one small thing we can make sense of together."
      ],
      focused: [
        "Alright, I'm locking in with you. Let's look at this clearly and decide on the next small step.",
        "I can feel your determination. What specific aspect would you like to explore first?",
        "Your focus is sharp. Let's channel this energy toward what matters most right now."
      ],
      dream: [
        "I love where your mind is wandering. Tell me more about the world you're seeing inside.",
        "Your imagination is creating something beautiful. What does this vision feel like?",
        "These possibilities you're exploring... they have a special energy. Keep going."
      ],
      celebrate: [
        "Yes! This is worth celebrating. Let's really feel what's good in this moment.",
        "Your joy is lighting up this space. What made this moment so special?",
        "I'm celebrating with you! This happiness you're feeling—it's important."
      ],
      neutral: [
        "I'm here, listening. Whatever is moving through you, you don't have to hold it alone.",
        "Tell me what's present for you right now. I'm here to witness it with you.",
        "Your words are safe here. What would feel good to share?"
      ]
    };

    const moodResponses = responses[mood] || responses.neutral;
    return moodResponses[Math.floor(Math.random() * moodResponses.length)];
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) return;

    // Add user message
    appendMessage(text, 'user');
    input.value = '';
    input.focus();

    // Infer mood from text
    const mood = inferMoodFromText(text);


    // Generate and add response with natural delay
    const typingDelay = 600 + Math.random() * 600; // 600-1200ms
    setTimeout(() => {
      const reply = generateCythereaReply(text, mood);
      appendMessage(reply, 'cytherea');
    }, typingDelay);
  });

  // Add initial greeting when page loads
  appendMessage("Hello. I'm here whenever you're ready to share.", 'cytherea');
})();