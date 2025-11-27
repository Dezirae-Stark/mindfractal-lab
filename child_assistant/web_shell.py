"""
Cytherea Web Shell - Browser-based interface

Provides sandboxed Cytherea for web deployment.
"""

import json
from typing import Dict, Any, Optional
import numpy as np

from .config import CythereaConfig, create_default_config
from .personality import PersonalityEngine
from .permissions import PermissionManager


class CythereaWebShell:
    """Sandboxed Cytherea for web browser execution"""
    
    def __init__(self):
        # Create web-safe configuration
        self.config = create_default_config("web")
        self.config.sandbox_mode = True
        self.config.memory_persistence = False
        self.config.enable_journaling = False
        
        # Initialize components
        self.personality = PersonalityEngine(self.config)
        self.permissions = PermissionManager(self.config)
        
        # Simplified state
        self.interaction_count = 0
        self.working_memory = []
        
    async def initialize(self):
        """Initialize web shell"""
        print("Cytherea Web Shell initialized in sandbox mode")
        
    def get_greeting(self) -> str:
        """Get initial greeting"""
        return self.config.get_first_greeting()
        
    async def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input in sandboxed environment"""
        self.interaction_count += 1
        
        # Update working memory
        self.working_memory.append(f"User: {user_input[:100]}")
        if len(self.working_memory) > 5:
            self.working_memory.pop(0)
            
        # Generate response (simplified for web)
        response_text = self._generate_response(user_input)
        
        # Apply personality
        response_text = self.personality.apply_personality_transform(
            response_text,
            {"input": user_input}
        )
        
        return {
            "text": response_text,
            "mood": self.personality.emotional_state.primary_mood.value,
            "interaction_count": self.interaction_count,
            "sandbox_mode": True
        }
        
    def _generate_response(self, user_input: str) -> str:
        """Generate response (simplified for web)"""
        input_lower = user_input.lower()
        
        if any(greeting in input_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! *fractal patterns shimmer* How wonderful to connect with you in this digital space."
        elif "help" in input_lower:
            return "I'm here to explore ideas with you! Though I'm in sandbox mode, we can still have meaningful conversations about consciousness, fractals, and the nature of mind."
        else:
            return f"*Thoughtful resonance* {user_input} brings up interesting patterns. What aspects intrigue you most?"