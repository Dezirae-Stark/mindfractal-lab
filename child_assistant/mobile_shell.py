"""
Cytherea Mobile Shell - Termux/Android interface

Provides full-featured Cytherea for mobile deployment.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from .interface import CythereaInterface
from .config import CythereaConfig, create_default_config


class CythereaMobileShell:
    """Full-featured Cytherea for mobile deployment"""
    
    def __init__(self, config_path: Optional[Path] = None):
        # Load or create configuration
        if config_path and config_path.exists():
            self.config = CythereaConfig.load(config_path)
        else:
            self.config = create_default_config("mobile")
            
        # Initialize interface
        self.interface = CythereaInterface(self.config)
        
    async def initialize(self):
        """Initialize mobile shell"""
        print(f"Initializing Cytherea Mobile for {self.config.primary_user}...")
        
        # Ensure directories exist
        if self.config.journal_path:
            self.config.journal_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.memory_path:
            self.config.memory_path.parent.mkdir(parents=True, exist_ok=True)
            
        print("Cytherea Mobile ready!")
        
    async def interactive_session(self):
        """Run interactive command-line session"""
        print(self.config.get_first_greeting())
        print("\n" + "="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ")
                
                if user_input.lower() in ['exit', 'quit', 'goodbye']:
                    farewell = await self.interface.shutdown()
                    print(f"\nCytherea: {farewell}")
                    break
                    
                # Process input
                response = await self.interface.process_input(user_input)
                
                # Display response
                print(f"\nCytherea: {response['text']}")
                
                # Show mood if changed significantly
                if 'mood' in response:
                    print(f"[Mood: {response['mood']}]")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Shutting down gracefully...")
                await self.interface.shutdown()
                break
            except Exception as e:
                print(f"\nError: {e}")
                
    async def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        introspection = await self.interface.get_introspection()
        
        return {
            "phase": self.config.current_phase.name,
            "mood": introspection["emotional"]["mood"],
            "coherence": introspection["consciousness"]["coherence"],
            "interactions": introspection["cognitive"]["interaction_count"],
            "needs": introspection["needs"]
        }