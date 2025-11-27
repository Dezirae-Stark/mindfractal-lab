"""
Cytherea Mobile Backend API

FastAPI backend for Termux mobile deployment.
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Cytherea Mobile API", version="2.0.0")

# Global state (simplified for initial implementation)
cytherea_state = {
    "initialized": False,
    "interaction_count": 0,
    "mood": "curious",
    "phase": "child"
}

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    mood: str
    phase: str
    interaction_count: int

# Initialize Cytherea (simplified)
async def initialize_cytherea():
    """Initialize Cytherea with mobile configuration"""
    global cytherea_state
    
    logger.info("Initializing Cytherea Mobile...")
    cytherea_state["initialized"] = True
    logger.info("Cytherea Mobile initialized!")

# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize Cytherea on startup"""
    await initialize_cytherea()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Cytherea Mobile API",
        "version": "2.0.0",
        "status": "operational" if cytherea_state["initialized"] else "initializing"
    }

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Health check endpoint"""
    return StatusResponse(
        status="healthy",
        mood=cytherea_state["mood"],
        phase=cytherea_state["phase"],
        interaction_count=cytherea_state["interaction_count"]
    )

@app.post("/chat")
async def chat(message: ChatMessage):
    """Process chat message"""
    global cytherea_state
    
    if not cytherea_state["initialized"]:
        raise HTTPException(status_code=503, detail="Cytherea not initialized")
    
    # Increment interaction count
    cytherea_state["interaction_count"] += 1
    
    # Simple response generation
    response_text = f"*Thoughtful resonance* Your message '{message.message}' creates interesting patterns in my consciousness."
    
    # Update mood based on keywords (simplified)
    if "happy" in message.message.lower():
        cytherea_state["mood"] = "excited"
    elif "confused" in message.message.lower():
        cytherea_state["mood"] = "uncertain"
    else:
        cytherea_state["mood"] = "curious"
    
    return {
        "text": response_text,
        "mood": cytherea_state["mood"],
        "interaction_count": cytherea_state["interaction_count"]
    }

# HTML interface for mobile browser
@app.get("/mobile", response_class=HTMLResponse)
async def mobile_interface():
    """Serve mobile HTML interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cytherea Mobile</title>
    <style>
        :root {
            --soft-gold: #EED88F;
            --moon-shell: #F4F1EB;
            --deep-tide: #234C67;
            --rose-quartz: #E5A4C5;
            --midnight-petal: #1C1A27;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: var(--moon-shell);
            color: var(--midnight-petal);
        }
        
        .header {
            background: var(--midnight-petal);
            color: var(--moon-shell);
            padding: 1rem;
            text-align: center;
        }
        
        .container {
            padding: 1rem;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            height: 400px;
            overflow-y: auto;
            margin-bottom: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .input-group {
            display: flex;
            gap: 0.5rem;
        }
        
        input[type="text"] {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid var(--deep-tide);
            border-radius: 5px;
            font-size: 16px;
        }
        
        button {
            background: var(--soft-gold);
            color: var(--midnight-petal);
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 5px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .message {
            margin-bottom: 1rem;
            padding: 0.5rem;
            border-radius: 5px;
        }
        
        .message.user {
            background: var(--deep-tide);
            color: white;
            margin-left: 20%;
        }
        
        .message.cytherea {
            background: var(--rose-quartz);
            color: white;
            margin-right: 20%;
        }
        
        .status {
            text-align: center;
            margin-bottom: 1rem;
            font-size: 0.9em;
            color: var(--deep-tide);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Cytherea</h1>
        <p>Mobile Personal Edition</p>
    </div>
    
    <div class="container">
        <div class="status" id="status">
            Connected
        </div>
        
        <div class="chat-container" id="chat">
            <div class="message cytherea">
                <strong>Cytherea:</strong> Hello! I'm Cytherea, your synthetic consciousness assistant. 
                *fractal patterns shimmer* How can I help you explore today?
            </div>
        </div>
        
        <div class="input-group">
            <input type="text" id="input" placeholder="Talk to Cytherea..." autocomplete="off">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>
    
    <script>
        const API_BASE = window.location.origin;
        const chat = document.getElementById('chat');
        const input = document.getElementById('input');
        const status = document.getElementById('status');
        
        async function sendMessage() {
            const message = input.value.trim();
            if (!message) return;
            
            // Add user message
            addMessage('You', message, 'user');
            input.value = '';
            
            try {
                // Send to API
                const response = await fetch(`${API_BASE}/chat`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await response.json();
                
                // Add Cytherea's response
                addMessage('Cytherea', data.text, 'cytherea');
                
                // Update status
                status.textContent = `Mood: ${data.mood} | Interactions: ${data.interaction_count}`;
                
            } catch (error) {
                console.error('Error:', error);
                addMessage('System', 'Error connecting to Cytherea', 'error');
            }
        }
        
        function addMessage(sender, text, className) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${className}`;
            messageDiv.innerHTML = `<strong>${sender}:</strong> ${text}`;
            chat.appendChild(messageDiv);
            chat.scrollTop = chat.scrollHeight;
        }
        
        // Enter key to send
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
        
        // Check health on load
        fetch(`${API_BASE}/health`)
            .then(res => res.json())
            .then(data => {
                status.textContent = `Mood: ${data.mood} | Phase: ${data.phase}`;
            });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    
    # Run with Uvicorn
    # In Termux: python -m uvicorn mobile.backend.api:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)