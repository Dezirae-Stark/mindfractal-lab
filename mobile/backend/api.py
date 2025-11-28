"""
Cytherea Mobile Backend API

FastAPI backend for Termux mobile deployment.
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
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

# Templates
templates = Jinja2Templates(directory="templates")

# Global state (simplified for initial implementation)
cytherea_state = {
    "initialized": False,
    "interaction_count": 0,
    "mood": "curious",
    "avatar_mood": "calm",  # Avatar mood state
    "phase": "child",
    "coherence": 0.85,
    "activity": "idle",
    "permissions_requested": []
}

# Request/Response models
class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    mood: str
    avatar_mood: str
    phase: str
    interaction_count: int
    coherence: float
    activity: str

class AvatarMoodRequest(BaseModel):
    mood: str  # calm, focused, dream, overload, celebrate

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
        avatar_mood=cytherea_state["avatar_mood"],
        phase=cytherea_state["phase"],
        interaction_count=cytherea_state["interaction_count"],
        coherence=cytherea_state["coherence"],
        activity=cytherea_state["activity"]
    )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current Cytherea status including avatar mood"""
    return StatusResponse(
        status="operational" if cytherea_state["initialized"] else "initializing",
        mood=cytherea_state["mood"],
        avatar_mood=cytherea_state["avatar_mood"],
        phase=cytherea_state["phase"],
        interaction_count=cytherea_state["interaction_count"],
        coherence=cytherea_state["coherence"],
        activity=cytherea_state["activity"]
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
    
    # Update mood and avatar mood based on keywords and context
    message_lower = message.message.lower()
    
    # Update emotional mood
    if any(word in message_lower for word in ["happy", "excited", "great", "wonderful"]):
        cytherea_state["mood"] = "excited"
        cytherea_state["avatar_mood"] = "celebrate"
    elif any(word in message_lower for word in ["confused", "lost", "overwhelmed"]):
        cytherea_state["mood"] = "uncertain"
        cytherea_state["avatar_mood"] = "overload"
    elif any(word in message_lower for word in ["analyze", "calculate", "solve", "think"]):
        cytherea_state["mood"] = "focused"
        cytherea_state["avatar_mood"] = "focused"
        cytherea_state["activity"] = "processing"
    elif any(word in message_lower for word in ["dream", "imagine", "wonder", "reflect"]):
        cytherea_state["mood"] = "contemplative"
        cytherea_state["avatar_mood"] = "dream"
        cytherea_state["activity"] = "reflecting"
    else:
        cytherea_state["mood"] = "curious"
        cytherea_state["avatar_mood"] = "calm"
        cytherea_state["activity"] = "listening"
    
    # Update coherence based on interaction complexity
    if len(message.message) > 100:
        cytherea_state["coherence"] = min(0.95, cytherea_state["coherence"] + 0.02)
    
    return {
        "text": response_text,
        "mood": cytherea_state["mood"],
        "avatar_mood": cytherea_state["avatar_mood"],
        "interaction_count": cytherea_state["interaction_count"],
        "coherence": cytherea_state["coherence"],
        "activity": cytherea_state["activity"]
    }

@app.post("/avatar/mood")
async def set_avatar_mood(mood_request: AvatarMoodRequest):
    """Set avatar mood directly"""
    global cytherea_state
    
    valid_moods = ["calm", "focused", "dream", "overload", "celebrate"]
    
    if mood_request.mood not in valid_moods:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mood. Must be one of: {', '.join(valid_moods)}"
        )
    
    cytherea_state["avatar_mood"] = mood_request.mood
    
    return {
        "status": "success",
        "avatar_mood": cytherea_state["avatar_mood"],
        "message": f"Avatar mood set to {mood_request.mood}"
    }

@app.get("/avatar/mood")
async def get_avatar_mood():
    """Get current avatar mood"""
    return {
        "avatar_mood": cytherea_state["avatar_mood"],
        "available_moods": ["calm", "focused", "dream", "overload", "celebrate"]
    }

# HTML interface for mobile browser
@app.get("/mobile", response_class=HTMLResponse)
async def mobile_interface(request: Request):
    """Serve mobile HTML interface"""
    # Read template file directly since it's a simple deployment
    template_path = Path(__file__).parent / "templates" / "mobile_interface.html"
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        # Fallback to simple HTML if template file not found
        return HTMLResponse("""
<!DOCTYPE html>
<html><head><title>Cytherea Mobile</title></head>
<body><h1>Cytherea Mobile</h1><p>Template file not found. Please check deployment.</p></body>
</html>
""")


if __name__ == "__main__":
    import uvicorn
    
    # Run with Uvicorn
    # In Termux: python -m uvicorn mobile.backend.api:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)