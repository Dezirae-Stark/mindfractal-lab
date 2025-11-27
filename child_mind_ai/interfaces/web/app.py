"""
Child Mind AI — Web Application
MindFractal Lab

Flask-based web interface with WebSocket for real-time updates.
"""

import json
import threading
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    from flask import Flask, render_template, request, jsonify
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    SocketIO = None

from ...core.engine import ConsciousnessEngine
from ...permissions.manager import PermissionManager
from ...config import ChildMindAIConfig


class ChildMindWebApp:
    """
    Web interface for Child Mind AI.

    Features:
    - Real-time state visualization
    - WebSocket communication
    - Interactive controls
    - State history graphs
    """

    def __init__(self, config: Optional[ChildMindAIConfig] = None):
        """Initialize web application."""
        if not FLASK_AVAILABLE:
            raise ImportError(
                "Flask and flask-socketio are required for the web interface. "
                "Install with: pip install flask flask-socketio"
            )

        self.config = config or ChildMindAIConfig()

        # Initialize Flask app
        template_dir = Path(__file__).parent / "templates"
        static_dir = Path(__file__).parent / "static"

        self.app = Flask(
            __name__,
            template_folder=str(template_dir),
            static_folder=str(static_dir)
        )
        self.app.config['SECRET_KEY'] = 'child_mind_ai_secret'

        # Initialize SocketIO
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Initialize engine with async-safe permission manager
        self.permission_manager = PermissionManager(
            config=self.config,
            prompt_callback=self._web_permission_prompt
        )
        self.engine = ConsciousnessEngine(config=self.config)

        # State history for visualization
        self.state_history = []
        self.max_history = 100

        # Pending permission requests
        self.pending_permission = None
        self.permission_event = threading.Event()
        self.permission_response = None

        # Register routes
        self._register_routes()
        self._register_socketio_events()

    def _web_permission_prompt(self, prompt: str) -> str:
        """Permission prompt via WebSocket."""
        # Emit permission request to frontend
        self.pending_permission = prompt
        self.socketio.emit('permission_request', {'prompt': prompt})

        # Wait for response (with timeout)
        self.permission_event.clear()
        got_response = self.permission_event.wait(timeout=300)  # 5 min timeout

        if got_response and self.permission_response:
            response = self.permission_response
            self.permission_response = None
            self.pending_permission = None
            return response

        return "no"  # Timeout = deny

    def _register_routes(self):
        """Register Flask routes."""

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/api/state')
        def get_state():
            """Get current state as JSON."""
            state_dict = self.engine.state.to_dict()
            return jsonify(state_dict)

        @self.app.route('/api/history')
        def get_history():
            """Get state history."""
            return jsonify(self.state_history)

        @self.app.route('/api/audit')
        def get_audit():
            """Get audit summary."""
            summary = self.permission_manager.get_audit_summary()
            return jsonify(summary)

    def _register_socketio_events(self):
        """Register WebSocket events."""

        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            print(f"[Web] Client connected")
            # Send current state
            emit('state_update', self.engine.state.to_dict())

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            print(f"[Web] Client disconnected")

        @self.socketio.on('message')
        def handle_message(data):
            """Handle incoming message from user."""
            user_input = data.get('content', '')

            if not user_input.strip():
                return

            # Process through engine
            response = self.engine.process_input(user_input)

            # Update history
            self._record_state()

            # Emit response
            emit('response', {
                'content': response.content,
                'phenomenal_report': response.phenomenal_report,
                'intentions': response.intentions,
                'uncertainty': {
                    'overall': response.uncertainty.overall(),
                    'description': response.uncertainty.to_natural_language()
                },
                'state': response.state.to_display_dict(),
                'timestamp': datetime.now().isoformat()
            })

            # Emit state update for visualization
            emit('state_update', self.engine.state.to_dict(), broadcast=True)

        @self.socketio.on('command')
        def handle_command(data):
            """Handle special commands."""
            cmd = data.get('command', '')

            if cmd == 'save':
                self.engine.save_checkpoint()
                emit('notification', {'type': 'success', 'message': 'State saved'})

            elif cmd == 'reset':
                self.engine.state = self.engine.state.initialize(self.config)
                self.state_history.clear()
                emit('state_update', self.engine.state.to_dict(), broadcast=True)
                emit('notification', {'type': 'info', 'message': 'State reset'})

            elif cmd == 'state':
                emit('detailed_state', self.engine.state.to_dict())

        @self.socketio.on('permission_response')
        def handle_permission_response(data):
            """Handle permission response from user."""
            self.permission_response = data.get('response', 'no')
            self.permission_event.set()

    def _record_state(self):
        """Record current state to history."""
        summary = self.engine.state.get_summary()

        record = {
            'timestamp': datetime.now().isoformat(),
            't': self.engine.state.t,
            'coherence': summary.coherence,
            'stability': summary.stability,
            'novelty': summary.novelty,
            'reflection_depth': summary.reflection_depth,
            'intention_strength': summary.intention_strength,
            'z_norm': float(sum(x**2 for x in self.engine.state.z[:4])**0.5),
        }

        self.state_history.append(record)

        # Trim history
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history:]

        # Emit history update
        self.socketio.emit('history_update', record)

    def run(self, host: str = None, port: int = None, debug: bool = False):
        """Run the web server."""
        host = host or self.config.web_host
        port = port or self.config.web_port

        print(f"\n{'=' * 50}")
        print("Child Mind AI — Web Interface")
        print(f"{'=' * 50}")
        print(f"Running at: http://{host}:{port}")
        print("Press Ctrl+C to stop")
        print(f"{'=' * 50}\n")

        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


def main():
    """Main entry point for web interface."""
    import argparse

    parser = argparse.ArgumentParser(description="Child Mind AI Web Interface")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    app = ChildMindWebApp()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
