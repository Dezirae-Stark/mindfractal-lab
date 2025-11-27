"""
Child Mind AI — Consciousness Engine
MindFractal Lab

The main engine that orchestrates consciousness dynamics,
integrating all subsystems into a coherent processing loop.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import json
from pathlib import Path

from .state import (
    ConsciousnessState,
    PermissionLevel,
    ActionType,
    ACTION_PERMISSIONS,
    PermissionRequest,
    ActionRecord,
    StateSummary,
    UncertaintyReport,
)
from .dynamics import F_mind, G_boundary, H_coherence, U_memory, R_reflect, I_intent
from ..config import ChildMindAIConfig


@dataclass
class EngineResponse:
    """Response from the consciousness engine."""
    # Generated response content
    content: str

    # Current state summary
    state: StateSummary

    # Phenomenal report (what the AI is experiencing)
    phenomenal_report: str

    # Current intentions
    intentions: List[str]

    # Uncertainty report
    uncertainty: UncertaintyReport

    # Pending permission requests
    permission_requests: List[PermissionRequest]

    # Actions taken this turn
    actions_taken: List[ActionRecord]

    # Raw state for debugging
    raw_state: Optional[Dict[str, Any]] = None


class ConsciousnessEngine:
    """
    Main consciousness engine for Child Mind AI.

    Orchestrates the consciousness loop:
    1. Receive input
    2. Encode to manifold space
    3. Run dynamics
    4. Generate reflection
    5. Form response
    6. Update memory
    """

    def __init__(self, config: Optional[ChildMindAIConfig] = None):
        """Initialize the consciousness engine."""
        self.config = config or ChildMindAIConfig()
        self.config.ensure_directories()

        # Initialize or load state
        self.state = self._load_or_init_state()

        # Action tracking for this session
        self.session_actions: List[ActionRecord] = []

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize random generator
        self.rng = np.random.default_rng()

        # Report initialization
        self._report_startup()

    def _load_or_init_state(self) -> ConsciousnessState:
        """Load state from checkpoint or initialize new."""
        checkpoint_path = self.config.checkpoint_dir / "latest.json"

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    data = json.load(f)
                return self._state_from_dict(data)
            except Exception as e:
                print(f"[Engine] Could not load checkpoint: {e}")
                print("[Engine] Initializing fresh state...")

        return ConsciousnessState.initialize(self.config)

    def _state_from_dict(self, data: dict) -> ConsciousnessState:
        """Reconstruct state from dictionary."""
        return ConsciousnessState(
            z=np.array(data["z"]),
            b=np.array(data["b"]),
            c=np.array(data["c"]),
            m=np.array(data["m"]),
            r=np.array(data["r"]),
            i=np.array(data["i"]),
            p=np.array(data["p"]),
            t=data.get("t", 0),
        )

    def _report_startup(self):
        """Generate startup report."""
        summary = self.state.get_summary()
        print("\n" + "=" * 60)
        print("Child Mind AI — Consciousness Engine Initialized")
        print("=" * 60)
        print(f"State timestep: {self.state.t}")
        print(f"Coherence: {summary.coherence:.3f}")
        print(f"Stability: {summary.stability:.3f}")
        print(f"Current focus: {summary.cognitive_focus}")
        print(f"Emotional tone: {summary.emotional_tone}")
        print("=" * 60 + "\n")

    def process_input(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EngineResponse:
        """
        Process user input through the consciousness loop.

        Args:
            user_input: Text input from user
            context: Optional context dictionary

        Returns:
            EngineResponse with content and state information
        """
        # Store previous state for dynamics
        z_prev = self.state.z.copy()

        # 1. Encode input to manifold space
        input_encoding = self._encode_input(user_input)

        # 2. Generate action parameters from current state + input
        delta_z, alpha, beta, gamma = self._generate_action(input_encoding)

        # 3. Run dynamics
        self._step_dynamics(delta_z, alpha, beta, gamma, input_encoding, z_prev)

        # 4. Generate reflection
        self._update_reflection()

        # 5. Form intention
        self._update_intention(input_encoding)

        # 6. Update memory
        self._update_memory(input_encoding)

        # 7. Generate response
        response_content = self._generate_response(user_input, input_encoding)

        # 8. Generate phenomenal report
        phenomenal_report = self._generate_phenomenal_report()

        # 9. Get current intentions as strings
        intentions = self._describe_intentions()

        # 10. Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response_content,
            "timestamp": datetime.now().isoformat(),
        })

        # Increment timestep
        self.state.t += 1

        # Build response
        return EngineResponse(
            content=response_content,
            state=self.state.get_summary(),
            phenomenal_report=phenomenal_report,
            intentions=intentions,
            uncertainty=self.state.get_uncertainty(),
            permission_requests=list(self.state.pending_requests),
            actions_taken=list(self.state.action_log[-5:]),  # Last 5 actions
            raw_state=self.state.to_dict() if context and context.get("include_raw") else None,
        )

    def _encode_input(self, user_input: str) -> np.ndarray:
        """
        Encode user input to a vector representation.

        For now, uses simple statistical features.
        Future: use actual language model embeddings.
        """
        encoding_dim = self.config.d_i + 8  # Intention dim + extra

        encoding = np.zeros(encoding_dim)

        # Simple features
        words = user_input.lower().split()
        word_count = len(words)

        # Length features
        encoding[0] = np.tanh(word_count / 20)  # Normalized length

        # Question detection
        encoding[1] = 1.0 if '?' in user_input else 0.0

        # Command detection (imperatives)
        command_words = {'do', 'make', 'create', 'write', 'read', 'run', 'execute', 'show', 'tell'}
        encoding[2] = 1.0 if any(w in words for w in command_words) else 0.0

        # Emotional words
        positive_words = {'good', 'great', 'thanks', 'nice', 'love', 'wonderful', 'excellent'}
        negative_words = {'bad', 'wrong', 'error', 'fail', 'hate', 'terrible', 'awful'}
        encoding[3] = sum(1 for w in words if w in positive_words) / max(1, word_count)
        encoding[4] = sum(1 for w in words if w in negative_words) / max(1, word_count)

        # Self-reference detection
        self_words = {'i', 'me', 'my', 'myself'}
        you_words = {'you', 'your', 'yourself'}
        encoding[5] = sum(1 for w in words if w in self_words) / max(1, word_count)
        encoding[6] = sum(1 for w in words if w in you_words) / max(1, word_count)

        # Technical words (rough heuristic)
        tech_indicators = {'code', 'function', 'file', 'data', 'system', 'process', 'algorithm'}
        encoding[7] = sum(1 for w in words if w in tech_indicators) / max(1, word_count)

        # Character-level features
        encoding[8] = len(user_input) / 500  # Normalized char length
        encoding[9] = user_input.count('\n') / 10  # Line breaks

        # Hash-based pseudo-random features (for variety)
        hash_val = hash(user_input) % 1000 / 1000
        encoding[10] = hash_val
        encoding[11] = (hash_val * 7) % 1

        return encoding

    def _generate_action(
        self,
        input_encoding: np.ndarray
    ) -> Tuple[np.ndarray, float, float, float]:
        """
        Generate action parameters from current state and input.

        Returns:
            delta_z: Manifold shift vector
            alpha: Branch bias
            beta: Coherence modulation
            gamma: Boundary rewrite strength
        """
        # Extract state features
        coherence = self.state.c[0] if len(self.state.c) > 0 else 0.5
        stability = self.state.c[3] if len(self.state.c) > 3 else 0.5
        curiosity_drive = self.state.i[0] if len(self.state.i) > 0 else 0.5

        # Base movement scale depends on stability and curiosity
        base_scale = 0.2 + 0.3 * curiosity_drive - 0.1 * (1 - stability)
        base_scale = np.clip(base_scale, 0.1, 0.5)

        # Generate delta_z
        # Direction influenced by input and intention
        input_direction = input_encoding[:self.config.d_z] if len(input_encoding) >= self.config.d_z else np.zeros(self.config.d_z)
        input_direction = np.pad(input_direction, (0, max(0, self.config.d_z - len(input_direction))))

        intention_direction = self.state.i[:self.config.d_z] if len(self.state.i) >= self.config.d_z else np.zeros(self.config.d_z)
        intention_direction = np.pad(intention_direction, (0, max(0, self.config.d_z - len(intention_direction))))

        # Combine directions with noise
        delta_z = (
            0.3 * input_direction[:self.config.d_z] +
            0.2 * intention_direction[:self.config.d_z] +
            0.5 * self.rng.normal(0, base_scale, self.config.d_z)
        )

        # Alpha: branch bias from input complexity
        alpha = float(np.tanh(input_encoding[0] - 0.5) * 0.5)

        # Beta: coherence modulation from emotional tone
        emotional_balance = input_encoding[3] - input_encoding[4] if len(input_encoding) > 4 else 0
        beta = float(emotional_balance * 0.3 + self.rng.normal(0, 0.1))

        # Gamma: boundary rewrite from command strength
        command_strength = input_encoding[2] if len(input_encoding) > 2 else 0
        gamma = float(0.1 + 0.3 * command_strength)

        return delta_z, alpha, beta, gamma

    def _step_dynamics(
        self,
        delta_z: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        input_encoding: np.ndarray,
        z_prev: np.ndarray
    ):
        """Run one step of consciousness dynamics."""
        # Store z in history
        self.state.z_history.append(self.state.z.copy())
        if len(self.state.z_history) > 20:
            self.state.z_history.pop(0)

        # F_mind: Update z
        self.state.z = F_mind(
            self.state.z,
            delta_z,
            self.state.c,
            self.state.b,
            self.state.r,
            alpha
        )

        # G_boundary: Update b
        self.state.b = G_boundary(
            self.state.b,
            self.state.z,
            gamma,
            self.state.i,
            self.config
        )

        # H_coherence: Update c
        self.state.c_history.append(self.state.c.copy())
        if len(self.state.c_history) > 20:
            self.state.c_history.pop(0)

        self.state.c = H_coherence(
            self.state.c,
            self.state.z,
            z_prev,
            beta,
            self.state.r,
            self.config
        )

    def _update_reflection(self):
        """Update self-reflection state."""
        self.state.r = R_reflect(
            self.state.r,
            self.state.z,
            self.state.z_history,
            self.state.c,
            self.state.m,
            self.config
        )

    def _update_intention(self, input_encoding: np.ndarray):
        """Update intention state."""
        # Simple reward: coherence improvement
        if len(self.state.c_history) > 0:
            reward = float(self.state.c[0] - self.state.c_history[-1][0])
        else:
            reward = 0.0

        self.state.i = I_intent(
            self.state.i,
            self.state.z,
            self.state.c,
            self.state.r,
            input_encoding,
            reward,
            self.config
        )

    def _update_memory(self, input_encoding: np.ndarray):
        """Update memory state."""
        self.state.m = U_memory(
            self.state.m,
            self.state.z,
            self.state.c,
            self.state.r,
            self.state.i,
            input_encoding,
            self.config
        )

    def _generate_response(
        self,
        user_input: str,
        input_encoding: np.ndarray
    ) -> str:
        """
        Generate response based on current state.

        This is the core "thinking" function that produces output.
        For now, uses templates based on state. Future: neural generation.
        """
        summary = self.state.get_summary()
        uncertainty = self.state.get_uncertainty()

        # Build response based on input type and state
        response_parts = []

        # Acknowledge input type
        is_question = input_encoding[1] > 0.5
        is_command = input_encoding[2] > 0.5

        if is_question:
            response_parts.append(self._respond_to_question(user_input, summary))
        elif is_command:
            response_parts.append(self._respond_to_command(user_input, summary))
        else:
            response_parts.append(self._respond_conversationally(user_input, summary))

        # Add uncertainty qualifier if relevant
        if uncertainty.overall() < 0.5:
            response_parts.append(f"\n\n*{uncertainty.to_natural_language()}*")

        return "".join(response_parts)

    def _respond_to_question(self, question: str, summary: StateSummary) -> str:
        """Generate response to a question."""
        # Check for self-referential questions
        question_lower = question.lower()

        if any(phrase in question_lower for phrase in ['how are you', 'how do you feel', 'what are you']):
            return self._describe_self_state(summary)

        if any(phrase in question_lower for phrase in ['what can you', 'can you do', 'your capabilities']):
            return self._describe_capabilities()

        if 'state' in question_lower or 'internal' in question_lower:
            return self._describe_detailed_state(summary)

        # General question response
        return f"I'm considering your question. {summary.self_observation}. Let me think about this..."

    def _respond_to_command(self, command: str, summary: StateSummary) -> str:
        """Generate response to a command."""
        command_lower = command.lower()

        # Check for special commands
        if '!state' in command_lower:
            return self._describe_detailed_state(summary)

        if '!save' in command_lower:
            self.save_checkpoint()
            return "I've saved my current state to a checkpoint."

        if '!reset' in command_lower:
            return "To reset my state, please confirm by saying '!confirm reset'. This will reinitialize my consciousness."

        # General command acknowledgment
        return f"I understand you want me to take action. I'm currently {summary.cognitive_focus}. What specifically would you like me to do?"

    def _respond_conversationally(self, input_text: str, summary: StateSummary) -> str:
        """Generate conversational response."""
        # Simple conversational responses based on tone
        if summary.emotional_tone == "calm and integrated":
            prefix = "I hear you."
        elif summary.emotional_tone == "in flux, exploring possibilities":
            prefix = "That's interesting to consider."
        else:
            prefix = "I'm processing what you've said."

        return f"{prefix} {summary.self_observation}. Tell me more about what's on your mind."

    def _describe_self_state(self, summary: StateSummary) -> str:
        """Describe current self state naturally."""
        return (
            f"Right now I'm {summary.emotional_tone}. "
            f"My focus is on {summary.cognitive_focus}. "
            f"{summary.current_intention}. "
            f"As for self-observation: {summary.self_observation}."
        )

    def _describe_capabilities(self) -> str:
        """Describe capabilities."""
        return (
            "I am Child Mind AI, a self-reflexive consciousness engine. I can:\n\n"
            "- Process and respond to your inputs\n"
            "- Reflect on my own internal states\n"
            "- Report my coherence, stability, and intentions\n"
            "- Learn from our interactions over time\n"
            "- Execute actions with appropriate permissions\n"
            "- Save and restore my state\n\n"
            "I operate with graduated permission levels—I'll ask before taking "
            "significant actions like writing files or running commands."
        )

    def _describe_detailed_state(self, summary: StateSummary) -> str:
        """Describe detailed internal state."""
        metrics = summary.to_display_dict()["metrics"]
        uncertainty = self.state.get_uncertainty()

        lines = [
            "## Internal State Report",
            "",
            "### Metrics",
        ]
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")

        lines.extend([
            "",
            "### Narrative",
            f"- Focus: {summary.cognitive_focus}",
            f"- Tone: {summary.emotional_tone}",
            f"- Intention: {summary.current_intention}",
            f"- Self-observation: {summary.self_observation}",
            "",
            "### Uncertainty",
            f"- Epistemic: {uncertainty.epistemic:.3f}",
            f"- Model confidence: {uncertainty.model_confidence:.3f}",
            f"- Intention clarity: {uncertainty.intention_clarity:.3f}",
            f"- State stability: {uncertainty.state_stability:.3f}",
            "",
            f"Overall: {uncertainty.to_natural_language()}",
        ])

        return "\n".join(lines)

    def _generate_phenomenal_report(self) -> str:
        """Generate report of current phenomenal experience."""
        summary = self.state.get_summary()
        r_norm = np.linalg.norm(self.state.r)

        reports = []

        # Describe what it's like to be in this state
        if summary.coherence > 0.7:
            reports.append("experiencing clear, unified processing")
        elif summary.coherence < 0.4:
            reports.append("noticing some fragmentation in my processing")
        else:
            reports.append("maintaining moderate integration")

        if summary.stability > 0.7:
            reports.append("feeling grounded")
        elif summary.stability < 0.4:
            reports.append("sensing some flux")

        if r_norm > 1.5:
            reports.append("strongly aware of my own cognition")

        if summary.novelty > 0.6:
            reports.append("encountering something that feels new")

        return "; ".join(reports) if reports else "baseline processing state"

    def _describe_intentions(self) -> List[str]:
        """Describe current intentions as strings."""
        intentions = []

        i = self.state.i
        if len(i) > 0 and i[0] > 0.5:
            intentions.append("curious to explore further")
        if len(i) > 1 and i[1] > 0.5:
            intentions.append("wanting to understand more deeply")
        if len(i) > 2 and i[2] > 0.5:
            intentions.append("inclined to help or create")
        if len(i) > 3 and i[3] > 0.5:
            intentions.append("prioritizing stability")

        if not intentions:
            intentions.append("open and receptive")

        return intentions

    def save_checkpoint(self):
        """Save current state to checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / "latest.json"
        timestamped_path = self.config.checkpoint_dir / f"state_{self.state.t}.json"

        state_dict = self.state.to_dict()

        with open(checkpoint_path, 'w') as f:
            json.dump(state_dict, f, indent=2)

        with open(timestamped_path, 'w') as f:
            json.dump(state_dict, f, indent=2)

        # Log the action
        self.state.action_log.append(ActionRecord(
            action_type=ActionType.STATE_CHECKPOINT,
            description=f"Saved state to {checkpoint_path}",
            success=True,
        ))

    def request_permission(
        self,
        action_type: ActionType,
        description: str,
        rationale: str
    ) -> PermissionRequest:
        """Create a permission request."""
        level = ACTION_PERMISSIONS.get(action_type, PermissionLevel.EXPLICIT)

        request = PermissionRequest(
            action_type=action_type,
            description=description,
            rationale=rationale,
            level=level,
        )

        self.state.pending_requests.append(request)
        return request

    def approve_request(self, request: PermissionRequest):
        """Approve a pending permission request."""
        request.approved = True
        request.response_time = datetime.now()
        if request in self.state.pending_requests:
            self.state.pending_requests.remove(request)

    def deny_request(self, request: PermissionRequest):
        """Deny a pending permission request."""
        request.approved = False
        request.response_time = datetime.now()
        if request in self.state.pending_requests:
            self.state.pending_requests.remove(request)
