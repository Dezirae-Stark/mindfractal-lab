"""
Child Mind AI — Configuration
MindFractal Lab

Central configuration for the consciousness engine.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os


@dataclass
class ChildMindAIConfig:
    """Configuration for Child Mind AI."""

    # === State Space Dimensions ===
    d_z: int = 64           # Core manifold dimension
    d_b: int = 32           # Boundary grid size (d_b x d_b)
    d_c: int = 16           # Coherence vector dimension
    d_m: int = 128          # Memory summary dimension
    d_r: int = 32           # Reflection state dimension
    d_i: int = 16           # Intention vector dimension
    d_p: int = 8            # Permission state dimension

    # === Dynamics Parameters ===
    coherence_target: float = 0.7
    memory_decay: float = 0.95
    reflection_depth: int = 3
    boundary_diffusion: float = 0.1

    # === Learning Parameters ===
    learning_rate: float = 0.001
    batch_size: int = 32
    episode_buffer_size: int = 1000
    update_frequency: int = 10

    # === Permission Settings ===
    auto_approve_reads: bool = True
    notify_on_sandbox_writes: bool = True
    require_explicit_for_network: bool = True
    require_explicit_for_shell: bool = True
    require_explicit_for_core_modify: bool = True

    # === Paths ===
    base_dir: Path = field(default_factory=lambda: Path.home() / ".child_mind_ai")
    sandbox_dir: Optional[Path] = None
    memory_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    audit_dir: Optional[Path] = None

    # === Interface Settings ===
    cli_show_state: bool = True
    cli_show_uncertainty: bool = True
    cli_state_update_interval: float = 0.5
    web_port: int = 8765
    web_host: str = "127.0.0.1"

    # === Backend Selection ===
    # Options: "numpy", "torch", "onnx", "auto"
    compute_backend: str = "auto"
    use_gpu: bool = False

    # === Safety Settings ===
    max_actions_per_turn: int = 10
    max_shell_command_length: int = 1000
    blocked_commands: List[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", "dd if=", ":(){:|:&};:",
        "chmod -R 777 /", "chown -R", "> /dev/sd",
    ])
    allowed_network_hosts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize derived paths."""
        self.base_dir = Path(self.base_dir)

        if self.sandbox_dir is None:
            self.sandbox_dir = self.base_dir / "sandbox"
        else:
            self.sandbox_dir = Path(self.sandbox_dir)

        if self.memory_dir is None:
            self.memory_dir = self.base_dir / "memory"
        else:
            self.memory_dir = Path(self.memory_dir)

        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.base_dir / "checkpoints"
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)

        if self.audit_dir is None:
            self.audit_dir = self.base_dir / "audit"
        else:
            self.audit_dir = Path(self.audit_dir)

    def ensure_directories(self):
        """Create all necessary directories."""
        for dir_path in [
            self.base_dir,
            self.sandbox_dir,
            self.memory_dir,
            self.checkpoint_dir,
            self.audit_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_state_dims(self) -> dict:
        """Return dictionary of all state dimensions."""
        return {
            "z": self.d_z,
            "b": (self.d_b, self.d_b),
            "c": self.d_c,
            "m": self.d_m,
            "r": self.d_r,
            "i": self.d_i,
            "p": self.d_p,
        }

    @classmethod
    def from_env(cls) -> "ChildMindAIConfig":
        """Create config from environment variables."""
        config = cls()

        # Override from environment
        if base := os.environ.get("CHILD_MIND_BASE_DIR"):
            config.base_dir = Path(base)

        if backend := os.environ.get("CHILD_MIND_BACKEND"):
            config.compute_backend = backend

        if os.environ.get("CHILD_MIND_USE_GPU", "").lower() == "true":
            config.use_gpu = True

        config.__post_init__()
        return config


# Default global config
DEFAULT_CONFIG = ChildMindAIConfig()
