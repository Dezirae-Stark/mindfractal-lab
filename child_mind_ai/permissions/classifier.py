"""
Child Mind AI — Action Classifier
MindFractal Lab

Classifies actions to determine required permission level.
"""

import re
from pathlib import Path
from typing import Optional, List
from ..core.state import ActionType, PermissionLevel, ACTION_PERMISSIONS
from ..config import ChildMindAIConfig


class ActionClassifier:
    """Classifies actions and determines permission requirements."""

    # Patterns for risky shell commands
    RISKY_PATTERNS = [
        r'rm\s+-rf',
        r'rm\s+-r',
        r'sudo\s+',
        r'chmod\s+-R',
        r'chown\s+-R',
        r'>\s*/dev/',
        r'mkfs',
        r'dd\s+if=',
        r':\(\)\{',  # Fork bomb
        r'wget.*\|.*sh',
        r'curl.*\|.*sh',
        r'eval\s+',
        r'exec\s+',
    ]

    # Safe read-only commands
    SAFE_COMMANDS = [
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'which', 'whereis',
        'pwd', 'whoami', 'date', 'cal', 'echo', 'printf', 'wc',
        'file', 'stat', 'du', 'df', 'free', 'uptime', 'uname',
        'python3 --version', 'pip --version', 'git status', 'git log',
        'git diff', 'git branch', 'env', 'printenv',
    ]

    def __init__(self, config: Optional[ChildMindAIConfig] = None):
        """Initialize classifier."""
        self.config = config or ChildMindAIConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns."""
        self.risky_regex = [re.compile(p, re.IGNORECASE) for p in self.RISKY_PATTERNS]

        # Add config-specified blocked commands
        for cmd in self.config.blocked_commands:
            try:
                self.risky_regex.append(re.compile(re.escape(cmd), re.IGNORECASE))
            except re.error:
                pass

    def classify_file_operation(
        self,
        path: str,
        operation: str  # 'read', 'write', 'delete'
    ) -> ActionType:
        """Classify a file operation."""
        path_obj = Path(path).resolve()
        sandbox_path = self.config.sandbox_dir.resolve()

        if operation == 'read':
            return ActionType.FILE_READ

        elif operation in ('write', 'delete'):
            # Check if within sandbox
            try:
                path_obj.relative_to(sandbox_path)
                return ActionType.FILE_WRITE_SANDBOX
            except ValueError:
                return ActionType.FILE_WRITE_EXTERNAL

        return ActionType.FILE_WRITE_EXTERNAL

    def classify_shell_command(self, command: str) -> ActionType:
        """Classify a shell command."""
        command_stripped = command.strip()

        # Check for risky patterns
        for pattern in self.risky_regex:
            if pattern.search(command_stripped):
                return ActionType.SHELL_RISKY

        # Check for safe commands
        for safe_cmd in self.SAFE_COMMANDS:
            if command_stripped.startswith(safe_cmd):
                return ActionType.SHELL_SAFE

        # Default to risky for unknown commands
        return ActionType.SHELL_RISKY

    def classify_network_request(
        self,
        url: str,
        method: str = 'GET'
    ) -> ActionType:
        """Classify a network request."""
        # All network requests are explicit by default
        return ActionType.NETWORK_REQUEST

    def is_core_modification(self, file_path: str) -> bool:
        """Check if a file operation would modify core code."""
        path_obj = Path(file_path).resolve()
        core_paths = [
            Path(__file__).parent.parent.resolve(),  # child_mind_ai package
        ]

        for core_path in core_paths:
            try:
                path_obj.relative_to(core_path)
                return True
            except ValueError:
                continue

        return False

    def get_permission_level(self, action_type: ActionType) -> PermissionLevel:
        """Get permission level for an action type."""
        return ACTION_PERMISSIONS.get(action_type, PermissionLevel.EXPLICIT)

    def describe_action(self, action_type: ActionType) -> str:
        """Get human-readable description of action type."""
        descriptions = {
            ActionType.INTERNAL_COMPUTE: "Internal computation",
            ActionType.FILE_READ: "Read file",
            ActionType.FILE_WRITE_SANDBOX: "Write file (sandbox)",
            ActionType.FILE_WRITE_EXTERNAL: "Write file (external)",
            ActionType.SHELL_SAFE: "Run safe shell command",
            ActionType.SHELL_RISKY: "Run potentially risky shell command",
            ActionType.NETWORK_REQUEST: "Make network request",
            ActionType.CORE_MODIFY: "Modify core code",
            ActionType.SPAWN_PROCESS: "Spawn new process",
            ActionType.MEMORY_UPDATE: "Update memory",
            ActionType.STATE_CHECKPOINT: "Save state checkpoint",
        }
        return descriptions.get(action_type, "Unknown action")

    def describe_permission_level(self, level: PermissionLevel) -> str:
        """Get human-readable description of permission level."""
        descriptions = {
            PermissionLevel.AUTO: "Automatic (no confirmation needed)",
            PermissionLevel.NOTIFY: "Notify (will proceed unless stopped)",
            PermissionLevel.ASK: "Ask (requires yes/no response)",
            PermissionLevel.EXPLICIT: "Explicit (requires typed confirmation)",
        }
        return descriptions.get(level, "Unknown level")
