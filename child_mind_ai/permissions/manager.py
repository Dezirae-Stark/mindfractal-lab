"""
Child Mind AI — Permission Manager
MindFractal Lab

Central manager for action permissions and approval flow.
"""

from typing import Optional, Callable, Tuple
from datetime import datetime
from ..core.state import (
    ActionType,
    PermissionLevel,
    PermissionRequest,
    ActionRecord,
)
from ..config import ChildMindAIConfig
from .classifier import ActionClassifier
from .audit import AuditLog


class PermissionManager:
    """
    Manages permission requests and approval flow.

    Handles graduated permission levels:
    - AUTO: Execute immediately
    - NOTIFY: Execute and notify user
    - ASK: Wait for yes/no
    - EXPLICIT: Require typed confirmation
    """

    def __init__(
        self,
        config: Optional[ChildMindAIConfig] = None,
        prompt_callback: Optional[Callable[[str], str]] = None
    ):
        """
        Initialize permission manager.

        Args:
            config: Configuration object
            prompt_callback: Function to prompt user for input.
                             Receives prompt string, returns user response.
        """
        self.config = config or ChildMindAIConfig()
        self.classifier = ActionClassifier(self.config)
        self.audit = AuditLog(self.config)

        # Callback for user prompts
        self.prompt_callback = prompt_callback or self._default_prompt

        # Pending requests
        self.pending_requests: list[PermissionRequest] = []

    def _default_prompt(self, prompt: str) -> str:
        """Default prompt using input()."""
        return input(prompt)

    def check_permission(
        self,
        action_type: ActionType,
        description: str,
        rationale: str = ""
    ) -> Tuple[bool, Optional[PermissionRequest]]:
        """
        Check if an action is permitted.

        Returns:
            (approved, request) tuple.
            If approved is True, action can proceed.
            Request contains details for logging.
        """
        level = self.classifier.get_permission_level(action_type)

        request = PermissionRequest(
            action_type=action_type,
            description=description,
            rationale=rationale,
            level=level,
        )

        if level == PermissionLevel.AUTO:
            request.approved = True
            request.response_time = datetime.now()
            self.audit.log_permission_request(request)
            return True, request

        elif level == PermissionLevel.NOTIFY:
            # Notify user but proceed
            self._notify_user(request)
            request.approved = True
            request.response_time = datetime.now()
            self.audit.log_permission_request(request)
            return True, request

        elif level == PermissionLevel.ASK:
            # Ask for yes/no
            approved = self._ask_permission(request)
            request.approved = approved
            request.response_time = datetime.now()
            self.audit.log_permission_request(request)
            return approved, request

        elif level == PermissionLevel.EXPLICIT:
            # Require explicit confirmation
            approved = self._explicit_confirmation(request)
            request.approved = approved
            request.response_time = datetime.now()
            self.audit.log_permission_request(request)
            return approved, request

        # Default deny
        request.approved = False
        request.response_time = datetime.now()
        self.audit.log_permission_request(request)
        return False, request

    def _notify_user(self, request: PermissionRequest):
        """Notify user of an action being taken."""
        action_desc = self.classifier.describe_action(request.action_type)
        print(f"\n[Notice] {action_desc}: {request.description}")
        if request.rationale:
            print(f"         Reason: {request.rationale}")

    def _ask_permission(self, request: PermissionRequest) -> bool:
        """Ask user for yes/no permission."""
        action_desc = self.classifier.describe_action(request.action_type)

        prompt = (
            f"\n{'=' * 50}\n"
            f"Permission Request: {action_desc}\n"
            f"{'=' * 50}\n"
            f"Action: {request.description}\n"
        )
        if request.rationale:
            prompt += f"Reason: {request.rationale}\n"
        prompt += f"\nAllow this action? [y/N]: "

        response = self.prompt_callback(prompt).strip().lower()
        return response in ('y', 'yes')

    def _explicit_confirmation(self, request: PermissionRequest) -> bool:
        """Require explicit typed confirmation."""
        action_desc = self.classifier.describe_action(request.action_type)
        level_desc = self.classifier.describe_permission_level(request.level)

        # Generate confirmation phrase
        confirm_phrase = self._generate_confirmation_phrase(request)

        prompt = (
            f"\n{'!' * 50}\n"
            f"EXPLICIT PERMISSION REQUIRED\n"
            f"{'!' * 50}\n"
            f"Action Type: {action_desc}\n"
            f"Level: {level_desc}\n"
            f"\nAction: {request.description}\n"
        )
        if request.rationale:
            prompt += f"Reason: {request.rationale}\n"

        prompt += (
            f"\n⚠️  This action requires explicit confirmation.\n"
            f"    Type '{confirm_phrase}' to proceed, or anything else to deny: "
        )

        response = self.prompt_callback(prompt).strip()

        # Log user response
        self.audit.log_action(
            action_type=request.action_type,
            description=request.description,
            permission_level=request.level,
            user_response=response,
        )

        return response.lower() == confirm_phrase.lower()

    def _generate_confirmation_phrase(self, request: PermissionRequest) -> str:
        """Generate a confirmation phrase for explicit requests."""
        phrases = {
            ActionType.NETWORK_REQUEST: "allow network",
            ActionType.SHELL_RISKY: "run command",
            ActionType.CORE_MODIFY: "modify core",
            ActionType.SPAWN_PROCESS: "spawn process",
            ActionType.FILE_WRITE_EXTERNAL: "write external",
        }
        return phrases.get(request.action_type, "confirm action")

    def record_action(
        self,
        action_type: ActionType,
        description: str,
        success: bool,
        result: Optional[str] = None,
        error: Optional[str] = None,
        request: Optional[PermissionRequest] = None
    ) -> ActionRecord:
        """Record a completed action."""
        record = ActionRecord(
            action_type=action_type,
            description=description,
            success=success,
            result=result,
            error=error,
            permission_request=request,
        )

        self.audit.log_action_record(record)
        return record

    def get_audit_summary(self) -> dict:
        """Get audit summary for current session."""
        return self.audit.get_summary()

    # Convenience methods for common action types

    def check_file_read(self, path: str) -> Tuple[bool, Optional[PermissionRequest]]:
        """Check permission to read a file."""
        action_type = self.classifier.classify_file_operation(path, 'read')
        return self.check_permission(
            action_type,
            f"Read file: {path}",
            "Gathering information"
        )

    def check_file_write(self, path: str, rationale: str = "") -> Tuple[bool, Optional[PermissionRequest]]:
        """Check permission to write a file."""
        # Check if this is core modification
        if self.classifier.is_core_modification(path):
            action_type = ActionType.CORE_MODIFY
        else:
            action_type = self.classifier.classify_file_operation(path, 'write')

        return self.check_permission(
            action_type,
            f"Write file: {path}",
            rationale
        )

    def check_shell_command(self, command: str, rationale: str = "") -> Tuple[bool, Optional[PermissionRequest]]:
        """Check permission to run a shell command."""
        action_type = self.classifier.classify_shell_command(command)
        return self.check_permission(
            action_type,
            f"Run command: {command}",
            rationale
        )

    def check_network_request(self, url: str, method: str = "GET", rationale: str = "") -> Tuple[bool, Optional[PermissionRequest]]:
        """Check permission for network request."""
        action_type = ActionType.NETWORK_REQUEST
        return self.check_permission(
            action_type,
            f"{method} request to: {url}",
            rationale
        )

    def check_core_modify(self, file_path: str, change_description: str) -> Tuple[bool, Optional[PermissionRequest]]:
        """Check permission to modify core code."""
        return self.check_permission(
            ActionType.CORE_MODIFY,
            f"Modify core file: {file_path}",
            change_description
        )
