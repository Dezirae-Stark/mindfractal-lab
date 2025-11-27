"""
Child Mind AI — Sandboxed Executor
MindFractal Lab

Safe execution environment for actions.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, Tuple
from ..config import ChildMindAIConfig
from ..permissions.manager import PermissionManager
from ..core.state import ActionType, ActionRecord


class SandboxedExecutor:
    """
    Sandboxed execution environment for Child Mind AI actions.

    All file operations are restricted to sandbox directory by default.
    Shell commands require permission based on risk classification.
    """

    def __init__(
        self,
        config: Optional[ChildMindAIConfig] = None,
        permission_manager: Optional[PermissionManager] = None
    ):
        """Initialize executor."""
        self.config = config or ChildMindAIConfig()
        self.config.ensure_directories()
        self.permissions = permission_manager or PermissionManager(config=self.config)

    def read_file(self, path: str) -> Tuple[bool, str, Optional[ActionRecord]]:
        """
        Read a file with permission check.

        Returns:
            (success, content_or_error, action_record)
        """
        approved, request = self.permissions.check_file_read(path)

        if not approved:
            record = self.permissions.record_action(
                ActionType.FILE_READ,
                f"Read {path}",
                success=False,
                error="Permission denied",
                request=request
            )
            return False, "Permission denied", record

        try:
            with open(path, 'r') as f:
                content = f.read()

            record = self.permissions.record_action(
                ActionType.FILE_READ,
                f"Read {path}",
                success=True,
                result=f"Read {len(content)} characters",
                request=request
            )
            return True, content, record

        except Exception as e:
            record = self.permissions.record_action(
                ActionType.FILE_READ,
                f"Read {path}",
                success=False,
                error=str(e),
                request=request
            )
            return False, str(e), record

    def write_file(
        self,
        path: str,
        content: str,
        rationale: str = ""
    ) -> Tuple[bool, str, Optional[ActionRecord]]:
        """
        Write a file with permission check.

        Returns:
            (success, message, action_record)
        """
        approved, request = self.permissions.check_file_write(path, rationale)

        if not approved:
            record = self.permissions.record_action(
                ActionType.FILE_WRITE_EXTERNAL,
                f"Write {path}",
                success=False,
                error="Permission denied",
                request=request
            )
            return False, "Permission denied", record

        try:
            # Ensure parent directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                f.write(content)

            record = self.permissions.record_action(
                request.action_type if request else ActionType.FILE_WRITE_EXTERNAL,
                f"Write {path}",
                success=True,
                result=f"Wrote {len(content)} characters",
                request=request
            )
            return True, f"Wrote {len(content)} characters to {path}", record

        except Exception as e:
            record = self.permissions.record_action(
                ActionType.FILE_WRITE_EXTERNAL,
                f"Write {path}",
                success=False,
                error=str(e),
                request=request
            )
            return False, str(e), record

    def run_command(
        self,
        command: str,
        rationale: str = "",
        timeout: int = 30
    ) -> Tuple[bool, str, Optional[ActionRecord]]:
        """
        Run a shell command with permission check.

        Returns:
            (success, output_or_error, action_record)
        """
        approved, request = self.permissions.check_shell_command(command, rationale)

        if not approved:
            record = self.permissions.record_action(
                ActionType.SHELL_RISKY,
                f"Run: {command}",
                success=False,
                error="Permission denied",
                request=request
            )
            return False, "Permission denied", record

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.config.sandbox_dir)  # Run in sandbox by default
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            success = result.returncode == 0

            record = self.permissions.record_action(
                request.action_type if request else ActionType.SHELL_SAFE,
                f"Run: {command}",
                success=success,
                result=output[:500] if output else "No output",
                error=result.stderr if not success else None,
                request=request
            )
            return success, output, record

        except subprocess.TimeoutExpired:
            record = self.permissions.record_action(
                ActionType.SHELL_RISKY,
                f"Run: {command}",
                success=False,
                error=f"Command timed out after {timeout}s",
                request=request
            )
            return False, f"Command timed out after {timeout}s", record

        except Exception as e:
            record = self.permissions.record_action(
                ActionType.SHELL_RISKY,
                f"Run: {command}",
                success=False,
                error=str(e),
                request=request
            )
            return False, str(e), record

    def sandbox_path(self, relative_path: str) -> Path:
        """Get absolute path within sandbox."""
        return self.config.sandbox_dir / relative_path

    def list_sandbox(self) -> list:
        """List files in sandbox directory."""
        return list(self.config.sandbox_dir.glob("**/*"))
