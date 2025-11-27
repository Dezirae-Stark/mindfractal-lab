"""
Child Mind AI — Audit Log
MindFractal Lab

Logging and audit trail for all actions.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from ..core.state import ActionType, PermissionLevel, ActionRecord, PermissionRequest
from ..config import ChildMindAIConfig


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    action_type: str
    description: str
    permission_level: str
    approved: Optional[bool]
    success: Optional[bool]
    result: Optional[str]
    error: Optional[str]
    user_response: Optional[str]
    session_id: str


class AuditLog:
    """Audit logging system for action tracking."""

    def __init__(self, config: Optional[ChildMindAIConfig] = None):
        """Initialize audit log."""
        self.config = config or ChildMindAIConfig()
        self.config.ensure_directories()

        # Generate session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Current session entries (in-memory)
        self.entries: List[AuditEntry] = []

        # Log file path
        self.log_file = self.config.audit_dir / f"audit_{self.session_id}.jsonl"

    def log_action(
        self,
        action_type: ActionType,
        description: str,
        permission_level: PermissionLevel,
        approved: Optional[bool] = None,
        success: Optional[bool] = None,
        result: Optional[str] = None,
        error: Optional[str] = None,
        user_response: Optional[str] = None
    ):
        """Log an action."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            action_type=action_type.name,
            description=description,
            permission_level=permission_level.name,
            approved=approved,
            success=success,
            result=result[:500] if result else None,  # Truncate long results
            error=error,
            user_response=user_response,
            session_id=self.session_id,
        )

        self.entries.append(entry)
        self._write_entry(entry)

    def log_permission_request(self, request: PermissionRequest):
        """Log a permission request."""
        self.log_action(
            action_type=request.action_type,
            description=request.description,
            permission_level=request.level,
            approved=request.approved,
        )

    def log_action_record(self, record: ActionRecord):
        """Log an action record."""
        permission_level = PermissionLevel.AUTO
        if record.permission_request:
            permission_level = record.permission_request.level

        self.log_action(
            action_type=record.action_type,
            description=record.description,
            permission_level=permission_level,
            approved=True if record.permission_request is None else record.permission_request.approved,
            success=record.success,
            result=record.result,
            error=record.error,
        )

    def _write_entry(self, entry: AuditEntry):
        """Write entry to log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(asdict(entry)) + '\n')
        except Exception as e:
            print(f"[Audit] Warning: Could not write to audit log: {e}")

    def get_session_entries(self) -> List[AuditEntry]:
        """Get all entries for current session."""
        return list(self.entries)

    def get_entries_by_type(self, action_type: ActionType) -> List[AuditEntry]:
        """Get entries filtered by action type."""
        return [e for e in self.entries if e.action_type == action_type.name]

    def get_denied_actions(self) -> List[AuditEntry]:
        """Get all denied action requests."""
        return [e for e in self.entries if e.approved is False]

    def get_failed_actions(self) -> List[AuditEntry]:
        """Get all failed actions."""
        return [e for e in self.entries if e.success is False]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        total = len(self.entries)
        by_type = {}
        approved_count = 0
        denied_count = 0
        success_count = 0
        failed_count = 0

        for entry in self.entries:
            by_type[entry.action_type] = by_type.get(entry.action_type, 0) + 1
            if entry.approved is True:
                approved_count += 1
            elif entry.approved is False:
                denied_count += 1
            if entry.success is True:
                success_count += 1
            elif entry.success is False:
                failed_count += 1

        return {
            "session_id": self.session_id,
            "total_actions": total,
            "by_type": by_type,
            "approved": approved_count,
            "denied": denied_count,
            "successful": success_count,
            "failed": failed_count,
        }

    def load_session(self, session_id: str) -> List[AuditEntry]:
        """Load entries from a previous session."""
        log_file = self.config.audit_dir / f"audit_{session_id}.jsonl"

        if not log_file.exists():
            return []

        entries = []
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entries.append(AuditEntry(**data))

        return entries

    def list_sessions(self) -> List[str]:
        """List all available audit sessions."""
        sessions = []
        for log_file in self.config.audit_dir.glob("audit_*.jsonl"):
            session_id = log_file.stem.replace("audit_", "")
            sessions.append(session_id)
        return sorted(sessions, reverse=True)
