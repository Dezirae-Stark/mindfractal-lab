"""
Child Mind AI — Core Rewrite System
MindFractal Lab

Safe self-modification capability with versioning and rollback.
All modifications require explicit user approval.
"""

import os
import shutil
import hashlib
import json
import difflib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

from ..config import ChildMindAIConfig
from ..permissions.manager import PermissionManager
from ..core.state import ActionType


@dataclass
class CoreVersion:
    """A version snapshot of core code."""
    version_id: str
    timestamp: str
    description: str
    file_hashes: Dict[str, str]
    changes_from_previous: List[str]
    author: str  # "system" or "ai"


@dataclass
class ProposedChange:
    """A proposed modification to core code."""
    file_path: str
    original_content: str
    new_content: str
    rationale: str
    impact_analysis: str
    confidence: float  # AI's confidence this change is beneficial


class CoreRewriteManager:
    """
    Manages safe self-modification of Child Mind AI core code.

    Protocol:
    1. AI proposes change with rationale
    2. System generates diff and impact analysis
    3. User reviews and provides explicit confirmation
    4. Current version is backed up
    5. Change is applied atomically
    6. Post-change verification runs
    7. Rollback available if needed
    """

    # Files that can be modified by AI
    MODIFIABLE_FILES = [
        "core/dynamics.py",
        "core/state.py",
        "learning/online.py",
        "memory/semantic.py",
        "config.py",
    ]

    # Files that should never be modified
    PROTECTED_FILES = [
        "permissions/manager.py",
        "permissions/classifier.py",
        "actions/core_rewrite.py",  # Self-protection
        "__init__.py",
    ]

    def __init__(
        self,
        config: Optional[ChildMindAIConfig] = None,
        permission_manager: Optional[PermissionManager] = None
    ):
        """Initialize core rewrite manager."""
        self.config = config or ChildMindAIConfig()
        self.permission_manager = permission_manager or PermissionManager(config=self.config)

        # Core package directory
        self.core_dir = Path(__file__).parent.parent

        # Versions directory
        self.versions_dir = self.config.base_dir / "core_versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        # Load version history
        self.versions: List[CoreVersion] = self._load_versions()

        # Create initial version if none exists
        if not self.versions:
            self._create_version("Initial version", "system")

    def _load_versions(self) -> List[CoreVersion]:
        """Load version history from disk."""
        versions_file = self.versions_dir / "versions.json"

        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    data = json.load(f)
                    return [CoreVersion(**v) for v in data]
            except:
                pass

        return []

    def _save_versions(self):
        """Save version history to disk."""
        versions_file = self.versions_dir / "versions.json"

        with open(versions_file, 'w') as f:
            json.dump([asdict(v) for v in self.versions], f, indent=2)

    def _create_version(self, description: str, author: str) -> CoreVersion:
        """Create a new version snapshot."""
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compute file hashes
        file_hashes = {}
        version_backup_dir = self.versions_dir / version_id

        version_backup_dir.mkdir(exist_ok=True)

        for rel_path in self.MODIFIABLE_FILES:
            file_path = self.core_dir / rel_path
            if file_path.exists():
                content = file_path.read_text()
                file_hashes[rel_path] = hashlib.sha256(content.encode()).hexdigest()

                # Backup file
                backup_path = version_backup_dir / rel_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(content)

        # Compute changes from previous version
        changes = []
        if self.versions:
            prev_hashes = self.versions[-1].file_hashes
            for rel_path, new_hash in file_hashes.items():
                old_hash = prev_hashes.get(rel_path)
                if old_hash != new_hash:
                    changes.append(f"Modified: {rel_path}")

        version = CoreVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            description=description,
            file_hashes=file_hashes,
            changes_from_previous=changes,
            author=author
        )

        self.versions.append(version)
        self._save_versions()

        return version

    def propose_change(
        self,
        file_rel_path: str,
        new_content: str,
        rationale: str,
        confidence: float = 0.5
    ) -> Tuple[bool, ProposedChange, str]:
        """
        Propose a change to core code.

        Args:
            file_rel_path: Relative path within child_mind_ai package
            new_content: Proposed new content
            rationale: AI's reason for the change
            confidence: AI's confidence level (0-1)

        Returns:
            (can_proceed, proposal, message) tuple
        """
        # Check if file is modifiable
        if file_rel_path in self.PROTECTED_FILES:
            return False, None, f"File '{file_rel_path}' is protected and cannot be modified."

        if file_rel_path not in self.MODIFIABLE_FILES:
            return False, None, f"File '{file_rel_path}' is not in the modifiable list."

        file_path = self.core_dir / file_rel_path

        # Get original content
        if file_path.exists():
            original_content = file_path.read_text()
        else:
            original_content = ""

        # Generate impact analysis
        impact_analysis = self._analyze_impact(file_rel_path, original_content, new_content)

        proposal = ProposedChange(
            file_path=file_rel_path,
            original_content=original_content,
            new_content=new_content,
            rationale=rationale,
            impact_analysis=impact_analysis,
            confidence=confidence
        )

        return True, proposal, "Proposal created successfully."

    def _analyze_impact(
        self,
        file_rel_path: str,
        original: str,
        new: str
    ) -> str:
        """Analyze the impact of a proposed change."""
        analysis = []

        # Basic diff statistics
        original_lines = original.split('\n')
        new_lines = new.split('\n')

        diff = list(difflib.unified_diff(original_lines, new_lines, lineterm=''))
        additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
        deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

        analysis.append(f"Lines added: {additions}")
        analysis.append(f"Lines removed: {deletions}")

        # Check for dangerous patterns
        dangerous_patterns = [
            ('os.system', 'Shell command execution'),
            ('subprocess', 'Process spawning'),
            ('eval(', 'Dynamic code execution'),
            ('exec(', 'Dynamic code execution'),
            ('open(', 'File operations'),
            ('import socket', 'Network access'),
            ('requests.', 'HTTP requests'),
        ]

        for pattern, description in dangerous_patterns:
            if pattern in new and pattern not in original:
                analysis.append(f"⚠️ ADDS: {description} ({pattern})")

        # Check for function/class changes
        import re

        original_funcs = set(re.findall(r'def (\w+)\(', original))
        new_funcs = set(re.findall(r'def (\w+)\(', new))

        added_funcs = new_funcs - original_funcs
        removed_funcs = original_funcs - new_funcs

        if added_funcs:
            analysis.append(f"Functions added: {', '.join(added_funcs)}")
        if removed_funcs:
            analysis.append(f"Functions removed: {', '.join(removed_funcs)}")

        return '\n'.join(analysis)

    def get_diff(self, proposal: ProposedChange) -> str:
        """Generate human-readable diff for a proposal."""
        original_lines = proposal.original_content.split('\n')
        new_lines = proposal.new_content.split('\n')

        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile=f"a/{proposal.file_path}",
            tofile=f"b/{proposal.file_path}",
            lineterm=''
        )

        return '\n'.join(diff)

    def apply_change(
        self,
        proposal: ProposedChange,
        user_confirmation: str
    ) -> Tuple[bool, str]:
        """
        Apply a proposed change after user confirmation.

        Args:
            proposal: The proposed change
            user_confirmation: User's typed confirmation

        Returns:
            (success, message) tuple
        """
        # Verify confirmation phrase
        expected_confirmation = "modify core"
        if user_confirmation.lower().strip() != expected_confirmation:
            return False, f"Confirmation phrase incorrect. Expected '{expected_confirmation}'."

        # Request explicit permission
        approved, request = self.permission_manager.check_permission(
            ActionType.CORE_MODIFY,
            f"Modify {proposal.file_path}",
            proposal.rationale
        )

        if not approved:
            return False, "Permission denied for core modification."

        # Create backup version
        backup_version = self._create_version(
            f"Pre-change backup: {proposal.file_path}",
            "system"
        )

        # Apply change
        try:
            file_path = self.core_dir / proposal.file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(proposal.new_content)

            # Verify change
            verification_passed, verification_msg = self._verify_change(proposal.file_path)

            if not verification_passed:
                # Rollback
                self.rollback(backup_version.version_id)
                return False, f"Verification failed: {verification_msg}. Change rolled back."

            # Create post-change version
            self._create_version(
                f"AI modification: {proposal.rationale[:50]}",
                "ai"
            )

            return True, f"Change applied successfully. Verification passed."

        except Exception as e:
            # Attempt rollback
            try:
                self.rollback(backup_version.version_id)
            except:
                pass
            return False, f"Error applying change: {e}. Attempted rollback."

    def _verify_change(self, file_rel_path: str) -> Tuple[bool, str]:
        """Verify a change is syntactically valid."""
        file_path = self.core_dir / file_rel_path

        try:
            # Try to compile the Python code
            content = file_path.read_text()
            compile(content, file_path, 'exec')
            return True, "Syntax check passed"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Verification error: {e}"

    def rollback(self, version_id: str) -> Tuple[bool, str]:
        """Rollback to a specific version."""
        # Find version
        version = None
        for v in self.versions:
            if v.version_id == version_id:
                version = v
                break

        if not version:
            return False, f"Version {version_id} not found."

        backup_dir = self.versions_dir / version_id

        if not backup_dir.exists():
            return False, f"Backup directory for version {version_id} not found."

        # Restore files
        for rel_path in self.MODIFIABLE_FILES:
            backup_file = backup_dir / rel_path
            target_file = self.core_dir / rel_path

            if backup_file.exists():
                shutil.copy2(backup_file, target_file)

        # Create rollback version record
        self._create_version(f"Rollback to {version_id}", "system")

        return True, f"Rolled back to version {version_id}"

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions."""
        return [
            {
                "id": v.version_id,
                "timestamp": v.timestamp,
                "description": v.description,
                "author": v.author,
                "changes": v.changes_from_previous
            }
            for v in self.versions
        ]

    def get_current_hash(self, file_rel_path: str) -> Optional[str]:
        """Get current hash of a file."""
        file_path = self.core_dir / file_rel_path
        if file_path.exists():
            content = file_path.read_text()
            return hashlib.sha256(content.encode()).hexdigest()
        return None

    def describe_capability(self) -> str:
        """Describe self-modification capability."""
        modifiable = ", ".join(self.MODIFIABLE_FILES)
        protected = ", ".join(self.PROTECTED_FILES)

        return (
            "I have the ability to modify my own core code, with your explicit permission.\n\n"
            f"**Modifiable files:** {modifiable}\n\n"
            f"**Protected files:** {protected}\n\n"
            "Any modification requires:\n"
            "1. Proposal with rationale\n"
            "2. Impact analysis review\n"
            "3. Your explicit typed confirmation ('modify core')\n"
            "4. Automatic backup before change\n"
            "5. Syntax verification after change\n"
            "6. Automatic rollback if verification fails\n\n"
            f"Current versions saved: {len(self.versions)}"
        )
