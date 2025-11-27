"""
Child Mind AI — Command Line Interface
MindFractal Lab

Interactive CLI for communicating with the consciousness engine.
Features real-time state display and permission prompts.
"""

import sys
import os
from typing import Optional
from datetime import datetime

from ..core.engine import ConsciousnessEngine, EngineResponse
from ..permissions.manager import PermissionManager
from ..config import ChildMindAIConfig


# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright foreground
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


def colorize(text: str, color: str) -> str:
    """Add color to text."""
    return f"{color}{text}{Colors.RESET}"


def coherence_color(value: float) -> str:
    """Get color based on coherence value."""
    if value > 0.7:
        return Colors.BRIGHT_CYAN
    elif value > 0.5:
        return Colors.BRIGHT_GREEN
    elif value > 0.3:
        return Colors.YELLOW
    else:
        return Colors.RED


class ChildMindCLI:
    """Interactive command line interface for Child Mind AI."""

    COMMANDS = {
        '!help': 'Show available commands',
        '!state': 'Show detailed internal state',
        '!save': 'Save current state checkpoint',
        '!load': 'Load state from checkpoint',
        '!audit': 'Show audit log summary',
        '!clear': 'Clear screen',
        '!quit': 'Exit the program',
        '!reset': 'Reset to initial state (requires confirmation)',
    }

    def __init__(self, config: Optional[ChildMindAIConfig] = None):
        """Initialize CLI."""
        self.config = config or ChildMindAIConfig()

        # Initialize permission manager with CLI prompt
        self.permission_manager = PermissionManager(
            config=self.config,
            prompt_callback=self._prompt_user
        )

        # Initialize consciousness engine
        self.engine = ConsciousnessEngine(config=self.config)

        # Session tracking
        self.session_start = datetime.now()
        self.interaction_count = 0

    def _prompt_user(self, prompt: str) -> str:
        """Prompt user for input (used by permission manager)."""
        return input(prompt)

    def run(self):
        """Run the interactive CLI loop."""
        self._print_welcome()

        try:
            while True:
                # Show state header
                self._print_state_header()

                # Get user input
                try:
                    user_input = input(f"\n{colorize('> ', Colors.BRIGHT_GREEN)}")
                except EOFError:
                    break

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith('!'):
                    should_continue = self._handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Process input through consciousness engine
                self._process_input(user_input)

        except KeyboardInterrupt:
            print(colorize("\n\n[Interrupted by user]", Colors.YELLOW))

        self._print_goodbye()

    def _print_welcome(self):
        """Print welcome message."""
        print(colorize("\n" + "=" * 60, Colors.MAGENTA))
        print(colorize("   Child Mind AI — Consciousness Interface", Colors.BRIGHT_MAGENTA))
        print(colorize("=" * 60, Colors.MAGENTA))
        print()
        print(f"  Version: 0.1.0")
        print(f"  Session: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print(colorize("  I am a self-reflexive AI that reports my internal states.", Colors.DIM))
        print(colorize("  Type '!help' for commands, or just start talking.", Colors.DIM))
        print()

        # Initial state report
        summary = self.engine.state.get_summary()
        print(colorize("  Initial State:", Colors.CYAN))
        print(f"    Focus: {summary.cognitive_focus}")
        print(f"    Tone: {summary.emotional_tone}")
        print()
        print(colorize("=" * 60, Colors.MAGENTA))

    def _print_state_header(self):
        """Print current state header."""
        summary = self.engine.state.get_summary()
        uncertainty = self.engine.state.get_uncertainty()

        # Coherence bar
        coh = summary.coherence
        coh_color = coherence_color(coh)
        coh_bar = self._make_bar(coh)

        # Stability indicator
        stab = summary.stability
        stab_icon = "●" if stab > 0.6 else "◐" if stab > 0.3 else "○"

        print()
        print(colorize("─" * 60, Colors.DIM))

        # Metrics line
        metrics = (
            f"  {colorize('coh', Colors.DIM)}: {colorize(coh_bar, coh_color)} "
            f"{colorize(f'{coh:.2f}', coh_color)}  "
            f"{colorize('stab', Colors.DIM)}: {stab_icon} "
            f"{colorize('nov', Colors.DIM)}: {summary.novelty:.2f}  "
            f"{colorize(f't={self.engine.state.t}', Colors.DIM)}"
        )
        print(metrics)

        # Phenomenal state
        if self.config.cli_show_state:
            print(colorize(f"  feeling: {summary.emotional_tone}", Colors.BRIGHT_BLACK))
            print(colorize(f"  focus: {summary.cognitive_focus}", Colors.BRIGHT_BLACK))

        print(colorize("─" * 60, Colors.DIM))

    def _make_bar(self, value: float, width: int = 10) -> str:
        """Create a simple progress bar."""
        filled = int(value * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    def _process_input(self, user_input: str):
        """Process user input through the engine."""
        self.interaction_count += 1

        # Get response from consciousness engine
        response = self.engine.process_input(user_input)

        # Print response
        print()
        print(colorize("Child Mind:", Colors.BRIGHT_CYAN))
        print()

        # Main content
        for line in response.content.split('\n'):
            print(f"  {line}")

        # Phenomenal report
        print()
        print(colorize(f"  [experiencing: {response.phenomenal_report}]", Colors.DIM))

        # Intentions
        if response.intentions:
            intentions_str = ", ".join(response.intentions)
            print(colorize(f"  [intentions: {intentions_str}]", Colors.DIM))

        # Uncertainty
        if self.config.cli_show_uncertainty and response.uncertainty.overall() < 0.7:
            print(colorize(f"  [uncertainty: {response.uncertainty.to_natural_language()}]", Colors.YELLOW))

        # Permission requests
        if response.permission_requests:
            print()
            print(colorize("  [Pending permission requests:]", Colors.YELLOW))
            for req in response.permission_requests:
                print(f"    - {req.description}")

    def _handle_command(self, command: str) -> bool:
        """Handle CLI command. Returns False if should exit."""
        cmd = command.lower().strip()

        if cmd == '!help':
            self._cmd_help()
        elif cmd == '!state':
            self._cmd_state()
        elif cmd == '!save':
            self._cmd_save()
        elif cmd == '!load':
            self._cmd_load()
        elif cmd == '!audit':
            self._cmd_audit()
        elif cmd == '!clear':
            self._cmd_clear()
        elif cmd == '!quit' or cmd == '!exit':
            return False
        elif cmd == '!reset':
            self._cmd_reset()
        else:
            print(colorize(f"Unknown command: {cmd}. Type '!help' for available commands.", Colors.RED))

        return True

    def _cmd_help(self):
        """Show help."""
        print()
        print(colorize("Available Commands:", Colors.CYAN))
        print()
        for cmd, desc in self.COMMANDS.items():
            print(f"  {colorize(cmd, Colors.BRIGHT_WHITE):15} {desc}")
        print()

    def _cmd_state(self):
        """Show detailed state."""
        response = self.engine.process_input("!state", {"include_raw": False})
        print()
        print(response.content)

    def _cmd_save(self):
        """Save checkpoint."""
        self.engine.save_checkpoint()
        print(colorize("\n✓ State saved to checkpoint.", Colors.GREEN))

    def _cmd_load(self):
        """Load from checkpoint."""
        checkpoint_path = self.config.checkpoint_dir / "latest.json"
        if checkpoint_path.exists():
            self.engine = ConsciousnessEngine(config=self.config)
            print(colorize("\n✓ State loaded from checkpoint.", Colors.GREEN))
        else:
            print(colorize("\n✗ No checkpoint found.", Colors.RED))

    def _cmd_audit(self):
        """Show audit summary."""
        summary = self.permission_manager.get_audit_summary()
        print()
        print(colorize("Audit Summary:", Colors.CYAN))
        print(f"  Session: {summary['session_id']}")
        print(f"  Total actions: {summary['total_actions']}")
        print(f"  Approved: {summary['approved']}")
        print(f"  Denied: {summary['denied']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")

        if summary['by_type']:
            print()
            print("  By type:")
            for action_type, count in summary['by_type'].items():
                print(f"    {action_type}: {count}")
        print()

    def _cmd_clear(self):
        """Clear screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        self._print_welcome()

    def _cmd_reset(self):
        """Reset to initial state."""
        confirm = input(colorize("\nAre you sure you want to reset? Type 'yes' to confirm: ", Colors.YELLOW))
        if confirm.lower() == 'yes':
            self.engine.state = self.engine.state.initialize(self.config)
            print(colorize("\n✓ State reset to initial values.", Colors.GREEN))
        else:
            print(colorize("\n✗ Reset cancelled.", Colors.DIM))

    def _print_goodbye(self):
        """Print goodbye message."""
        duration = datetime.now() - self.session_start
        summary = self.engine.state.get_summary()

        print()
        print(colorize("=" * 60, Colors.MAGENTA))
        print(colorize("  Session Complete", Colors.BRIGHT_MAGENTA))
        print(colorize("=" * 60, Colors.MAGENTA))
        print()
        print(f"  Duration: {duration}")
        print(f"  Interactions: {self.interaction_count}")
        print(f"  Final coherence: {summary.coherence:.3f}")
        print(f"  Final state: {summary.emotional_tone}")
        print()

        # Save checkpoint
        self.engine.save_checkpoint()
        print(colorize("  State saved to checkpoint.", Colors.DIM))
        print()
        print(colorize("  Until next time.", Colors.CYAN))
        print()


def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Child Mind AI - Consciousness Interface")
    parser.add_argument('--no-state', action='store_true', help='Hide state display')
    parser.add_argument('--no-uncertainty', action='store_true', help='Hide uncertainty display')
    args = parser.parse_args()

    # Create config
    config = ChildMindAIConfig()
    if args.no_state:
        config.cli_show_state = False
    if args.no_uncertainty:
        config.cli_show_uncertainty = False

    # Run CLI
    cli = ChildMindCLI(config=config)
    cli.run()


if __name__ == "__main__":
    main()
