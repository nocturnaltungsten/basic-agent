"""Terminal tool with safety features."""

import re
import subprocess

from ..exceptions import ToolError, UserCancellationError
from .base import Tool


class TerminalTool(Tool):
    """Execute terminal commands with safety features."""

    @property
    def name(self) -> str:
        return "terminal"

    @property
    def description(self) -> str:
        return "Execute terminal/shell commands with smart confirmation for destructive operations"

    def execute(self, command: str) -> str:
        """Execute a terminal command with safety checks.

        Args:
            command: Shell command to execute

        Returns:
            Command output or error message

        Raises:
            ToolError: If command execution fails
            UserCancellationError: If user cancels destructive operation
        """
        if self._is_destructive_command(command):
            return self._handle_destructive_command(command)

        return self._execute_command(command)

    def _is_destructive_command(self, command: str) -> bool:
        """Check if a command is potentially destructive."""
        destructive_patterns = [
            "rm ",
            "rm\\t",
            "rmdir",
            "mv ",
            "mv\\t",
            "dd ",
            "dd\\t",
            "shred",
            "wipe",
            "format",
            "mkfs",
            "> ",
            ">\\t",
            "truncate",
            "chown",
            "chmod 000",
            "chmod 777",
        ]

        command_lower = command.lower().strip()
        return any(pattern in command_lower for pattern in destructive_patterns)

    def _classify_command_risk(self, command: str) -> str:
        """Classify the risk level of a destructive command."""
        command_lower = command.lower()

        high_risk = [
            "rm -rf /",
            "rm -rf *",
            "dd if=",
            "mkfs",
            "format",
            "shred",
            "chmod 777",
            "chmod 000",
        ]

        medium_risk = ["rm -rf", "rm -r", "rmdir", "mv ", "rm "]

        for pattern in high_risk:
            if pattern in command_lower:
                return "high"

        for pattern in medium_risk:
            if pattern in command_lower:
                return "medium"

        return "low"

    def _get_user_confirmation(self, command: str, risk_level: str) -> bool:
        """Get user confirmation for potentially destructive commands."""
        risk_symbols = {"low": "⚠️", "medium": "🚨", "high": "💀"}

        print(f"\\n{risk_symbols.get(risk_level, '⚠️')} POTENTIALLY DESTRUCTIVE COMMAND DETECTED")
        print(f"Command: {command}")
        print(f"Risk Level: {risk_level.upper()}")

        if risk_level == "high":
            print("⚠️  This command could cause irreversible damage!")
        elif risk_level == "medium":
            print("⚠️  This command could modify or delete files.")

        while True:
            try:
                response = input("\\nDo you want to proceed? (yes/no/details): ").lower().strip()

                if response in ["yes", "y"]:
                    return True
                elif response in ["no", "n"]:
                    return False
                elif response in ["details", "d"]:
                    self._show_command_details(command)
                    continue
                else:
                    print("Please enter 'yes', 'no', or 'details'")
            except (KeyboardInterrupt, EOFError):
                print("\\nOperation cancelled.")
                return False

    def _show_command_details(self, command: str) -> None:
        """Show detailed breakdown of command components."""
        print("\\nCommand breakdown:")
        print(f"  Full command: {command}")

        if "rm" in command.lower():
            print("  - Contains 'rm': Will delete files/directories")
        if "mv" in command.lower():
            print("  - Contains 'mv': Will move/rename files (can overwrite)")
        if "-r" in command or "-rf" in command:
            print("  - Recursive flag: Will affect directories and all contents")
        if "-f" in command:
            print("  - Force flag: Will not prompt for individual confirmations")

    def _handle_destructive_command(self, command: str) -> str:
        """Handle destructive commands with confirmation and alternatives."""
        # Check for find+rm pattern and offer safer alternative
        if "find" in command and "rm" in command and "-exec" in command:
            return self._handle_find_rm_pattern(command)

        # Get user confirmation
        risk_level = self._classify_command_risk(command)
        if not self._get_user_confirmation(command, risk_level):
            raise UserCancellationError("Operation cancelled by user.")

        return self._execute_command(command)

    def _handle_find_rm_pattern(self, command: str) -> str:
        """Handle find...rm patterns with safer alternatives."""
        try:
            # Parse find command to extract pattern
            find_match = re.search(
                r'find\s+([^\s]+).*?-name\s+["\']([^"\']+)["\'].*?-mtime\s+\+(\d+)', command
            )

            if find_match:
                search_path = find_match.group(1)
                name_pattern = find_match.group(2)
                days = find_match.group(3)

                print("🔄 SAFER ALTERNATIVE AVAILABLE")
                print("Instead of permanently deleting files, I can move them to trash.")
                print(f"Original command: {command}")

                while True:
                    try:
                        choice = (
                            input(
                                "Choose: (s)afer trash deletion, (p)roceed with original, (c)ancel: "
                            )
                            .lower()
                            .strip()
                        )
                        if choice in ["s", "safer", "safe"]:
                            return self._safe_delete_files(search_path, name_pattern, int(days))
                        elif choice in ["p", "proceed", "original"]:
                            break  # Continue to normal confirmation
                        elif choice in ["c", "cancel"]:
                            raise UserCancellationError("Operation cancelled by user.")
                        else:
                            print("Please enter 's' for safer, 'p' to proceed, or 'c' to cancel")
                    except (KeyboardInterrupt, EOFError):
                        raise UserCancellationError("Operation cancelled by user.")
        except Exception:
            pass  # Fall through to normal confirmation

        # Continue with normal destructive command handling
        risk_level = self._classify_command_risk(command)
        if not self._get_user_confirmation(command, risk_level):
            raise UserCancellationError("Operation cancelled by user.")

        return self._execute_command(command)

    def _safe_delete_files(self, search_path: str, pattern: str, days: int) -> str:
        """Safely delete files by moving to trash."""
        from .file_operations import safe_delete_files  # Avoid circular import

        find_pattern = f"{search_path} -type f -name '{pattern}' -mtime +{days}"
        return safe_delete_files(find_pattern)

    def _execute_command(self, command: str) -> str:
        """Execute a command and return output."""
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)

            output = result.stdout.strip() if result.stdout else ""
            error = result.stderr.strip() if result.stderr else ""

            if result.returncode == 0:
                return output if output else "Command executed successfully (no output)"
            else:
                error_msg = error if error else f"Command failed (exit code {result.returncode})"
                return error_msg

        except subprocess.TimeoutExpired:
            raise ToolError(f"Command '{command}' timed out after 30 seconds")
        except Exception as e:
            raise ToolError(f"Error executing command '{command}': {e}")
