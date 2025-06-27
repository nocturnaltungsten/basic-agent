"""Memory management for the Basic Agent framework."""

import json
import re
from pathlib import Path
from typing import Any

from .exceptions import MemoryError


class MemoryManager:
    """Manages both short-term and long-term memory for the agent."""

    def __init__(self, long_term_path: str, short_term_cap: int = 10240):
        """Initialize memory manager.

        Args:
            long_term_path: Path to long-term memory JSON file
            short_term_cap: Maximum characters for short-term memory
        """
        self.long_term_path = Path(long_term_path)
        self.short_term_cap = short_term_cap
        self.short_term_memory = ""
        self.long_term_memory = self._load_long_term_memory()

    def _load_long_term_memory(self) -> dict[str, Any]:
        """Load long-term memory from JSON file.

        Returns:
            Dictionary containing long-term memory data

        Raises:
            MemoryError: If unable to load memory file
        """
        if not self.long_term_path.exists():
            return {}

        try:
            with self.long_term_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                # Handle empty files gracefully
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            raise MemoryError(f"Corrupted long-term memory file: {e}") from e
        except Exception as e:
            raise MemoryError(f"Failed to load long-term memory: {e}") from e

    def save_long_term_memory(self) -> None:
        """Save long-term memory to JSON file.

        Raises:
            MemoryError: If unable to save memory file
        """
        try:
            # Ensure directory exists
            self.long_term_path.parent.mkdir(parents=True, exist_ok=True)

            with self.long_term_path.open("w", encoding="utf-8") as f:
                json.dump(self.long_term_memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise MemoryError(f"Failed to save long-term memory: {e}") from e

    def update_memory(self, user_input: str, agent_response: str) -> None:
        """Update both short-term and long-term memory.

        Args:
            user_input: User's input message
            agent_response: Agent's response message
        """
        # Update short-term memory
        interaction = f"\nUser: {user_input}\nAgent: {agent_response}"
        self.short_term_memory += interaction

        # Truncate if too long
        if len(self.short_term_memory) > self.short_term_cap:
            self.short_term_memory = self.short_term_memory[-self.short_term_cap :]

        # Extract and store important information
        self._extract_important_info(user_input)

    def _extract_important_info(self, user_input: str) -> None:
        """Extract important information for long-term storage.

        Args:
            user_input: User's input to analyze
        """
        # Remember user's name if mentioned
        if "my name is" in user_input.lower():
            name_match = re.search(r"my name is (\\w+)", user_input.lower())
            if name_match:
                self.long_term_memory["user_name"] = name_match.group(1).title()
                self.save_long_term_memory()

    def get_memory_context(self) -> str:
        """Get formatted memory context for LLM.

        Returns:
            Formatted memory context string
        """
        context_parts = []

        if self.short_term_memory:
            context_parts.append(f"Recent conversation: {self.short_term_memory}")

        if self.long_term_memory:
            context_parts.append(f"Important information: {json.dumps(self.long_term_memory)}")

        return "\n".join(context_parts)

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term_memory = ""

    def clear_long_term(self) -> None:
        """Clear long-term memory and delete file."""
        self.long_term_memory = {}
        if self.long_term_path.exists():
            self.long_term_path.unlink()

    def get_stats(self) -> dict[str, Any]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "short_term_size": len(self.short_term_memory),
            "short_term_cap": self.short_term_cap,
            "short_term_usage_pct": round(
                len(self.short_term_memory) / self.short_term_cap * 100, 1
            ),
            "long_term_entries": len(self.long_term_memory),
            "long_term_file_exists": self.long_term_path.exists(),
        }
