"""Configuration management for the Basic Agent framework."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


@dataclass
class ToolConfig:
    """Configuration for a single tool."""

    name: str
    description: str


@dataclass
class AgentConfig:
    """Agent configuration with validation and type safety."""

    # Model configuration
    lm_studio_model: str | None = None

    # Memory configuration
    memory_short_term_cap: int = 10240
    memory_long_term_path: str = "long_term_memory.json"

    # System configuration
    system_prompt: str = (
        "You are a helpful AI agent with access to tools for file operations, "
        "terminal commands, and web search. For safety, I ask for user confirmation "
        "before executing potentially destructive commands, and offer safer alternatives "
        "when available (like moving files to trash instead of permanent deletion). "
        "Use these tools to help users accomplish their tasks efficiently and safely."
    )

    # Tool configuration
    tools: list[ToolConfig] = field(
        default_factory=lambda: [
            ToolConfig(
                "terminal",
                "Execute terminal/shell commands with smart confirmation for destructive operations",
            ),
            ToolConfig("create_file", "Create new files with content"),
            ToolConfig("read_file", "Read contents of existing files"),
            ToolConfig("write_file", "Write/overwrite file contents"),
            ToolConfig(
                "delete_files", "Safely delete files by moving to trash (not permanent deletion)"
            ),
            ToolConfig("list_files", "List files in directories with optional filtering"),
            ToolConfig("web_search", "Search the web for information"),
        ]
    )

    @classmethod
    def from_file(cls, config_path: str = "config.json") -> "AgentConfig":
        """Load configuration from a JSON file."""
        try:
            path = Path(config_path)
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")

            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return cls.from_dict(data)

        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}") from e

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentConfig":
        """Create configuration from a dictionary."""
        try:
            # Convert tools list to ToolConfig objects
            tools_data = data.get("tools", [])
            tools = [ToolConfig(**tool) for tool in tools_data]

            # Create config with tools
            config_data = data.copy()
            config_data["tools"] = tools

            return cls(**config_data)

        except TypeError as e:
            raise ConfigurationError(f"Invalid configuration structure: {e}") from e

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "lm_studio_model": self.lm_studio_model,
            "memory_short_term_cap": self.memory_short_term_cap,
            "memory_long_term_path": self.memory_long_term_path,
            "system_prompt": self.system_prompt,
            "tools": [{"name": tool.name, "description": tool.description} for tool in self.tools],
        }

    def save_to_file(self, config_path: str = "config.json") -> None:
        """Save configuration to a JSON file."""
        try:
            path = Path(config_path)
            with path.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}") from e

    def validate(self) -> None:
        """Validate the configuration."""
        if self.memory_short_term_cap <= 0:
            raise ConfigurationError("memory_short_term_cap must be positive")

        if not self.memory_long_term_path:
            raise ConfigurationError("memory_long_term_path cannot be empty")

        if not self.system_prompt.strip():
            raise ConfigurationError("system_prompt cannot be empty")

        if not self.tools:
            raise ConfigurationError("At least one tool must be configured")

        # Validate tool names are unique
        tool_names = [tool.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ConfigurationError("Tool names must be unique")


def load_config(config_path: str = "config.json") -> AgentConfig:
    """Load and validate agent configuration."""
    config = AgentConfig.from_file(config_path)
    config.validate()
    return config
