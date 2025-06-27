"""Basic Agent - A foundational AI agent framework with LM Studio integration.

This package provides a secure, modular AI agent system with:
- LM Studio LLM integration
- Safe file operations with trash protection
- Interactive confirmation for destructive commands
- Extensible tool system
- Persistent memory management
"""

__version__ = "0.1.0"
__author__ = "Basic Agent Project"

from .agent import BasicAgent
from .config import AgentConfig, load_config
from .dev_mode import DevModeTracker, is_dev_mode_enabled
from .exceptions import AgentError, ConfigurationError, ModelError, ToolError

__all__ = [
    "BasicAgent",
    "AgentConfig",
    "load_config",
    "DevModeTracker",
    "is_dev_mode_enabled",
    "AgentError",
    "ConfigurationError",
    "ModelError",
    "ToolError",
]
