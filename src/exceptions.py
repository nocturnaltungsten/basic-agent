"""Exception classes for the Basic Agent framework."""


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class ConfigurationError(AgentError):
    """Raised when there are configuration-related issues."""

    pass


class ModelError(AgentError):
    """Raised when there are model-related issues."""

    pass


class ToolError(AgentError):
    """Raised when there are tool execution issues."""

    pass


class MemoryError(AgentError):
    """Raised when there are memory management issues."""

    pass


class UserCancellationError(AgentError):
    """Raised when the user cancels an operation."""

    pass
