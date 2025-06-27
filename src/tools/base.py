"""Base classes for the tool system."""

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for identification."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result as string
        """
        pass

    def __call__(self, **kwargs: Any) -> str:
        """Make tool callable."""
        return self.execute(**kwargs)
