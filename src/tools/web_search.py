"""Web search tool (placeholder implementation)."""

from .base import Tool


class WebSearchTool(Tool):
    """Search the web for information (placeholder for future implementation)."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for information"

    def execute(self, query: str) -> str:
        """Search the web for information.

        Args:
            query: Search query

        Returns:
            Search results (placeholder)
        """
        return f"Web search functionality not yet implemented. Your query was: '{query}'"
