"""Tool system for the Basic Agent framework."""

from .base import Tool
from .file_operations import (
    CreateFileTool,
    DeleteFilesTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from .terminal import TerminalTool
from .web_search import WebSearchTool

# Tool registry for easy access
AVAILABLE_TOOLS = {
    "terminal": TerminalTool(),
    "create_file": CreateFileTool(),
    "read_file": ReadFileTool(),
    "write_file": WriteFileTool(),
    "delete_files": DeleteFilesTool(),
    "list_files": ListFilesTool(),
    "web_search": WebSearchTool(),
}

__all__ = [
    "Tool",
    "TerminalTool",
    "CreateFileTool",
    "ReadFileTool",
    "WriteFileTool",
    "DeleteFilesTool",
    "ListFilesTool",
    "WebSearchTool",
    "AVAILABLE_TOOLS",
]
