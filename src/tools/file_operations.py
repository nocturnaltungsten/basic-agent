"""File operation tools with safety features."""

import platform
import shutil
import subprocess
from pathlib import Path

from ..exceptions import ToolError
from .base import Tool


def safe_delete_files(file_pattern: str) -> str:
    """Safely delete files by moving them to trash instead of permanent deletion.

    Args:
        file_pattern: Find pattern for files to delete

    Returns:
        Status message about deletion

    Raises:
        ToolError: If deletion fails
    """
    try:
        # First, find the files that match the pattern
        find_cmd = f"find {file_pattern} 2>/dev/null"
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return f"No files found matching pattern: {file_pattern}"

        files = result.stdout.strip().split("\\n") if result.stdout.strip() else []
        files = [f for f in files if f.strip()]  # Remove empty lines

        if not files:
            return f"No files found matching pattern: {file_pattern}"

        # Move files to trash based on OS
        moved_count = 0

        if platform.system() == "Darwin":  # macOS
            moved_count = _move_to_trash_macos(files)
        else:  # Linux/other
            moved_count = _move_to_trash_linux(files)

        return f"Moved {moved_count} file(s) to Trash"

    except Exception as e:
        raise ToolError(f"Error moving files to trash: {e}")


def _move_to_trash_macos(files: list[str]) -> int:
    """Move files to trash on macOS."""
    # Check if trash command exists
    check_trash = subprocess.run("which trash", shell=True, capture_output=True, text=True)

    if check_trash.returncode == 0:
        # Use trash command
        for file_path in files:
            subprocess.run(f"trash '{file_path}'", shell=True, capture_output=True, text=True)
        return len(files)
    else:
        # Use native macOS trash
        moved_count = 0
        for file_path in files:
            mv_cmd = f"mv '{file_path}' ~/.Trash/"
            mv_result = subprocess.run(mv_cmd, shell=True, capture_output=True, text=True)
            if mv_result.returncode == 0:
                moved_count += 1
        return moved_count


def _move_to_trash_linux(files: list[str]) -> int:
    """Move files to trash on Linux."""
    trash_dir = Path.home() / ".local" / "share" / "Trash" / "files"
    trash_dir.mkdir(parents=True, exist_ok=True)

    moved_count = 0
    for file_path in files:
        try:
            filename = Path(file_path).name
            target_path = trash_dir / filename

            # Handle name conflicts
            counter = 1
            while target_path.exists():
                stem = Path(filename).stem
                suffix = Path(filename).suffix
                target_path = trash_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.move(file_path, target_path)
            moved_count += 1
        except Exception:
            continue  # Skip files that can't be moved

    return moved_count


class CreateFileTool(Tool):
    """Create new files with content."""

    @property
    def name(self) -> str:
        return "create_file"

    @property
    def description(self) -> str:
        return "Create new files with content"

    def execute(self, path: str, content: str = "") -> str:
        """Create a new file with the given content.

        Args:
            path: File path to create
            content: Content to write to file

        Returns:
            Success or error message
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with file_path.open("w", encoding="utf-8") as f:
                f.write(content)

            return f"File {path} created successfully"
        except PermissionError:
            raise ToolError(f"Permission denied creating {path}")
        except Exception as e:
            raise ToolError(f"Error creating file {path}: {e}")


class ReadFileTool(Tool):
    """Read the contents of existing files."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read contents of existing files"

    def execute(self, path: str) -> str:
        """Read the contents of a file.

        Args:
            path: File path to read

        Returns:
            File contents or error message
        """
        try:
            file_path = Path(path)

            if not file_path.exists():
                raise ToolError(f"File {path} not found")

            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()

            return content if content else "File is empty"
        except PermissionError:
            raise ToolError(f"Permission denied reading {path}")
        except UnicodeDecodeError:
            raise ToolError(f"Cannot read {path}: file appears to be binary")
        except Exception as e:
            raise ToolError(f"Error reading file {path}: {e}")


class WriteFileTool(Tool):
    """Write content to files (overwrites existing content)."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write/overwrite file contents"

    def execute(self, path: str, content: str) -> str:
        """Write content to a file (overwrites existing content).

        Args:
            path: File path to write
            content: Content to write to file

        Returns:
            Success or error message
        """
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with file_path.open("w", encoding="utf-8") as f:
                f.write(content)

            return f"Content written to {path} successfully"
        except PermissionError:
            raise ToolError(f"Permission denied writing to {path}")
        except Exception as e:
            raise ToolError(f"Error writing to file {path}: {e}")


class DeleteFilesTool(Tool):
    """Safely delete files by moving to trash."""

    @property
    def name(self) -> str:
        return "delete_files"

    @property
    def description(self) -> str:
        return "Safely delete files by moving to trash (not permanent deletion)"

    def execute(self, path_pattern: str, older_than_days: int | None = None) -> str:
        """Safely delete files by moving them to trash.

        Args:
            path_pattern: Path pattern for files to delete
            older_than_days: Only delete files older than this many days

        Returns:
            Status message about deletion
        """
        try:
            # Build find command based on parameters
            if older_than_days:
                find_pattern = f"{path_pattern} -type f -mtime +{older_than_days}"
            else:
                find_pattern = f"{path_pattern} -type f"

            return safe_delete_files(find_pattern)
        except Exception as e:
            raise ToolError(f"Error deleting files: {e}")


class ListFilesTool(Tool):
    """List files in directories with optional filtering."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List files in directories with optional filtering"

    def execute(self, path: str, pattern: str | None = None, show_hidden: bool = False) -> str:
        """List files in a directory with optional filtering.

        Args:
            path: Directory path to list
            pattern: Optional file pattern to filter by
            show_hidden: Whether to show hidden files

        Returns:
            File listing or error message
        """
        try:
            if pattern:
                cmd = f"find '{path}' -name '{pattern}' -type f"
            else:
                if show_hidden:
                    cmd = f"ls -la '{path}'"
                else:
                    cmd = f"ls -l '{path}'"

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                output = result.stdout.strip()
                return output if output else "No files found"
            else:
                raise ToolError(f"Error listing files: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            raise ToolError("Timeout while listing files")
        except Exception as e:
            raise ToolError(f"Error listing files: {e}")
