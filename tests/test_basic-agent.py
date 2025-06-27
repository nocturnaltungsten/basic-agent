"""Basic smoke tests for the agent."""

import subprocess
import sys


def test_import_main():
    """Test that main module can be imported without errors."""
    try:
        import main

        assert hasattr(main, "BasicAgent")
        assert hasattr(main, "main")
    except ImportError as e:
        assert False, f"Failed to import main module: {e}"


def test_python_syntax():
    """Test that main.py has valid Python syntax."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "main.py"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"Syntax error in main.py: {result.stderr}"
