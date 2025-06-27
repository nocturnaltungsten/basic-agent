"""Test the new modular architecture."""

from unittest.mock import patch

import pytest
from src.config import AgentConfig, ToolConfig
from src.exceptions import ConfigurationError
from src.memory import MemoryManager


def test_tool_config():
    """Test ToolConfig dataclass."""
    tool = ToolConfig("test_tool", "A test tool")
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"


def test_agent_config_defaults():
    """Test AgentConfig with default values."""
    config = AgentConfig()
    assert config.lm_studio_model is None
    assert config.memory_short_term_cap == 10240
    assert config.memory_long_term_path == "long_term_memory.json"
    assert len(config.tools) == 7  # Default tools


def test_agent_config_validation():
    """Test AgentConfig validation."""
    config = AgentConfig()
    config.validate()  # Should not raise

    # Test invalid memory cap
    config.memory_short_term_cap = -1
    with pytest.raises(ConfigurationError):
        config.validate()

    # Test empty path
    config.memory_short_term_cap = 1000
    config.memory_long_term_path = ""
    with pytest.raises(ConfigurationError):
        config.validate()


def test_memory_manager_initialization():
    """Test MemoryManager initialization."""
    with patch("src.memory.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        memory = MemoryManager("test_memory.json", 1000)
        assert memory.short_term_cap == 1000
        assert memory.short_term_memory == ""
        assert memory.long_term_memory == {}


def test_memory_manager_update():
    """Test memory update functionality."""
    with patch("src.memory.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        memory = MemoryManager("test_memory.json", 50)  # Small cap for testing

        # Test normal update
        memory.update_memory("Hello", "Hi there")
        assert "User: Hello" in memory.short_term_memory
        assert "Agent: Hi there" in memory.short_term_memory

        # Test truncation
        long_input = "A" * 100
        long_response = "B" * 100
        memory.update_memory(long_input, long_response)

        # Should be truncated to last 50 characters
        assert len(memory.short_term_memory) == 50


def test_memory_context():
    """Test memory context generation."""
    with patch("src.memory.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        memory = MemoryManager("test_memory.json")

        # Empty memory
        assert memory.get_memory_context() == ""

        # With short-term memory
        memory.short_term_memory = "Test conversation"
        context = memory.get_memory_context()
        assert "Recent conversation: Test conversation" in context

        # With long-term memory
        memory.long_term_memory = {"user_name": "Alice"}
        context = memory.get_memory_context()
        assert "Important information:" in context
        assert "Alice" in context


def test_memory_stats():
    """Test memory statistics."""
    with patch("src.memory.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        memory = MemoryManager("test_memory.json", 1000)
        memory.short_term_memory = "Test" * 50  # 200 characters

        stats = memory.get_stats()
        assert stats["short_term_size"] == 200
        assert stats["short_term_cap"] == 1000
        assert stats["short_term_usage_pct"] == 20.0
        assert stats["long_term_entries"] == 0
